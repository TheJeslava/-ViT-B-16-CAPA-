import copy
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ViT_B_16_Weights, vit_b_16

from modeling import FrozenInitCrossAttention


@dataclass
class CVPTExpertAdapterOptions:
    method: str = "cvpt_expert_adapter"
    pretrained: bool = True
    prompt_tokens: int = 8
    prompt_init_std: float = 0.02
    cross_attn_trainable: bool = False
    cvpt_residual_scale: float = 1.0
    num_experts: int = 6
    top_k: int = 2
    adapter_bottleneck: int = 64
    router_hidden_dim: int = 16
    router_noise_std: float = 0.01
    entropy_alpha: float = 0.01
    adapter_scale_init: float = 1e-3


def options_from_config(
    config: Dict,
    method: Optional[str] = None,
    pretrained: Optional[bool] = None,
) -> CVPTExpertAdapterOptions:
    cfg = copy.deepcopy(config.get("cvpt_expert_adapter", {}))
    if method and method in config:
        method_cfg = config.get(method, {})
        for key, value in method_cfg.items():
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                cfg[key].update(value)
            else:
                cfg[key] = value
    cvpt = cfg.get("cvpt", {})
    moe = cfg.get("moe_adapter", {})
    opts = CVPTExpertAdapterOptions()
    opts.method = method or "cvpt_expert_adapter"
    opts.pretrained = bool(config.get("project", {}).get("pretrained_weights", True))
    if pretrained is not None:
        opts.pretrained = pretrained
    opts.prompt_tokens = int(cvpt.get("prompt_tokens", opts.prompt_tokens))
    opts.prompt_init_std = float(cvpt.get("prompt_init_std", opts.prompt_init_std))
    opts.cross_attn_trainable = bool(cvpt.get("cross_attn_trainable", opts.cross_attn_trainable))
    opts.cvpt_residual_scale = float(cvpt.get("residual_scale", opts.cvpt_residual_scale))
    opts.num_experts = int(moe.get("num_experts", opts.num_experts))
    opts.top_k = int(moe.get("top_k", opts.top_k))
    opts.adapter_bottleneck = int(moe.get("bottleneck", opts.adapter_bottleneck))
    opts.router_hidden_dim = int(moe.get("router_hidden_dim", opts.router_hidden_dim))
    opts.router_noise_std = float(moe.get("router_noise_std", opts.router_noise_std))
    opts.entropy_alpha = float(moe.get("entropy_alpha", opts.entropy_alpha))
    opts.adapter_scale_init = float(moe.get("adapter_scale_init", opts.adapter_scale_init))
    return opts


class LayerPromptBank(nn.Module):
    def __init__(self, num_layers: int, tokens: int, dim: int, init_std: float) -> None:
        super().__init__()
        self.prompts = nn.Parameter(torch.empty(num_layers, tokens, dim))
        nn.init.normal_(self.prompts, mean=0.0, std=init_std)

    def forward(self, layer_idx: int, batch_size: int) -> torch.Tensor:
        return self.prompts[layer_idx].unsqueeze(0).expand(batch_size, -1, -1)


class AdaptMLPExpert(nn.Module):
    def __init__(self, dim: int, bottleneck: int, scale_init: float) -> None:
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim)
        self.scale = nn.Parameter(torch.ones(1) * scale_init)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * self.up(self.act(self.down(x)))


class TopKExpertAdapter(nn.Module):
    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int,
        bottleneck: int,
        router_hidden_dim: int,
        noise_std: float,
        scale_init: float,
    ) -> None:
        super().__init__()
        if top_k < 1 or top_k > num_experts:
            raise ValueError(f"top_k must be in [1, num_experts], got top_k={top_k}, num_experts={num_experts}")
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.router = nn.Sequential(
            nn.Linear(dim, router_hidden_dim),
            nn.GELU(),
            nn.Linear(router_hidden_dim, num_experts),
        )
        self.experts = nn.ModuleList(
            [AdaptMLPExpert(dim, bottleneck, scale_init=scale_init) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled = x[:, 1:, :].mean(dim=1)
        logits = self.router(pooled)
        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std
        full_gates = F.softmax(logits, dim=-1)
        top_values, top_indices = torch.topk(full_gates, k=self.top_k, dim=-1)
        sparse_gates = torch.zeros_like(full_gates)
        sparse_gates.scatter_(dim=-1, index=top_indices, src=top_values)
        sparse_gates = sparse_gates / sparse_gates.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        adapter_out = torch.einsum("be,bend->bnd", sparse_gates, expert_outputs)
        return adapter_out, sparse_gates


class CVPTExpertAdapterViT(nn.Module):
    """CVPT branch plus Top-k expert AdaptFormer branch inside every ViT block."""

    def __init__(self, num_classes: int, options: CVPTExpertAdapterOptions) -> None:
        super().__init__()
        self.options = copy.deepcopy(options)
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if options.pretrained else None
        self.vit = vit_b_16(weights=weights)
        hidden_dim = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(hidden_dim, num_classes)
        self.hidden_dim = hidden_dim
        self.num_layers = len(self.vit.encoder.layers)

        self.cvpt_norms = nn.ModuleList([nn.LayerNorm(hidden_dim, eps=1e-6) for _ in range(self.num_layers)])
        self.prompts = LayerPromptBank(self.num_layers, options.prompt_tokens, hidden_dim, options.prompt_init_std)
        self.cross_attn = nn.ModuleList(
            [
                FrozenInitCrossAttention(block.self_attention, trainable=options.cross_attn_trainable)
                for block in self.vit.encoder.layers
            ]
        )
        self.expert_adapters = nn.ModuleList(
            [
                TopKExpertAdapter(
                    dim=hidden_dim,
                    num_experts=options.num_experts,
                    top_k=options.top_k,
                    bottleneck=options.adapter_bottleneck,
                    router_hidden_dim=options.router_hidden_dim,
                    noise_std=options.router_noise_std,
                    scale_init=options.adapter_scale_init,
                )
                for _ in range(self.num_layers)
            ]
        )
        self._configure_trainable()

    def _configure_trainable(self) -> None:
        for param in self.vit.parameters():
            param.requires_grad = False
        for param in self.vit.heads.parameters():
            param.requires_grad = True
        for module in (self.cvpt_norms, self.prompts, self.expert_adapters):
            for param in module.parameters():
                param.requires_grad = True
        if self.options.cross_attn_trainable:
            for param in self.cross_attn.parameters():
                param.requires_grad = True

    def forward(self, images: torch.Tensor, return_aux: bool = False):
        x = self._tokens_from_images(images)
        gates: List[torch.Tensor] = []
        for layer_idx, block in enumerate(self.vit.encoder.layers):
            x = self._forward_block(block, x, layer_idx, gates)
        x = self.vit.encoder.ln(x)
        logits = self.vit.heads(x[:, 0])
        if not return_aux:
            return logits
        aux = self._make_aux(gates, logits.device)
        return logits, aux

    def _tokens_from_images(self, images: torch.Tensor) -> torch.Tensor:
        x = self.vit._process_input(images)
        batch = x.shape[0]
        cls = self.vit.class_token.expand(batch, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.vit.encoder.pos_embedding
        return self.vit.encoder.dropout(x)

    def _forward_block(
        self,
        block: nn.Module,
        x: torch.Tensor,
        layer_idx: int,
        gates: List[torch.Tensor],
    ) -> torch.Tensor:
        attn_in = block.ln_1(x)
        attn_out, _ = block.self_attention(attn_in, attn_in, attn_in, need_weights=False)
        x = x + block.dropout(attn_out)

        cvpt_in = self.cvpt_norms[layer_idx](x)
        prompt = self.prompts(layer_idx, x.shape[0])
        x = x + self.options.cvpt_residual_scale * self.cross_attn[layer_idx](cvpt_in, prompt)

        mlp_in = block.ln_2(x)
        native_mlp = block.mlp(mlp_in)
        expert_out, gate = self.expert_adapters[layer_idx](mlp_in)
        gates.append(gate)
        x = x + native_mlp + expert_out
        return x

    def _make_aux(self, gates: List[torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
        zero = torch.zeros((), device=device)
        if not gates:
            return {
                "routing_loss": zero,
                "mean_normalized_entropy": zero,
                "effective_experts": zero,
                "gates": None,
            }
        gate_tensor = torch.stack(gates, dim=1)
        entropy = -(gate_tensor * gate_tensor.clamp_min(1e-8).log()).sum(dim=-1)
        norm_entropy = entropy / math.log(gate_tensor.shape[-1])
        routing_loss = zero
        if self.options.entropy_alpha > 0:
            routing_loss = routing_loss - self.options.entropy_alpha * norm_entropy.mean()
        return {
            "routing_loss": routing_loss,
            "mean_normalized_entropy": norm_entropy.mean(),
            "effective_experts": entropy.exp().mean(),
            "gates": gate_tensor,
        }


class ExpertAdapterPoolViT(nn.Module):
    """Top-k expert AdaptFormer branch without CVPT prompts or cross-attention."""

    def __init__(self, num_classes: int, options: CVPTExpertAdapterOptions) -> None:
        super().__init__()
        self.options = copy.deepcopy(options)
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if options.pretrained else None
        self.vit = vit_b_16(weights=weights)
        hidden_dim = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(hidden_dim, num_classes)
        self.hidden_dim = hidden_dim
        self.num_layers = len(self.vit.encoder.layers)

        self.expert_adapters = nn.ModuleList(
            [
                TopKExpertAdapter(
                    dim=hidden_dim,
                    num_experts=options.num_experts,
                    top_k=options.top_k,
                    bottleneck=options.adapter_bottleneck,
                    router_hidden_dim=options.router_hidden_dim,
                    noise_std=options.router_noise_std,
                    scale_init=options.adapter_scale_init,
                )
                for _ in range(self.num_layers)
            ]
        )
        self._configure_trainable()

    def _configure_trainable(self) -> None:
        for param in self.vit.parameters():
            param.requires_grad = False
        for param in self.vit.heads.parameters():
            param.requires_grad = True
        for param in self.expert_adapters.parameters():
            param.requires_grad = True

    def forward(self, images: torch.Tensor, return_aux: bool = False):
        x = self._tokens_from_images(images)
        gates: List[torch.Tensor] = []
        for layer_idx, block in enumerate(self.vit.encoder.layers):
            x = self._forward_block(block, x, layer_idx, gates)
        x = self.vit.encoder.ln(x)
        logits = self.vit.heads(x[:, 0])
        if not return_aux:
            return logits
        aux = self._make_aux(gates, logits.device)
        return logits, aux

    def _tokens_from_images(self, images: torch.Tensor) -> torch.Tensor:
        x = self.vit._process_input(images)
        batch = x.shape[0]
        cls = self.vit.class_token.expand(batch, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.vit.encoder.pos_embedding
        return self.vit.encoder.dropout(x)

    def _forward_block(
        self,
        block: nn.Module,
        x: torch.Tensor,
        layer_idx: int,
        gates: List[torch.Tensor],
    ) -> torch.Tensor:
        attn_in = block.ln_1(x)
        attn_out, _ = block.self_attention(attn_in, attn_in, attn_in, need_weights=False)
        x = x + block.dropout(attn_out)

        mlp_in = block.ln_2(x)
        native_mlp = block.mlp(mlp_in)
        expert_out, gate = self.expert_adapters[layer_idx](mlp_in)
        gates.append(gate)
        x = x + native_mlp + expert_out
        return x

    def _make_aux(self, gates: List[torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
        zero = torch.zeros((), device=device)
        if not gates:
            return {
                "routing_loss": zero,
                "mean_normalized_entropy": zero,
                "effective_experts": zero,
                "gates": None,
            }
        gate_tensor = torch.stack(gates, dim=1)
        entropy = -(gate_tensor * gate_tensor.clamp_min(1e-8).log()).sum(dim=-1)
        norm_entropy = entropy / math.log(gate_tensor.shape[-1])
        routing_loss = zero
        if self.options.entropy_alpha > 0:
            routing_loss = routing_loss - self.options.entropy_alpha * norm_entropy.mean()
        return {
            "routing_loss": routing_loss,
            "mean_normalized_entropy": norm_entropy.mean(),
            "effective_experts": entropy.exp().mean(),
            "gates": gate_tensor,
        }


def build_cvpt_expert_adapter_model(
    num_classes: int,
    config: Dict,
    method: Optional[str] = None,
    pretrained: Optional[bool] = None,
) -> CVPTExpertAdapterViT:
    options = options_from_config(config, method=method, pretrained=pretrained)
    return CVPTExpertAdapterViT(num_classes=num_classes, options=options)


def build_expert_adapter_pool_model(
    num_classes: int,
    config: Dict,
    method: Optional[str] = None,
    pretrained: Optional[bool] = None,
) -> ExpertAdapterPoolViT:
    options = options_from_config(config, method=method, pretrained=pretrained)
    return ExpertAdapterPoolViT(num_classes=num_classes, options=options)
