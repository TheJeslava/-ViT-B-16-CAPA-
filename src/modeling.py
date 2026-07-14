import copy
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ViT_B_16_Weights, vit_b_16


SUPPORTED_METHODS = {
    "linear_probe",
    "full_finetune",
    "vpt",
    "cvpt",
    "shared_cvpt",
    "routed_cvpt",
    "para_cvpt",
    "adaptformer",
    "ssf",
    "lora",
    "cvpt_expert_adapter",
    "expert_adapter_pool",
}


@dataclass
class ModelOptions:
    method: str = "para_cvpt"
    pretrained: bool = True
    prompt_tokens: int = 8
    num_experts: int = 12
    router_hidden_dim: int = 16
    router_noise_std: float = 0.01
    expert_dropout: float = 0.10
    entropy_alpha: float = 0.02
    entropy_enabled: bool = True
    load_kl_enabled: bool = False
    load_kl_beta: float = 0.01
    prompt_init_std: float = 0.02
    expert_scope: str = "global_12_layers"
    cross_attn_trainable: bool = False
    residual_scale: float = 1.0
    adapter_bottleneck: int = 64
    lora_rank: int = 8
    lora_alpha: float = 16.0


def options_from_config(config: Dict, method: Optional[str] = None, pretrained: Optional[bool] = None) -> ModelOptions:
    para = config.get("para_cvpt", {})
    router = para.get("router", {})
    bank = para.get("expert_bank", {})
    collapse = para.get("collapse_guard", {})
    entropy = collapse.get("entropy_loss", {})
    load_kl = collapse.get("batch_load_kl", {})
    baselines = config.get("baselines", {})

    opts = ModelOptions()
    opts.method = method or config.get("method", "para_cvpt")
    opts.pretrained = config.get("project", {}).get("pretrained_weights", True)
    if pretrained is not None:
        opts.pretrained = pretrained

    opts.prompt_tokens = int(bank.get("tokens_per_expert_t", para.get("prompt_tokens", opts.prompt_tokens)))
    opts.num_experts = int(bank.get("num_experts_m", opts.num_experts))
    opts.prompt_init_std = float(bank.get("init_std", opts.prompt_init_std))
    opts.expert_scope = str(bank.get("scope", opts.expert_scope))
    opts.router_hidden_dim = int(router.get("hidden_dim", opts.router_hidden_dim))
    opts.router_noise_std = float(router.get("train_noise_std", opts.router_noise_std))
    opts.expert_dropout = float(router.get("expert_dropout", opts.expert_dropout))
    opts.cross_attn_trainable = not bool(para.get("cvpt_cross_attention", {}).get("freeze_qkv_o", True))
    opts.residual_scale = float(para.get("cvpt_cross_attention", {}).get("residual_scale", {}).get("value", 1.0))
    opts.entropy_enabled = bool(entropy.get("enabled", opts.entropy_enabled))
    opts.entropy_alpha = float(entropy.get("alpha", opts.entropy_alpha))
    opts.load_kl_enabled = bool(load_kl.get("enabled", opts.load_kl_enabled))
    opts.load_kl_beta = float(load_kl.get("beta", opts.load_kl_beta))
    opts.adapter_bottleneck = int(baselines.get("adaptformer", {}).get("bottleneck", opts.adapter_bottleneck))
    opts.lora_rank = int(baselines.get("lora", {}).get("rank", opts.lora_rank))
    opts.lora_alpha = float(baselines.get("lora", {}).get("alpha", opts.lora_alpha))
    return opts


class StaticPromptBank(nn.Module):
    """Prompt bank for CVPT variants without dynamic routing."""

    def __init__(self, num_layers: int, tokens: int, dim: int, shared: bool, init_std: float) -> None:
        super().__init__()
        self.shared = shared
        shape = (1, tokens, dim) if shared else (num_layers, tokens, dim)
        self.prompts = nn.Parameter(torch.empty(shape))
        nn.init.normal_(self.prompts, mean=0.0, std=init_std)

    def forward(self, layer_idx: int, batch_size: int) -> torch.Tensor:
        if self.shared:
            prompt = self.prompts[0]
        else:
            prompt = self.prompts[layer_idx]
        return prompt.unsqueeze(0).expand(batch_size, -1, -1)


class RoutedPromptBank(nn.Module):
    """ParaX-style shared expert center with one lightweight router per layer."""

    def __init__(
        self,
        num_layers: int,
        dim: int,
        num_experts: int,
        tokens_per_expert: int,
        router_hidden_dim: int,
        scope: str,
        noise_std: float,
        expert_dropout: float,
        init_std: float,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.noise_std = noise_std
        self.expert_dropout = expert_dropout
        self.layer_to_group = self._make_layer_to_group(num_layers, scope)
        num_groups = max(self.layer_to_group) + 1

        self.expert_banks = nn.Parameter(torch.empty(num_groups, num_experts, tokens_per_expert, dim))
        nn.init.normal_(self.expert_banks, mean=0.0, std=init_std)
        self.routers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, router_hidden_dim),
                    nn.GELU(),
                    nn.Linear(router_hidden_dim, num_experts),
                )
                for _ in range(num_layers)
            ]
        )

    @staticmethod
    def _make_layer_to_group(num_layers: int, scope: str) -> List[int]:
        if scope in {"global", "global_12_layers"}:
            return [0 for _ in range(num_layers)]
        if scope in {"layer", "layer_wise"}:
            return list(range(num_layers))
        if scope.startswith("group_") and scope.endswith("_layers"):
            group_size = int(scope.split("_")[1])
            return [idx // group_size for idx in range(num_layers)]
        raise ValueError(f"Unsupported expert scope: {scope}")

    def forward(self, x: torch.Tensor, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled = x[:, 1:, :].mean(dim=1)
        logits = self.routers[layer_idx](pooled)
        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std
        gates = F.softmax(logits, dim=-1)
        gates = self._apply_expert_dropout(gates)
        bank = self.expert_banks[self.layer_to_group[layer_idx]]
        prompt = torch.einsum("bm,mtd->btd", gates, bank)
        return prompt, gates

    def _apply_expert_dropout(self, gates: torch.Tensor) -> torch.Tensor:
        if not self.training or self.expert_dropout <= 0:
            return gates
        keep = torch.rand_like(gates) > self.expert_dropout
        empty = keep.sum(dim=-1, keepdim=True) == 0
        if empty.any():
            keep = torch.where(empty, torch.ones_like(keep), keep)
        gates = gates * keep.to(gates.dtype)
        return gates / gates.sum(dim=-1, keepdim=True).clamp_min(1e-6)


class FrozenInitCrossAttention(nn.Module):
    """CVPT cross-attention initialized from a pretrained self-attention layer."""

    def __init__(self, source_attn: nn.MultiheadAttention, trainable: bool) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=source_attn.embed_dim,
            num_heads=source_attn.num_heads,
            dropout=source_attn.dropout,
            batch_first=True,
        )
        with torch.no_grad():
            self.attn.in_proj_weight.copy_(source_attn.in_proj_weight)
            if source_attn.in_proj_bias is not None:
                self.attn.in_proj_bias.copy_(source_attn.in_proj_bias)
            self.attn.out_proj.weight.copy_(source_attn.out_proj.weight)
            self.attn.out_proj.bias.copy_(source_attn.out_proj.bias)
        for param in self.attn.parameters():
            param.requires_grad = trainable

    def forward(self, x: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(x, prompt, prompt, need_weights=False)
        return out


class AdaptFormerAdapter(nn.Module):
    def __init__(self, dim: int, bottleneck: int) -> None:
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim)
        self.scale = nn.Parameter(torch.ones(1) * 1e-3)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * self.up(self.act(self.down(x)))


class SSFLayer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma + self.beta


class LoRAQKV(nn.Module):
    def __init__(self, dim: int, num_heads: int, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.q_down = nn.Linear(dim, rank, bias=False)
        self.q_up = nn.Linear(rank, dim, bias=False)
        self.v_down = nn.Linear(dim, rank, bias=False)
        self.v_up = nn.Linear(rank, dim, bias=False)
        nn.init.kaiming_uniform_(self.q_down.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.v_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.q_up.weight)
        nn.init.zeros_(self.v_up.weight)

    def forward(self, attn: nn.MultiheadAttention, x: torch.Tensor) -> torch.Tensor:
        qkv = F.linear(x, attn.in_proj_weight, attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_up(self.q_down(x)) * self.scaling
        v = v + self.v_up(self.v_down(x)) * self.scaling

        batch, tokens, _ = q.shape
        q = q.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        probs = scores.softmax(dim=-1)
        probs = self.dropout(probs)
        out = torch.matmul(probs, v)
        out = out.transpose(1, 2).reshape(batch, tokens, self.dim)
        return F.linear(out, attn.out_proj.weight, attn.out_proj.bias)


class PEFTVisionTransformer(nn.Module):
    def __init__(self, num_classes: int, options: ModelOptions) -> None:
        super().__init__()
        if options.method not in SUPPORTED_METHODS:
            raise ValueError(f"Unsupported method '{options.method}'. Choose from {sorted(SUPPORTED_METHODS)}")
        self.options = copy.deepcopy(options)
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if options.pretrained else None
        self.vit = vit_b_16(weights=weights)
        hidden_dim = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(hidden_dim, num_classes)

        self.hidden_dim = hidden_dim
        self.num_layers = len(self.vit.encoder.layers)
        first_attn = self.vit.encoder.layers[0].self_attention
        self.num_heads = first_attn.num_heads
        self.routing_layers = {"para_cvpt", "routed_cvpt"}
        self.cross_methods = {"cvpt", "shared_cvpt", "para_cvpt", "routed_cvpt"}

        if options.method == "vpt":
            self.vpt_prompts = nn.Parameter(torch.empty(options.prompt_tokens, hidden_dim))
            nn.init.normal_(self.vpt_prompts, mean=0.0, std=options.prompt_init_std)
        if options.method in {"cvpt", "shared_cvpt"}:
            self.static_prompts = StaticPromptBank(
                self.num_layers,
                options.prompt_tokens,
                hidden_dim,
                shared=options.method == "shared_cvpt",
                init_std=options.prompt_init_std,
            )
        if options.method in self.routing_layers:
            scope = "layer_wise" if options.method == "routed_cvpt" else options.expert_scope
            self.routed_prompts = RoutedPromptBank(
                self.num_layers,
                hidden_dim,
                options.num_experts,
                options.prompt_tokens,
                options.router_hidden_dim,
                scope,
                options.router_noise_std,
                options.expert_dropout,
                options.prompt_init_std,
            )
        if options.method in self.cross_methods:
            self.cross_attn = nn.ModuleList(
                [
                    FrozenInitCrossAttention(block.self_attention, trainable=options.cross_attn_trainable)
                    for block in self.vit.encoder.layers
                ]
            )
        if options.method == "adaptformer":
            self.adapters = nn.ModuleList(
                [AdaptFormerAdapter(hidden_dim, options.adapter_bottleneck) for _ in range(self.num_layers)]
            )
        if options.method == "ssf":
            self.ssf_attn = nn.ModuleList([SSFLayer(hidden_dim) for _ in range(self.num_layers)])
            self.ssf_mlp = nn.ModuleList([SSFLayer(hidden_dim) for _ in range(self.num_layers)])
        if options.method == "lora":
            self.lora = nn.ModuleList(
                [
                    LoRAQKV(hidden_dim, self.num_heads, options.lora_rank, options.lora_alpha, block.self_attention.dropout)
                    for block in self.vit.encoder.layers
                ]
            )

        self._configure_trainable()

    def _configure_trainable(self) -> None:
        if self.options.method == "full_finetune":
            for param in self.parameters():
                param.requires_grad = True
            return

        for param in self.vit.parameters():
            param.requires_grad = False
        for param in self.vit.heads.parameters():
            param.requires_grad = True

        trainable_modules = []
        for name in ("vpt_prompts", "static_prompts", "routed_prompts", "adapters", "ssf_attn", "ssf_mlp", "lora"):
            module = getattr(self, name, None)
            if module is not None:
                trainable_modules.append(module)
        if self.options.method in self.cross_methods and self.options.cross_attn_trainable:
            trainable_modules.append(self.cross_attn)

        for module in trainable_modules:
            if isinstance(module, nn.Parameter):
                module.requires_grad = True
            else:
                for param in module.parameters():
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
        x = self.vit.encoder.dropout(x)
        if self.options.method == "vpt":
            prompts = self.vpt_prompts.unsqueeze(0).expand(batch, -1, -1)
            x = torch.cat([x[:, :1], prompts, x[:, 1:]], dim=1)
        return x

    def _forward_block(
        self,
        block: nn.Module,
        x: torch.Tensor,
        layer_idx: int,
        gates: List[torch.Tensor],
    ) -> torch.Tensor:
        y = block.ln_1(x)
        if self.options.method == "lora":
            attn_out = self.lora[layer_idx](block.self_attention, y)
        else:
            attn_out, _ = block.self_attention(y, y, y, need_weights=False)
        x = x + block.dropout(attn_out)

        if self.options.method in self.cross_methods:
            if self.options.method in self.routing_layers:
                prompt, gate = self.routed_prompts(x, layer_idx)
                gates.append(gate)
            else:
                prompt = self.static_prompts(layer_idx, x.shape[0])
            cross = self.cross_attn[layer_idx](x, prompt)
            x = x + self.options.residual_scale * cross

        if self.options.method == "ssf":
            x = self.ssf_attn[layer_idx](x)

        y = block.ln_2(x)
        mlp_out = block.mlp(y)
        if self.options.method == "adaptformer":
            x = x + mlp_out + self.adapters[layer_idx](y)
        else:
            x = x + mlp_out

        if self.options.method == "ssf":
            x = self.ssf_mlp[layer_idx](x)
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
        if self.options.entropy_enabled and self.options.entropy_alpha > 0:
            routing_loss = routing_loss - self.options.entropy_alpha * norm_entropy.mean()
        if self.options.load_kl_enabled and self.options.load_kl_beta > 0:
            mean_load = gate_tensor.mean(dim=(0, 1)).clamp_min(1e-8)
            uniform = torch.full_like(mean_load, 1.0 / mean_load.numel())
            load_kl = (mean_load * (mean_load.log() - uniform.log())).sum()
            routing_loss = routing_loss + self.options.load_kl_beta * load_kl
        return {
            "routing_loss": routing_loss,
            "mean_normalized_entropy": norm_entropy.mean(),
            "effective_experts": entropy.exp().mean(),
            "gates": gate_tensor,
        }


def build_model(num_classes: int, config: Dict, method: Optional[str] = None, pretrained: Optional[bool] = None) -> PEFTVisionTransformer:
    selected_method = method or config.get("method")
    if selected_method == "cvpt_expert_adapter":
        from cvpt_expert_adapter import build_cvpt_expert_adapter_model

        return build_cvpt_expert_adapter_model(num_classes, config, method=selected_method, pretrained=pretrained)
    if selected_method == "expert_adapter_pool":
        from cvpt_expert_adapter import build_expert_adapter_pool_model

        return build_expert_adapter_pool_model(num_classes, config, method=selected_method, pretrained=pretrained)
    options = options_from_config(config, method=method, pretrained=pretrained)
    return PEFTVisionTransformer(num_classes=num_classes, options=options)


def count_parameters(model: nn.Module) -> Dict[str, float]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "total_m": total / 1e6,
        "trainable_m": trainable / 1e6,
        "trainable_percent_of_86m": trainable / 86_000_000 * 100,
    }


def estimate_flops_g(model: nn.Module, image_size: int, device: torch.device) -> Optional[float]:
    try:
        from fvcore.nn import FlopCountAnalysis
    except Exception:
        return None
    was_training = model.training
    model.eval()
    dummy = torch.zeros(1, 3, image_size, image_size, device=device)
    try:
        flops = FlopCountAnalysis(model, dummy).total() / 1e9
    except Exception:
        flops = None
    model.train(was_training)
    return flops
