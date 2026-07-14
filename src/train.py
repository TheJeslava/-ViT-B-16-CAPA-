import argparse
import copy
import csv
import json
import logging
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml

from data import build_dataloaders, seed_everything
from modeling import build_model, count_parameters, estimate_flops_g


def parse_args():
    parser = argparse.ArgumentParser(description="Train Para-CVPT and related PEFT baselines.")
    parser.add_argument("--config", default="configs/para_cvpt_single_seed.yaml", help="Path to YAML config.")
    parser.add_argument("--dataset", default=None, help="Dataset key. Defaults to config experiment.dataset.")
    parser.add_argument("--method", default=None, help="Method key. Defaults to config experiment.method.")
    parser.add_argument("--ablation", default=None, choices=["core_structure", "collapse_guard", "capacity", "scope"])
    parser.add_argument("--data-root", default=None, help="Dataset root. Defaults to config paths.data_root.")
    parser.add_argument("--output-root", default=None, help="Output root. Defaults to config outputs.root.")
    parser.add_argument("--device", default=None, help="cuda, cpu, or auto.")
    parser.add_argument("--seed", type=int, default=None, help="Single seed override.")
    parser.add_argument("--epochs", type=int, default=None, help="Epoch override.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch-size override.")
    parser.add_argument("--download", action="store_true", help="Allow torchvision datasets to download.")
    parser.add_argument("--no-pretrained", action="store_true", help="Do not load ImageNet-1k pretrained ViT weights.")
    parser.add_argument("--limit-train-batches", type=int, default=None, help="Debug: limit train batches per epoch.")
    parser.add_argument("--limit-eval-batches", type=int, default=None, help="Debug: limit eval batches.")
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_config(config: Dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)


def setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(str(run_dir))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(run_dir / "train.log", mode="w", encoding="utf-8")
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def select_device(name: Optional[str]) -> torch.device:
    if name is None or name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def make_optimizer(config: Dict, model: nn.Module) -> torch.optim.Optimizer:
    opt_cfg = config.get("training", {}).get("optimizer", {})
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found.")
    name = opt_cfg.get("name", "adamw").lower()
    lr = float(opt_cfg.get("lr", 5e-4))
    weight_decay = float(opt_cfg.get("weight_decay", 0.05))
    betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


class EpochLRScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, epochs: int, warmup: int) -> None:
        self.optimizer = optimizer
        self.epochs = epochs
        self.warmup = warmup
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self, epoch: int) -> None:
        factor = self._factor(epoch)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * factor

    def _factor(self, epoch: int) -> float:
        if self.warmup > 0 and epoch <= self.warmup:
            return epoch / self.warmup
        if self.epochs <= self.warmup:
            return 1.0
        progress = (epoch - self.warmup) / max(1, self.epochs - self.warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def make_scheduler(config: Dict, optimizer: torch.optim.Optimizer, epochs: int):
    sched_cfg = config.get("training", {}).get("scheduler", {})
    if sched_cfg.get("name", "cosine").lower() == "none":
        return None
    warmup = int(sched_cfg.get("warmup_epochs", 0))
    return EpochLRScheduler(optimizer, epochs=epochs, warmup=warmup)


def autocast_context(device: torch.device, enabled: bool):
    return autocast_context_with_dtype(device, enabled, None)


def autocast_context_with_dtype(device: torch.device, enabled: bool, dtype: Optional[torch.dtype]):
    if hasattr(torch, "amp"):
        return torch.amp.autocast(device_type=device.type, enabled=enabled, dtype=dtype)
    return torch.cuda.amp.autocast(enabled=enabled, dtype=dtype)


def make_grad_scaler(device: torch.device, enabled: bool):
    if hasattr(torch, "amp"):
        return torch.amp.GradScaler(device.type, enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def amp_dtype_from_config(config: Dict, device: torch.device) -> Optional[torch.dtype]:
    if device.type != "cuda":
        return None
    dtype_name = str(config.get("training", {}).get("amp_dtype", "float16")).lower()
    if dtype_name in {"fp16", "float16", "half"}:
        return torch.float16
    if dtype_name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype_name in {"none", "auto", ""}:
        return None
    raise ValueError(f"Unsupported amp_dtype: {dtype_name}")


def apply_performance_options(config: Dict) -> Dict:
    perf = config.get("performance", {})
    allow_tf32 = bool(perf.get("allow_tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    matmul_precision = perf.get("matmul_precision", "high")
    if hasattr(torch, "set_float32_matmul_precision") and matmul_precision:
        torch.set_float32_matmul_precision(str(matmul_precision))
    return {
        "allow_tf32": allow_tf32,
        "matmul_precision": matmul_precision,
        "cudnn_benchmark": bool(perf.get("cudnn_benchmark", True)),
        "deterministic": bool(perf.get("deterministic", False)),
        "compile": bool(perf.get("compile", False)),
    }


def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().sum().item()


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: Optional[torch.dtype],
    grad_clip_norm: float,
    limit_batches: Optional[int],
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_routing = 0.0
    total_correct = 0.0
    total_samples = 0
    entropy_values: List[float] = []
    start = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for batch_idx, (images, targets) in enumerate(loader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context_with_dtype(device, amp_enabled, amp_dtype):
            logits, aux = model(images, return_aux=True)
            ce_loss = criterion(logits, targets)
            routing_loss = aux["routing_loss"]
            loss = ce_loss + routing_loss
        scaler.scale(loss).backward()
        if grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [param for param in model.parameters() if param.requires_grad],
                max_norm=grad_clip_norm,
            )
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.numel()
        total_loss += float(loss.detach().item()) * batch_size
        total_ce += float(ce_loss.detach().item()) * batch_size
        total_routing += float(routing_loss.detach().item()) * batch_size
        total_correct += accuracy_top1(logits.detach(), targets)
        total_samples += batch_size
        if aux.get("gates") is not None:
            entropy_values.append(float(aux["mean_normalized_entropy"].detach().item()))

    peak_memory = torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
    return {
        "loss": total_loss / max(1, total_samples),
        "ce_loss": total_ce / max(1, total_samples),
        "routing_loss": total_routing / max(1, total_samples),
        "acc": total_correct / max(1, total_samples) * 100,
        "mean_router_entropy": float(np.mean(entropy_values)) if entropy_values else 0.0,
        "epoch_time_s": time.time() - start,
        "peak_memory_gb": peak_memory,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: Optional[torch.dtype],
    limit_batches: Optional[int],
    collect_gates: bool = False,
) -> Dict:
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0
    gates: List[torch.Tensor] = []
    entropy_values: List[float] = []

    for batch_idx, (images, targets) in enumerate(loader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast_context_with_dtype(device, amp_enabled, amp_dtype):
            logits, aux = model(images, return_aux=True)
            loss = criterion(logits, targets)
        batch_size = targets.numel()
        total_loss += float(loss.item()) * batch_size
        total_correct += accuracy_top1(logits, targets)
        total_samples += batch_size
        if aux.get("gates") is not None:
            entropy_values.append(float(aux["mean_normalized_entropy"].item()))
            if collect_gates:
                gates.append(aux["gates"].detach().cpu())

    gate_tensor = torch.cat(gates, dim=0) if gates else None
    return {
        "loss": total_loss / max(1, total_samples),
        "acc": total_correct / max(1, total_samples) * 100,
        "mean_router_entropy": float(np.mean(entropy_values)) if entropy_values else 0.0,
        "gates": gate_tensor,
    }


def write_metrics_csv(path: Path, rows: List[Dict]) -> None:
    fieldnames = [
        "epoch",
        "lr",
        "train_loss",
        "train_ce_loss",
        "train_routing_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "train_router_entropy",
        "val_router_entropy",
        "epoch_time_s",
        "peak_memory_gb",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def save_summary(path: Path, summary: Dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def should_keep_checkpoint(config: Dict, method: str) -> bool:
    policy = config.get("outputs", {}).get("checkpoint_policy", {})
    keep = set(policy.get("save_best_for", ["para_cvpt", "cvpt"]))
    return method in keep


def state_dict_to_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def save_router_artifacts(run_dir: Path, gates: Optional[torch.Tensor], max_samples: int = 3) -> Dict:
    if gates is None:
        return {}
    fig_dir = run_dir / "visualizations"
    fig_dir.mkdir(parents=True, exist_ok=True)
    arr = gates.numpy()
    np.savez(run_dir / "router_stats.npz", gates=arr)

    mean_entropy = entropy_np(arr).mean()
    effective_experts = float(np.exp(entropy_np(arr)).mean())
    load = arr.mean(axis=(0, 1))
    load_min = float(load.min())
    load_ratio = float(load.max() / max(load_min, 1e-8))

    for idx in range(min(max_samples, arr.shape[0])):
        plt.figure(figsize=(8, 4))
        plt.imshow(arr[idx], aspect="auto", cmap="viridis")
        plt.colorbar(label="gate weight")
        plt.xlabel("Expert")
        plt.ylabel("Transformer layer")
        plt.title(f"Routing heatmap sample {idx}")
        plt.tight_layout()
        plt.savefig(fig_dir / f"route_heatmap_sample_{idx}.png", dpi=200)
        plt.close()

    plt.figure(figsize=(7, 4))
    plt.bar(np.arange(load.shape[0]), load)
    plt.xlabel("Expert")
    plt.ylabel("Mean gate weight")
    plt.title("Expert load over test set")
    plt.tight_layout()
    plt.savefig(fig_dir / "expert_load_bar.png", dpi=200)
    plt.close()

    return {
        "router_mean_entropy": float(mean_entropy),
        "router_effective_experts": effective_experts,
        "expert_load_mean": load.tolist(),
        "expert_load_min": load_min,
        "expert_load_max_min_ratio": load_ratio,
    }


def entropy_np(gates: np.ndarray) -> np.ndarray:
    return -(gates * np.log(np.clip(gates, 1e-8, 1.0))).sum(axis=-1)


def save_convergence_plot(run_dir: Path, rows: List[Dict], method: str) -> None:
    if not rows:
        return
    fig_dir = run_dir / "visualizations"
    fig_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in rows]
    vals = [row["val_acc"] for row in rows]
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, vals, marker="o", linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Validation top-1 (%)")
    plt.title(f"Convergence: {method}")
    plt.tight_layout()
    plt.savefig(fig_dir / "convergence_curve.png", dpi=200)
    plt.close()


def run_experiment(
    base_config: Dict,
    method: str,
    dataset_key: str,
    run_name: str,
    args,
    config_overrides: Optional[Dict] = None,
) -> Dict:
    config = copy.deepcopy(base_config)
    if config_overrides:
        merge_dict(config, config_overrides)
    method_training_overrides = config.get("training", {}).get("method_overrides", {}).get(method, {})
    if method_training_overrides:
        merge_dict(config.setdefault("training", {}), method_training_overrides)
    seed = int(args.seed if args.seed is not None else config.get("training", {}).get("seed", 0))
    epochs = int(args.epochs if args.epochs is not None else config.get("training", {}).get("epochs", 100))
    batch_size = int(args.batch_size if args.batch_size is not None else config.get("training", {}).get("batch_size", 64))
    data_root = args.data_root or config.get("paths", {}).get("data_root", "data")
    output_root = Path(args.output_root or config.get("outputs", {}).get("root", "outputs"))
    run_dir = output_root / dataset_key / run_name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(run_dir)

    config.setdefault("experiment", {})
    config["experiment"].update({"method": method, "dataset": dataset_key, "seed": seed, "run_name": run_name})
    config.setdefault("training", {})
    config["training"].update({"seed": seed, "epochs": epochs, "batch_size": batch_size})
    save_config(config, run_dir / "config.yaml")

    perf_state = apply_performance_options(config)
    seed_everything(
        seed,
        deterministic=bool(config.get("performance", {}).get("deterministic", False)),
        benchmark=bool(config.get("performance", {}).get("cudnn_benchmark", True)),
    )
    device = select_device(args.device)
    logger.info("Run directory: %s", run_dir)
    logger.info("Method=%s dataset=%s seed=%d device=%s", method, dataset_key, seed, device)

    data = build_dataloaders(
        config=config,
        dataset_key=dataset_key,
        data_root=data_root,
        seed=seed,
        batch_size=batch_size,
        num_workers=int(config.get("data", {}).get("num_workers", 8)),
        download=bool(args.download),
    )
    logger.info("Dataset sizes: train=%d val=%d test=%d", data.train_size, data.val_size, data.test_size)
    logger.info(
        "Loader/perf: batch_size=%d workers=%d pin_memory=%s persistent_workers=%s prefetch_factor=%s tf32=%s matmul_precision=%s benchmark=%s deterministic=%s",
        batch_size,
        int(config.get("data", {}).get("num_workers", 8)),
        bool(config.get("data", {}).get("pin_memory", True)),
        bool(config.get("data", {}).get("persistent_workers", True)),
        config.get("data", {}).get("prefetch_factor", 4),
        perf_state["allow_tf32"],
        perf_state["matmul_precision"],
        perf_state["cudnn_benchmark"],
        perf_state["deterministic"],
    )

    model = build_model(
        data.num_classes,
        config,
        method=method,
        pretrained=False if args.no_pretrained else None,
    ).to(device)
    if getattr(model.options, "pretrained", False):
        logger.info("ViT init weights: torchvision ViT_B_16_Weights.IMAGENET1K_V1")
    else:
        logger.info("ViT init weights: random_init (--no-pretrained or pretrained_weights=false)")
    params = count_parameters(model)
    model_for_state = model
    if bool(config.get("performance", {}).get("compile", False)) and hasattr(torch, "compile"):
        model = torch.compile(model)
    logger.info(
        "Trainable params: %.4fM (%.4f%% of ViT-B/16 86M)",
        params["trainable_m"],
        params["trainable_percent_of_86m"],
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(config, model)
    scheduler = make_scheduler(config, optimizer, epochs)
    amp_enabled = bool(config.get("training", {}).get("amp", True)) and device.type == "cuda"
    amp_dtype = amp_dtype_from_config(config, device)
    logger.info("AMP: enabled=%s dtype=%s", amp_enabled, amp_dtype)
    scaler = make_grad_scaler(device, amp_enabled)
    grad_clip_norm = float(config.get("training", {}).get("grad_clip_norm", 1.0))
    flops_g = estimate_flops_g(model, int(config.get("project", {}).get("input_size", 224)), device)

    rows: List[Dict] = []
    best_val = -1.0
    best_epoch = -1
    best_state = None
    best_ckpt = run_dir / "best_checkpoint.pth"
    keep_ckpt = should_keep_checkpoint(config, method)
    peak_memory = 0.0

    for epoch in range(1, epochs + 1):
        if scheduler is not None:
            scheduler.step(epoch)
        train_stats = train_one_epoch(
            model,
            data.train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            amp_enabled,
            amp_dtype,
            grad_clip_norm,
            args.limit_train_batches,
        )
        val_stats = evaluate(
            model,
            data.val_loader,
            criterion,
            device,
            amp_enabled,
            amp_dtype,
            args.limit_eval_batches,
            collect_gates=False,
        )
        lr = optimizer.param_groups[0]["lr"]
        peak_memory = max(peak_memory, train_stats["peak_memory_gb"])
        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_stats["loss"],
            "train_ce_loss": train_stats["ce_loss"],
            "train_routing_loss": train_stats["routing_loss"],
            "train_acc": train_stats["acc"],
            "val_loss": val_stats["loss"],
            "val_acc": val_stats["acc"],
            "train_router_entropy": train_stats["mean_router_entropy"],
            "val_router_entropy": val_stats["mean_router_entropy"],
            "epoch_time_s": train_stats["epoch_time_s"],
            "peak_memory_gb": train_stats["peak_memory_gb"],
        }
        rows.append(row)
        write_metrics_csv(run_dir / "metrics.csv", rows)
        logger.info(
            "Epoch %03d/%03d | train %.2f%% loss %.4f | val %.2f%% loss %.4f | entropy %.3f | %.1fs",
            epoch,
            epochs,
            row["train_acc"],
            row["train_loss"],
            row["val_acc"],
            row["val_loss"],
            row["val_router_entropy"],
            row["epoch_time_s"],
        )

        if val_stats["acc"] > best_val:
            best_val = val_stats["acc"]
            best_epoch = epoch
            if keep_ckpt:
                torch.save(
                    {
                        "model": model_for_state.state_dict(),
                        "method": method,
                        "dataset": dataset_key,
                        "seed": seed,
                        "epoch": epoch,
                        "val_acc": best_val,
                        "config": config,
                    },
                    best_ckpt,
                )
            else:
                best_state = state_dict_to_cpu(model_for_state)

    if keep_ckpt and best_ckpt.exists():
        checkpoint = torch.load(best_ckpt, map_location=device, weights_only=False)
        model_for_state.load_state_dict(checkpoint["model"])
    elif best_state is not None:
        model_for_state.load_state_dict(best_state)
    else:
        logger.warning("No best state was captured; testing current model.")

    test_stats = evaluate(
        model,
        data.test_loader,
        criterion,
        device,
        amp_enabled,
        amp_dtype,
        args.limit_eval_batches,
        collect_gates=True,
    )
    router_summary = save_router_artifacts(run_dir, test_stats.get("gates"))
    save_convergence_plot(run_dir, rows, method)

    epoch_times = [row["epoch_time_s"] for row in rows]
    summary = {
        "method": method,
        "run_name": run_name,
        "dataset": dataset_key,
        "seed": seed,
        "num_classes": data.num_classes,
        "train_size": data.train_size,
        "val_size": data.val_size,
        "test_size": data.test_size,
        "best_epoch": best_epoch,
        "best_val_top1": best_val,
        "test_top1_at_best_val": test_stats["acc"],
        "test_loss_at_best_val": test_stats["loss"],
        "trainable_params_m": params["trainable_m"],
        "trainable_params_percent_of_86m": params["trainable_percent_of_86m"],
        "total_params_m": params["total_m"],
        "inference_flops_g": flops_g,
        "peak_train_memory_gb": peak_memory,
        "mean_epoch_time_s": float(np.mean(epoch_times)) if epoch_times else 0.0,
        "checkpoint_saved": keep_ckpt,
    }
    summary.update(router_summary)
    save_summary(run_dir / "summary.json", summary)
    logger.info("Best val %.2f%% at epoch %d | test %.2f%%", best_val, best_epoch, test_stats["acc"])
    return summary


def merge_dict(target: Dict, updates: Dict) -> Dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            merge_dict(target[key], value)
        else:
            target[key] = value
    return target


def build_ablation_runs(config: Dict, ablation: str, args) -> List[Tuple[str, str, Dict]]:
    runs: List[Tuple[str, str, Dict]] = []
    if ablation == "core_structure":
        variants = config.get("ablations", {}).get("core_structure", {}).get("variants", {})
        method_map = {
            "cvpt": "cvpt",
            "shared_cvpt": "shared_cvpt",
            "routed_cvpt": "routed_cvpt",
            "para_cvpt": "para_cvpt",
        }
        for name in variants:
            runs.append((name, method_map[name], {}))
    elif ablation == "collapse_guard":
        variants = config.get("ablations", {}).get("collapse_guard", {}).get("variants", {})
        for name, values in variants.items():
            overrides = {
                "para_cvpt": {
                    "collapse_guard": {
                        "entropy_loss": {"enabled": bool(values.get("entropy_loss", False))},
                    },
                    "router": {
                        "train_noise_std": float(values.get("train_noise_std", 0.0)),
                        "expert_dropout": float(values.get("expert_dropout", 0.0)),
                    },
                }
            }
            runs.append((name, "para_cvpt", overrides))
    elif ablation == "capacity":
        grid = config.get("ablations", {}).get("capacity_and_scope", {}).get("capacity_grid", [])
        for item in grid:
            m = int(item["num_experts_m"])
            t = int(item["tokens_per_expert_t"])
            name = f"capacity_m{m}_t{t}"
            overrides = {"para_cvpt": {"expert_bank": {"num_experts_m": m, "tokens_per_expert_t": t}}}
            runs.append((name, "para_cvpt", overrides))
    elif ablation == "scope":
        grid = config.get("ablations", {}).get("capacity_and_scope", {}).get("scope_grid", [])
        for scope in grid:
            overrides = {"para_cvpt": {"expert_bank": {"scope": scope}}}
            runs.append((f"scope_{scope}", "para_cvpt", overrides))
    else:
        raise ValueError(f"Unknown ablation: {ablation}")
    return runs


def write_aggregate_table(output_root: Path, dataset_key: str, summaries: List[Dict], table_name: str) -> None:
    table_dir = output_root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    path = table_dir / table_name
    fields = [
        "dataset",
        "method",
        "run_name",
        "seed",
        "best_epoch",
        "best_val_top1",
        "test_top1_at_best_val",
        "trainable_params_m",
        "trainable_params_percent_of_86m",
        "inference_flops_g",
        "peak_train_memory_gb",
        "mean_epoch_time_s",
        "router_mean_entropy",
        "router_effective_experts",
        "expert_load_min",
        "expert_load_max_min_ratio",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary.get(field, "") for field in fields})


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset_key = args.dataset or config.get("experiment", {}).get("dataset", "cifar100_full")
    method = args.method or config.get("experiment", {}).get("method", "para_cvpt")
    output_root = Path(args.output_root or config.get("outputs", {}).get("root", "outputs"))

    if args.ablation:
        summaries = []
        ablation_cfg = config.get("ablations", {}).get(args.ablation, {})
        dataset_for_ablation = args.dataset or ablation_cfg.get("dataset_first", dataset_key)
        for run_name, run_method, overrides in build_ablation_runs(config, args.ablation, args):
            summaries.append(run_experiment(config, run_method, dataset_for_ablation, run_name, args, overrides))
        write_aggregate_table(output_root, dataset_for_ablation, summaries, f"{args.ablation}_results.csv")
    else:
        summary = run_experiment(config, method, dataset_key, method, args)
        write_aggregate_table(output_root, dataset_key, [summary], "main_results.csv")


if __name__ == "__main__":
    main()
