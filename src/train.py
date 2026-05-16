"""
Training script for VizWiz AnswerTherapy grounding model.

Usage:
    python src/train.py --config configs/base.yaml [--overrides key=value ...]

Overrides example:
    python src/train.py --config configs/base.yaml \
        model.num_queries=10 \
        training.epochs=20 \
        loss.lambda_box=5.0

Training loop:
    1. Frozen Qwen2.5-VL extracts hidden states for (image, question)
    2. GroundingHead decodes N box predictions + objectness + single/multi
    3. HungarianMatcher assigns predictions to GT boxes
    4. GroundingLoss computes L1 + GIoU + BCE_obj + CE_sm
    5. Backprop only through unfrozen parameters (projection + decoder + heads)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---- project imports ----
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from datasets.vizwiz_answertherapy import VizWizAnswerTherapyDataset, collate_fn, build_datasets
from models.qwen_backbone import QwenBackbone
from models.grounding_head import GroundingHead
from models.matcher import HungarianMatcher
from models.losses import GroundingLoss
from utils.metrics import GroundingMetrics, format_metrics


# -----------------------------------------------------------------------
# Config loading
# -----------------------------------------------------------------------

def load_config(path: str, overrides: list[str] | None = None) -> dict:
    """Load YAML config and apply dot-notation overrides."""
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if overrides:
        for ov in overrides:
            key, _, val = ov.partition("=")
            _set_nested(cfg, key.split("."), _parse_val(val))
    return cfg


def _set_nested(d: dict, keys: list[str], val: Any) -> None:
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = val


def _parse_val(s: str) -> Any:
    """Try int → float → bool → str."""
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    if s.lower() == "null":
        return None
    return s


# -----------------------------------------------------------------------
# Seed
# -----------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------
# Model construction
# -----------------------------------------------------------------------

def build_model(cfg: dict, device: torch.device) -> tuple[QwenBackbone, GroundingHead]:
    """Instantiate backbone and grounding head from config."""
    m = cfg["model"]
    abl = cfg.get("ablation", {})

    _dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = _dtype_map.get(m.get("torch_dtype", "float16"), torch.float16)

    max_memory = None
    max_memory_per_gpu = m.get("max_memory_per_gpu")
    if max_memory_per_gpu and torch.cuda.is_available():
        max_memory = {i: max_memory_per_gpu for i in range(torch.cuda.device_count())}

    backbone = QwenBackbone(
        model_name=m["backbone"],
        freeze=m.get("freeze_backbone", True),
        torch_dtype=torch_dtype,
        device_map=m.get("device_map", "auto"),
        max_memory=max_memory,
        attn_implementation=m.get("attn_implementation", "eager"),
        max_pixels=m.get("max_pixels"),
        min_pixels=m.get("min_pixels"),
    )

    d_vlm = m.get("d_vlm") or backbone.d_vlm

    d_dec = abl.get("d_dec") or m.get("d_dec", 256)
    num_layers = abl.get("num_decoder_layers") or m.get("num_decoder_layers", 6)
    num_queries = m.get("num_queries") or m.get("num_queries_default", 8)
    use_sm = abl.get("use_single_multi_head", True)
    if use_sm is None:
        use_sm = True

    head = GroundingHead(
        d_vlm=d_vlm,
        d_dec=d_dec,
        num_queries=num_queries,
        num_decoder_layers=num_layers,
        nheads=m.get("nheads", 8),
        dropout=m.get("dropout", 0.1),
        use_single_multi_head=use_sm,
    ).to(device)

    # Optional LoRA
    lora_cfg = m.get("lora", {})
    if lora_cfg.get("enabled", False) and not m.get("freeze_backbone", True):
        backbone.enable_lora(
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("lora_alpha", 32),
            lora_dropout=lora_cfg.get("lora_dropout", 0.05),
            target_modules=lora_cfg.get("target_modules"),
        )

    return backbone, head


# -----------------------------------------------------------------------
# Optimizer and scheduler
# -----------------------------------------------------------------------

def build_optimizer(cfg: dict, backbone: QwenBackbone, head: GroundingHead) -> torch.optim.Optimizer:
    opt_cfg = cfg["training"]["optimizer"]
    lr      = float(opt_cfg.get("lr", 1e-4))
    lr_lora = float(opt_cfg.get("lr_lora", 1e-5))
    wd      = float(opt_cfg.get("weight_decay", 1e-4))
    betas   = tuple(opt_cfg.get("betas", [0.9, 0.999]))

    param_groups = [{"params": head.parameters(), "lr": lr}]

    # If LoRA is enabled, add backbone LoRA params with lower LR
    try:
        lora_params = list(backbone.lora_parameters())
        if lora_params:
            param_groups.append({"params": lora_params, "lr": lr_lora})
    except Exception:
        pass

    return torch.optim.AdamW(param_groups, weight_decay=wd, betas=betas)


def build_scheduler(cfg: dict, optimizer: torch.optim.Optimizer, total_steps: int):
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    sch_cfg = cfg["training"].get("scheduler", {})
    warmup  = sch_cfg.get("warmup_steps", 100)
    min_lr  = float(sch_cfg.get("min_lr", 1e-6))

    warmup_sched = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup, 1), eta_min=min_lr)
    return SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup])


# -----------------------------------------------------------------------
# Checkpoint helpers
# -----------------------------------------------------------------------

class CheckpointManager:
    def __init__(self, save_dir: Path, save_top_k: int = 3) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_top_k = save_top_k
        self._history: list[tuple[float, Path]] = []   # (val_loss, path)

    def save(
        self,
        epoch: int,
        val_loss: float,
        head: GroundingHead,
        optimizer: torch.optim.Optimizer,
        metrics: dict,
    ) -> Path:
        ckpt_path = self.save_dir / f"epoch_{epoch:03d}_loss_{val_loss:.4f}.pt"
        torch.save({
            "epoch": epoch,
            "val_loss": val_loss,
            "head_state_dict": head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        }, ckpt_path)

        self._history.append((val_loss, ckpt_path))
        self._history.sort(key=lambda x: x[0])

        # Remove excess checkpoints (keep best k)
        while len(self._history) > self.save_top_k:
            _, old_path = self._history.pop()
            if old_path.exists():
                old_path.unlink()

        # Always write a "latest" symlink
        latest = self.save_dir / "latest.pt"
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(ckpt_path.name)

        return ckpt_path

    def best_path(self) -> Path | None:
        if not self._history:
            return None
        return self._history[0][1]


# -----------------------------------------------------------------------
# Training epoch
# -----------------------------------------------------------------------

def train_epoch(
    backbone: QwenBackbone,
    head: GroundingHead,
    matcher: HungarianMatcher,
    criterion: GroundingLoss,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loader: DataLoader,
    device: torch.device,
    grad_accum: int,
    grad_clip: float,
    log_every: int,
    epoch: int,
    writer=None,
    wandb_run=None,
    global_step: list[int] | None = None,
) -> float:
    head.train()
    backbone.eval()

    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    head_dtype = next(head.parameters()).dtype
    for step, batch in enumerate(pbar):
        images    = batch["images"]
        questions = batch["questions"]
        gt_boxes  = batch["gt_boxes"]       # list of tensors
        sm_labels = batch["single_multi"].to(device)  # Long[B]

        # ---- Forward through frozen backbone ----
        hidden, key_pad = backbone(images, questions, device=device)
        hidden  = hidden.to(device=device, dtype=head_dtype)
        hidden = _check_nonfinite(hidden, name="backbone hidden states")
        key_pad = key_pad.to(device)

        # ---- Grounding head ----
        out = head(hidden, key_padding_mask=key_pad)
        pred_boxes  = out["pred_boxes"]
        pred_obj    = out["pred_obj_logits"]
        pred_sm     = out["pred_single_multi_logits"]

        # ---- Hungarian matching (no_grad internally) ----
        match = matcher(pred_boxes.float(), pred_obj.float(), gt_boxes)

        # ---- Loss ----
        loss_dict = criterion(
            pred_boxes=pred_boxes.float(),
            pred_obj_logits=pred_obj.float(),
            pred_sm_logits=pred_sm.float(),
            gt_boxes_list=gt_boxes,
            sm_labels=sm_labels,
            match_result=match,
        )
        loss = loss_dict.total / grad_accum

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Loss is {loss.item()} at epoch={epoch} step={step}. "
                f"Check for NaN in backbone hidden states (set torch_dtype: bfloat16)."
            )

        loss.backward()

        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            nn.utils.clip_grad_norm_(head.parameters(), grad_clip)
            optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

        loss_val = loss_dict.total.item()
        total_loss += loss_val
        n_batches  += 1

        pbar.set_postfix({
            "loss": f"{loss_val:.4f}",
            "l1":   f"{loss_dict.l1.item():.3f}",
            "giou": f"{loss_dict.giou.item():.3f}",
            "obj":  f"{loss_dict.obj.item():.3f}",
            "sm":   f"{loss_dict.sm.item():.3f}",
        })

        if global_step is not None:
            global_step[0] += 1
            gs = global_step[0]
            if gs % log_every == 0:
                log_data = {
                    "train/loss_total": loss_dict.total.item(),
                    "train/loss_l1":    loss_dict.l1.item(),
                    "train/loss_giou":  loss_dict.giou.item(),
                    "train/loss_obj":   loss_dict.obj.item(),
                    "train/loss_sm":    loss_dict.sm.item(),
                    "train/lr":         optimizer.param_groups[0]["lr"],
                }
                if writer:
                    for k, v in log_data.items():
                        writer.add_scalar(k, v, gs)
                if wandb_run:
                    wandb_run.log(log_data, step=gs)

    return total_loss / max(n_batches, 1)


# -----------------------------------------------------------------------
# Validation epoch
# -----------------------------------------------------------------------

@torch.no_grad()
def val_epoch(
    backbone: QwenBackbone,
    head: GroundingHead,
    matcher: HungarianMatcher,
    criterion: GroundingLoss,
    loader: DataLoader,
    device: torch.device,
    obj_threshold: float,
    iou_thresholds: list[float],
    epoch: int,
    writer=None,
    wandb_run=None,
    global_step: int = 0,
) -> tuple[float, dict]:
    head.eval()
    backbone.eval()
    total_loss = 0.0
    n_batches  = 0
    metrics = GroundingMetrics(iou_thresholds=iou_thresholds, obj_threshold=obj_threshold)

    pbar = tqdm(loader, desc=f"Epoch {epoch} [val]", leave=False)
    head_dtype = next(head.parameters()).dtype
    for batch in pbar:
        images    = batch["images"]
        questions = batch["questions"]
        gt_boxes  = batch["gt_boxes"]
        sm_labels = batch["single_multi"].to(device)

        hidden, key_pad = backbone(images, questions, device=device)
        hidden  = hidden.to(device=device, dtype=head_dtype)
        hidden = _check_nonfinite(hidden, name="backbone hidden states")
        key_pad = key_pad.to(device)

        out = head(hidden, key_padding_mask=key_pad)
        pred_boxes = out["pred_boxes"]
        pred_obj   = out["pred_obj_logits"]
        pred_sm    = out["pred_single_multi_logits"]

        match = matcher(pred_boxes.float(), pred_obj.float(), gt_boxes)

        loss_dict = criterion(
            pred_boxes=pred_boxes.float(),
            pred_obj_logits=pred_obj.float(),
            pred_sm_logits=pred_sm.float(),
            gt_boxes_list=gt_boxes,
            sm_labels=sm_labels,
            match_result=match,
        )

        total_loss += loss_dict.total.item()
        n_batches  += 1

        metrics.update(
            pred_boxes=pred_boxes,
            pred_obj_logits=pred_obj,
            pred_sm_logits=pred_sm,
            gt_boxes_list=gt_boxes,
            sm_labels=sm_labels,
        )

    avg_loss = total_loss / max(n_batches, 1)
    metric_results = metrics.compute()

    if writer:
        writer.add_scalar("val/loss", avg_loss, global_step)
        for thr_key in [f"iou{int(t*100)}" for t in iou_thresholds]:
            d = metric_results.get("all", {}).get(thr_key, {})
            for k, v in d.items():
                if isinstance(v, (int, float)):
                    writer.add_scalar(f"val/{thr_key}_{k}", v, global_step)
        sm = metric_results.get("single_multi_cls", {})
        for k, v in sm.items():
            if isinstance(v, (int, float)):
                writer.add_scalar(f"val/sm_{k}", v, global_step)

    if wandb_run:
        log_flat = {"val/loss": avg_loss}
        _flatten_dict(metric_results, log_flat, prefix="val/")
        wandb_run.log(log_flat, step=global_step)

    return avg_loss, metric_results


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--overrides", nargs="*", default=[])
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    # parse_known_args lets users pass bare key=value overrides without --overrides:
    #   python src/train.py --config configs/base.yaml training.epochs=5
    args, extra = parser.parse_known_args()
    # Merge explicit --overrides and any bare key=value extras
    all_overrides = (args.overrides or []) + [e for e in extra if "=" in e]
    args.overrides = all_overrides

    cfg = load_config(args.config, args.overrides)

    set_seed(cfg["training"].get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Output dirs ----
    out_dir  = Path(cfg["output"]["dir"])
    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    log_dir  = Path(cfg["output"]["log_dir"])
    for d in [out_dir, ckpt_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Save config snapshot
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # ---- Logging ----
    writer = None
    if cfg["logging"].get("use_tensorboard", True):
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(log_dir))

    wandb_run = None
    if cfg["logging"].get("use_wandb", False):
        import wandb
        wandb_run = wandb.init(
            project=cfg["logging"].get("wandb_project", "vizwiz-grounding"),
            entity=cfg["logging"].get("wandb_entity"),
            config=cfg,
        )

    # ---- Datasets and loaders ----
    train_ds, val_ds = build_datasets(cfg)
    print(f"Train samples: {len(train_ds)},  Val samples: {len(val_ds)}")

    batch_size  = cfg["training"]["batch_size"]
    num_workers = cfg["training"].get("num_workers", 2)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["eval"].get("batch_size", batch_size),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # ---- Model ----
    backbone, head = build_model(cfg, device)
    print(f"Backbone d_vlm={backbone.d_vlm},  Head num_queries={head.num_queries}")

    trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
    total_bb  = sum(p.numel() for p in backbone.parameters())
    print(f"Trainable params (head): {trainable:,}")
    print(f"Backbone total params  : {total_bb:,} (frozen={cfg['model'].get('freeze_backbone', True)})")

    # ---- Optimizer / Scheduler / Loss ----
    matcher  = HungarianMatcher(**cfg.get("matcher", {}))
    criterion = GroundingLoss(**cfg.get("loss", {}))
    optimizer = build_optimizer(cfg, backbone, head)

    grad_accum = cfg["training"].get("grad_accum_steps", 1)
    epochs     = cfg["training"]["epochs"]
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_steps = epochs * steps_per_epoch

    scheduler = build_scheduler(cfg, optimizer, total_steps)

    # ---- Resume ----
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        head.load_state_dict(ckpt["head_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {ckpt['epoch']}")

    # ---- Checkpoint manager ----
    ckpt_mgr = CheckpointManager(
        ckpt_dir,
        save_top_k=cfg["logging"].get("save_top_k", 3),
    )

    log_every  = cfg["logging"].get("log_every", 10)
    eval_every = cfg["logging"].get("eval_every", 1)
    save_every = cfg["logging"].get("save_every", 1)
    grad_clip  = cfg["training"].get("grad_clip", 1.0)
    iou_thr    = cfg["eval"].get("iou_thresholds", [0.3, 0.5])
    obj_thr    = cfg["eval"].get("obj_threshold", 0.5)

    global_step = [0]

    # ---------------------------------------------------------------
    # Main training loop
    # ---------------------------------------------------------------
    for epoch in range(start_epoch, epochs):
        train_loss = train_epoch(
            backbone=backbone, head=head, matcher=matcher, criterion=criterion,
            optimizer=optimizer, scheduler=scheduler, loader=train_loader,
            device=device, grad_accum=grad_accum,
            grad_clip=grad_clip, log_every=log_every,
            epoch=epoch, writer=writer, wandb_run=wandb_run, global_step=global_step,
        )
        print(f"[Epoch {epoch:3d}] train_loss={train_loss:.4f}")

        if (epoch + 1) % eval_every == 0:
            val_loss, metric_results = val_epoch(
                backbone=backbone, head=head, matcher=matcher, criterion=criterion,
                loader=val_loader, device=device, obj_threshold=obj_thr,
                iou_thresholds=iou_thr, epoch=epoch,
                writer=writer, wandb_run=wandb_run, global_step=global_step[0],
            )
            print(f"[Epoch {epoch:3d}] val_loss={val_loss:.4f}")
            print(format_metrics(metric_results))

            if (epoch + 1) % save_every == 0:
                path = ckpt_mgr.save(epoch, val_loss, head, optimizer, metric_results)
                print(f"Saved checkpoint: {path}")

    if writer:
        writer.close()
    if wandb_run:
        wandb_run.finish()

    print(f"\nTraining complete. Best checkpoint: {ckpt_mgr.best_path()}")


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _check_nonfinite(x: torch.Tensor, name: str) -> torch.Tensor:
    """Abort if more than 1% of values are non-finite; warn and clamp otherwise."""
    mask = ~torch.isfinite(x)
    if not mask.any():
        return x
    bad = int(mask.sum())
    frac = bad / x.numel()
    if frac > 0.01:
        raise RuntimeError(
            f"{name} has {bad}/{x.numel()} ({frac*100:.1f}%) non-finite values. "
            f"This is catastrophic — training on zeroed features is useless.\n"
            f"Fix: set  model.torch_dtype: bfloat16  in configs/base.yaml.\n"
            f"Qwen2.5-VL overflows in float16; bfloat16 uses fp32 compute on T4."
        )
    print(f"Warning: {bad} non-finite values ({frac*100:.2f}%) in {name}, clamping.")
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

def _flatten_dict(d: dict, out: dict, prefix: str = "") -> None:
    for k, v in d.items():
        if isinstance(v, dict):
            _flatten_dict(v, out, prefix=f"{prefix}{k}/")
        elif isinstance(v, (int, float)):
            out[f"{prefix}{k}"] = v


if __name__ == "__main__":
    main()
