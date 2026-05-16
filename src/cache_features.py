"""One-time backbone feature caching for fast training.

Since the Qwen2.5-VL backbone is frozen, hidden states for every
(image, question) pair are identical every epoch.  This script runs
the backbone once and saves results to disk so that subsequent training
epochs skip the expensive 3B-parameter forward pass entirely.

Usage:
    python src/cache_features.py --config configs/base.yaml \\
        --cache_dir /kaggle/working/feature_cache

    # Optional batch size override (default 4):
    python src/cache_features.py --config configs/base.yaml \\
        --cache_dir /kaggle/working/feature_cache --batch_size 2

Then train using the cache:
    python src/train.py --config configs/base.yaml \\
        --cache_dir /kaggle/working/feature_cache

Disk usage: ~2 MB per sample in bfloat16 (varies with image resolution).
For ~5000 samples at max_pixels=262144 expect roughly 10 GB total.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from datasets.vizwiz_answertherapy import collate_fn, build_datasets
from models.qwen_backbone import QwenBackbone
from train import load_config


@torch.no_grad()
def cache_split(
    backbone: QwenBackbone,
    dataset,
    cache_dir: Path,
    split: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> None:
    feat_dir = cache_dir / split
    feat_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,          # sequential so idx matches filename
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
    )

    manifest: list[dict] = []
    global_idx = 0

    pbar = tqdm(loader, desc=f"Caching {split}")
    for batch in pbar:
        images    = batch["images"]
        questions = batch["questions"]
        gt_boxes  = batch["gt_boxes"]
        sm_labels = batch["single_multi"]
        image_ids = batch["image_ids"]
        orig_sizes = batch["orig_sizes"]

        # [B, L, d_vlm], [B, L]
        hidden, key_pad = backbone(images, questions, device=device)
        hidden  = hidden.cpu().to(torch.bfloat16)   # compact on-disk dtype
        key_pad = key_pad.cpu()                      # bool

        B = hidden.shape[0]
        for i in range(B):
            torch.save(
                {"hidden": hidden[i], "key_pad": key_pad[i]},
                feat_dir / f"{global_idx:06d}.pt",
            )
            gt_b = gt_boxes[i]
            manifest.append({
                "image_id":    image_ids[i],
                "gt_boxes":    gt_b.tolist(),
                "single_multi": int(sm_labels[i].item()),
                "orig_size":   list(orig_sizes[i]),
            })
            global_idx += 1

        pbar.set_postfix({"saved": global_idx})

    manifest_path = cache_dir / f"{split}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({"d_vlm": backbone.d_vlm, "num_samples": global_idx, "samples": manifest}, f)

    print(f"[{split}] Cached {global_idx} samples → {feat_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-extract backbone features for fast training")
    parser.add_argument("--config",     default="configs/base.yaml")
    parser.add_argument("--cache_dir",  required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--overrides",  nargs="*", default=[])
    args, extra = parser.parse_known_args()
    all_overrides = (args.overrides or []) + [e for e in extra if "=" in e]

    cfg = load_config(args.config, all_overrides)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    m = cfg["model"]
    _dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = _dtype_map.get(m.get("torch_dtype", "bfloat16"), torch.bfloat16)

    max_memory = None
    if m.get("max_memory_per_gpu") and torch.cuda.is_available():
        max_memory = {i: m["max_memory_per_gpu"] for i in range(torch.cuda.device_count())}

    backbone = QwenBackbone(
        model_name=m["backbone"],
        freeze=True,
        torch_dtype=torch_dtype,
        device_map=m.get("device_map", "balanced"),
        max_memory=max_memory,
        attn_implementation=m.get("attn_implementation", "sdpa"),
        max_pixels=m.get("max_pixels"),
        min_pixels=m.get("min_pixels"),
    )
    backbone.eval()
    print(f"Backbone d_vlm={backbone.d_vlm}")

    num_workers = cfg["training"].get("num_workers", 2)
    train_ds, val_ds = build_datasets(cfg)
    total = len(train_ds) + len(val_ds)
    print(f"Train: {len(train_ds)},  Val: {len(val_ds)}")

    # Rough disk estimate: 500 tokens × d_vlm × 2 bytes (bfloat16) per sample
    est_gb = (500 * backbone.d_vlm * 2 * total) / (1024 ** 3)
    print(f"Estimated disk usage: ~{est_gb:.1f} GB")

    cache_split(backbone, train_ds, cache_dir, "train", args.batch_size, num_workers, device)
    cache_split(backbone, val_ds,   cache_dir, "val",   args.batch_size, num_workers, device)

    print(f"\nDone. Train with:")
    print(f"  python src/train.py --config configs/base.yaml --cache_dir {cache_dir}")


if __name__ == "__main__":
    main()
