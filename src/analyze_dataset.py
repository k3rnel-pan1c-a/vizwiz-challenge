"""
Dataset analysis script for VizWiz AnswerTherapy.

Computes per-sample GT bounding box counts, then recommends N (number of
DETR queries) using the rule:

    N_raw  = max(p99 + 2, 8)
    N      = round N_raw up to the nearest clean value in {8, 10, 16, 20}

Run:
    python src/analyze_dataset.py \
        --data_root /kaggle/input/datasets/abdelrhmanshaheen/answer-therapy \
        [--include_vqa]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

# Local import — works when run from project root or with PYTHONPATH set.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.boxes import polygon_to_xyxy, deduplicate_boxes


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def load_json(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def derive_gt_boxes(sample: dict, iou_threshold: float = 0.85) -> list:
    """Return deduplicated xyxy GT boxes for a single sample.

    Polygons that overlap heavily (IoU > iou_threshold) are collapsed to
    a single representative box.  This correctly reduces 2-4 identical
    annotator polygons on 'single' samples to 1 GT box.
    """
    polygons = sample.get("grounding_labels", [])
    raw_boxes = [polygon_to_xyxy(p) for p in polygons if p]
    if not raw_boxes:
        return []
    return deduplicate_boxes(raw_boxes, iou_threshold=iou_threshold)


def _clean_n(n_raw: int) -> int:
    """Round up to the nearest clean value in {8, 10, 16, 20}."""
    for clean in [8, 10, 16, 20]:
        if clean >= n_raw:
            return clean
    return n_raw   # larger than 20 — keep as is


def analyze_samples(samples: list[dict], iou_threshold: float = 0.85) -> np.ndarray:
    """Return array of GT box counts per sample."""
    counts = []
    for s in samples:
        boxes = derive_gt_boxes(s, iou_threshold=iou_threshold)
        counts.append(len(boxes))
    return np.array(counts, dtype=int)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze VizWiz AnswerTherapy dataset")
    parser.add_argument("--data_root", default="/kaggle/input/datasets/abdelrhmanshaheen/answer-therapy")
    parser.add_argument("--include_vqa", action="store_true",
                        help="Also include VQA_train.json / VQA_val.json")
    parser.add_argument("--iou_threshold", type=float, default=0.85,
                        help="IoU threshold for polygon deduplication")
    parser.add_argument("--splits", nargs="+", default=["train", "val"],
                        choices=["train", "val"])
    args = parser.parse_args()

    root = Path(args.data_root)

    # ---- collect samples from chosen splits ----
    all_samples: list[dict] = []
    per_split: dict[str, list[dict]] = {}

    split_files: dict[str, list[str]] = {
        "train": ["VizWiz_train.json"],
        "val":   ["VizWiz_val.json"],
    }
    if args.include_vqa:
        split_files["train"].append("VQA_train.json")
        split_files["val"].append("VQA_val.json")

    for split in args.splits:
        split_samples: list[dict] = []
        for fname in split_files.get(split, []):
            fpath = root / fname
            if fpath.exists():
                loaded = load_json(fpath)
                # Filter samples that actually have grounding labels
                labeled = [s for s in loaded if s.get("grounding_labels")]
                split_samples.extend(labeled)
                print(f"  Loaded {len(labeled)} labeled samples from {fname}")
            else:
                print(f"  WARNING: {fpath} not found — skipping")
        per_split[split] = split_samples
        all_samples.extend(split_samples)

    print(f"\nTotal labeled samples across selected splits: {len(all_samples)}\n")

    # ---- per-split stats ----
    for split, samples in per_split.items():
        if not samples:
            continue
        counts = analyze_samples(samples, iou_threshold=args.iou_threshold)
        print_stats(split, counts, samples)

    # ---- combined stats + N recommendation ----
    all_counts = analyze_samples(all_samples, iou_threshold=args.iou_threshold)
    print_stats("COMBINED", all_counts, all_samples)

    p99 = int(np.percentile(all_counts, 99))
    n_raw = max(p99 + 2, 8)
    n_clean = _clean_n(n_raw)
    print("=" * 60)
    print(f"RECOMMENDED N (DETR queries):")
    print(f"  p99 GT boxes   = {p99}")
    print(f"  N_raw          = max({p99} + 2, 8) = {n_raw}")
    print(f"  N (clean)      = {n_clean}   <- use this in configs/base.yaml")
    print("=" * 60)
    print(f"\nTo apply: set  model.num_queries: {n_clean}  in configs/base.yaml\n")


def print_stats(split: str, counts: np.ndarray, samples: list[dict]) -> None:
    n = len(counts)
    if n == 0:
        return

    # ---- histogram ----
    hist = Counter(counts.tolist())

    print("=" * 60)
    print(f"Split: {split}  ({n} samples)")
    print("=" * 60)
    print("GT box count histogram:")
    for k in sorted(hist.keys()):
        bar = "#" * min(40, hist[k])
        print(f"  {k:3d} GT box(es): {hist[k]:5d} samples  {bar}")

    # ---- summary statistics ----
    print(f"\nSummary statistics (GT boxes per sample):")
    print(f"  mean   : {counts.mean():.3f}")
    print(f"  median : {np.median(counts):.1f}")
    print(f"  max    : {counts.max()}")
    print(f"  p90    : {np.percentile(counts, 90):.1f}")
    print(f"  p95    : {np.percentile(counts, 95):.1f}")
    print(f"  p99    : {np.percentile(counts, 99):.1f}")

    zero = (counts == 0).sum()
    one  = (counts == 1).sum()
    more = (counts > 1).sum()
    print(f"\n  % with 0 GT boxes   : {100 * zero / n:.1f}%")
    print(f"  % with 1 GT box     : {100 * one  / n:.1f}%")
    print(f"  % with >1 GT boxes  : {100 * more / n:.1f}%")

    # ---- binary_label distribution ----
    bl = Counter(s.get("binary_label", "unknown") for s in samples)
    print(f"\n  binary_label distribution:")
    for label, cnt in sorted(bl.items()):
        print(f"    {label}: {cnt} ({100 * cnt / n:.1f}%)")

    print()


if __name__ == "__main__":
    main()
