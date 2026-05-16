"""
Dataset loader for VizWiz AnswerTherapy grounding task.

Each sample returns:
    image         : PIL.Image (RGB)
    question      : str
    gt_boxes      : Float[K, 4]  normalized cxcywh in [0, 1], K = num GT boxes
    single_multi  : int  0 = single grounding, 1 = multiple grounding
    image_id      : str  (for evaluation / visualization)
    orig_size     : (H, W) original image size

GT boxes are derived by:
  1. converting each polygon annotation to its bounding box
  2. deduplicating overlapping boxes (IoU > 0.85) so identical
     annotator polygons on 'single' samples collapse to 1 GT box
  3. normalizing to cx, cy, w, h in [0, 1]

The single_multi label comes from the dataset's 'binary_label' field
("single" → 0, "multiple" → 1).  When that field is absent the label
is derived from the deduplicated box count: 0 if ≤ 1 box, else 1.

VQA samples use a different image naming convention
(COCO_train2014_{image_id:012d}.jpg), handled automatically.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


# -----------------------------------------------------------------------
# Cached dataset (uses pre-extracted backbone features)
# -----------------------------------------------------------------------

class CachedVizWizDataset(Dataset):
    """Loads pre-extracted backbone features written by cache_features.py.

    Directory layout expected:
        cache_dir/{split}/{idx:06d}.pt       — {"hidden": Tensor[L, d_vlm], "key_pad": Tensor[L]}
        cache_dir/{split}_manifest.json      — {"d_vlm": int, "samples": [...]}
    """

    def __init__(self, cache_dir: str | Path, split: str) -> None:
        self.feat_dir = Path(cache_dir) / split
        manifest_path = Path(cache_dir) / f"{split}_manifest.json"
        with open(manifest_path) as f:
            data = json.load(f)
        self.d_vlm   = data["d_vlm"]
        self.samples = data["samples"]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        meta = self.samples[idx]
        feat = torch.load(self.feat_dir / f"{idx:06d}.pt", weights_only=True)

        gt_b = meta["gt_boxes"]
        gt_boxes = torch.tensor(gt_b, dtype=torch.float32) if gt_b else torch.zeros(0, 4)

        return {
            "hidden":       feat["hidden"],           # [L, d_vlm]  bfloat16
            "key_pad":      feat["key_pad"],           # [L]  bool
            "gt_boxes":     gt_boxes,                  # [K, 4]
            "single_multi": int(meta["single_multi"]),
            "image_id":     meta["image_id"],
            "orig_size":    tuple(meta["orig_size"]),
        }


def cached_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate pre-cached features, right-padding sequences to the batch max length."""
    hidden_list  = [item["hidden"]  for item in batch]   # each [L_i, d_vlm]
    key_pad_list = [item["key_pad"] for item in batch]   # each [L_i]  bool

    max_len = max(h.shape[0] for h in hidden_list)
    d_vlm   = hidden_list[0].shape[1]
    B       = len(batch)
    dtype   = hidden_list[0].dtype

    hidden_out  = torch.zeros(B, max_len, d_vlm, dtype=dtype)
    key_pad_out = torch.ones(B, max_len, dtype=torch.bool)   # True = padding

    for i, (h, kp) in enumerate(zip(hidden_list, key_pad_list)):
        L = h.shape[0]
        hidden_out[i, :L]  = h
        key_pad_out[i, :L] = kp

    gt_boxes  = [item["gt_boxes"]     for item in batch]
    sm_labels = torch.tensor([item["single_multi"] for item in batch], dtype=torch.long)
    image_ids = [item["image_id"]     for item in batch]
    orig_sizes = [item["orig_size"]   for item in batch]

    return {
        "hidden":       hidden_out,    # [B, max_L, d_vlm]
        "key_pad":      key_pad_out,   # [B, max_L]
        "gt_boxes":     gt_boxes,
        "single_multi": sm_labels,
        "image_ids":    image_ids,
        "orig_sizes":   orig_sizes,
    }

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.boxes import (
    deduplicate_boxes,
    polygon_to_xyxy,
    polygon_to_normalized_cxcywh,
)


_BINARY_MAP = {"single": 0, "multiple": 1}
_DEDUP_IOU = 0.85


class VizWizAnswerTherapyDataset(Dataset):
    """PyTorch dataset for VizWiz AnswerTherapy grounding.

    Args:
        json_paths: one or more paths to annotation JSON files.
        img_dirs: corresponding image directories (one per json_paths entry,
                  or a single directory applied to all).
        processor: optional Qwen2.5-VL processor; if supplied images are
                   preprocessed here and the raw PIL image is also returned.
                   If None, raw PIL images are returned for external preprocessing.
        max_size: optional (width, height) to resize images to before returning.
        split: "train" | "val" | "test" — controls whether GT labels are expected.
    """

    def __init__(
        self,
        json_paths: list[str | Path],
        img_dirs: list[str | Path] | str | Path,
        processor=None,
        max_size: tuple[int, int] | None = None,
        split: str = "train",
    ) -> None:
        self.processor = processor
        self.max_size = max_size
        self.split = split

        # Normalize img_dirs to a list aligned with json_paths
        if isinstance(img_dirs, (str, Path)):
            img_dirs = [img_dirs] * len(json_paths)
        assert len(img_dirs) == len(json_paths), "img_dirs must match json_paths length"

        self.samples: list[dict] = []
        for jpath, idir in zip(json_paths, img_dirs):
            loaded = _load_json(Path(jpath))
            for sample in loaded:
                sample["_img_dir"] = str(idir)
                self.samples.append(sample)

    # ------------------------------------------------------------------ #
    # Dataset interface
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        image_id = str(sample["image_id"])
        img_dir = Path(sample["_img_dir"])
        img_path = _resolve_image_path(image_id, img_dir)

        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        if self.max_size is not None:
            image = _resize_image(image, self.max_size)

        question = str(sample.get("question", ""))

        item: dict[str, Any] = {
            "image": image,
            "question": question,
            "image_id": image_id,
            "orig_size": (orig_h, orig_w),
        }

        # GT boxes and label are only available for labeled splits
        if self.split != "test" and sample.get("grounding_labels"):
            gt_boxes, single_multi = _process_groundings(sample)
            item["gt_boxes"] = gt_boxes            # Float[K, 4] cxcywh normalized
            item["single_multi"] = single_multi    # int {0, 1}
        else:
            item["gt_boxes"] = torch.zeros(0, 4)
            item["single_multi"] = 0

        return item


# -----------------------------------------------------------------------
# Collate function
# -----------------------------------------------------------------------

def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Custom collate that handles variable-length GT box tensors.

    Images and questions are left as lists (the Qwen processor handles
    batching). GT boxes remain as a list of tensors (variable K per sample).
    """
    images    = [item["image"] for item in batch]
    questions = [item["question"] for item in batch]
    gt_boxes  = [item["gt_boxes"] for item in batch]    # list of Float[K_i, 4]
    sm_labels = torch.tensor([item["single_multi"] for item in batch], dtype=torch.long)
    image_ids = [item["image_id"] for item in batch]
    orig_sizes = [item["orig_size"] for item in batch]

    return {
        "images": images,
        "questions": questions,
        "gt_boxes": gt_boxes,       # list of tensors, one per sample
        "single_multi": sm_labels,  # Long[B]
        "image_ids": image_ids,
        "orig_sizes": orig_sizes,
    }


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------

def _load_json(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _resolve_image_path(image_id: str, img_dir: Path) -> Path:
    """Resolve the image file path handling both VizWiz and VQA naming."""
    # Direct name (VizWiz files include extension)
    if image_id.endswith(".jpg") or image_id.endswith(".png"):
        return img_dir / image_id

    # VQA: image_id is a numeric string — COCO naming convention
    try:
        numeric_id = int(image_id)
        return img_dir / f"COCO_train2014_{numeric_id:012d}.jpg"
    except ValueError:
        pass

    # Fallback: try as-is with .jpg extension
    return img_dir / f"{image_id}.jpg"


def _resize_image(image: Image.Image, max_size: tuple[int, int]) -> Image.Image:
    """Resize image so neither dimension exceeds max_size, preserving aspect ratio."""
    max_w, max_h = max_size
    w, h = image.size
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)
    return image


def _process_groundings(sample: dict) -> tuple[Tensor, int]:
    """Convert grounding_labels polygons to deduplicated normalized cxcywh boxes.

    Returns:
        gt_boxes     : Float[K, 4]  normalized cxcywh.
        single_multi : int  0 = single, 1 = multiple.
    """
    w = float(sample["width"])
    h = float(sample["height"])
    polygons: list[list[dict]] = sample["grounding_labels"]

    # Convert each polygon to its xyxy bounding box (absolute pixels)
    raw_xyxy = [polygon_to_xyxy(p) for p in polygons if p]
    # Deduplicate (removes repeated annotator polygons for single-label samples)
    dedup_xyxy = deduplicate_boxes(raw_xyxy, iou_threshold=_DEDUP_IOU)

    # Normalize and convert to cxcywh
    boxes_cxcywh: list[list[float]] = []
    for x1, y1, x2, y2 in dedup_xyxy:
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        # Clamp to [0, 1]
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        bw = max(0.0, min(1.0, bw))
        bh = max(0.0, min(1.0, bh))
        boxes_cxcywh.append([cx, cy, bw, bh])

    if boxes_cxcywh:
        gt_boxes = torch.tensor(boxes_cxcywh, dtype=torch.float32)
    else:
        gt_boxes = torch.zeros(0, 4, dtype=torch.float32)

    # Single/multiple label: prefer the dataset field, fall back to box count
    binary_label = sample.get("binary_label", "")
    if binary_label in _BINARY_MAP:
        single_multi = _BINARY_MAP[binary_label]
    else:
        single_multi = 0 if len(boxes_cxcywh) <= 1 else 1

    return gt_boxes, single_multi


# -----------------------------------------------------------------------
# Factory helper
# -----------------------------------------------------------------------

def build_datasets(
    cfg: dict,
    processor=None,
    max_size: tuple[int, int] | None = None,
) -> tuple["VizWizAnswerTherapyDataset", "VizWizAnswerTherapyDataset"]:
    """Build train and val datasets from config dict.

    Expected config keys (mirror configs/base.yaml):
        data.root, data.train_json, data.val_json,
        data.train_json_extra (optional), data.val_json_extra (optional),
        data.train_img_dir, data.val_img_dir
    """
    root = Path(cfg["data"]["root"])
    train_img_dir = root / cfg["data"]["train_img_dir"]
    val_img_dir   = root / cfg["data"]["val_img_dir"]

    train_jsons = [root / cfg["data"]["train_json"]]
    val_jsons   = [root / cfg["data"]["val_json"]]
    train_dirs  = [train_img_dir]
    val_dirs    = [val_img_dir]

    if cfg["data"].get("train_json_extra"):
        train_jsons.append(root / cfg["data"]["train_json_extra"])
        train_dirs.append(train_img_dir)
    if cfg["data"].get("val_json_extra"):
        val_jsons.append(root / cfg["data"]["val_json_extra"])
        val_dirs.append(val_img_dir)

    train_ds = VizWizAnswerTherapyDataset(
        json_paths=train_jsons,
        img_dirs=train_dirs,
        processor=processor,
        max_size=max_size,
        split="train",
    )
    val_ds = VizWizAnswerTherapyDataset(
        json_paths=val_jsons,
        img_dirs=val_dirs,
        processor=processor,
        max_size=max_size,
        split="val",
    )
    return train_ds, val_ds
