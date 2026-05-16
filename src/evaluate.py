"""
Evaluation script for VizWiz AnswerTherapy grounding model.

Loads a trained GroundingHead checkpoint and evaluates on the val set.
Produces:
    - Full metrics report (JSON + printed)
    - Qualitative visualizations organized by outcome type

Usage:
    python src/evaluate.py \
        --config configs/base.yaml \
        --checkpoint outputs/checkpoints/latest.pt \
        [--obj_threshold 0.5] \
        [--iou_thresholds 0.3 0.5] \
        [--save_viz] \
        [--viz_dir outputs/visualizations] \
        [--max_viz 20]
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

from datasets.vizwiz_answertherapy import build_datasets, collate_fn
from models.qwen_backbone import QwenBackbone
from models.grounding_head import GroundingHead
from models.matcher import HungarianMatcher
from models.losses import GroundingLoss
from utils.metrics import GroundingMetrics, format_metrics
from utils.visualization import QualitativeSaver
from train import load_config, build_model


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",         default="configs/base.yaml")
    parser.add_argument("--checkpoint",     required=True)
    parser.add_argument("--obj_threshold",  type=float, default=None)
    parser.add_argument("--iou_thresholds", type=float, nargs="+", default=None)
    parser.add_argument("--save_viz",       action="store_true")
    parser.add_argument("--viz_dir",        default=None)
    parser.add_argument("--max_viz",        type=int, default=20)
    parser.add_argument("--split",          default="val", choices=["train", "val"])
    args = parser.parse_args()

    cfg = load_config(args.config)

    obj_thr = args.obj_threshold or cfg["eval"].get("obj_threshold", 0.5)
    iou_thr = args.iou_thresholds or cfg["eval"].get("iou_thresholds", [0.3, 0.5])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Model ----
    backbone, head = build_model(cfg, device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # ---- Dataset ----
    train_ds, val_ds = build_datasets(cfg)
    eval_ds = val_ds if args.split == "val" else train_ds

    loader = DataLoader(
        eval_ds,
        batch_size=cfg["eval"].get("batch_size", 4),
        shuffle=False,
        num_workers=cfg["eval"].get("num_workers", 2),
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # ---- Visualization saver ----
    viz_saver = None
    if args.save_viz:
        viz_dir = Path(args.viz_dir or cfg["output"].get("viz_dir", "outputs/visualizations"))
        viz_saver = QualitativeSaver(
            save_dir=viz_dir,
            max_per_type=args.max_viz,
            iou_threshold=iou_thr[0] if iou_thr else 0.5,
            obj_threshold=obj_thr,
        )
        print(f"Saving visualizations to: {viz_dir}")

    # ---- Evaluate ----
    metrics = GroundingMetrics(iou_thresholds=iou_thr, obj_threshold=obj_thr)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            images    = batch["images"]
            questions = batch["questions"]
            gt_boxes  = batch["gt_boxes"]
            sm_labels = batch["single_multi"].to(device)
            image_ids = batch["image_ids"]

            hidden, key_pad = backbone(images, questions, device=device)
            hidden  = hidden.to(device)
            key_pad = key_pad.to(device)

            out = head(hidden, key_padding_mask=key_pad)
            pred_boxes = out["pred_boxes"]
            pred_obj   = out["pred_obj_logits"]
            pred_sm    = out["pred_single_multi_logits"]

            metrics.update(
                pred_boxes=pred_boxes,
                pred_obj_logits=pred_obj,
                pred_sm_logits=pred_sm,
                gt_boxes_list=gt_boxes,
                sm_labels=sm_labels,
            )

            # ---- Visualization ----
            if viz_saver is not None:
                sm_preds = pred_sm.argmax(dim=1)
                scores   = torch.sigmoid(pred_obj)
                for b in range(len(images)):
                    viz_saver.add(
                        image=images[b],
                        pred_boxes=pred_boxes[b].cpu(),
                        pred_scores=scores[b].cpu(),
                        gt_boxes=gt_boxes[b].cpu(),
                        sm_pred=int(sm_preds[b]),
                        sm_gt=int(sm_labels[b]),
                        image_id=image_ids[b],
                        question=questions[b],
                    )

    # ---- Report ----
    results = metrics.compute()
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(format_metrics(results))

    # Save JSON report
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"eval_{args.split}_thr{int(obj_thr*100)}.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nReport saved to: {report_path}")

    # ---- Per-threshold summary ----
    print("\n--- Quick Summary ---")
    for subset in ["all", "single", "multiple"]:
        if subset not in results:
            continue
        print(f"\n  Subset: {subset} ({results[subset].get('num_samples', '?')} samples)")
        for thr_key in [f"iou{int(t*100)}" for t in iou_thr]:
            d = results[subset].get(thr_key, {})
            print(f"    @IoU={thr_key}: "
                  f"P={d.get('precision','?'):.3f} "
                  f"R={d.get('recall','?'):.3f} "
                  f"F1={d.get('f1','?'):.3f} "
                  f"mIoU_matched={d.get('mean_iou_matched','?'):.3f}")

    sm_cls = results.get("single_multi_cls", {})
    print(f"\n  Single/Multi cls: "
          f"Acc={sm_cls.get('accuracy','?'):.3f}  "
          f"F1={sm_cls.get('f1','?'):.3f}")
    cm = sm_cls.get("confusion_matrix", {})
    if cm:
        print(f"  Confusion matrix: TN={cm['TN']} FP={cm['FP']} FN={cm['FN']} TP={cm['TP']}")

    diag = results.get("diagnostic", {})
    print(f"\n  SM head / box count agreement: {diag.get('sm_head_vs_box_count_agreement','?'):.3f}")


if __name__ == "__main__":
    main()
