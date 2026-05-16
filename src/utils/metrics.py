"""
Evaluation metrics for VizWiz AnswerTherapy grounding.

Metrics computed:
    Box grounding:
        - Recall@IoU (per threshold: 0.3, 0.5)
        - Precision@IoU (per threshold)
        - F1@IoU (per threshold)
        - Mean IoU of matched prediction/GT pairs
        - Number of predictions kept after objectness thresholding

    Single/multiple classification:
        - Accuracy
        - Precision / Recall / F1 (class 1 = multiple)
        - Confusion matrix [2 x 2]

    Subset analysis:
        - All metrics split by GT single/multiple label

    Diagnostic:
        - Predicted single/multiple label vs. number of kept boxes

All inputs use normalized cxcywh boxes (IoU computed in xyxy space).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from torch import Tensor

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.boxes import cxcywh_to_xyxy, box_iou


# -----------------------------------------------------------------------
# Sample-level accumulator
# -----------------------------------------------------------------------

class GroundingMetrics:
    """Accumulate predictions across the dataset, then compute metrics.

    Usage:
        metrics = GroundingMetrics(iou_thresholds=[0.3, 0.5], obj_threshold=0.5)
        for batch in val_loader:
            metrics.update(...)
        report = metrics.compute()
    """

    def __init__(
        self,
        iou_thresholds: list[float] | None = None,
        obj_threshold: float = 0.5,
    ) -> None:
        self.iou_thresholds = iou_thresholds or [0.3, 0.5]
        self.obj_threshold = obj_threshold
        self.reset()

    def reset(self) -> None:
        self._records: list[dict] = []

    def update(
        self,
        pred_boxes: list[Tensor] | Tensor,
        pred_obj_logits: list[Tensor] | Tensor,
        pred_sm_logits: Tensor,
        gt_boxes_list: list[Tensor],
        sm_labels: Tensor,
    ) -> None:
        """Add a batch of predictions.

        Args:
            pred_boxes      : Float[B, N, 4] cxcywh or list of Float[N,4].
            pred_obj_logits : Float[B, N] raw logits or list of Float[N].
            pred_sm_logits  : Float[B, 2] raw logits.
            gt_boxes_list   : list of Float[K_i, 4] GT boxes per sample.
            sm_labels       : Long[B] GT single/multiple labels.
        """
        if isinstance(pred_boxes, Tensor):
            pred_boxes = list(pred_boxes)
        if isinstance(pred_obj_logits, Tensor):
            pred_obj_logits = list(pred_obj_logits)

        B = len(pred_boxes)
        sm_preds = pred_sm_logits.argmax(dim=1).cpu()

        for b in range(B):
            pb   = pred_boxes[b].cpu()        # [N, 4] cxcywh
            pol  = pred_obj_logits[b].cpu()   # [N]
            gt_b = gt_boxes_list[b].cpu()     # [K, 4] cxcywh
            sm_label = int(sm_labels[b])
            sm_pred  = int(sm_preds[b])

            scores = torch.sigmoid(pol)   # [N]
            keep   = scores > self.obj_threshold
            kept_pb     = pb[keep]        # [Nk, 4]
            kept_scores = scores[keep]    # [Nk]

            self._records.append({
                "kept_pb":     kept_pb,
                "kept_scores": kept_scores,
                "gt_boxes":    gt_b,
                "sm_label":    sm_label,
                "sm_pred":     sm_pred,
                "n_kept":      int(keep.sum()),
                "n_gt":        gt_b.size(0),
            })

    # ------------------------------------------------------------------ #
    # Compute
    # ------------------------------------------------------------------ #

    def compute(self) -> dict:
        """Compute all metrics and return as a nested dict."""
        if not self._records:
            return {}

        results: dict = {}

        # ---- split by GT label ----
        subsets = {
            "all":      self._records,
            "single":   [r for r in self._records if r["sm_label"] == 0],
            "multiple": [r for r in self._records if r["sm_label"] == 1],
        }

        for subset_name, records in subsets.items():
            if not records:
                continue
            results[subset_name] = self._compute_subset(records)

        # ---- single/multiple classification (on full set) ----
        sm_metrics = self._compute_sm_metrics(self._records)
        results["single_multi_cls"] = sm_metrics

        # ---- diagnostic: sm head vs. kept box count ----
        results["diagnostic"] = self._diagnostic(self._records)

        return results

    def _compute_subset(self, records: list[dict]) -> dict:
        n = len(records)
        out: dict = {"num_samples": n}

        # ---- box grounding metrics per IoU threshold ----
        for thr in self.iou_thresholds:
            tp_counts = []
            fp_counts = []
            fn_counts = []
            matched_ious = []

            for r in records:
                kept_pb = r["kept_pb"]   # [Nk, 4]
                gt_b    = r["gt_boxes"]  # [K, 4]
                K = gt_b.size(0)
                Nk = kept_pb.size(0)

                if K == 0 and Nk == 0:
                    tp_counts.append(0)
                    fp_counts.append(0)
                    fn_counts.append(0)
                    continue
                if K == 0:
                    tp_counts.append(0)
                    fp_counts.append(Nk)
                    fn_counts.append(0)
                    continue
                if Nk == 0:
                    tp_counts.append(0)
                    fp_counts.append(0)
                    fn_counts.append(K)
                    continue

                kp_xyxy = cxcywh_to_xyxy(kept_pb)   # [Nk, 4]
                gt_xyxy = cxcywh_to_xyxy(gt_b)      # [K, 4]
                iou_mat = box_iou(kp_xyxy, gt_xyxy)  # [Nk, K]

                max_iou_per_pred = iou_mat.max(dim=1).values   # [Nk]
                max_iou_per_gt   = iou_mat.max(dim=0).values   # [K]

                n_tp = int((max_iou_per_pred >= thr).sum())
                n_fp = Nk - n_tp
                n_fn = int((max_iou_per_gt < thr).sum())

                tp_counts.append(n_tp)
                fp_counts.append(n_fp)
                fn_counts.append(n_fn)

                # Collect per-pair matched IoUs for mean IoU
                matched_ious.extend(
                    max_iou_per_pred[max_iou_per_pred >= thr].tolist()
                )

            total_tp = sum(tp_counts)
            total_fp = sum(fp_counts)
            total_fn = sum(fn_counts)

            precision = total_tp / (total_tp + total_fp + 1e-9)
            recall    = total_tp / (total_tp + total_fn + 1e-9)
            f1 = 2 * precision * recall / (precision + recall + 1e-9)

            thr_key = f"iou{int(thr*100)}"
            out[thr_key] = {
                "precision":  round(precision, 4),
                "recall":     round(recall, 4),
                "f1":         round(f1, 4),
                "tp":         total_tp,
                "fp":         total_fp,
                "fn":         total_fn,
                "mean_iou_matched": round(float(np.mean(matched_ious)) if matched_ious else 0.0, 4),
            }

        # ---- average number of kept boxes ----
        out["avg_kept"] = round(float(np.mean([r["n_kept"] for r in records])), 3)
        out["avg_gt"]   = round(float(np.mean([r["n_gt"]   for r in records])), 3)

        return out

    def _compute_sm_metrics(self, records: list[dict]) -> dict:
        gt_labels   = [r["sm_label"] for r in records]
        pred_labels = [r["sm_pred"]  for r in records]

        n = len(gt_labels)
        acc = sum(g == p for g, p in zip(gt_labels, pred_labels)) / n

        # Binary metrics for class 1 (multiple)
        tp = sum(g == 1 and p == 1 for g, p in zip(gt_labels, pred_labels))
        fp = sum(g == 0 and p == 1 for g, p in zip(gt_labels, pred_labels))
        fn = sum(g == 1 and p == 0 for g, p in zip(gt_labels, pred_labels))
        tn = sum(g == 0 and p == 0 for g, p in zip(gt_labels, pred_labels))

        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)

        return {
            "accuracy":    round(acc, 4),
            "precision":   round(precision, 4),   # for class "multiple"
            "recall":      round(recall, 4),
            "f1":          round(f1, 4),
            "confusion_matrix": {
                "TN": tn, "FP": fp,
                "FN": fn, "TP": tp,
            },
        }

    def _diagnostic(self, records: list[dict]) -> dict:
        """Compare sm_head prediction to number of kept boxes."""
        sm_single_keptgt1  = 0  # head says single, but kept > 1 box
        sm_multi_keptle1   = 0  # head says multiple, but kept <= 1 box
        sm_agree_count     = 0  # head agrees with box count
        total = len(records)

        for r in records:
            pred_sm = r["sm_pred"]
            n_kept  = r["n_kept"]
            box_sm  = 0 if n_kept <= 1 else 1
            if pred_sm == box_sm:
                sm_agree_count += 1
            if pred_sm == 0 and n_kept > 1:
                sm_single_keptgt1 += 1
            if pred_sm == 1 and n_kept <= 1:
                sm_multi_keptle1 += 1

        return {
            "sm_head_vs_box_count_agreement": round(sm_agree_count / total, 4),
            "sm_says_single_but_kept_gt1": sm_single_keptgt1,
            "sm_says_multiple_but_kept_le1": sm_multi_keptle1,
        }


# -----------------------------------------------------------------------
# Formatting helper
# -----------------------------------------------------------------------

def format_metrics(results: dict, indent: int = 0) -> str:
    """Pretty-print nested metrics dict."""
    lines = []
    pad = "  " * indent
    for k, v in results.items():
        if isinstance(v, dict):
            lines.append(f"{pad}{k}:")
            lines.append(format_metrics(v, indent + 1))
        else:
            lines.append(f"{pad}{k}: {v}")
    return "\n".join(lines)
