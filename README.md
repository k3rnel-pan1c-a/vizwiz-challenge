# VizWiz AnswerTherapy — DETR-style Answer Grounding

Graduation project: given an image and a question, predict **which regions of
the image contain the answer**, without generating bounding boxes as text.

---

## The Task

The [VizWiz AnswerTherapy](https://vizwiz.org/) dataset provides image–question pairs where
human annotators have drawn polygon outlines around the part of the image that
visually answers the question. The task is to predict a **set of bounding
boxes** that covers all answer regions, and to classify whether the sample
requires **single** or **multiple** groundings.

### Why this is hard

- The number of ground-truth boxes varies per sample (1–3 in this dataset).
- A fixed-output model (e.g., predict exactly one box) cannot handle this.
- Generating boxes as text tokens introduces quantization artifacts and
  makes set prediction non-trivial.

---

## Architecture

```
Image + Question
       │
       ▼
┌──────────────────────────────┐
│  Qwen2.5-VL-7B (frozen)     │  ← visual–language feature extractor
│  output_hidden_states=True   │
└──────────────┬───────────────┘
               │  hidden states H  [B, L, d_vlm=3584]
               ▼
        input_proj (Linear + LayerNorm)
               │  memory  [B, L, d_dec=256]
               ▼
  ┌─────────────────────────────┐
  │  Transformer Decoder (DETR) │  ← N learnable object queries
  │  6 layers, 8 heads          │     attend to memory via cross-attention
  └──────┬───────┬──────────────┘
         │       │
    ┌────┘  ┌────┘
    ▼       ▼
  box_head  obj_head    sm_head (pools queries → 2 logits)
  [B,N,4]   [B,N]       [B,2]
  cxcywh   raw logit    raw logit
  sigmoid                         ← applied only inside forward()
```

### Why Qwen2.5-VL as a feature extractor?

Qwen2.5-VL is a state-of-the-art vision–language model that:
- Natively understands images and text together
- Produces rich contextual embeddings over the joint sequence
- Supports dynamic image resolution (no fixed grid)

We use it **read-only** (frozen weights) as a powerful encoder, then train
only a small DETR-style decoder on top.

### Why a DETR-style set decoder?

DETR (Detection Transformer) uses **N learnable object queries** that each
compete to explain a different region of the image. Training via Hungarian
matching ensures:
- Each GT box is assigned to at most one query
- No duplicate predictions for the same object
- Variable-size output (kept boxes depend on objectness scores)

This avoids the fixed-output limitation and does not require text tokens.

### What are learnable object queries?

Each query is a trainable embedding vector of dimension `d_dec`. During
cross-attention, each query learns to "look" at different parts of the
image–text memory to propose a bounding box. There are N=8 queries (chosen
from dataset statistics; see below).

### What is objectness?

Each query also predicts a raw **objectness logit**. After training, the
sigmoid of this logit is a score in (0, 1) indicating how confident the
model is that this query corresponds to a real answer region. At inference,
only queries with objectness score > threshold (default 0.5) are reported.

---

## N Selection

Running `python src/analyze_dataset.py --include_vqa` on the full labeled
set (4 440 samples) reveals:

| Statistic | Value |
|-----------|-------|
| Max GT boxes per sample | 3 |
| p99 GT boxes | 2 |
| % with 1 GT box | 86.4 % |
| % with >1 GT box | 13.6 % |

Rule applied:
```
N_raw  = max(p99 + 2, 8)   → max(2+2, 8) = 8
N_clean = round up to {8, 10, 16, 20}  → 8
```

**Recommended N = 8.** Override in `configs/base.yaml` → `model.num_queries`.

---

## Single/Multiple Auxiliary Head

The model includes a small MLP that pools the N decoder query outputs and
predicts whether the sample is a *single-grounding* (one answer region) or
*multiple-grounding* (several distinct regions) case.

**Important:** this head is **auxiliary** only. It classifies the sample but
does **not** control how many boxes are kept. The box count at inference is
determined entirely by objectness thresholding.

This means the head and the box count can disagree — this is reported as a
diagnostic metric (`sm_says_single_but_kept_gt1`, etc.).

### Why not derive the single/multiple label from box count?

The dataset provides an authoritative `binary_label` field from human
annotators, which we use directly. Box count after thresholding is noisy
(depends on threshold, model calibration), so using it as a label would
introduce circular dependencies and instability.

---

## Why Hungarian Matching?

A naive loss would compare each predicted box to each GT box independently,
leading to many-to-one matches and duplicate detections. Hungarian matching
finds the **optimal one-to-one assignment** between N predicted queries and K
GT boxes (K ≤ N). This:
- Prevents multiple queries from redundantly predicting the same GT box
- Forces the model to distribute predictions across all GT boxes
- Enables clean unmatched queries to be trained as "no object"

The matching is done under `torch.no_grad()`, so assignment indices are
detached. The losses computed *after* matching remain fully differentiable.

---

## Project Structure

```
project/
  configs/
    base.yaml                   ← all hyperparameters
  src/
    analyze_dataset.py          ← dataset statistics + N recommendation
    train.py                    ← training loop
    evaluate.py                 ← evaluation + qualitative examples
    infer.py                    ← single-image / batch inference
    datasets/
      vizwiz_answertherapy.py   ← dataset loader + collate_fn
    models/
      qwen_backbone.py          ← frozen Qwen2.5-VL wrapper
      grounding_head.py         ← DETR decoder + all heads
      matcher.py                ← Hungarian matcher
      losses.py                 ← L1 + GIoU + BCE_obj + CE_sm
    utils/
      boxes.py                  ← polygon/bbox conversions, IoU, GIoU
      visualization.py          ← draw boxes, qualitative saver
      metrics.py                ← precision/recall/F1, confusion matrix
  README.md
```

---

## Installation

```bash
pip install torch torchvision
pip install transformers accelerate
pip install scipy pillow tqdm pyyaml
pip install tensorboard            # optional logging
pip install wandb                  # optional logging
pip install peft                   # only if using LoRA
```

---

## Dataset Analysis

```bash
cd project
python src/analyze_dataset.py \
    --data_root /kaggle/input/datasets/abdelrhmanshaheen/answer-therapy \
    --include_vqa \
    --splits train val
```

Output: histogram + statistics + recommended N.

---

## Training

```bash
cd project
python src/train.py --config configs/base.yaml

# Override specific values:
python src/train.py --config configs/base.yaml \
    model.num_queries=10 \
    training.epochs=30 \
    loss.lambda_box=5.0 \
    training.batch_size=2
```

The backbone is frozen by default; only the projection layer, DETR decoder,
and three heads are trained. To enable LoRA:

```yaml
# configs/base.yaml
model:
  freeze_backbone: false
  lora:
    enabled: true
    r: 16
    lora_alpha: 32
```

---

## Evaluation

```bash
python src/evaluate.py \
    --config configs/base.yaml \
    --checkpoint outputs/checkpoints/latest.pt \
    --obj_threshold 0.5 \
    --iou_thresholds 0.3 0.5 \
    --save_viz \
    --viz_dir outputs/visualizations \
    --max_viz 30
```

Qualitative examples are saved per outcome type:
- `true_pos/`   — correctly localized boxes (IoU ≥ threshold)
- `false_pos/`  — predicted boxes with no matching GT
- `false_neg/`  — GT boxes not covered by any prediction
- `duplicate/`  — multiple predictions matched to the same GT
- `sm_conflict/` — SM head says single but >1 box kept (or vice versa)

---

## Inference

Single image:
```bash
python src/infer.py \
    --config configs/base.yaml \
    --checkpoint outputs/checkpoints/latest.pt \
    --image path/to/image.jpg \
    --question "What is this?" \
    --obj_threshold 0.5 \
    --out_dir outputs/inference
```

Batch from JSON:
```bash
python src/infer.py \
    --config configs/base.yaml \
    --checkpoint outputs/checkpoints/latest.pt \
    --json_file my_samples.json \
    --img_dir data/images
```

Output JSON per sample:
```json
{
  "image_id": "foo.jpg",
  "question": "What is this?",
  "single_multi_prediction": "single",
  "single_multi_probabilities": {"single": 0.87, "multiple": 0.13},
  "kept_boxes_xyxy": [[x1, y1, x2, y2], ...],
  "kept_box_scores": [0.93, ...],
  "num_kept_boxes": 1
}
```

---

## Ablation Guide

All ablations are controlled via config overrides:

| Ablation | Override |
|----------|----------|
| Number of queries | `model.num_queries=5` (or 10, 16, 20) |
| Frozen vs LoRA | `model.freeze_backbone=false model.lora.enabled=true` |
| Without SM head | `ablation.use_single_multi_head=false` |
| Objectness threshold | `--obj_threshold 0.3` (eval/infer arg) |
| Decoder layers | `ablation.num_decoder_layers=3` |
| d_dec | `ablation.d_dec=128` |
| Loss weights | `loss.lambda_box=2.0 loss.lambda_giou=1.0` |

---

## Model Output Contract

```python
{
    "pred_boxes":               # Float[B, N, 4]  normalized cxcywh in [0,1]
    "pred_obj_logits":          # Float[B, N]     raw objectness logits
    "pred_single_multi_logits": # Float[B, 2]     raw logits (0=single, 1=multi)
}
```

- `pred_obj_logits` are **raw logits** — apply `sigmoid()` at inference
- `pred_single_multi_logits` are **raw logits** — apply `softmax()` at inference
- Boxes are in **normalized cxcywh** — convert with `cxcywh_to_xyxy() * (W, H, W, H)` for pixels
- No masks used anywhere
- No answer text used anywhere in the model

---

## Key Implementation Notes

- Polygons → bounding boxes via min/max of vertices
- Identical annotator polygons on "single" samples deduplicated at IoU > 0.85
- Boxes normalized to `[0, 1]` by image `width` and `height`
- Internal format: `cx, cy, w, h`; only converted to `xyxy` for IoU/GIoU
- Matching is `no_grad`; losses after matching are differentiable
- Zero-GT samples handled: all queries trained as no-object, SM loss still computed
- Variable batch GT counts handled with `list[Tensor]` (not padded)
