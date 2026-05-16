"""
Qwen2.5-VL backbone wrapper.

Used purely as a frozen feature extractor: image + question text
are fed in, and the last hidden states H of shape [B, L, d_vlm]
are returned.  No text is generated.

Key design choices:
- output_hidden_states=True  — we read the last layer's hidden states
- The processor handles dynamic image tokens automatically (no fixed grid)
- Variable-length sequences across the batch are padded by the processor;
  we expose the attention mask so the decoder can use it as a
  key_padding_mask (True = position is padding)
- The entire Qwen2.5-VL model is frozen; only the projection layer +
  DETR decoder + heads are trained

Optional LoRA support (requires peft):
    backbone = QwenBackbone(model_name, freeze=False)
    backbone.enable_lora(r=16, lora_alpha=32, ...)
"""

from __future__ import annotations

import warnings
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class QwenBackbone(nn.Module):
    """Qwen2.5-VL visual-language feature extractor.

    Args:
        model_name: HuggingFace model identifier, e.g. "Qwen/Qwen2.5-VL-7B-Instruct".
        freeze: if True (default), freeze all backbone parameters.
        torch_dtype: dtype for backbone weights (torch.bfloat16 recommended).
        device_map: passed to from_pretrained; use "auto" for multi-GPU.
        attn_implementation: "flash_attention_2" when Flash-Attn is installed,
                             otherwise "eager" (default).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        freeze: bool = True,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str | None = "auto",
        attn_implementation: str = "eager",
    ) -> None:
        super().__init__()

        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )

        model_kwargs: dict = {
            "torch_dtype": torch_dtype,
            "output_hidden_states": True,
            "attn_implementation": attn_implementation,
        }
        if device_map is not None:
            model_kwargs["device_map"] = device_map

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            trust_remote_code=True,
            **model_kwargs,
        )

        # Infer d_vlm from the model config.
        # Qwen2.5-VL stores the LM hidden size under text_config, not top-level.
        _cfg = self.model.config
        self.d_vlm: int = (
            getattr(_cfg, "hidden_size", None)
            or getattr(getattr(_cfg, "text_config", None), "hidden_size", None)
            or getattr(getattr(_cfg, "language_config", None), "hidden_size", None)
            or 3584  # Qwen2.5-VL-7B fallback
        )

        if freeze:
            self._freeze()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def forward(
        self,
        images: list,
        questions: list[str],
        device: torch.device | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Extract last-layer hidden states for a batch of image-question pairs.

        Args:
            images: list of PIL.Image objects (B items).
            questions: list of question strings (B items).
            device: target device; inferred from model parameters if None.

        Returns:
            hidden_states : Float[B, L, d_vlm]  — last layer hidden states.
            key_pad_mask  : Bool[B, L]           — True where position is padding.
        """
        if device is None:
            device = next(self.model.parameters()).device

        messages = _build_messages(images, questions)
        texts = [
            self.processor.apply_chat_template(
                [msg], tokenize=False, add_generation_prompt=True
            )
            for msg in messages
        ]

        # Processor returns pixel_values, input_ids, attention_mask, etc.
        inputs = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad() if not self.training else _null_ctx():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        # Last hidden states: [B, L, d_vlm]
        hidden_states: Tensor = outputs.hidden_states[-1]

        # attention_mask: 1 = real token, 0 = padding
        # key_padding_mask for PyTorch MultiheadAttention: True = IGNORE (padding)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            key_pad_mask = attention_mask == 0   # Bool[B, L]
        else:
            key_pad_mask = torch.zeros(
                hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device
            )

        return hidden_states, key_pad_mask

    # ------------------------------------------------------------------ #
    # LoRA helpers
    # ------------------------------------------------------------------ #

    def enable_lora(
        self,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: list[str] | None = None,
    ) -> None:
        """Wrap the backbone with LoRA adapters (requires peft)."""
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as e:
            raise ImportError("Install peft: pip install peft") from e

        if target_modules is None:
            target_modules = ["q_proj", "v_proj"]

        # Unfreeze first (LoRA will re-freeze base weights internally)
        for p in self.model.parameters():
            p.requires_grad = False

        lora_cfg = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            task_type="CAUSAL_LM",
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_cfg)
        self.model.print_trainable_parameters()

    def lora_parameters(self):
        """Iterate over LoRA trainable parameters (for a separate LR group)."""
        for name, p in self.model.named_parameters():
            if p.requires_grad and "lora_" in name:
                yield p

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _freeze(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _build_messages(images: list, questions: list[str]) -> list[list[dict]]:
    """Build Qwen2.5-VL chat messages for each image-question pair."""
    messages = []
    for img, q in zip(images, questions):
        messages.append([
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": q},
                ],
            }
        ])
    return messages


class _null_ctx:
    """No-op context manager (used when model.training is True)."""
    def __enter__(self): return self
    def __exit__(self, *_): pass
