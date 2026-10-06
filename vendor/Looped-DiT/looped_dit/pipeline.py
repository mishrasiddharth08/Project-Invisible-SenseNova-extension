"""Text encoder, checkpoint loading and image generation."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from transformers import AutoTokenizer, T5EncoderModel

from .config import TrainConfig
from .diffusion import euler_sample
from .model import LoopedMMDiT


class TextEncoder:
    """Frozen FLAN-T5 encoder; prompts are padded to prompt_length tokens."""

    def __init__(self, name: str, prompt_length: int, device: torch.device):
        self.prompt_length = prompt_length
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(name, model_max_length=prompt_length)
        self.model = T5EncoderModel.from_pretrained(name).to(device).eval().requires_grad_(False)

    def tokenize(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenizer(
            prompts, max_length=self.prompt_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return tokens["input_ids"].to(self.device), tokens["attention_mask"].to(self.device)

    @torch.no_grad()
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


def load_model(
    checkpoint: str | Path, device: torch.device, dtype: torch.dtype = torch.bfloat16, weights: str = "ema"
) -> tuple[LoopedMMDiT, TrainConfig]:
    """Build the model described by a checkpoint's config and load its (EMA) weights."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = TrainConfig.from_dict(ckpt["config"])
    model = LoopedMMDiT(**cfg.model_kwargs())
    model.load_state_dict(ckpt[weights])
    return model.to(device=device, dtype=dtype).eval().requires_grad_(False), cfg


def to_pil(images: torch.Tensor) -> list[Image.Image]:
    images = (images.float().clamp(-1, 1) * 127.5 + 128.0).clamp(0, 255).to(torch.uint8)
    return [Image.fromarray(image) for image in images.permute(0, 2, 3, 1).cpu().numpy()]


@torch.no_grad()
def generate(
    model: LoopedMMDiT,
    text_encoder: TextEncoder,
    prompts: list[str],
    image_size: int = 512,
    steps: int = 100,
    cfg_scale: float = 6.0,
    num_loops: int | None = None,
    noise_scale: float = 2.0,
) -> list[Image.Image]:
    """One image per prompt; seed torch beforehand for reproducible noise."""
    input_ids, mask = text_encoder.tokenize(prompts)
    dtype = next(model.parameters()).dtype
    text = text_encoder.encode(input_ids, mask).to(dtype)
    images = euler_sample(model, text, mask, image_size, steps, cfg_scale, noise_scale, num_loops)
    return to_pil(images)
