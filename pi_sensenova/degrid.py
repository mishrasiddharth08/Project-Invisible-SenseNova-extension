"""Optional final-image 2px grid cleanup for SenseNova.

The filter is CPU-only and disabled unless the caller explicitly enables it.
Clean images are returned as the original PIL object, avoiding a lossy or
unnecessary round trip. Large images use haloed tiles to bound peak memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from ._degrid.core import NEGLIGIBLE_AMP, auto_limit, extract_grid, lattice_amp

_TILE = 512
_HALO = 4
_FULL_MAX_SIDE = 1024
_MAX_SAMPLES = 1_000_000


@dataclass(frozen=True)
class DeGridStats:
    grid_255: float
    limit: float
    skipped: bool
    tiled: bool


def _as_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)


def _tile_boxes(width: int, height: int):
    for top in range(0, height, _TILE):
        bottom = min(top + _TILE, height)
        for left in range(0, width, _TILE):
            right = min(left + _TILE, width)
            yield left, top, right, bottom


def _tile_corr(image: Image.Image, box: tuple[int, int, int, int]):
    left, top, right, bottom = box
    outer_left = max(0, left - _HALO)
    outer_top = max(0, top - _HALO)
    outer_right = min(image.width, right + _HALO)
    outer_bottom = min(image.height, bottom + _HALO)
    # The pinned core deliberately bypasses images <= 8px. A final narrow
    # tile therefore borrows more source pixels from its opposite side.
    if outer_right - outer_left <= 8:
        outer_left = max(0, outer_right - 9)
    if outer_bottom - outer_top <= 8:
        outer_top = max(0, outer_bottom - 9)
    outer = (outer_left, outer_top, outer_right, outer_bottom)
    tensor = _as_tensor(image.crop(outer))
    corr = extract_grid(tensor)
    x0, y0 = left - outer[0], top - outer[1]
    return tensor[:, :, y0:y0 + bottom - top, x0:x0 + right - left], corr[:, :, y0:y0 + bottom - top, x0:x0 + right - left]


def _measure_tiled(image: Image.Image):
    sums = torch.zeros((3, 4), dtype=torch.float64)
    counts = torch.zeros(4, dtype=torch.int64)
    samples: list[torch.Tensor] = []
    pixels = image.width * image.height * 3
    stride = max(1, (pixels + _MAX_SAMPLES - 1) // _MAX_SAMPLES)

    for box in _tile_boxes(image.width, image.height):
        _, corr = _tile_corr(image, box)
        left, top, _, _ = box
        for local_y in (0, 1):
            global_y = top + local_y
            for local_x in (0, 1):
                global_x = left + local_x
                phase = (global_y & 1) * 2 + (global_x & 1)
                part = corr[0, :, local_y::2, local_x::2]
                sums[:, phase] += part.double().sum(dim=(1, 2))
                counts[phase] += part.shape[1] * part.shape[2]
        samples.append(corr.abs().reshape(-1)[::stride].float().clone())

    means = sums / counts.clamp_min(1).unsqueeze(0)
    amp = (means.amax(1) - means.amin(1)).amax().item()
    sampled = torch.cat(samples)
    limit = float((torch.quantile(sampled, 0.75) * 3.0).clamp(0.004, 0.05).item())
    return amp, limit


def _apply_full(image: Image.Image):
    tensor = _as_tensor(image)
    corr = extract_grid(tensor)
    amp = float(lattice_amp(corr)[0].amax().item())
    limit = float(auto_limit(corr)[0].item())
    if amp < NEGLIGIBLE_AMP:
        return image, DeGridStats(amp * 255.0, limit, True, False)
    cleaned = (tensor - corr.clamp(-limit, limit)).clamp(0.0, 1.0)
    array = cleaned[0].permute(1, 2, 0).mul(255.0).round().byte().numpy()
    result = Image.fromarray(array, "RGB")
    result.info.update(image.info)
    return result, DeGridStats(amp * 255.0, limit, False, False)


def _apply_tiled(image: Image.Image):
    amp, limit = _measure_tiled(image)
    if amp < NEGLIGIBLE_AMP:
        return image, DeGridStats(amp * 255.0, limit, True, True)
    output = Image.new("RGB", image.size)
    for box in _tile_boxes(image.width, image.height):
        tensor, corr = _tile_corr(image, box)
        cleaned = (tensor - corr.clamp(-limit, limit)).clamp(0.0, 1.0)
        array = cleaned[0].permute(1, 2, 0).mul(255.0).round().byte().numpy()
        output.paste(Image.fromarray(array, "RGB"), box[:2])
    output.info.update(image.info)
    return output, DeGridStats(amp * 255.0, limit, False, True)


def apply_with_stats(image: Image.Image, *, enabled: bool = False):
    """Return ``(image, stats)``; disabled and clean inputs are untouched."""
    if not enabled:
        return image, DeGridStats(0.0, 0.0, True, False)
    if not isinstance(image, Image.Image):
        raise TypeError("degrid expects a PIL image")
    if min(image.size) <= 8:
        return image, DeGridStats(0.0, 0.0, True, False)
    if max(image.size) <= _FULL_MAX_SIDE:
        return _apply_full(image)
    return _apply_tiled(image)


def apply(image: Image.Image, *, enabled: bool = False) -> Image.Image:
    """Apply automatic cleanup when explicitly enabled."""
    return apply_with_stats(image, enabled=enabled)[0]
