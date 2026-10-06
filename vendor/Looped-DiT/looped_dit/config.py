"""Training configuration: one flat dataclass, loaded from YAML."""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

# Config values may reference ${VAR}. VAR is read from the environment; these
# three fall back to directories inside the repository.
DEFAULT_ROOTS = {
    "DATA_ROOT": REPO_ROOT / "data",
    "OUTPUT_ROOT": REPO_ROOT / "outputs",
    "ASSET_ROOT": REPO_ROOT / "eval_assets",
}

DEEP_SUPERVISION_WEIGHTINGS = ("final_plus_mean", "exponential", "uniform")


def expand_vars(value: Any) -> Any:
    """Recursively substitute ${VAR} in strings; unknown variables raise."""
    if isinstance(value, str):

        def substitute(match: re.Match) -> str:
            name = match.group(1)
            if name in os.environ:
                return os.environ[name]
            if name in DEFAULT_ROOTS:
                return str(DEFAULT_ROOTS[name])
            raise KeyError(f"undefined variable ${{{name}}} in {value!r}")

        return re.sub(r"\$\{(\w+)\}", substitute, value)
    if isinstance(value, list):
        return [expand_vars(v) for v in value]
    if isinstance(value, dict):
        return {k: expand_vars(v) for k, v in value.items()}
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    return expand_vars(data)


@dataclass
class TrainConfig:
    # Architecture (MiniT2I MMDiT backbone).
    image_size: int = 512
    patch_size: int = 32
    hidden_size: int = 768
    num_heads: int = 12
    head_dim: int = 64
    mlp_ratio: float = 2.6667
    pca_channels: int = 128
    text_preamble_depth: int = 2
    loop_split: list[int] = field(default_factory=lambda: [6, 5, 6])  # pre-loop / looped / post-loop blocks

    # Looped-DiT components; each can be switched off for ablations.
    num_loops: int = 4  # loop depth N at training time; 1 = no looping
    share_loop_weights: bool = True  # False: separate core blocks per pass (the "deeper" baseline)
    deep_supervision: bool = True  # flow loss on every loop exit, not only the last
    deep_supervision_weighting: str = "final_plus_mean"  # final_plus_mean | exponential | uniform
    use_xsa: bool = True  # exclusive self-attention in the looped blocks
    use_attn_gate: bool = False  # head-wise sigmoid attention gate in the looped blocks

    # Frozen text encoder.
    text_encoder: str = "google/flan-t5-large"
    text_dim: int = 1024
    prompt_length: int = 256

    # Flow matching: x_t = t x + (1 - t) eps, eps ~ N(0, noise_scale^2), t ~ logit-normal.
    noise_scale: float = 2.0
    t_logit_mean: float = -0.8
    t_logit_std: float = 0.8
    label_drop_rate: float = 0.1  # prompt dropout for classifier-free guidance

    # Data: "chunks" reads pre-tokenized tensor chunks (pretraining); "webdataset"
    # mixes image/caption WebDataset sources by weight (fine-tuning).
    dataset: str = "chunks"
    chunk_dirs: list[str] = field(default_factory=list)
    webdataset_sources: dict[str, dict[str, Any]] = field(default_factory=dict)  # name -> {path, weight}
    shuffle_buffer: int = 1000
    num_workers: int = 6
    prefetch_factor: int = 4

    # Optimization. Gradient accumulation = batch_size / (micro_batch_size * world size).
    batch_size: int = 1024
    micro_batch_size: int = 32
    num_steps: int = 250_000
    warmup_steps: int = 5_000
    learning_rate: float = 4e-4
    adam_beta2: float = 0.95
    weight_decay: float = 0.0
    max_grad_norm: float = 0.1
    ema_decay: float = 0.99995
    amp_dtype: str = "bf16"
    seed: int = 42

    # Logging and checkpoints.
    log_every: int = 100
    sample_every: int = 10_000
    ckpt_every: int = 10_000
    ckpt_keep_every: int = 50_000  # checkpoints at multiples of this are never pruned
    keep_last: int = 3

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> TrainConfig:
        cfg = cls()
        names = {f.name for f in fields(cls)}
        for key, value in values.items():
            if key not in names:
                raise KeyError(f"unknown config key: {key}")
            default = getattr(cfg, key)
            if type(default) in (int, float) and isinstance(value, str):  # YAML reads "2e-4" as a string
                value = type(default)(float(value))
            elif type(default) is float and type(value) is int:
                value = float(value)
            setattr(cfg, key, value)
        cfg.validate()
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: dict[str, Any] | None = None) -> TrainConfig:
        return cls.from_dict({**load_yaml(path), **(overrides or {})})

    def validate(self) -> None:
        if len(self.loop_split) != 3:
            raise ValueError(f"loop_split must have three entries (pre, core, post), got {self.loop_split}")
        if self.deep_supervision_weighting not in DEEP_SUPERVISION_WEIGHTINGS:
            raise ValueError(f"deep_supervision_weighting must be one of {DEEP_SUPERVISION_WEIGHTINGS}")
        if self.deep_supervision and self.num_loops < 2:
            raise ValueError("deep_supervision needs num_loops >= 2 (it supervises the intermediate loops)")
        if self.dataset not in ("chunks", "webdataset"):
            raise ValueError(f"dataset must be 'chunks' or 'webdataset', got {self.dataset!r}")
        if self.amp_dtype not in ("bf16", "fp32"):  # fp16 would need loss scaling
            raise ValueError(f"amp_dtype must be bf16 or fp32, got {self.amp_dtype!r}")

    def model_kwargs(self) -> dict[str, Any]:
        return dict(
            image_size=self.image_size,
            patch_size=self.patch_size,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            mlp_ratio=self.mlp_ratio,
            pca_channels=self.pca_channels,
            text_dim=self.text_dim,
            text_preamble_depth=self.text_preamble_depth,
            loop_split=tuple(self.loop_split),
            num_loops=self.num_loops,
            share_loop_weights=self.share_loop_weights,
            use_xsa=self.use_xsa,
            use_attn_gate=self.use_attn_gate,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
