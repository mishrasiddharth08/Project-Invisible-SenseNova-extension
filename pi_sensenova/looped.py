"""Extension-owned Looped-DiT EMA inference, offline T5 and bounded memory.

Euler equations follow OpenSenseNova/Looped-DiT at UPSTREAM_REVISION (MIT).
No host package versions or global Torch/Transformers methods are changed.
"""
import contextlib
import gc
import math
import sys
import threading
import time
import weakref
from pathlib import Path

import torch

from .detect import identity, is_looped
from .engine import LOCK, Cancelled
from .streaming import BlockStream

UPSTREAM_REVISION = "65a7705ad2954fa127721bd1131ea8f865688848"


def upstream():
    root = Path(__file__).resolve().parents[1] / "vendor" / "Looped-DiT"
    if not (root / "looped_dit" / "model.py").is_file():
        raise RuntimeError("Bundled Looped-DiT runtime is missing; restore the extension.")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import looped_dit
    if not Path(looped_dit.__file__).resolve().is_relative_to(root):
        raise RuntimeError("A different Looped-DiT runtime is already loaded; restart Forge.")
    from looped_dit.config import TrainConfig
    from looped_dit.model import LoopedMMDiT
    from looped_dit.pipeline import to_pil
    return TrainConfig, LoopedMMDiT, to_pil


@torch.inference_mode()
def sample(model, text, mask, image_size, steps, cfg_scale, num_loops,
           noise_scale=2.0, callback=lambda *_: False, preview_callback=None):
    """Official Euler math with per-step cancellation and evolving RGB previews."""
    if steps < 1:
        raise ValueError("Looped-DiT needs at least one step.")
    if callback(0, steps):
        raise Cancelled("Generation stopped before sampling.")
    b, device = text.shape[0], text.device
    x = torch.randn(b, 3, image_size, image_size, device=device, dtype=text.dtype) * noise_scale
    ts = torch.linspace(0.0, 1.0, steps + 1, device=device)
    null_mask = torch.zeros_like(mask)
    amp_dtype = text.dtype if text.dtype in (torch.float16, torch.bfloat16) else torch.float32
    for i in range(steps):
        if callback(i, steps):
            raise Cancelled("Generation stopped; incomplete image discarded.")
        t0, t1 = ts[i], ts[i + 1]
        t = torch.full((b,), float(t0), device=device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=x.is_cuda and amp_dtype != torch.float32):
            pred = model(x, text, mask, num_loops=num_loops)
            if cfg_scale != 1.0:
                uncond = model(x, text, null_mask, num_loops=num_loops)
                pred = uncond + (pred - uncond) * cfg_scale
        v = (pred - x) / (1.0 - t[:, None, None, None]).clamp_min(0.05)
        x = x + v * (t1 - t0)
        if callback(i + 1, steps):
            raise Cancelled("Generation stopped; incomplete image discarded.")
        if preview_callback:
            preview_callback(lambda: x.detach().clamp(-1, 1), i + 1, steps)
    return x.clamp(-1, 1)


class LoopedEngine:
    def __init__(self):
        self.model = self.text_encoder = self.tokenizer = self.cfg = self.key = None
        self.mode = None
        self.metrics = {}
        self.release_requested = threading.Event()

    def request_unload(self):
        if self.model is None:
            return
        target = weakref.ref(self.model)
        self.release_requested.set()
        def release():
            with LOCK:
                if self.model is target():
                    self.unload()
        if LOCK.acquire(blocking=False):
            try:
                release()
            finally:
                LOCK.release()
        else:
            threading.Thread(target=release, name="Looped-DiT-unload", daemon=True).start()

    def unload(self):
        with LOCK:
            self.model = self.text_encoder = self.tokenizer = self.cfg = self.key = None
            self.mode = None
            self.metrics = {}
            self.release_requested.clear()
            module = sys.modules.get("looped_dit.model")
            if module is not None:
                module._ROPE_CACHE.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def load(self, path, settings):
        with LOCK:
            checkpoint = Path(path).resolve()
            resources = (Path(settings.resources).expanduser().resolve() if settings.resources
                         else checkpoint.parent / "flan-t5-large")
            if not is_looped(checkpoint):
                raise ValueError("Select a valid official Looped-DiT EMA checkpoint.")
            if not resources.is_dir():
                raise ValueError("Download FLAN-T5-Large into flan-t5-large beside the checkpoint, or select its local folder in Advanced.")
            if not torch.cuda.is_available():
                raise RuntimeError("Looped-DiT needs a compatible NVIDIA CUDA GPU.")
            key = (identity(checkpoint), identity(resources), settings.memory)
            if self.model is not None and self.key == key:
                return
            self.unload()
            started = time.perf_counter()
            try:
                Config, Model, _ = upstream()
                state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
                self.cfg = Config.from_dict(state["config"])
                if self.cfg.image_size != 512 or self.cfg.text_dim != 1024 or self.cfg.text_encoder != "google/flan-t5-large":
                    raise ValueError("This checkpoint does not match the supported 512px FLAN-T5-Large architecture.")
                with torch.device("meta"):
                    self.model = Model(**self.cfg.model_kwargs())
                self.model.load_state_dict(state["ema"], strict=True, assign=True)
                # Nonpersistent positional buffer is absent from state_dict.
                from looped_dit.model import sincos_2d
                self.model.pos_embed = sincos_2d(self.cfg.hidden_size, 512 // self.cfg.patch_size)[None]
                self.model = self.model.to(dtype=torch.bfloat16).eval().requires_grad_(False)
                del state
                from transformers import AutoTokenizer, T5EncoderModel
                self.tokenizer = AutoTokenizer.from_pretrained(str(resources), local_files_only=True)
                self.text_encoder = T5EncoderModel.from_pretrained(str(resources), local_files_only=True,
                                                                  use_safetensors=True).eval().requires_grad_(False)
                if self.text_encoder.config.d_model != self.cfg.text_dim:
                    raise ValueError("Text encoder dimension does not match the Looped-DiT checkpoint.")
                free, _ = torch.cuda.mem_get_info()
                weight_bytes = sum(t.numel() * t.element_size() for m in (self.model, self.text_encoder) for t in m.parameters())
                self.mode = "full" if settings.memory == "Full" or (settings.memory == "Auto" and free >= weight_bytes + 2 * 2**30) else "low"
                if self.mode == "full":
                    self.model.to("cuda")
                    self.text_encoder.to("cuda")
                self.key = key
                self.metrics = {"load_seconds": time.perf_counter() - started, "upstream_revision": UPSTREAM_REVISION}
            except Exception:
                self.unload()
                raise

    def generate(self, *, prompt, width, height, steps, cfg, seed, settings, source=None,
                 callback=lambda *_: False, preview_callback=None):
        if source is not None or settings.fast or settings.think or settings.adapter:
            raise ValueError("Looped-DiT supports text-to-image only; disable U1.5 fast adapter and Think mode.")
        depth = float(settings.loop_depth)
        if isinstance(settings.loop_depth, bool) or not math.isfinite(depth) or not depth.is_integer() or not 1 <= depth <= 16:
            raise ValueError("Loop depth must be a whole number from 1 to 16.")
        if (width, height) != (512, 512):
            raise ValueError("Official Looped-DiT models require 512×512. Use Looped-DiT settings.")
        with LOCK:
            if self.model is None:
                raise RuntimeError("Load a Looped-DiT checkpoint before generating.")
            if not self.model.share_loop_weights and int(depth) > self.model.num_loops:
                raise ValueError("This unshared checkpoint cannot exceed its trained loop depth.")
            original = callback
            callback = lambda *args: self.release_requested.is_set() or original(*args)
            if callback(0, steps):
                raise Cancelled("Generation stopped before text encoding.")
            start = time.perf_counter()
            try:
                _, _, to_pil = upstream()
                torch.cuda.reset_peak_memory_stats()
                with torch.inference_mode(), torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    torch.manual_seed(seed)
                    tokens = self.tokenizer([prompt], max_length=self.cfg.prompt_length, padding="max_length",
                                            truncation=True, return_tensors="pt")
                    ids, mask = tokens["input_ids"].to("cuda"), tokens["attention_mask"].to("cuda")
                    self.text_encoder.to("cuda")
                    try:
                        text = self.text_encoder(input_ids=ids, attention_mask=mask).last_hidden_state.to(torch.bfloat16)
                    finally:
                        if self.mode != "full":
                            self.text_encoder.to("cpu")
                    offload = contextlib.nullcontext(self.model) if self.mode == "full" else BlockStream(self.model, layers_attr="blocks")
                    with offload as model:
                        pixels = sample(model, text, mask, 512, steps, cfg, int(settings.loop_depth),
                                        self.cfg.noise_scale, callback, preview_callback)
                        images = to_pil(pixels)
                    del pixels, text, ids, mask
                torch.cuda.synchronize()
                self.metrics.update(generation_seconds=time.perf_counter() - start,
                                    generation_peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                                    memory_mode=self.mode, loop_depth=int(settings.loop_depth))
                return images
            except Exception:
                self.unload()
                raise


LOOPED_ENGINE = LoopedEngine()
