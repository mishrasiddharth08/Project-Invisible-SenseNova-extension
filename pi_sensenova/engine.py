"""Official pipeline lifetime, safe offload, scoped progress and cancellation."""
import contextlib
import gc
import inspect
import sys
import threading
import time
import weakref
from pathlib import Path

import torch

from .adapters import RuntimeAdapter
from .detect import identity, resolve, header
from .settings import memory_mode

LOCK = threading.RLock()
UPSTREAM_REVISION = "f9e5a684262efd5c49c9239735fe269ffb984045"


class Cancelled(RuntimeError):
    pass


def upstream():
    src = Path(__file__).resolve().parents[1] / "vendor" / "SenseNova-U1" / "src"
    if not src.is_dir():
        raise RuntimeError("Bundled SenseNova runtime is missing. Restore the extension package.")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import sensenova_u1
    if not Path(sensenova_u1.__file__).resolve().is_relative_to(src):
        raise RuntimeError("Another SenseNova runtime was imported first. Restart with only one SenseNova extension enabled.")
    from sensenova_u1 import utils
    return utils


@contextlib.contextmanager
def progress(model, total, callback, preview_callback=None):
    """The pinned official sampler unpatchifies once per completed flow step."""
    missing = object()
    previous = model.__dict__.get("unpatchify", missing)
    original = model.unpatchify
    completed = 0
    estimate = None
    predictor = getattr(model, "_t2i_predict_v", None)
    previous_predictor = model.__dict__.get("_t2i_predict_v", missing)

    def predict(*args, **kwargs):
        nonlocal estimate
        velocity = predictor(*args, **kwargs)
        if estimate is None:
            # First pass is the conditioned image prediction. Flow advances
            # from t=0 (noise) to t=1 (clean RGB): x_clean = z + (1-t)*v.
            estimate = (args[5].detach(), args[4].detach(), velocity.detach())
        return velocity

    def wrapped(*args, **kwargs):
        nonlocal completed, estimate
        # Cancel before allocating even the cheap pixel reshape.
        if callback(completed, total):
            raise Cancelled("Generation stopped; incomplete image discarded.")
        result = original(*args, **kwargs)
        completed += 1
        if callback(completed, total):
            raise Cancelled("Generation stopped; incomplete image discarded.")
        if preview_callback is not None:
            def frame():
                # Real evolving pixels: noise to the exact final sample.
                return result
            preview_callback(frame, completed, total)
        estimate = None
        return result

    model.unpatchify = wrapped
    if preview_callback is not None and callable(predictor):
        model._t2i_predict_v = predict
    try:
        if callback(0, total):
            raise Cancelled("Generation stopped before sampling.")
        yield
    finally:
        if previous is missing:
            del model.unpatchify
        else:
            model.unpatchify = previous
        if preview_callback is not None and callable(predictor):
            if previous_predictor is missing:
                del model._t2i_predict_v
            else:
                model._t2i_predict_v = previous_predictor


def to_images(tensor):
    from PIL import Image
    tensor = tensor.detach().float().add(1).mul(127.5).clamp(0, 255).to(torch.uint8).cpu()
    if tensor.ndim != 4 or tensor.shape[1] != 3:
        raise RuntimeError(f"Unexpected official image tensor shape: {tuple(tensor.shape)}")
    return [Image.fromarray(t.permute(1, 2, 0).numpy()) for t in tensor]


class Engine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.key = None
        self.mode = None
        self.adapter_key = None
        self.adapter = RuntimeAdapter()
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
            threading.Thread(target=release, name="SenseNova-unload", daemon=True).start()

    def unload(self):
        with LOCK:
            self.adapter.remove()
            self.model = self.tokenizer = self.key = self.adapter_key = None
            self.mode = None
            self.metrics = {}
            self.release_requested.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def load(self, path, settings):
        utils = upstream()
        p, resources, weight_bytes = resolve(path, settings.resources)
        if not torch.cuda.is_available():
            raise RuntimeError("SenseNova generation needs a compatible CUDA GPU.")
        key = (identity(p), identity(resources), settings.memory)
        if self.model is not None and self.key == key:
            return
        self.unload()
        free_bytes, _ = torch.cuda.mem_get_info()
        self.mode = memory_mode(free_bytes, weight_bytes, settings.memory)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        try:
            quantized = p.suffix == ".safetensors" and any(k.endswith(".comfy_quant") for k in header(p))
            if p.suffix == ".gguf":
                from .gguf import load
                self.model, self.tokenizer = load(p, resources, "cuda" if self.mode == "full" else "cpu")
            elif quantized:
                from .quantized import load
                self.model, self.tokenizer = load(p, resources, "cuda" if self.mode == "full" else "cpu")
            else:
                self.model, self.tokenizer = utils.load_model_and_tokenizer(
                    str(p), model_resources=str(resources), dtype=torch.bfloat16,
                    device="cuda", for_offload=self.mode != "full",
                )
            self.model.eval()
            self.model.requires_grad_(False)
            if p.suffix == ".gguf":
                from .compat import repair_gguf_dtype_sentinels
                repaired = repair_gguf_dtype_sentinels(self.model)
                print(f"[Invisible-SenseNova] Repaired GGUF activation dtype sentinels: {repaired}")
            self.key = key
        except Exception:
            self.unload()
            raise
        torch.cuda.synchronize()
        self.metrics = {"load_seconds": time.perf_counter() - start,
                        "load_peak_gib": torch.cuda.max_memory_allocated() / 2**30}
        print(f"[Invisible-SenseNova] Loaded {p.name}: {self.mode}, {self.metrics}")

    def configure_adapter(self, settings):
        from .paths import default_adapter
        adapter_path = settings.adapter or (default_adapter() if settings.fast else "")
        wanted = identity(adapter_path) if settings.fast and adapter_path else None
        if settings.fast and not wanted:
            raise ValueError("Fast mode requires the local official 8-step adapter path.")
        if wanted == self.adapter_key:
            return
        self.adapter.remove()
        self.adapter_key = None
        if wanted:
            from safetensors.torch import load_file
            count = self.adapter.attach(self.model, load_file(str(Path(adapter_path).resolve())))
            if self.mode == "full":
                for factors in self.adapter.factors:
                    factors[:] = [f.to(device="cuda", dtype=torch.bfloat16) for f in factors]
            self.adapter_key = wanted
            print(f"[Invisible-SenseNova] Fast adapter attached to {count} modules; base weights unchanged.")

    def offload(self):
        if self.mode == "full":
            return contextlib.nullcontext(self.model)
        if any(hasattr(m, "quant_format") or type(m).__name__ == "GGUFLinear" for m in self.model.modules()):
            from .streaming import BlockStream
            print("[Invisible-SenseNova] Packed weights: safe synchronous block streaming; full precision and steps retained.")
            return BlockStream(self.model)
        utils = upstream()
        kwargs = dict(layers_attr="language_model.model.layers", target_device=torch.device("cuda"),
                      keep_generation_resident=self.mode == "fast", fast_vram_fraction=0.90,
                      fast_vram_headroom_gib=2.0, fast_activation_reserve_gib=6.0)
        if self.mode == "low":
            return utils.offload_layers_sync(self.model, **kwargs)
        return utils.offload_layers_async(self.model, prefetch_count=2, **kwargs)

    def generate(self, *, prompt, width, height, steps, cfg, seed, settings, source=None, callback=lambda *_: False, preview_callback=None):
        original_callback = callback
        callback = lambda *args: self.release_requested.is_set() or original_callback(*args)
        with LOCK:
            if self.model is None:
                raise RuntimeError("Load a SenseNova checkpoint before generating.")
            self.configure_adapter(settings)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            try:
                with self.offload() as model, torch.inference_mode(), progress(model, steps, callback, preview_callback):
                    kwargs = dict(image_size=(width, height), num_steps=steps, cfg_scale=cfg,
                                  timestep_shift=float(settings.shift), cfg_norm="none", cfg_interval=(0, 1),
                                  batch_size=1, seed=seed, think_mode=settings.think)
                    if source is None:
                        result = model.t2i_generate(self.tokenizer, prompt, **kwargs)
                    else:
                        result = model.it2i_generate(self.tokenizer, prompt, [source.convert("RGB")], **kwargs)
                    if callback(steps, steps):
                        raise Cancelled("Generation stopped; incomplete image discarded.")
                    if isinstance(result, tuple):
                        result = result[0]
                    images = to_images(result)
                torch.cuda.synchronize()
                self.metrics.update(generation_seconds=time.perf_counter() - started,
                                    generation_peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                                    memory_mode=self.mode)
                print(f"[Invisible-SenseNova] Completed: {self.metrics}")
                return images
            except Exception:
                # Upstream may retain offload/KV state after an interrupted forward.
                # Discard the affected pipeline instead of reusing uncertain state.
                self.unload()
                raise


ENGINE = Engine()
