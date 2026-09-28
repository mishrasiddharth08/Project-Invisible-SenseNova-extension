"""Reversible official low-rank adapters, including quantized base modules."""
import torch
from torch.nn import functional as F


class RuntimeAdapter:
    def __init__(self):
        self.handles = []
        self.factors = []

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.factors.clear()

    def attach(self, model, tensors):
        if self.handles:
            raise RuntimeError("Remove the previous adapter before attaching another.")
        modules = dict(model.named_modules())
        plans = []
        consumed = set()
        for key, down in tensors.items():
            if not key.endswith(".lora_down.weight"):
                continue
            prefix = key.removesuffix(".lora_down.weight")
            target = prefix.removeprefix("diffusion_model.")
            up_key, alpha_key = prefix + ".lora_up.weight", prefix + ".alpha"
            if target not in modules or up_key not in tensors:
                raise ValueError(f"Adapter target or paired tensor missing: {target}")
            up = tensors[up_key]
            mod = modules[target]
            if down.ndim != 2 or up.ndim != 2 or down.shape[0] != up.shape[1]:
                raise ValueError(f"Unsupported adapter tensor shapes: {target}")
            shape = getattr(getattr(mod, "weight", None), "shape", None)
            expected = (getattr(mod, "out_features", up.shape[0]), getattr(mod, "in_features", down.shape[1]))
            if not hasattr(mod, "in_features") and shape is not None and len(shape) == 2:
                expected = tuple(shape)
            if expected != (up.shape[0], down.shape[1]):
                raise ValueError(f"Adapter does not match base model dimensions: {target}")
            scale = float(tensors.get(alpha_key, down.shape[0])) / down.shape[0]
            if not torch.isfinite(down).all() or not torch.isfinite(up).all() or not torch.isfinite(torch.tensor(scale)):
                raise ValueError("Adapter contains non-finite values.")
            plans.append((mod, down, up, scale))
            consumed.update((key, up_key, alpha_key))
        if not plans or set(tensors) - consumed:
            raise ValueError("Adapter is empty or contains unsupported/unmatched tensors.")
        try:
            for mod, down, up, scale in plans:
                # Keep bounded CPU factors. GPU residency follows this module's mode.
                factors = [down.detach(), up.detach()]
                self.factors.append(factors)

                def forward(module, args, output, factors=factors, scale=scale):
                    x = args[0]
                    a, b = (t.to(device=x.device, dtype=x.dtype) for t in factors)
                    return output + F.linear(F.linear(x, a), b).to(output.dtype) * scale

                self.handles.append(mod.register_forward_hook(forward))
        except Exception:
            self.remove()
            raise
        return len(plans)
