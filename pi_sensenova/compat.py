"""Narrow, instance-owned compatibility repairs for the pinned upstream."""
import torch


def repair_gguf_dtype_sentinels(model):
    """Upstream reads these weight dtypes to cast activations; packed bytes lie.

    Materialize only the small dtype-sentinel linears through diffusers' real
    GGUF decoder. The large trunk remains quantized; no activation is cast to
    integer bytes, and no rotation/scale arithmetic is guessed.
    """
    from diffusers.quantizers.gguf.utils import GGUFLinear, dequantize_gguf_tensor
    repaired = []
    for name, module in list(model.named_modules()):
        sentinel = name.endswith(("timestep_embedder.mlp.0", "noise_scale_embedder.mlp.0", "fm_head.net.input_proj"))
        if not sentinel or not isinstance(module, GGUFLinear):
            continue
        weight = dequantize_gguf_tensor(module.weight).to(torch.bfloat16)
        replacement = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=module.bias is not None,
                                      device="meta", dtype=torch.bfloat16)
        replacement.weight = torch.nn.Parameter(weight, requires_grad=False)
        if module.bias is not None:
            replacement.bias = torch.nn.Parameter(module.bias.detach().to(torch.bfloat16), requires_grad=False)
        parent, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(parent), leaf, replacement)
        repaired.append(name)
    return repaired
