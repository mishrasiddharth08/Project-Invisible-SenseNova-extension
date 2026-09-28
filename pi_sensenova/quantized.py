"""Load descriptor-tagged weights using Forge's own math, without global patches."""
import torch


def load(path, resources, device):
    try:
        from backend.operations import mixed_precision_ops
        from backend.state_dict import detect_quantization
        from backend.quant_ops import QUANT_ALGOS
    except Exception as error:
        raise RuntimeError("This checkpoint requires compatible Forge quantization APIs. Select GGUF if this Forge build cannot load it.") from error
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    import json

    state = load_file(str(path), device="cpu")
    descriptor = detect_quantization(state, is_unet=True)
    if descriptor is None:
        raise ValueError("Quantized checkpoint has no recognized per-layer descriptors.")
    formats = {json.loads(value.numpy().tobytes())["format"] for key, value in state.items() if key.endswith(".comfy_quant")}
    if formats - set(QUANT_ALGOS):
        raise ValueError(f"Forge lacks these quantization formats: {formats - set(QUANT_ALGOS)}")
    config = AutoConfig.from_pretrained(str(resources), local_files_only=True)
    with init_empty_weights():
        model = AutoModel.from_config(config)
    ops = mixed_precision_ops(quant_config=dict(descriptor), compute_dtype=torch.bfloat16,
                              full_precision_mm=False, disabled=[])
    # Instantiate only engine-owned Linear objects. torch.nn is never rebound.
    for name, module in list(model.named_modules()):
        if isinstance(module, torch.nn.Linear):
            parent, _, leaf = name.rpartition(".")
            replacement = ops.Linear(module.in_features, module.out_features, bias=module.bias is not None,
                                     device="cpu", dtype=torch.bfloat16)
            setattr(model.get_submodule(parent), leaf, replacement)
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if missing or unexpected:
        raise ValueError(f"Checkpoint architecture mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}")
    del state
    remaining = [n for n, t in (*model.named_parameters(), *model.named_buffers()) if t.is_meta]
    if remaining:
        raise ValueError(f"Checkpoint left uninitialized model tensors: {remaining[:5]}")
    from backend.quant_ops import QuantizedTensor
    for name, parameter in list(model.named_parameters()):
        if isinstance(parameter, QuantizedTensor) or "scale" in name.rsplit(".", 1)[-1]:
            continue
        if parameter.is_floating_point() and parameter.dtype != torch.bfloat16:
            parent, _, leaf = name.rpartition(".")
            setattr(model.get_submodule(parent), leaf, torch.nn.Parameter(parameter.to(torch.bfloat16), requires_grad=False))
    model.eval().requires_grad_(False)
    model = model.to(device=device)
    tokenizer = AutoTokenizer.from_pretrained(str(resources), local_files_only=True)
    return model, tokenizer
