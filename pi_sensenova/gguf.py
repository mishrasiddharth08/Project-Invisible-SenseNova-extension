"""Local GGUF loading; independent of cached optional-dependency probes."""
import torch


def load(path, resources, device):
    import gguf
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from diffusers.quantizers.gguf.utils import (
        GGUFParameter, GGUFLinear, SUPPORTED_GGUF_QUANT_TYPES,
        _replace_with_gguf_linear, dequantize_gguf_tensor,
    )

    config = AutoConfig.from_pretrained(str(resources), local_files_only=True)
    with init_empty_weights():
        model = AutoModel.from_config(config)
    expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    state = {}
    reader = gguf.GGUFReader(str(path))
    for entry in reader.tensors:
        if entry.name in state or entry.name not in expected:
            raise ValueError(f"Unexpected/duplicate GGUF tensor: {entry.name}")
        if tuple(reversed(entry.shape.tolist())) != expected[entry.name]:
            raise ValueError(f"GGUF tensor shape mismatch: {entry.name}")
        quant = entry.tensor_type not in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16)
        if quant and entry.tensor_type not in SUPPORTED_GGUF_QUANT_TYPES:
            raise ValueError(f"Unsupported GGUF format: {entry.tensor_type}")
        value = torch.from_numpy(entry.data.copy())
        state[entry.name] = GGUFParameter(value, quant_type=entry.tensor_type) if quant else value
    del reader
    if set(state) != set(expected):
        raise ValueError(f"Missing GGUF tensors: {sorted(set(expected) - set(state))[:5]}")
    _replace_with_gguf_linear(model, torch.bfloat16, state)
    # Explicitly assign verified shapes because packed Linear storage has a
    # different physical shape from its logical matrix. No host classes change.
    for name, value in state.items():
        parent, _, leaf = name.rpartition(".")
        module = model.get_submodule(parent)
        if isinstance(value, GGUFParameter):
            if not (isinstance(module, GGUFLinear) and leaf == "weight"):
                value = dequantize_gguf_tensor(value).to(torch.bfloat16)
        else:
            value = value.to(torch.bfloat16) if value.is_floating_point() else value
        value = value.to(device)
        if leaf in module._parameters:
            module._parameters[leaf] = torch.nn.Parameter(value, requires_grad=False)
        elif leaf in module._buffers:
            module._buffers[leaf] = value
        else:
            raise ValueError(f"GGUF target is not a parameter/buffer: {name}")
    del state
    remaining = [name for name, value in (*model.named_parameters(), *model.named_buffers()) if value.is_meta]
    if remaining:
        raise ValueError(f"Uninitialized GGUF model tensors: {remaining[:5]}")
    tokenizer = AutoTokenizer.from_pretrained(str(resources), local_files_only=True)
    return model.eval(), tokenizer
