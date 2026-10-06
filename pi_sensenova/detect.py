"""Local checkpoint truth and cheap identities; never download on Generate."""
import json
import struct
from functools import lru_cache
from pathlib import Path


def identity(path):
    p = Path(path).expanduser().resolve()
    files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.is_file() and f.suffix in (".json", ".safetensors", ".gguf", ".model"))
    return str(p), tuple((str(f), f.stat().st_size, f.stat().st_mtime_ns) for f in files)


def header(path):
    p = Path(path)
    st = p.stat()
    return _header(str(p.resolve()), st.st_size, st.st_mtime_ns)


@lru_cache(maxsize=64)
def _header(path, size, mtime):
    with open(path, "rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError("Incomplete checkpoint header.")
        length = struct.unpack("<Q", raw)[0]
        if length > 64 * 1024 * 1024 or length + 8 > size:
            raise ValueError("Invalid safetensors header length.")
        data = json.loads(stream.read(length))
    return data


@lru_cache(maxsize=64)
def _gguf_identity(path, size, mtime):
    from gguf import GGUFReader
    reader = GGUFReader(path, "r")
    names = [t.name for t in reader.tensors]
    return any("fm_modules" in n or "fm_head" in n for n in names) and any("language_model" in n or "blk." in n for n in names)


def is_ours(path):
    p = Path(path)
    try:
        if p.suffix.lower() == ".pt":
            return is_looped(p)
        if p.is_dir() or p.name == "config.json":
            cfg = json.loads((p / "config.json" if p.is_dir() else p).read_text(encoding="utf-8"))
            return cfg.get("model_type") == "neo_chat"
        if p.suffix in (".safetensors", ".sft"):
            keys = header(p)
            return any(k.startswith("fm_modules.") or "fm_head." in k for k in keys) and any("language_model.model.layers." in k for k in keys)
        if p.suffix == ".gguf":
            # GGUF identity is read from metadata; the filename is not proof.
            st = p.stat()
            return _gguf_identity(str(p.resolve()), st.st_size, st.st_mtime_ns)
    except (OSError, ValueError, ImportError, KeyError, TypeError):
        return False
    return False


def is_looped(path):
    """Inspect tensor/config signatures without trusting filenames or pickle code."""
    p = Path(path)
    if p.suffix.lower() != ".pt" or not p.is_file():
        return False
    st = p.stat()
    return _looped_identity(str(p.resolve()), st.st_size, st.st_mtime_ns)


@lru_cache(maxsize=16)
def _looped_identity(path, size, mtime):
    try:
        import torch
        # Meta mapping reads tensor descriptions, without allocating their storage.
        checkpoint = torch.load(path, map_location="meta", weights_only=True, mmap=True)
        cfg = checkpoint.get("config", {})
        weights = checkpoint.get("ema", {})
        return (isinstance(cfg, dict) and isinstance(weights, dict)
                and cfg.get("image_size", 512) == 512
                and cfg.get("patch_size", 32) in (16, 32)
                and isinstance(cfg.get("loop_split"), (list, tuple))
                and len(cfg["loop_split"]) == 3
                and all(k in weights and isinstance(weights[k], torch.Tensor)
                        for k in ("img_embed.proj1.weight", "txt_embed.weight", "mask_token", "final.weight")))
    except Exception:
        return False


def resolve(path, resources=""):
    p = Path(path).expanduser().resolve()
    if p.name == "config.json":
        p = p.parent
    if not p.exists():
        raise ValueError(f"Checkpoint does not exist: {p}")
    r = Path(resources).expanduser().resolve() if resources else (p if p.is_dir() else p.parent)
    if not resources and not (r / "config.json").is_file() and (r / "SenseNova-resources" / "config.json").is_file():
        r = r / "SenseNova-resources"
    if not (r / "config.json").is_file() or not is_ours(r):
        raise ValueError("Select the local SenseNova config/tokenizer folder in Advanced. No resources are downloaded during generation.")
    if not ((r / "tokenizer.json").is_file() or ((r / "vocab.json").is_file() and (r / "merges.txt").is_file())):
        raise ValueError(f"Missing tokenizer files in {r}; finish Model Setup first.")
    if p.is_dir():
        index = p / "model.safetensors.index.json"
        if index.is_file():
            mapping = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
            files = sorted({(p / f).resolve() for f in mapping.values()})
            if any(not f.is_relative_to(p) for f in files):
                raise ValueError("Checkpoint shard paths must stay inside their model folder.")
        else:
            files = [p / "model.safetensors"]
        if not files or any(not f.is_file() for f in files):
            raise ValueError("The SenseNova snapshot has missing weight shards.")
    else:
        files = [p]
    if p.suffix != ".gguf":
        for f in files:
            h = header(f)
            # The upstream single-file loader casts raw FP8 without scales; refuse it.
            dtypes = {v.get("dtype") for k, v in h.items() if k != "__metadata__" and k.endswith("weight")}
            marked_int8 = "I8" in dtypes and all(d in {"BF16", "F16", "F32", "I8"} for d in dtypes) and any(k.endswith(".comfy_quant") for k in h)
            if dtypes - {"BF16", "F16", "F32"} and not marked_int8:
                raise ValueError("This quantized safetensors layout has no validated SenseNova loader. Use dense BF16/FP16 or a supported GGUF plus matching resources.")
    return p, r, sum(f.stat().st_size for f in files)
