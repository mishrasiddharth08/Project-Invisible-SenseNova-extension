"""Search paths only. No writes or implicit network access."""
from pathlib import Path


def model_roots():
    try:
        from modules import paths, shared
        base = Path(paths.models_path)
        return [base / "Stable-diffusion", base / "SENSENOVA", *map(Path, shared.cmd_opts.ckpt_dirs)]
    except (ImportError, AttributeError) as error:
        print(f"[Invisible-SenseNova] Forge paths unavailable: {error}")
        return []


def default_adapter():
    try:
        from modules import paths
        p = Path(paths.models_path) / "Lora" / "SenseNova" / "SenseNova-U1.5-8B-MoT-LoRA-8step.safetensors"
        return str(p.resolve()) if p.is_file() else ""
    except (ImportError, AttributeError):
        return ""
