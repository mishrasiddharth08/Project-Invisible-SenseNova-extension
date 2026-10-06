"""Checkpoint dropdown ownership, including complete HF snapshot directories."""
from functools import wraps
from pathlib import Path

from .detect import is_ours
from .paths import model_roots


def selected(p=None):
    try:
        from modules import sd_models, shared
        override = getattr(p, "override_settings", {}) or {}
        value = override.get("sd_model_checkpoint", shared.opts.sd_model_checkpoint)
        ci = sd_models.checkpoint_aliases.get(value) or sd_models.checkpoints_list.get(value)
        path = getattr(ci, "filename", value)
        return str(path) if path and is_ours(path) else None
    except (ImportError, AttributeError, TypeError):
        return None


def register():
    from modules import sd_models
    known = {str(Path(c.filename).resolve()) for c in sd_models.checkpoints_list.values()}
    for root in model_roots():
        if not root.is_dir():
            continue
        from .detect import is_looped
        for checkpoint in root.rglob("*.pt"):
            if str(checkpoint.resolve()) in known or not is_looped(checkpoint):
                continue
            ci = sd_models.CheckpointInfo(str(checkpoint.resolve()))
            ci.name = f"Looped-DiT / {checkpoint.stem}"
            ci.title = ci.name
            ci.ids += [ci.name, str(checkpoint.resolve())]
            ci.register()
            known.add(str(checkpoint.resolve()))
        for config in root.rglob("config.json"):
            if str(config.resolve()) in known or not is_ours(config):
                continue
            folder = config.parent
            if not ((folder / "model.safetensors").is_file() or (folder / "model.safetensors.index.json").is_file()):
                continue
            ci = sd_models.CheckpointInfo(str(config.resolve()))
            ci.name = f"SenseNova / {folder.name}"
            ci.title = ci.name
            ci.ids += [ci.name, str(folder.resolve()), str(config.resolve())]
            ci.register()
            known.add(str(config.resolve()))


def install():
    try:
        from modules import sd_models
        original = sd_models.list_models
        if getattr(original, "_pi_sensenova", False):
            return

        @wraps(original)
        def listed(*args, **kwargs):
            result = original(*args, **kwargs)
            try:
                register()
            except Exception as error:
                print(f"[Invisible-SenseNova] Extra checkpoint registration skipped: {error}")
            return result

        listed._pi_sensenova = True
        sd_models.list_models = listed
        register()
    except (ImportError, AttributeError, OSError) as error:
        print(f"[Invisible-SenseNova] Checkpoint registration unavailable: {error}")
