"""Direct UI/API takeover. Other checkpoints delegate without changed arguments."""
import inspect
import sys
from functools import wraps
from pathlib import Path

from . import selection
from .settings import DEFAULTS, Settings


def ui_settings(p, runner=None):
    runner = runner or getattr(p, "scripts", None)
    args = getattr(p, "script_args", ()) or ()
    for script in getattr(runner, "alwayson_scripts", ()):
        if getattr(script, "pi_sensenova", False):
            values = args[script.args_from:script.args_to]
            if values:
                return Settings.from_ui(values)
    from .config import read
    return Settings.from_ui(read())


def takeover(p, path, runner=None):
    from .generate import generate
    return generate(p, path, ui_settings(p, runner))


def release():
    # Don't import torch or the official package on ordinary model generations.
    for name, engine_name in (("pi_sensenova.engine", "ENGINE"), ("pi_sensenova.looped", "LOOPED_ENGINE")):
        module = sys.modules.get(name)
        engine = getattr(module, engine_name, None)
        if engine is not None:
            engine.unload()


def install_selection_release():
    """Chain Forge's setting callback; selection releases memory before Generate."""
    from modules import shared
    info = shared.opts.data_labels.get("sd_model_checkpoint")
    if info is None or getattr(info.onchange, "_sn_selection_release", False):
        return
    original = info.onchange
    def changed():
        for name, engine_name in (("pi_sensenova.engine", "ENGINE"), ("pi_sensenova.looped", "LOOPED_ENGINE")):
            module = sys.modules.get(name)
            engine = getattr(module, engine_name, None)
            if engine is not None:
                engine.request_unload()
        if original is not None:
            return original()
    changed._sn_selection_release = True
    info.onchange = changed


def process_guard(original):
    @wraps(original)
    def process(p, *args, **kwargs):
        path = selection.selected(p)
        if path:
            return takeover(p, path)
        release()
        return original(p, *args, **kwargs)
    process._pi_sensenova_route = True
    return process


def bind_aliases(*unused):
    # Each host alias owns its delegate: another extension may already have
    # wrapped processing.process_images after the API imported an older copy.
    for name in ("modules.api.api", "modules.txt2img", "modules.img2img"):
        module = sys.modules.get(name)
        original = getattr(module, "process_images", None)
        if callable(original) and not getattr(original, "_pi_sensenova_route", False):
            try:
                inspect.signature(original).bind(object())
            except (TypeError, ValueError):
                print(f"[Invisible-SenseNova] Incompatible API alias left untouched: {name}")
                continue
            module.process_images = process_guard(original)


def install():
    try:
        from modules import processing
        from modules.scripts import ScriptRunner
        inspect.signature(ScriptRunner.run).bind(object(), object())
        inspect.signature(processing.process_images).bind(object())
    except (ImportError, AttributeError, TypeError, ValueError) as error:
        print(f"[Invisible-SenseNova] Integration disabled: incompatible Forge interfaces: {error}")
        return False
    if getattr(ScriptRunner, "_pi_sensenova", False):
        return True
    original_run = ScriptRunner.run
    original_process = processing.process_images

    @wraps(original_run)
    def run(runner, p, *args, **kwargs):
        path = selection.selected(p)
        if path:
            return takeover(p, path, runner)
        release()
        return original_run(runner, p, *args, **kwargs)

    ScriptRunner.run = run
    ScriptRunner._pi_sensenova = True
    processing.process_images = process_guard(original_process)
    bind_aliases()
    print("[Invisible-SenseNova] Direct Generate/API routing installed; no discarded Forge pass.")
    return True


def install_unload():
    try:
        from modules import sd_models
        original = sd_models.unload_model_weights
        if getattr(original, "_pi_sensenova_unload", False):
            return
        @wraps(original)
        def unload(*args, **kwargs):
            release()
            return original(*args, **kwargs)
        unload._pi_sensenova_unload = True
        sd_models.unload_model_weights = unload
    except (ImportError, AttributeError) as error:
        print(f"[Invisible-SenseNova] Native unload integration unavailable: {error}; use the SenseNova Unload button.")
