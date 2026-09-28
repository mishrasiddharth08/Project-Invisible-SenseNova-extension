"""Versioned extension-owned defaults; atomic local writes only."""
import json
import os
import threading
from pathlib import Path
from .settings import DEFAULTS, UI_KEYS, Settings

PATH = Path(__file__).resolve().parents[1] / "config.json"
_LOCK = threading.Lock()


def read():
    try:
        data = json.loads(PATH.read_text(encoding="utf-8"))
        if data.get("version") != 2:
            return DEFAULTS
        values = tuple(data.get(key, default) for key, default in zip(UI_KEYS, DEFAULTS))
        Settings.from_ui(values)
        return values
    except (OSError, ValueError, TypeError):
        return DEFAULTS


def save(*values):
    Settings.from_ui(values)
    data = dict(zip(UI_KEYS, values), version=2)
    tmp = PATH.with_suffix(".tmp")
    with _LOCK:
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, PATH)
