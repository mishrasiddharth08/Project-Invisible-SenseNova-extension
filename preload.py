"""Expose the bundled GGUF parser before dependency availability is cached."""
import importlib.util
import sys
from pathlib import Path


def preload(parser):
    # Preserve an existing installation. No imports of torch/diffusers, package
    # installs, version overrides or Forge source changes during preloading.
    deps = Path(__file__).resolve().parent / "vendor" / "deps"
    if importlib.util.find_spec("gguf") is None and deps.is_dir():
        sys.path.append(str(deps))
