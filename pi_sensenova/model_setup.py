"""Opt-in SenseNova model catalog. Importing this module never uses the network."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import threading
import json
from pathlib import Path


CATALOG = {
    "Official SenseNova U1.5 8B MoT": {
        "repo": "sensenova/SenseNova-U1.5-8B-MoT",
        "kind": "folder",
        "target": "SenseNova-U1.5-8B-MoT",
    },
    "SenseNova U1.5 8B MoT Q8 GGUF": {
        "repo": "realrebelai/SenseNova-U1.5-8B_GGUFs",
        "kind": "gguf",
        "file": "SenseNova-U1.5-8B-MoT-Q8_0.gguf",
        "target": "SenseNova-U1.5-8B-MoT-Q8_0.gguf",
    },
}

OFFICIAL_REPO = "sensenova/SenseNova-U1.5-8B-MoT"
RESOURCE_TARGET = "SenseNova-resources"
MODEL_PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.model", "*.tiktoken"]
RESOURCE_PATTERNS = ["*.json", "*.txt", "*.model", "*.tiktoken"]
_DOWNLOAD_LOCK = threading.Lock()


def _models_dir() -> Path:
    from modules import paths
    return Path(paths.models_path) / "Stable-diffusion"


def _entry(name: str) -> dict:
    try:
        return CATALOG[name]
    except KeyError as error:
        raise ValueError(f"Unknown SenseNova model: {name}") from error


def _revision(api, repo: str) -> str:
    revision = str(api.model_info(repo_id=repo).sha or "")
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", revision):
        raise RuntimeError(f"Hugging Face did not return an immutable revision for {repo}.")
    return revision


def instructions(name: str) -> str:
    item = _entry(name)
    root = _models_dir()
    if item["kind"] == "folder":
        target = root / item["target"]
        return (f"Download **https://huggingface.co/{item['repo']}** and place the complete model in "
                f"`{target}`. Python files are not needed.")
    target = root / item["target"]
    resources = root / RESOURCE_TARGET
    return (f"Download **https://huggingface.co/{item['repo']}/blob/main/{item['file']}** to `{target}`. "
            f"Also download config/tokenizer files from **https://huggingface.co/{OFFICIAL_REPO}** into "
            f"`{resources}`. Python files are not needed.")


def _install_snapshot(snapshot_download, repo: str, revision: str, target: Path,
                      patterns: list[str], staging: Path, validator) -> None:
    if target.exists():
        raise FileExistsError(f"Already exists; kept unchanged: {target}")
    staged = staging / target.name
    snapshot_download(repo_id=repo, revision=revision, local_dir=str(staged),
                      allow_patterns=patterns)
    validator(staged)
    os.rename(staged, target)


def _validate_model(folder: Path) -> None:
    if not (folder / "config.json").is_file():
        raise RuntimeError("Downloaded model is missing config.json.")
    weights = list(folder.rglob("*.safetensors"))
    if not weights:
        raise RuntimeError("Downloaded model has no safetensors weights.")
    for index in folder.rglob("*.safetensors.index.json"):
        try:
            names = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise RuntimeError(f"Invalid safetensors index: {index.name}") from error
        missing = [name for name in names if not (index.parent / name).is_file()]
        if missing:
            raise RuntimeError(f"Missing safetensors shard: {missing[0]}")


def _validate_resources(folder: Path) -> None:
    required = (folder / "config.json", folder / "tokenizer_config.json")
    if not all(path.is_file() for path in required):
        raise RuntimeError("SenseNova resources need config.json and tokenizer_config.json.")
    vocab = ("tokenizer.json", "tokenizer.model", "vocab.json", "vocab.txt")
    if not any((folder / name).is_file() for name in vocab):
        raise RuntimeError("SenseNova resources are missing tokenizer vocabulary files.")


def _refresh() -> None:
    try:
        from modules import sd_models
        sd_models.list_models()
    except (ImportError, AttributeError):
        pass


def download(name: str):
    """Download one explicitly selected catalog item; yields UI-friendly status text."""
    item = _entry(name)
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    with _DOWNLOAD_LOCK:
        root = _models_dir()
        root.mkdir(parents=True, exist_ok=True)
        yield f"Checking immutable revision for {name}…"
        api = HfApi()
        revision = _revision(api, item["repo"])
        staging = Path(tempfile.mkdtemp(prefix=".sensenova-download-", dir=root))
        try:
            if item["kind"] == "folder":
                yield "Downloading selected model…"
                _install_snapshot(snapshot_download, item["repo"], revision,
                                  root / item["target"], MODEL_PATTERNS, staging,
                                  _validate_model)
            else:
                target = root / item["target"]
                if target.exists():
                    raise FileExistsError(f"Already exists; kept unchanged: {target}")
                resources = root / RESOURCE_TARGET
                if resources.exists():
                    _validate_resources(resources)
                else:
                    yield "Downloading official config/tokenizer resources…"
                    official_revision = _revision(api, OFFICIAL_REPO)
                    _install_snapshot(snapshot_download, OFFICIAL_REPO, official_revision,
                                      resources, RESOURCE_PATTERNS, staging,
                                      _validate_resources)
                yield "Downloading selected GGUF…"
                staged_file = Path(hf_hub_download(
                    repo_id=item["repo"], filename=item["file"], revision=revision,
                    local_dir=str(staging)))
                if not staged_file.is_file() or staged_file.stat().st_size < 4:
                    raise RuntimeError("Downloaded GGUF is empty or incomplete.")
                os.rename(staged_file, target)
            _refresh()
            yield f"Ready: {root / item['target']}"
        finally:
            shutil.rmtree(staging, ignore_errors=True)
