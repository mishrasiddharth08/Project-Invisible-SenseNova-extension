"""Read-only startup check. Never install or upgrade Forge's dependencies."""
from pathlib import Path

root = Path(__file__).resolve().parent
if not (root / "vendor" / "SenseNova-U1" / "src").is_dir():
    print("[Invisible-SenseNova] Runtime missing: restore the complete extension package.")
