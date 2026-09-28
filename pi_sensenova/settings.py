"""UI -> immutable run settings. No model or host imports."""
from dataclasses import dataclass
import math

UI_KEYS = ("resources", "memory", "fast", "adapter", "natural", "think", "shift", "degrid")
DEFAULTS = ("", "Auto", False, "", True, False, 3.0, False)
NATURAL = (
    "Photographic treatment: natural skin texture with visible pores and fine facial hair, "
    "subtle irregularities, realistic fabric weave and material texture, physically plausible "
    "soft lighting, restrained contrast and natural color. Preserve the requested subject "
    "and composition. No beauty retouching, waxy skin, airbrushing or artificial gloss."
)


@dataclass(frozen=True)
class Settings:
    resources: str = ""
    memory: str = "Auto"
    fast: bool = False
    adapter: str = ""
    natural: bool = True
    think: bool = False
    shift: float = 3.0
    degrid: bool = False

    @classmethod
    def from_ui(cls, values):
        if len(values) == 7:  # Existing API clients and saved UI settings.
            values = (*values, False)
        if len(values) != len(UI_KEYS):
            raise ValueError("SenseNova controls changed; fully restart Forge before generating.")
        result = cls(**dict(zip(UI_KEYS, values)))
        if not isinstance(result.degrid, bool):
            raise ValueError("DeGrid must be enabled or disabled.")
        if result.memory not in ("Auto", "Full", "Fast offload", "Balanced", "Low VRAM"):
            raise ValueError("Unknown SenseNova memory mode.")
        if not math.isfinite(float(result.shift)) or not 0.1 <= float(result.shift) <= 10:
            raise ValueError("Timestep shift must be between 0.1 and 10.")
        return result


def compose(prompt, negative, natural=False):
    result = str(prompt).strip()
    if not result:
        raise ValueError("Write a prompt before generating with SenseNova.")
    if natural:
        result += "\n\n" + NATURAL
    if str(negative or "").strip():
        result += "\n\nAvoid: " + str(negative).strip()
    return result


def dimensions(width, height):
    """Respect explicit sizes; never silently snap 1024 to 2048."""
    width, height = int(width), int(height)
    if min(width, height) < 256 or max(width, height) > 4096:
        raise ValueError("SenseNova dimensions must be 256–4096 pixels.")
    if width % 32 or height % 32:
        raise ValueError("SenseNova width and height must be multiples of 32.")
    return width, height


def memory_mode(free_bytes, weight_bytes, requested="Auto"):
    """Budget actual weights + activation headroom, rather than GPU marketing name."""
    if requested != "Auto":
        return {"Full": "full", "Fast offload": "fast", "Balanced": "balanced", "Low VRAM": "low"}[requested]
    gib = 1024 ** 3
    if free_bytes >= weight_bytes + 6 * gib:
        return "full"
    if free_bytes >= 16 * gib:
        return "fast"
    return "balanced" if free_bytes >= 10 * gib else "low"
