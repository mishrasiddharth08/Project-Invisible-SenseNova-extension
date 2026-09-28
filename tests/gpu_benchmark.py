"""Real-weight benchmark: actual time, peak allocation, and inspectable PNGs."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "vendor" / "deps")]
from pi_sensenova.engine import ENGINE
from pi_sensenova.settings import Settings, compose

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--resources", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--size", type=int, default=1024)
parser.add_argument("--memory", default="Auto")
parser.add_argument("--seed", type=int, default=4217)
parser.add_argument("--adapter", default="")
parser.add_argument("--cfg", type=float, default=3.0)
parser.add_argument("--forge", default="")
parser.add_argument("--prompt", default="Candid documentary photograph of an adult woman in her thirties at a small outdoor flower market, holding a bunch of daisies with both hands. Freckles, individual hair strands, a slightly creased linen shirt. Soft overcast daylight, muted natural colors, unretouched skin, realistic hands, subtle background detail, 50mm lens, eye-level medium portrait.")
args = parser.parse_args()
if args.forge:
    sys.path.insert(0, str(Path(args.forge).resolve()))
    sys.argv = sys.argv[:1]  # Forge parses its own CLI flags during import.
settings = Settings(resources=args.resources, memory=args.memory, fast=bool(args.adapter), adapter=args.adapter)
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
try:
    ENGINE.load(args.model, settings)
    result = ENGINE.generate(prompt=args.prompt, width=args.size, height=args.size, steps=args.steps,
                             cfg=args.cfg, seed=args.seed, settings=settings,
                             callback=lambda step, total: print(f"step {step}/{total}", flush=True) if step else False)
    for i, img in enumerate(result):
        img.save(out / f"image-{i}.png")
    (out / "metrics.json").write_text(json.dumps({**vars(args), **ENGINE.metrics}, indent=2), encoding="utf-8")
finally:
    ENGINE.unload()
