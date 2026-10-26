"""results/real/*/windows.png (2 MB each, written by run_real_data.py) -> windows.jpg (about 0.7 MB, quality 80, full colour resolution).

The PNGs are git-ignored; the JPEGs are what is committed.  A palette-reduced PNG is not an option: it turns the blue energy-detector boxes green.

    python scripts/shrink_real_figures.py
"""
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

for png in sorted((ROOT / "results/real").glob("*/windows.png")):
    Image.open(png).convert("RGB").save(png.with_suffix(".jpg"), quality=80, subsampling=0, optimize=True)
    print(png.parent.name, f"{png.with_suffix('.jpg').stat().st_size / 1e3:.0f} kB")
