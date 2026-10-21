"""Clean-up and export: transparent edges, 1:1 PNG, nearest-neighbour upscale, dark preview."""

from pathlib import Path

import numpy as np
from PIL import Image

PREVIEW_BG = (24, 24, 28, 255)


def _neighbours(op):
    p = np.pad(op, 1)
    return p[:-2, 1:-1].astype(int) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]


def outside_background(bg_like):
    """Cells of bg_like connected to the grid border (enclosed bg-coloured areas stay)."""
    out = np.zeros_like(bg_like)
    out[0], out[-1], out[:, 0], out[:, -1] = bg_like[0], bg_like[-1], bg_like[:, 0], bg_like[:, -1]
    while True:
        p = np.pad(out, 1)
        grown = bg_like & (out | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:])
        if (grown == out).all():
            return out
        out = grown


def remove_strays(rgba, bg=None, tol=60):
    """Clear isolated opaque pixels and, if bg is given, background-coloured pixels on the outer edge.

    Returns the number of pixels cleared.
    """
    cleared = 0
    while True:
        op = rgba[..., 3] > 0
        stray = op & (_neighbours(op) == 0)
        if bg is not None:
            edge = op & (_neighbours(op) < 4)
            stray |= edge & (np.abs(rgba[..., :3].astype(int) - bg).sum(2) < tol)
        if not stray.any():
            return cleared
        rgba[stray] = 0
        cleared += int(stray.sum())


def trim(rgba):
    ys, xs = np.where(rgba[..., 3] > 0)
    return rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def save_outputs(rgba, out_dir, name, width=1000):
    """Write <name>_<w>x<h>.png, <name>_<W>w.png and preview_dark.png; return their paths and the scale."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = rgba.shape[:2]
    img = Image.fromarray(rgba.astype(np.uint8), "RGBA")
    scale = max(1, round(width / w))
    big = img.resize((w * scale, h * scale), Image.NEAREST)
    small_path = out_dir / f"{name}_{w}x{h}.png"
    big_path = out_dir / f"{name}_{w * scale}w.png"
    img.save(small_path)
    big.save(big_path)
    margin = max(16, big.width // 25)
    pv = Image.new("RGBA", (big.width + 2 * margin, big.height + 2 * margin), PREVIEW_BG)
    pv.alpha_composite(big, (margin, margin))
    pv_path = out_dir / "preview_dark.png"
    pv.convert("RGB").save(pv_path)
    return {"sprite": str(small_path), "upscaled": str(big_path), "preview": str(pv_path), "scale": scale}
