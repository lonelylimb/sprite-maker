"""python -m sprite_tools INPUT [--colour-source IMAGE] ...

Turns a screenshot / upscale of pixel art, or a bead chart, into a clean sprite:
1:1 PNG, ~1000px nearest-neighbour PNG, dark preview and report.json.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

from . import beadchart, export, grid as gridmod, palette


def _slug(path):
    return re.sub(r"[^a-z0-9]+", "_", Path(path).stem.lower()).strip("_") or "sprite"


def from_image(rgba, bg_tol, max_colours, nx=None, ny=None, clear_holes=False):
    """Sprite from an upscaled image. Returns (rgba uint8, report dict, grid, cell colours, kept cells)."""
    rgb = rgba[..., :3]
    bg = gridmod.background_colour(rgb)
    box = gridmod.bbox(gridmod.foreground_mask(rgba, bg, bg_tol))
    g = gridmod.fit_grid(rgb, box, nx, ny)
    cols, alpha = gridmod.sample_cells(rgba, g)
    bg_like = (alpha < 128) | (np.abs(cols - bg).sum(2) <= bg_tol)
    outside = export.outside_background(bg_like)
    keep = ~bg_like if clear_holes else ~outside
    C, lab = palette.cluster_colours(cols[keep], max_colours)
    out = np.zeros(cols.shape[:2] + (4,), np.uint8)
    out[keep, :3] = C[lab].round()
    out[keep, 3] = 255
    notes = []
    enclosed = bg_like & ~outside
    if clear_holes and enclosed.any():
        notes.append(f"made {int(enclosed.sum())} enclosed background-coloured pixels transparent (--clear-holes)")
    elif enclosed.any():
        ys, xs = np.where(enclosed)
        notes.append(f"{int(enclosed.sum())} pixels are close to the background colour but enclosed by the art, so they "
                     f"were kept (around x {xs.min()}-{xs.max()}, y {ys.min()}-{ys.max()} before trimming); "
                     "use --clear-holes if they are background showing through")
    report = {"mode": "image", "grid": vars(g), "background": [int(v) for v in bg], "notes": notes}
    return out, report, g, cols, keep


def from_chart(rgba, colour_source, bg_tol):
    ch = beadchart.read_chart(rgba)
    lab = ch.labels
    ys, xs = np.where(lab >= 0)
    lab = lab[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    filled = lab >= 0
    colours = ch.colours.copy()
    notes = list(ch.notes)
    if ch.relabelled:
        cells = ", ".join(f"r{r - ys.min() + 1}c{c - xs.min() + 1}" for r, c, _, _ in ch.relabelled)
        notes.append(f"{len(ch.relabelled)} cells relabelled by printed-code shape (watermarks or misread colours): {cells}")
    if colour_source is not None:
        src = gridmod.load_rgba(colour_source)
        _, rep, g, cols, keep = from_image(src, bg_tol, 64, lab.shape[1], lab.shape[0])
        agree = (keep == filled).mean()
        notes.append(f"colour source aligned on a {g.nx}x{g.ny} grid, {g.px:.3f}x{g.py:.3f}px per pixel; "
                     f"outline agrees with the chart on {agree:.1%} of cells")
        if agree < 0.9:
            notes.append("WARNING: colour source does not line up with the chart; chart colours were used instead")
        else:
            med = palette.group_medians(lab[filled], cols[filled], len(colours))
            ok = ~np.isnan(med).any(1)
            colours[ok] = med[ok]
    out = np.zeros(lab.shape + (4,), np.uint8)
    out[filled, :3] = colours[lab[filled]].round()
    out[filled, 3] = 255
    report = {"mode": "chart", "chart_grid": vars(ch.grid), "legend_entries": len(ch.swatches),
              "legend_counts": ch.counts, "notes": notes}
    return out, report


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m sprite_tools", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("--colour-source", "--color-source", help="clean image of the same art to take colours from (chart mode)")
    ap.add_argument("--mode", choices=["auto", "chart", "image"], default="auto")
    ap.add_argument("--name", help="output name (default: from the input file name)")
    ap.add_argument("--out", help="output folder (default: sprites/<name>)")
    ap.add_argument("--width", type=int, default=1000, help="target width of the upscaled PNG")
    ap.add_argument("--bg-tol", type=int, default=40, help="max sum |dRGB| from the background colour to count as background")
    ap.add_argument("--max-colours", type=int, default=64)
    ap.add_argument("--keep-all", action="store_true", help="keep separate parts instead of only the largest shape")
    ap.add_argument("--keep-fringe", action="store_true", help="keep edge-only blend colours (image mode)")
    ap.add_argument("--clear-holes", action="store_true",
                    help="make enclosed background-coloured areas transparent too (image mode)")
    a = ap.parse_args(argv)

    name = a.name or _slug(a.input)
    out_dir = a.out or f"sprites/{name}"
    rgba = gridmod.load_rgba(a.input)

    mode = a.mode
    if mode == "auto":
        # charts are mostly white paper; anything else is read as a plain image
        mode = "chart" if (rgba[..., :3].min(2) > 240).mean() > 0.15 else "image"
    fallback = None
    if mode == "chart":
        try:
            out, report = from_chart(rgba, a.colour_source, a.bg_tol)
        except beadchart.ChartError as e:
            if a.mode == "chart":
                raise SystemExit(f"could not read chart: {e}")
            fallback = f"read as a plain image because chart reading failed: {e}"
            mode = "image"
    if mode == "image":
        out, report, *_ = from_image(rgba, a.bg_tol, a.max_colours, clear_holes=a.clear_holes)
        if fallback:
            report["notes"].insert(0, fallback)
        if a.colour_source:
            report["notes"].append("--colour-source is only used in chart mode; ignored")

    if not a.keep_all:
        for size, (x, y) in export.keep_largest(out):
            report["notes"].append(f"dropped a separate {size}-pixel part around x {x}, y {y} (before trimming); "
                                   "use --keep-all to keep separate parts")
    if mode == "image" and not a.keep_fringe:
        for colour, n in export.remove_fringe(out):
            report["notes"].append(f"removed {n} edge pixels of fringe colour {colour} (only used around the outside); "
                                   "use --keep-fringe to keep them")
    bg = report.get("background")
    cleared = export.remove_strays(out, np.array(bg) if bg else None, a.bg_tol * 1.5)
    if cleared:
        report["notes"].append(f"removed {cleared} stray edge pixels")
    out = export.trim(out)
    files = export.save_outputs(out, out_dir, name, a.width)
    h, w = out.shape[:2]
    report.update({"pixel_size": [w, h], "colours": int(len(np.unique(out[out[..., 3] > 0][:, :3], axis=0))),
                   "files": files})
    Path(out_dir, "report.json").write_text(json.dumps(report, indent=2, default=float))

    print(f"{name}: {w}x{h} pixels, {report['colours']} colours, upscaled x{files['scale']} ({mode} mode)")
    for k in ("sprite", "upscaled", "preview"):
        print(f"  {files[k]}")
    for n in report["notes"]:
        print(f"  - {n}")
