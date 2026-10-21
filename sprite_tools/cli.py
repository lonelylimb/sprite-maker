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


def _count_error(labels, counts):
    """Distance between assigned cells per legend entry and the legend counts it could read, or None."""
    known = [k for k, c in enumerate(counts) if c is not None]
    if len(known) < len(counts) / 2:
        return None
    used = np.bincount(labels, minlength=len(counts))
    return int(sum(abs(used[k] - counts[k]) for k in known))


def _label_from_source(ch, art, labels, cols, tol=30):
    """Use the clean picture to correct chart labels. Returns (art, labels, notes).

    A new labelling is kept only if it matches the legend counts better than the
    chart's own (when enough counts were read to tell).
    """
    chart_lab = labels[art]
    X = cols[art]
    near = palette.nearest(X, ch.swatches)
    dist = np.linalg.norm(X - ch.swatches[near], axis=1)
    med = palette.group_medians(chart_lab, X, len(ch.swatches))
    sizes = np.bincount(chart_lab, minlength=len(ch.swatches))
    off = np.abs(med - ch.swatches).sum(1)
    typical = np.nanmedian(np.where(sizes >= 3, off, np.nan))
    clear = np.zeros(len(chart_lab), bool)
    if typical < 40:
        # the picture shows the bead colours themselves: read each cell from it, except
        # where an overlay spoils the picture (nothing within tol) and the chart decides
        new = np.where(dist <= tol, near, chart_lab)
        # an entry that looks nothing like its swatch in the picture is a clear bead
        # with the background showing through: leave those cells transparent
        see = [k for k in range(len(off)) if sizes[k] >= 3 and off[k] > 150]
        clear = np.isin(chart_lab, see) & (dist > tol)
        how = (f"labelled {int((dist <= tol).sum())} cells from the colour source's bead colours "
               f"and {int((dist > tol).sum())} from the chart where the source is covered")
    else:
        new = palette.relabel_two_views(chart_lab, ch.fills[art], X)
        how = f"{int((new != chart_lab).sum())} cells relabelled using chart and colour source together"
    old_err = _count_error(chart_lab[~clear], ch.counts)
    new_err = _count_error(new[~clear], ch.counts)
    keep = new_err < old_err if old_err is not None else typical < 40
    notes = []
    if not keep:
        notes.append("kept the chart's labels: the colour source did not bring them closer to the legend counts")
        return art, labels, notes
    notes.append(how + (f" (legend count mismatch {old_err} -> {new_err})" if old_err is not None else ""))
    if clear.any():
        names = ", ".join(str(tuple(int(v) for v in ch.swatches[k])) for k in see)
        notes.append(f"{int(clear.sum())} cells of legend colour {names} show the background in the colour "
                     "source (clear beads), so they were left transparent")
    labels[art] = new
    idx = np.argwhere(art)[clear]
    art = art.copy()
    art[idx[:, 0], idx[:, 1]] = False
    labels[~art] = -1
    ch.labels = labels.copy()
    notes += [n for n in beadchart.count_notes(ch) if "legend says" in n or "total" in n]
    return art, labels, notes


def from_chart(rgba, colour_source, cells=None):
    ch = beadchart.read_chart(rgba, cells)
    art = ch.labels >= 0
    labels = ch.labels.copy()
    colours = ch.colours.copy()
    notes = list(ch.notes)
    cols = None
    if colour_source is not None:
        src = gridmod.load_rgba(colour_source)
        g, score = gridmod.align_to_cells(src, ch.fills, art)
        cols, _ = gridmod.sample_cells(src, g, core=0.35)
        notes.append(f"colour source aligned at {g.px:.3f}x{g.py:.3f}px per pixel; "
                     f"colour correlation with the chart {score:.2f}")
        if score < 0.5:
            notes.append("WARNING: colour source does not line up with the chart; chart colours were used instead")
            cols = None
    if ch.faded:
        # white beads get pale printed codes, so they look washed out like the non-art cells
        near = art.copy()
        added = 0
        while cols is not None:
            p = np.pad(art, 1)
            touch = (p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]) & ~art
            # a light cell next to the art that the picture shows in a bead colour
            bead = np.linalg.norm(cols - ch.swatches[palette.nearest(cols.reshape(-1, 3), ch.swatches)]
                                  .reshape(cols.shape), axis=2) <= 30
            add = touch & (cols.mean(2) > 150) & bead
            if not add.any():
                break
            art |= add
            art = ~export.outside_background(~art)
            added += int(add.sum())
        labels[art & (labels < 0)] = ch.all_labels[art & (labels < 0)]
        if added:
            notes.append(f"added {added} light cells next to the art that show a bead colour in the colour source "
                         f"(light beads printed with pale codes); art is now {int(art.sum())} cells")
        elif cols is None:
            p = np.pad(near, 1)
            n = int(((p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]) & ch.light).sum())
            if n:
                notes.append(f"{n} near-white cells touch the art but have pale codes; they may be white beads "
                             "(pass --colour-source to decide)")
    if cols is not None:
        art, labels, more = _label_from_source(ch, art, labels, cols)
        if not more[0].startswith("kept"):
            notes = [n for n in notes if "legend says" not in n and "legend total" not in n]
        notes += more
    ys, xs = np.where(art)
    y0, x0 = ys.min(), xs.min()
    crop = (slice(y0, ys.max() + 1), slice(x0, xs.max() + 1))
    lab, art = labels[crop], art[crop]
    if ch.relabelled:
        where = ", ".join(f"r{r - y0 + 1}c{c - x0 + 1}" for r, c, _, _ in ch.relabelled)
        notes.append(f"{len(ch.relabelled)} cells relabelled by printed-code shape (watermarks or misread colours): {where}")
    if cols is not None:
        med = palette.group_medians(lab[art], cols[crop][art], len(colours))
        ok = ~np.isnan(med).any(1)
        colours[ok] = med[ok]
    out = np.zeros(lab.shape + (4,), np.uint8)
    out[art, :3] = colours[lab[art]].round()
    out[art, 3] = 255
    report = {"mode": "chart", "chart_grid": vars(ch.grid), "faded_chart": bool(ch.faded),
              "legend_entries": len(ch.swatches), "legend_counts": ch.counts, "notes": notes}
    return out, report


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m sprite_tools", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("--colour-source", "--color-source", help="clean image of the same art to take colours from (chart mode)")
    ap.add_argument("--mode", choices=["auto", "chart", "image"], default="auto")
    ap.add_argument("--cells", help="chart grid size as WxH when the chart states it, e.g. 52x52")
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
            cells = tuple(int(v) for v in a.cells.lower().split("x")) if a.cells else None
            out, report = from_chart(rgba, a.colour_source, cells)
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
