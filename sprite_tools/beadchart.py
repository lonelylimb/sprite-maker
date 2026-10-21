"""Read a bead / cross-stitch chart: grid of coloured cells with printed codes and a colour legend."""

from dataclasses import dataclass, field

import numpy as np

from . import ocr
from .grid import Grid, estimate_pitch, refine_axis
from .palette import match_to_legend


class ChartError(ValueError):
    pass


@dataclass
class Chart:
    grid: Grid                 # grid of the art area (header rows/columns removed)
    labels: np.ndarray         # (ny, nx) legend index per cell, -1 for empty cells
    colours: np.ndarray        # (n_legend, 3) legend colours as they appear in the cells
    swatches: np.ndarray       # (n_legend, 3) legend colours as printed in the legend
    counts: list               # legend bead counts read by OCR (None where unread / OCR missing)
    relabelled: list = field(default_factory=list)  # (row, col, old, new), 0-based in the art grid
    notes: list = field(default_factory=list)


def _line_profiles(lum):
    """Per column / row: how many pixels are a thin dark line (darker than 2px either side)."""
    d = 8
    v = (lum < np.roll(lum, 2, 1) - d) & (lum < np.roll(lum, -2, 1) - d)
    h = (lum < np.roll(lum, 2, 0) - d) & (lum < np.roll(lum, -2, 0) - d)
    v[:, :2] = v[:, -2:] = False
    h[:2] = h[-2:] = False
    return v.sum(0).astype(float), h.sum(1).astype(float)


def _axis_lines(profile):
    """(first line position, pitch, cell count) of the longest run of evenly spaced grid lines."""
    n_px = len(profile)
    p = estimate_pitch(profile + 0.0, 0, n_px, min_pitch=6)
    offs = np.arange(0, p, 0.1)
    k = np.arange(int(n_px / p) + 1)
    pos = offs[:, None] + p * k[None, :]
    score = np.interp(pos, np.arange(n_px), profile, left=0, right=0).sum(1)
    o = offs[score.argmax()]
    pos = o + p * k
    pos = pos[pos < n_px - 1]
    strength = np.array([profile[max(0, int(x) - 1):int(x) + 2].max() for x in pos])
    strong = strength > 0.35 * np.median(np.sort(strength)[-max(3, len(strength) // 2):])
    best, run_start, cur = (0, 0), None, 0
    for i, s in enumerate(list(strong) + [False]):
        if s and run_start is None:
            run_start = i
        if not s and run_start is not None:
            if i - run_start > best[1] - best[0]:
                best = (run_start, i)
            run_start = None
    a, b = best
    n = b - a - 1
    if n < 4:
        raise ChartError("no regular grid lines found")
    o, p = refine_axis(profile, pos[a] + 0.5, n, p, spread=0.01)
    return o, p, n


def _text_mask(blk, fill):
    """0..1 mask of the printed code in a cell or legend box filled with fill."""
    lum, fl = blk.mean(2), fill.mean()
    # codes are printed dark on light cells and white on dark cells; watermarks are
    # translucent white, so they never look like dark text and are too faint to pass
    # the white-text threshold
    dark = np.clip((fl - lum - 25) / 60, 0, 1)
    room = max(255 - fl, 1)
    light = np.clip((lum - fl - 0.6 * room) / (0.3 * room), 0, 1)
    return dark if dark.sum() >= light.sum() else light


def _text_width(mask):
    xs = np.where(mask.max(0) > 0.4)[0]
    return xs.max() - xs.min() + 1 if len(xs) else 0


def _read_cell(rgb, box, inset, size):
    """Fill colour (robust to light watermark strokes) and a size-cropped text mask for one cell."""
    l, t, r, b = box
    blk = rgb[t + inset:b - inset, l + inset:r - inset][:size[0], :size[1]]
    strip = np.concatenate([blk[:2].reshape(-1, 3), blk[-2:].reshape(-1, 3)])
    sl = strip.mean(1)
    fill = np.median(strip[sl <= np.median(sl)], 0)
    return fill, _text_mask(blk, fill)


def _legend(rgb, y_from, pitch):
    """Legend swatches below the grid: list of (colour, code text width, count_box)."""
    h, w = rgb.shape[:2]
    nonwhite = rgb.min(2) < 244
    frac = nonwhite.mean(1)
    rows = [(y_from <= y) and (0.02 < frac[y] < 0.9) for y in range(h)]
    bands, start = [], None
    for y, on in enumerate(rows + [False]):
        if on and start is None:
            start = y
        if not on and start is not None:
            if y - start >= pitch * 0.6:
                bands.append((start, y))
            start = None
    items = []
    for ya, yb in bands:
        # rows just inside the box border, above and below the printed code
        strip = np.concatenate([rgb[ya + 3:ya + 5], rgb[yb - 5:yb - 3]])
        white = strip.mean(0).min(1) > 244
        segs, s = [], None
        for x, wt in enumerate(list(white) + [True]):
            if not wt and s is None:
                s = x
            if wt and s is not None:
                if x - s >= pitch * 0.8:
                    segs.append((s, x))
                s = None
        for i, (s, e) in enumerate(segs):
            colour = np.median(strip[:, s + 2:e - 2].reshape(-1, 3), 0)
            text = _text_mask(rgb[ya + 3:yb - 3, s + 2:e - 2], colour)
            nxt = segs[i + 1][0] if i + 1 < len(segs) else min(w, e + 2 * (e - s))
            items.append((colour, _text_width(text), (e, ya, nxt, yb)))
    return items


def _repair_by_code_shape(labels, fills, masks, k=7, min_votes=4):
    """Relabel cells whose printed code looks like the code of a different colour.

    Cells with the same printed code are the same colour, so a cell whose colour
    was shifted by a watermark is relabelled to the colour that its k most
    similar-looking cells (code shape first, colour second) agree on.
    """
    M = masks.reshape(len(masks), -1)
    sq = (M ** 2).sum(1)
    out = labels.copy()
    changed = []
    for s in range(0, len(M), 512):
        td = (sq[s:s + 512, None] + sq[None] - 2 * M[s:s + 512] @ M.T) / M.shape[1]
        cd = np.linalg.norm(fills[s:s + 512, None] - fills[None], axis=2)
        score = td * 100 + cd / 10
        score[np.arange(len(score)), np.arange(s, s + len(score))] = np.inf
        nn = np.argsort(score, 1)[:, :k]
        for i, row in enumerate(nn):
            votes = np.bincount(labels[row], minlength=labels.max() + 1)
            j = s + i
            if votes.max() >= min_votes and votes.argmax() != labels[j]:
                out[j] = votes.argmax()
                changed.append(j)
    return out, changed


def _split_merged(labels, masks, colours, widths, counts):
    """Split cells shared by two legend colours too close to tell apart, using code length.

    When a legend entry got no cells (or far fewer than the legend says) and its
    nearest legend colour has roughly both entries' counts, the printed codes in that group are split by text
    width (e.g. "A6" vs "A10"); the wider group goes to the entry whose legend
    code is wider. Edits labels in place and returns notes.
    """
    notes = []
    used = np.bincount(labels, minlength=len(colours))
    cell_w = np.array([_text_width(m) for m in masks])
    starved = [z for z in range(len(colours))
               if used[z] == 0 or (counts[z] is not None and used[z] < 0.5 * counts[z])]
    for z in starved:
        others = [k for k in range(len(colours)) if k != z and used[k] > 0]
        m = min(others, key=lambda k: np.linalg.norm(colours[k] - colours[z]))
        if widths[z] == widths[m]:
            continue
        both = used[m] + used[z]
        if counts[z] is not None and counts[m] is not None:
            if abs(both - counts[z] - counts[m]) > max(3, 0.1 * both):
                continue
        idx = np.where((labels == m) | (labels == z))[0]
        cut = (cell_w[idx].min() + cell_w[idx].max()) / 2
        wide = cell_w[idx] > cut
        if wide.all() or not wide.any():
            continue
        wide_entry, narrow_entry = (z, m) if widths[z] > widths[m] else (m, z)
        labels[idx[wide]] = wide_entry
        labels[idx[~wide]] = narrow_entry
        notes.append(f"legend entries {m + 1} and {z + 1} are almost the same colour; split their "
                     f"{len(idx)} cells by printed code length ({int((~wide).sum())} narrow, {int(wide.sum())} wide)")
        used = np.bincount(labels, minlength=len(colours))
    return notes


def read_chart(rgba):
    rgb = rgba[..., :3]
    vx, hy = _line_profiles(rgb.mean(2))
    ox, px, nx = _axis_lines(vx)
    oy, py, ny = _axis_lines(hy)
    grid = Grid(ox, px, nx, oy, py, ny)
    inset = max(2, int(round(min(px, py) * 0.13)))

    size = (int(py) - 2 * inset, int(px) - 2 * inset)
    fills = np.zeros((ny, nx, 3))
    masks = np.zeros((ny, nx) + size)
    for r in range(ny):
        for c in range(nx):
            fills[r, c], masks[r, c] = _read_cell(rgb, grid.cell_box(c, r), inset, size)

    # drop number headers: an outer row/column whose cells are all the same non-white grey
    def uniform(v):
        return np.abs(v - np.median(v, 0)).sum(1).max() < 25 and v.min() < 235
    r0, r1, c0, c1 = 0, ny, 0, nx
    for _ in range(2):
        if uniform(fills[r0, c0:c1]):
            r0 += 1
        if uniform(fills[r1 - 1, c0:c1]):
            r1 -= 1
        if uniform(fills[r0:r1, c0]):
            c0 += 1
        if uniform(fills[r0:r1, c1 - 1]):
            c1 -= 1
    fills, masks = fills[r0:r1, c0:c1], masks[r0:r1, c0:c1]
    grid = Grid(ox + px * c0, px, c1 - c0, oy + py * r0, py, r1 - r0)

    perimeter = np.concatenate([fills[0], fills[-1], fills[:, 0], fills[:, -1]])
    empty_col = np.median(perimeter, 0)
    empty = np.abs(fills - empty_col).sum(2) < 30

    items = _legend(rgb, int(grid.y0 + grid.py * grid.ny) + 2, min(px, py))
    if len(items) < 2:
        raise ChartError("no colour legend found below the grid")
    swatches = np.array([col for col, _, _ in items])
    widths = [wd for _, wd, _ in items]
    counts = [ocr.read_number(rgb, box) for _, _, box in items]

    X = fills[~empty]
    lab, colours = match_to_legend(X, swatches)
    lab2, changed = _repair_by_code_shape(lab, X, masks[~empty])
    split_notes = _split_merged(lab2, masks[~empty], colours, widths, counts)
    labels = np.full(empty.shape, -1)
    labels[~empty] = lab2
    rc = np.argwhere(~empty)
    relabelled = [(int(rc[j][0]), int(rc[j][1]), int(lab[j]), int(lab2[j])) for j in changed]

    chart = Chart(grid, labels, colours, swatches, counts, relabelled, split_notes)
    used = np.bincount(lab2, minlength=len(swatches))
    if all(c is not None for c in counts):
        total = sum(counts)
        chart.notes.append(f"legend total {total} beads, found {int((~empty).sum())} filled cells")
        for k, (want, got) in enumerate(zip(counts, used)):
            if want != got:
                chart.notes.append(f"legend entry {k + 1} (colour {tuple(int(v) for v in swatches[k])}): "
                                   f"legend says {want}, assigned {got}")
    for k in np.where(used == 0)[0]:
        chart.notes.append(f"legend entry {k + 1} (colour {tuple(int(v) for v in swatches[k])}) "
                           "matched no cells; it is probably too close to another legend colour to tell apart")
    return chart
