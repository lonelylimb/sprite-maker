"""Read a bead / cross-stitch chart: grid of coloured cells with printed codes and a colour legend."""

import re
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from . import ocr
from .export import components, outside_background
from .grid import Grid, estimate_pitch, refine_axis
from .palette import apply_affine, cluster_colours, fit_affine, match_to_legend, nearest


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
    fills: np.ndarray = None       # (ny, nx, 3) cell colours as read
    all_labels: np.ndarray = None  # (ny, nx) nearest legend entry for every cell, art or not
    faded: bool = False            # chart prints every cell and fades the ones that are not art
    light: np.ndarray = None       # faded charts: near-white non-art cells that may be white beads


def _line_profiles(lum):
    """Per column / row: how many pixels are a thin dark line (darker than 2px either side)."""
    d = 8
    v = (lum < np.roll(lum, 2, 1) - d) & (lum < np.roll(lum, -2, 1) - d)
    h = (lum < np.roll(lum, 2, 0) - d) & (lum < np.roll(lum, -2, 0) - d)
    v[:, :2] = v[:, -2:] = False
    h[:2] = h[-2:] = False
    return v.sum(0).astype(float), h.sum(1).astype(float)


def _axis_lines(profile, n=None):
    """(first line position, pitch, cell count) of the grid along one axis.

    Without n: the longest run of evenly spaced strong lines (a single faint line
    inside a run does not break it). With n: the window of n cells
    whose boundary lines are strongest overall.
    """
    n_px = len(profile)
    p0 = estimate_pitch(profile + 0.0, 0, n_px, min_pitch=6)

    def lines(p):
        offs = np.arange(0, p, 0.1)
        k = np.arange(int(n_px / p) + 1)
        score = np.interp(offs[:, None] + p * k[None, :], np.arange(n_px), profile, left=0, right=0).sum(1)
        pos = offs[score.argmax()] + p * k
        pos = pos[pos < n_px - 1]
        return pos, np.array([profile[max(0, int(x) - 1):int(x) + 2].max() for x in pos])

    if n:
        # a row of printed codes can look like a grid line halfway through each cell, halving
        # the pitch: try multiples and keep the one whose n cells have the strongest lines
        best = None
        for m in (1, 2, 3):
            pos, strength = lines(p0 * m)
            if len(pos) < n + 1:
                continue
            a = int(np.argmax([strength[i:i + n + 1].sum() for i in range(len(pos) - n)]))
            med = np.median(strength[a:a + n + 1])
            if best is None or med > 1.5 * best[0]:
                best = (med, p0 * m, pos, a)
        if best is None:
            raise ChartError(f"found only {len(lines(p0)[0]) - 1} cells along an axis, expected {n}")
        _, p, pos, a = best
    else:
        p = p0
        pos, strength = lines(p)
        strong = strength > 0.35 * np.median(np.sort(strength)[-max(3, len(strength) // 2):])
        # bridge a single faint line only deep inside a run, so the grid never
        # grows into legend boxes that happen to line up with it
        st = strong.copy()
        for i in range(3, len(st) - 3):
            if not st[i] and st[i - 3:i].all() and st[i + 1:i + 4].all():
                strong[i] = True
        best, run_start = (0, 0), None
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
    """Read one cell.

    Returns the fill colour (robust to light watermark strokes), a size-cropped text
    mask, how dark the darkest printed pixels are (crisp codes are near black, faded
    ones grey) and the share of strongly yellow pixels (outlined watermark lettering).
    """
    l, t, r, b = box
    blk = rgb[t + inset:b - inset, l + inset:r - inset][:size[0], :size[1]]
    strip = np.concatenate([blk[:2].reshape(-1, 3), blk[-2:].reshape(-1, 3)])
    sl = strip.mean(1)
    fill = np.median(strip[sl <= np.median(sl)], 0)
    px = blk.reshape(-1, 3)
    return fill, _text_mask(blk, fill), np.percentile(px.mean(1), 3), _is_watermark_yellow(px).mean()


def _valley(values, bin_width=5):
    """Threshold at the emptiest point between the two main peaks of a histogram."""
    hist, edges = np.histogram(values, bins=np.arange(0, 256 + bin_width, bin_width))
    smooth = np.convolve(hist, [1, 2, 1], "same")
    order = np.argsort(smooth)[::-1]
    a = order[0]
    b = next(i for i in order[1:] if abs(i - a) > 3)
    lo, hi = sorted((a, b))
    v = lo + int(np.argmin(smooth[lo:hi + 1]))
    return edges[v] + bin_width / 2


def _is_watermark_yellow(px):
    """Strong yellow of outlined watermark lettering (bead colours are never this saturated)."""
    return ((px[..., 0] + px[..., 1]) / 2 - px[..., 2] > 90) & (np.abs(px[..., 0] - px[..., 1]) < 45)


def _runs(flags):
    """(start, end) of each run of True values."""
    out, s = [], None
    for i, f in enumerate(list(flags) + [False]):
        if f and s is None:
            s = i
        if not f and s is not None:
            out.append((s, i))
            s = None
    return out


def _column_fill(band, tol=40):
    """Dominant colour of each column of band (box fill beats printed text and border lines).

    The pixel with the most similar pixels (within tol, sum of |dRGB|) in its column
    wins, so compression noise inside the fill does not split it up.
    """
    out = np.zeros((band.shape[1], 3))
    for x in range(band.shape[1]):
        px = band[:, x]
        near = np.abs(px[:, None] - px[None]).sum(2) <= tol
        out[x] = np.median(px[near[near.sum(1).argmax()]], 0)
    return out


def _legend(rgb, y_from, pitch):
    """Legend swatches below the grid: list of (colour, code text width, (box, count box)).

    Handles legends with a separate white count box after each swatch and legends
    with code and count inside one swatch, with or without gaps between swatches.
    """
    h, w = rgb.shape[:2]
    frac = (rgb.min(2) < 244).mean(1)
    rows = (np.arange(h) >= y_from) & (frac > 0.02)
    # legend rows can touch; split where a whole pixel row changes sharply (box edges)
    edge = np.r_[0, np.abs(np.diff(rgb, axis=0)).sum(2).mean(1)] > 180
    bands = []
    for ya, yb in _runs(rows):
        cuts = [ya] + [y for y in range(ya + 1, yb) if edge[y]] + [yb]
        bands += list(zip(cuts[:-1], cuts[1:]))
    items = []
    for ya, yb in bands:
        if yb - ya < pitch * 0.6 or _is_watermark_yellow(rgb[ya:yb]).mean() > 0.05:
            continue
        col = _column_fill(rgb[ya:yb])
        # a number header is one flat colour across; a legend row changes colour box to box
        chunks = np.array([np.median(col[x:x + 10], 0) for x in range(0, w - 10, 10)])
        if np.percentile(chunks.sum(1), 90) - np.percentile(chunks.sum(1), 10) < 30 and frac[ya:yb].mean() > 0.9:
            continue
        # boxes are separated by white gaps, checked on the plainest rows (no border, no text) ...
        inner = np.arange(ya + 1, yb - 1)
        strip = rgb[np.sort(inner[np.argsort(frac[inner], kind="stable")[:4]])]
        nonwhite = strip.mean(0).min(1) <= 244
        boxes = []
        for s, e in _runs(nonwhite):
            if e - s < pitch * 0.8:
                continue
            # ... or touch: split where two long steady stretches of different colour meet
            jump = np.r_[np.inf, np.abs(np.diff(col[s:e], axis=0)).sum(1)]
            cuts = [x for x in range(e - s) if jump[x] > 40] + [e - s]
            steady = [(s + a, s + b, np.median(col[s + a:s + b], 0)) for a, b in zip(cuts[:-1], cuts[1:])
                      if b - a >= pitch * 0.8]
            starts = [s] + [b0 for (_, _, c0), (b0, _, c1) in zip(steady, steady[1:])
                            if np.abs(c0 - c1).sum() > 40]
            for a, b in zip(starts, starts[1:] + [e]):
                c = np.median(col[a + 1:b - 1], 0)
                if c.min() > 244:
                    continue
                if boxes and a - boxes[-1][1] < pitch * 0.6 and np.abs(boxes[-1][2] - c).sum() < 12:
                    a0 = boxes[-1][0]  # a near-white box broken up by its own text
                    boxes[-1] = (a0, b, np.median(col[a0 + 1:b - 1], 0))
                else:
                    boxes.append((a, b, c))
        for i, (s, e, c) in enumerate(boxes):
            colour = c
            text = _text_mask(rgb[ya + 2:yb - 2, s + 2:e - 2], colour)
            nxt = boxes[i + 1][0] if i + 1 < len(boxes) else min(w, e + 2 * (e - s))
            items.append((colour, _text_width(text), ((s, ya, e, yb), (e, ya, nxt, yb))))
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


def _two_groups(masks):
    """Split code masks into two groups of similar-looking codes (2-means)."""
    M = masks.reshape(len(masks), -1)
    a = np.argmax(((M - M.mean(0)) ** 2).sum(1))
    b = np.argmax(((M - M[a]) ** 2).sum(1))
    C = M[[a, b]].astype(float)
    for _ in range(20):
        g = ((M[:, None] - C[None]) ** 2).sum(2).argmin(1)
        if g.all() or not g.any():
            break
        C = np.array([M[g == 0].mean(0), M[g == 1].mean(0)])
    return g == 1


def _split_merged(labels, masks, colours, widths, counts):
    """Split cells shared by two legend colours too close to tell apart, using the printed codes.

    When a legend entry got no cells (or far fewer than the legend says) and its
    nearest legend colour has roughly both entries' counts, the codes in that
    group are split in two: by text width when the legend codes differ in width
    (e.g. "A6" vs "A10"), otherwise by code shape with the group sizes matched to
    the legend counts. Edits labels in place and returns notes.
    """
    notes = []
    used = np.bincount(labels, minlength=len(colours))
    cell_w = np.array([_text_width(m) for m in masks])
    starved = [z for z in range(len(colours))
               if used[z] == 0 or (counts[z] is not None and used[z] < 0.5 * counts[z])]
    for z in starved:
        others = [k for k in range(len(colours)) if k != z and used[k] > 0]
        m = min(others, key=lambda k: np.linalg.norm(colours[k] - colours[z]))
        both = used[m] + used[z]
        known = counts[z] is not None and counts[m] is not None
        if known and abs(both - counts[z] - counts[m]) > max(3, 0.1 * both):
            continue
        idx = np.where((labels == m) | (labels == z))[0]
        if abs(widths[z] - widths[m]) >= 2:
            cut = (cell_w[idx].min() + cell_w[idx].max()) / 2
            grp = cell_w[idx] > cut
            first, second = (z, m) if widths[z] > widths[m] else (m, z)
            how = "printed code length"
        elif known:
            grp = _two_groups(masks[idx])
            n1 = int(grp.sum())
            # give each shape group the entry whose legend count it matches best
            first, second = (z, m) if abs(n1 - counts[z]) <= abs(n1 - counts[m]) else (m, z)
            how = "printed code shape, matched to the legend counts"
        else:
            continue
        if grp.all() or not grp.any():
            continue
        labels[idx[grp]] = first
        labels[idx[~grp]] = second
        notes.append(f"legend entries {m + 1} and {z + 1} are almost the same colour; split their {len(idx)} "
                     f"cells by {how} ({int((labels[idx] == m).sum())} / {int((labels[idx] == z).sum())})")
        used = np.bincount(labels, minlength=len(colours))
    return notes


def _read_cells(rgb, cells):
    """Grid of the chart and per cell: fill colour, code mask, darkest print, watermark-yellow share."""
    vx, hy = _line_profiles(rgb.mean(2))
    ox, px, nx = _axis_lines(vx, cells and cells[0])
    oy, py, ny = _axis_lines(hy, cells and cells[1])
    grid = Grid(ox, px, nx, oy, py, ny)
    inset = max(2, int(round(min(px, py) * 0.13)))

    size = (int(py) - 2 * inset, int(px) - 2 * inset)
    fills = np.zeros((ny, nx, 3))
    masks = np.zeros((ny, nx) + size)
    dark = np.zeros((ny, nx))
    yellow = np.zeros((ny, nx))
    for r in range(ny):
        for c in range(nx):
            fills[r, c], masks[r, c], dark[r, c], yellow[r, c] = _read_cell(rgb, grid.cell_box(c, r), inset, size)
    return grid, fills, masks, dark, yellow


def read_chart(rgba, cells=None):
    """Read a chart. cells=(nx, ny) fixes the grid size when the chart states it."""
    rgb = rgba[..., :3]
    grid, fills, masks, dark, yellow = _read_cells(rgb, cells)
    ox, px, nx, oy, py, ny = grid.x0, grid.px, grid.nx, grid.y0, grid.py, grid.ny

    if not cells:
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
        dark, yellow = dark[r0:r1, c0:c1], yellow[r0:r1, c0:c1]
        grid = Grid(ox + px * c0, px, c1 - c0, oy + py * r0, py, r1 - r0)

    notes = []
    blank = np.abs(fills - 255).sum(2) < 30
    faded = blank.mean() < 0.05
    light = None
    if faded:
        # every cell is printed; the art has crisp near-black codes, the rest is washed out
        wm = yellow > 0.1
        thr = _valley(dark[~wm])
        art = (dark < thr) & ~wm
        art = ~outside_background(~art)
        lab_cc, sizes = components(art)
        art = lab_cc == int(np.argmax(sizes)) + 1
        light = (fills.min(2) > 225) & ~art
        notes.append(f"faded-style chart: {int(art.sum())} cells with crisp codes (darkest print < {thr:.0f}) "
                     f"taken as the art; {int(wm.sum())} cells under yellow watermark lettering ignored unless enclosed")
    else:
        perimeter = np.concatenate([fills[0], fills[-1], fills[:, 0], fills[:, -1]])
        art = np.abs(fills - np.median(perimeter, 0)).sum(2) >= 30
    empty = ~art

    items = _legend(rgb, int(grid.y0 + grid.py * grid.ny) + 2, min(px, py))
    if len(items) < 2:
        raise ChartError("no colour legend found below the grid")
    swatches = np.array([col for col, _, _ in items])
    widths = [wd for _, wd, _ in items]
    counts = [ocr.read_count(rgb, box, after) for _, _, (box, after) in items]

    X = fills[~empty]
    lab, colours = match_to_legend(X, swatches)
    lab2, changed = _repair_by_code_shape(lab, X, masks[~empty])
    split_notes = _split_merged(lab2, masks[~empty], colours, widths, counts)
    labels = np.full(empty.shape, -1)
    labels[~empty] = lab2
    all_labels = nearest(fills.reshape(-1, 3), colours).reshape(empty.shape)
    all_labels[~empty] = lab2
    rc = np.argwhere(~empty)
    relabelled = [(int(rc[j][0]), int(rc[j][1]), int(lab[j]), int(lab2[j])) for j in changed]

    chart = Chart(grid, labels, colours, swatches, counts, relabelled, notes + split_notes,
                  fills, all_labels, faded, light)
    chart.notes += count_notes(chart)
    return chart


def count_notes(chart):
    """Compare assigned cells per legend entry with the legend counts."""
    notes = []
    lab = chart.labels[chart.labels >= 0]
    used = np.bincount(lab, minlength=len(chart.swatches))
    counts = chart.counts
    if all(c is not None for c in counts):
        notes.append(f"legend total {sum(counts)} beads, found {len(lab)} filled cells")
    elif any(c is not None for c in counts):
        notes.append(f"read {sum(c is not None for c in counts)} of {len(counts)} legend counts")
    for k, (want, got) in enumerate(zip(counts, used)):
        if want is not None and want != got:
            notes.append(f"legend entry {k + 1} (colour {tuple(int(v) for v in chart.swatches[k])}): "
                         f"legend says {want}, assigned {got}")
    for k in np.where(used == 0)[0]:
        notes.append(f"legend entry {k + 1} (colour {tuple(int(v) for v in chart.swatches[k])}) "
                     "matched no cells; it is probably too close to another legend colour to tell apart")
    return notes


def _match_clusters(C, sizes, beads, counts, rounds=6):
    """Pair printed colour clusters with legend beads. Returns {bead index: cluster index}.

    Chart apps print beads in their own shades, so the official colours are only a
    starting point: a colour transform from official to printed colours is fitted on
    the pairs found so far and the pairing is redone. Where the legend states counts,
    a cluster whose size is far from a bead's count is a poor match for it.
    """
    T = beads.copy()
    have = np.array([n is not None for n in counts])
    want = np.array([n or 0 for n in counts], float)
    pairs = {}
    for _ in range(rounds):
        cost = np.linalg.norm(C[:, None] - T[None], axis=2) / 40
        ratio = np.abs(np.log((sizes[:, None] + 1) / (want[None] + 1)))
        cost = cost + np.where(have[None], 2 * ratio, 0)
        new, free_c, free_b = {}, set(range(len(C))), set(range(len(beads)))
        for i, j in zip(*np.unravel_index(np.argsort(cost, axis=None), cost.shape)):
            if i in free_c and j in free_b:
                new[int(j)] = int(i)
                free_c.discard(i)
                free_b.discard(j)
            if not free_b or not free_c:
                break
        if new == pairs:
            break
        pairs = new
        js = list(pairs)
        if len(js) >= 6:
            w = np.maximum(sizes[[pairs[j] for j in js]], 1)
            T = apply_affine(beads, fit_affine(beads[js], C[[pairs[j] for j in js]], w))
    return pairs


def _paper(fills):
    """Blank paper: light and neutral (cells crossed by a thick grid line come out light grey)."""
    return (fills.min(2) > 200) & (np.ptp(fills, 2) < 14)


_GH, _GW = 6, 4  # a printed character, resampled: rows x (columns + 1 gap column)


def _printed_forms(code):
    """How a chart may print a code: 'F02' as 'F2' or 'F02'."""
    m = re.fullmatch(r"([A-Z]*)(\d+)", code)
    return [code] if not m else list(dict.fromkeys([f"{m[1]}{int(m[2])}", code]))


def _code_crop(blk, fill):
    """Tight crop of the printed code in a cell (0..1 ink), or None when no code is visible.

    Ink is measured relative to the cell's own fill, so dark codes on dark beads read as well
    as on light ones.
    """
    lum, fl = blk.mean(2), fill.mean()
    dark = np.clip(((fl - lum) / max(fl, 1) - 0.2) / 0.3, 0, 1)
    light = np.clip(((lum - fl) / max(255 - fl, 1) - 0.5) / 0.3, 0, 1)
    m = dark if dark.sum() >= light.sum() else light
    xs, ys = np.where(m.max(0) > 0.5)[0], np.where(m.max(1) > 0.5)[0]
    if len(xs) < 2 or len(ys) < 3:
        return None
    return m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _glyphs(crop, n):
    """The crop resampled to n characters wide; one (_GH, _GW - 1) patch per character."""
    im = Image.fromarray((crop * 255).astype(np.uint8)).resize((_GW * n - 1, _GH), Image.BILINEAR)
    z = np.asarray(im) / 255.0
    return np.stack([z[:, _GW * i:_GW * i + _GW - 1] for i in range(n)])


def _read_printed_codes(rgb, grid, fills, cells, lab, codes, beads, counts, rounds=3, margin=0.3, near=15):
    """Check each cell's label against the code printed in it.

    Codes are ~5px tall, too small for OCR, but every chart prints them in one font: the
    reference image of each character (e.g. the '3' of A3, H23 and F13) is the median of
    that character over all cells, taken per code first so that one wrongly named group
    cannot outvote the others. A cell then reads as the legend code whose characters
    (and length: F3 is two characters, F13 three) best match its print.

    A label group whose cells mostly read as another code is renamed: colour clustering
    can pair near-identical printed shades with the wrong codes. A single cell is then
    scored on its print and colour together, and moves only to a code whose printed colour
    is at most `near` further from the cell's colour than its current code's: where colours
    clearly differ the colour is trusted, since watermarks can hide part of the print, and
    where prints look alike at this size (2 and 3) the colour decides. Where the legend
    gives counts, no rename or move may take them further from the legend.
    Returns the new labels and the renamed groups as (old code, new code, cells).
    """
    n = len(codes)
    crops = {}
    inset = 1
    for i, (r, c) in enumerate(cells):
        l, t, rr, b = grid.cell_box(c, r)
        cr = _code_crop(rgb[t + inset:b - inset, l + inset:rr - inset].astype(float), fills[i])
        if cr is not None:
            crops[i] = cr
    if len(crops) < 20:
        return lab, []
    idx = np.array(sorted(crops))
    widths = np.array([crops[i].shape[1] for i in idx])

    def lengths(printed):
        # median printed width per code length, from the current labels; how far widths are from it
        ln = np.array([len(printed[lab[i]]) for i in idx])
        ks = np.unique(ln)
        med = np.array([np.median(widths[ln == k]) for k in ks])
        read = ks[np.abs(widths[:, None] - med[None]).argmin(1)]
        return read, np.abs(widths - med[np.searchsorted(ks, ln)]).mean()

    # the print style ('F2' or 'F02') whose code lengths explain the measured widths best
    printed, read_len, best = None, None, np.inf
    for f in ([_printed_forms(c)[0] for c in codes], [_printed_forms(c)[-1] for c in codes]):
        rl, err = lengths(f)
        if err < best:
            printed, read_len, best = f, rl, err
    glyphs = {i: _glyphs(crops[i], k) for i, k in zip(idx, read_len)}

    def count_error(labels):
        used = np.bincount(labels, minlength=n)
        return sum(abs(int(used[j]) - c) for j, c in enumerate(counts) if c is not None)

    cur = lab.copy()
    for _ in range(rounds):
        per = {}
        for i in idx:
            p = printed[cur[i]]
            if len(glyphs[i]) == len(p):
                for pos, ch in enumerate(p):
                    per.setdefault(ch, {}).setdefault(cur[i], []).append(glyphs[i][pos])
        ref = {ch: np.median([np.median(v, 0) for v in byc.values()], 0) for ch, byc in per.items()}
        # text cost of every cell against every code
        cost = np.full((len(idx), n), 5.0)
        for a, i in enumerate(idx):
            g = glyphs[i]
            for j, p in enumerate(printed):
                if len(p) == len(g):
                    cost[a, j] = np.mean([((g[pos] - ref[ch]) ** 2).mean() * 10 if ch in ref else 2
                                          for pos, ch in enumerate(p)])
        reading = cost.argmin(1)
        new = cur.copy()
        # whole groups first: colour clusters paired with the wrong code. Where the legend
        # gives counts, a rename must not take them further from the legend (a code printed
        # too faintly to read must not be merged away into a look-alike)
        moves = []
        for j in np.unique(cur):
            here = cur[idx] == j
            if here.sum() < 3:
                continue
            votes = np.bincount(reading[here], minlength=n)
            k = int(votes.argmax())
            if k != j and votes[k] >= 0.5 * here.sum():
                moves.append((votes[k] / here.sum(), j, k))
        pending = sorted(moves, reverse=True)
        while pending:
            for m in pending:
                _, j, k = m
                trial = np.where(new == j, k, new)
                if count_error(trial) <= count_error(new):
                    new = trial
                    pending.remove(m)
                    break
            else:
                break
        # then single cells, to codes their colour does not rule out; a code with
        # no cells yet gets the printed shade its official colour maps to on this chart
        # (from the cells that also read as the code, so stray cells such as header numbers
        # coloured like a bead do not pull its colour away)
        sure = np.zeros(len(lab), bool)
        sure[idx] = reading == new[idx]
        have = np.array([((new == j) & sure).sum() >= 3 for j in range(n)])
        model = np.array([np.median(fills[(new == j) & sure], 0) if have[j] else beads[j] for j in range(n)])
        if have.sum() >= 6 and not have.all():
            hj = np.where(have)[0]
            fit = fit_affine(beads[hj], model[hj], np.array([((new == j) & sure).sum() for j in hj], float))
            model[~have] = apply_affine(beads[~have], fit)
        cand = []
        for a, i in enumerate(idx):
            j = new[i]
            d = np.linalg.norm(model - fills[i], axis=1)
            both = np.where(d < d[j] + near, cost[a] + d / 20, np.inf)  # colour does not rule it out
            k = int(both.argmin())
            if k != j and both[j] - both[k] > margin:
                cand.append((both[j] - both[k], i, j, k))
        # clearest reads first; a move must not take both codes further from the legend counts
        used = np.bincount(new, minlength=n)
        for _, i, j, k in sorted(cand, reverse=True):
            if None not in (counts[j], counts[k]) and used[k] >= counts[k] and used[j] <= counts[j]:
                continue
            new[i] = k
            used[j] -= 1
            used[k] += 1
        if np.array_equal(new, cur):
            break
        cur = new
    renamed = []
    for j in np.unique(lab):
        here = lab == j
        k = int(np.bincount(cur[here], minlength=n).argmax())
        if here.sum() >= 3 and k != j and (cur[here] == k).mean() >= 0.5:
            renamed.append((codes[j], codes[k], int(here.sum())))
    return cur, renamed


def read_chart_codes(rgba, cells, codes, counts, beads, background=()):
    """Read a chart whose legend codes (and counts) are given; skips legend detection and OCR.

    beads are the official colours of codes; the returned Chart's colours are those, so
    the sprite uses true bead colours. background lists legend indices of beads that
    fill the chart's background (transparent).
    """
    rgb = rgba[..., :3]
    grid, fills, masks, dark, yellow = _read_cells(rgb, cells)
    n = len(codes)
    paper = _paper(fills)
    # white beads print as blank paper: the blank cells enclosed by the art are one more printed colour
    cc, sizes = components(~paper)
    main = cc == int(np.argmax(sizes)) + 1 if len(sizes) else ~paper
    holes = paper & ~outside_background(~main)
    use = ~paper | holes
    X = fills[use]
    # fine clusters: chart apps can print two beads only a few RGB steps apart
    C, lab = cluster_colours(fills[~paper], max_colours=min(128, 4 * n), merge_tol=6)
    if holes.any():
        C = np.vstack([C, np.median(fills[holes], 0)])
        lab = np.r_[lab, np.full(int(holes.sum()), len(C) - 1)]
        order = np.r_[np.where((~paper)[use])[0], np.where(holes[use])[0]]
        lab = lab[np.argsort(order)]
    # a watermark shifts a cell's colour but not its printed code: cells move to the cluster that
    # cells with the same-looking code agree on, so cluster sizes come close to the legend counts
    first = lab.copy()
    lab, changed = _repair_by_code_shape(lab, X, masks[use])
    sizes = np.bincount(lab, minlength=len(C)).astype(float)
    C = np.array([np.median(X[lab == k], 0) if sizes[k] else C[k] for k in range(len(C))])
    pairs = _match_clusters(C, sizes, beads, counts)
    notes = []
    missing = [codes[j] for j in range(n) if j not in pairs]
    if missing:
        notes.append(f"no printed colour found for {', '.join(missing)}")
    # clusters left over (stray shades) join the nearest paired cluster
    js = sorted(pairs)
    paired = np.array([pairs[j] for j in js])
    to_bead = np.array(js)[nearest(C, C[paired])]
    to_bead[paired] = js
    lab = to_bead[lab]
    # undo moves that took a cell from a bead short of its legend count to one over it
    used = np.bincount(lab, minlength=n)
    for j in changed:
        a, b = to_bead[first[j]], lab[j]
        if a != b and None not in (counts[a], counts[b]) and used[b] > counts[b] and used[a] < counts[a]:
            used[a] += 1
            used[b] -= 1
            lab[j] = a
    relabel_idx = {j for j in changed if to_bead[first[j]] != lab[j]}
    # once more on bead labels (several fine clusters can share a bead and split the votes above),
    # taking a change only where it brings the legend counts closer
    lab2, changed = _repair_by_code_shape(lab, X, masks[use])
    used = np.bincount(lab, minlength=n)
    for j in changed:
        a, b = lab[j], lab2[j]
        if counts[a] is None or counts[b] is None or (used[a] > counts[a] and used[b] < counts[b]):
            used[a] -= 1
            used[b] += 1
            lab[j] = b
            relabel_idx.add(j)
    rc = np.argwhere(use)
    # the printed codes have the last word over colour where they are readable
    read, renamed = _read_printed_codes(rgb, grid, X, rc, lab, codes, beads, counts)
    for a, b, k in renamed:
        notes.append(f"printed codes: the {k} cells labelled {a} by colour read as {b}; renamed")
    by_code = np.where(read != lab)[0]
    lone = [j for j in by_code if not any(codes[lab[j]] == a for a, _, _ in renamed)]
    if lone:
        notes.append(f"printed codes: {len(lone)} more cells relabelled by their print and colour")
    relabel_idx = set(relabel_idx) | set(lone)
    lab = read
    relabel_idx = sorted(j for j in relabel_idx if to_bead[first[j]] != lab[j])
    if holes.any():
        hl = lab[holes[use]]
        names = ", ".join(f"{codes[k]} x{int((hl == k).sum())}" for k in np.unique(hl))
        notes.append(f"{int(holes.sum())} blank cells enclosed by the art read as white beads ({names})")

    labels = np.full(paper.shape, -1)
    labels[use] = lab
    relabelled = [(int(rc[j][0]), int(rc[j][1]), -1, int(lab[j])) for j in relabel_idx]
    bg = np.isin(labels, list(background))
    if bg.any():
        notes.append(f"{int(bg.sum())} cells of background bead(s) {', '.join(codes[k] for k in background)} "
                     "made transparent")
    labels[bg] = -1

    chart = Chart(grid, labels, beads.copy(), beads.copy(), list(counts), relabelled, notes, fills,
                  labels.copy(), False, None)
    chart.codes = list(codes)
    chart.background = list(background)
    return chart
