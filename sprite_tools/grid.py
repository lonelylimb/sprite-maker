"""Find the art-pixel grid in an upscaled image and sample one colour per cell."""

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class Grid:
    x0: float  # left edge of the first cell, in image pixels
    px: float  # cell width
    nx: int
    y0: float
    py: float
    ny: int

    def cell_box(self, c, r):
        """Integer (left, top, right, bottom) of cell (c, r)."""
        return (int(round(self.x0 + self.px * c)), int(round(self.y0 + self.py * r)),
                int(round(self.x0 + self.px * (c + 1))), int(round(self.y0 + self.py * (r + 1))))


def load_rgba(path):
    """Image as a float (H, W, 4) array."""
    return np.asarray(Image.open(path).convert("RGBA")).astype(float)


def background_colour(rgb):
    """Median colour of the 2px image border."""
    border = np.concatenate([rgb[:2].reshape(-1, 3), rgb[-2:].reshape(-1, 3),
                             rgb[:, :2].reshape(-1, 3), rgb[:, -2:].reshape(-1, 3)])
    return np.median(border, 0)


def foreground_mask(rgba, bg, tol=40):
    """Pixels that are opaque and differ from the background by more than tol (sum of |dRGB|)."""
    return (rgba[..., 3] >= 128) & (np.abs(rgba[..., :3] - bg).sum(2) > tol)


def bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("nothing found that differs from the background")
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def edge_profiles(rgb, box):
    """Colour-change strength between neighbouring columns / rows inside box.

    gx[i] is the edge between columns i and i+1 (whole-image coordinates).
    """
    x0, y0, x1, y1 = box
    gx = np.abs(np.diff(rgb, axis=1)).sum(2)[y0:y1].sum(0)
    gy = np.abs(np.diff(rgb, axis=0)).sum(2)[:, x0:x1].sum(1)
    return gx, gy


def _boundary_score(profile, offsets, pitch, n):
    """Mean profile value on the n+1 cell boundaries for each offset."""
    b = offsets[:, None] + pitch * np.arange(n + 1)[None, :] - 0.5
    return np.interp(b, np.arange(len(profile)), profile, left=0, right=0).mean(1)


def refine_axis(profile, start, n, pitch, spread=0.03):
    """Best (offset, pitch) for n cells whose first boundary sits near start."""
    best = (-1, start, pitch)
    for p in np.arange(pitch * (1 - spread), pitch * (1 + spread), 0.002):
        offs = np.arange(start - p / 2, start + p / 2, 0.1)
        s = _boundary_score(profile, offs, p, n)
        i = s.argmax()
        if s[i] > best[0]:
            best = (s[i], offs[i], p)
    return best[1], best[2]


def estimate_pitch(profile, start, end, min_pitch=2.0):
    """Cell size along one axis, from the periodicity of the edge profile.

    The strongest FFT peak may be a harmonic of the real pitch, so multiples and
    fractions of it are scored and the smallest pitch that explains the edges
    about as well as the best one wins (a multiple of the true pitch scores the
    same, a fraction scores much lower).
    """
    seg = profile[start:end] - profile[start:end].mean()
    f = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(len(seg))
    ok = freqs > 0
    ok &= 1 / np.where(freqs > 0, freqs, 1) >= min_pitch
    peak = 1 / freqs[ok][f[ok].argmax()]
    cands = sorted({round(peak * k / d, 3) for k in (1, 2, 3, 4) for d in (1, 2, 3, 4)
                    if peak * k / d >= min_pitch and peak * k / d < (end - start) / 4})
    scored = []
    for p in cands:
        offs = np.arange(start - p / 2, start + p / 2, 0.1)
        n = max(1, int(round((end - start) / p)))
        scored.append((_boundary_score(profile, offs, p, n).max(), p))
    top = max(s for s, _ in scored)
    return min(p for s, p in scored if s >= 0.85 * top)


def fit_grid(rgb, box, nx=None, ny=None):
    """Grid covering box. nx/ny fix the cell counts when they are known."""
    x0, y0, x1, y1 = box
    gx, gy = edge_profiles(rgb, box)
    out = []
    for prof, a, b, n in ((gx, x0, x1, nx), (gy, y0, y1, ny)):
        p = (b - a) / n if n else estimate_pitch(prof, a, b)
        n = n or max(1, int(round((b - a) / p)))
        o, p = refine_axis(prof, a, n, (b - a) / n)
        out.append((o, p, n))
    (ox, px, nx), (oy, py, ny) = out
    return Grid(ox, px, nx, oy, py, ny)


def sample_cells(rgba, grid, core=0.4):
    """Median colour and alpha of the central core of every cell -> (ny, nx, 3), (ny, nx)."""
    h, w = rgba.shape[:2]
    cols = np.zeros((grid.ny, grid.nx, 3))
    alpha = np.zeros((grid.ny, grid.nx))
    for r in range(grid.ny):
        cy = grid.y0 + grid.py * (r + 0.5)
        hy = max(1.0, grid.py * core / 2)
        ys = slice(max(0, int(round(cy - hy))), min(h, int(round(cy + hy)) + 1))
        for c in range(grid.nx):
            cx = grid.x0 + grid.px * (c + 0.5)
            hx = max(1.0, grid.px * core / 2)
            xs = slice(max(0, int(round(cx - hx))), min(w, int(round(cx + hx)) + 1))
            blk = rgba[ys, xs].reshape(-1, 4)
            cols[r, c] = np.median(blk[:, :3], 0)
            alpha[r, c] = np.median(blk[:, 3])
    return cols, alpha
