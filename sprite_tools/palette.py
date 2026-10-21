"""Colour snapping: clustering, legend matching and recolouring from a second image."""

import numpy as np


def nearest(X, P):
    """Index of the nearest palette colour for each row of X."""
    d = (X ** 2).sum(1)[:, None] + (P ** 2).sum(1)[None, :] - 2 * X @ P.T
    return d.argmin(1)


def kmeans(X, centers, iters=30):
    C = np.array(centers, float)
    for _ in range(iters):
        lab = nearest(X, C)
        new = np.array([np.median(X[lab == k], 0) if (lab == k).any() else C[k] for k in range(len(C))])
        if np.allclose(new, C):
            break
        C = new
    return C, nearest(X, C)


def _kmeanspp(X, k, seed=0):
    rng = np.random.default_rng(seed)
    C = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d = np.min(((X[:, None] - np.array(C)[None]) ** 2).sum(2), 1)
        C.append(X[rng.choice(len(X), p=d / d.sum())] if d.sum() > 0 else X[rng.integers(len(X))])
    return np.array(C)


def cluster_colours(X, max_colours=64, merge_tol=14):
    """Reduce noisy samples to a small palette.

    Starts with up to max_colours k-means clusters, then merges clusters whose
    centres are closer than merge_tol (Euclidean RGB), so noise from blur and
    compression collapses but genuinely different shades stay apart.
    """
    uniq = np.unique(X.round(), axis=0)
    k = min(max_colours, len(uniq))
    C, lab = kmeans(X, _kmeanspp(X, k))
    while True:
        d = np.linalg.norm(C[:, None] - C[None], axis=2) + np.eye(len(C)) * 1e9
        i, j = np.unravel_index(d.argmin(), d.shape)
        if d[i, j] >= merge_tol or len(C) == 1:
            break
        lab[lab == j] = i
        keep = [k for k in range(len(C)) if k != j]
        lab = np.searchsorted(keep, lab)
        C = np.array([np.median(X[lab == k], 0) for k in range(len(keep))])
    return C, lab


def fit_affine(src, dst, weights=None):
    """3x4 colour transform (as a 4x3 matrix) mapping src colours to dst colours."""
    A = np.hstack([src, np.ones((len(src), 1))])
    w = np.ones(len(src)) if weights is None else np.asarray(weights, float)
    sw = np.sqrt(w)[:, None]
    M, *_ = np.linalg.lstsq(A * sw, dst * sw, rcond=None)
    return M


def apply_affine(P, M):
    return np.clip(np.hstack([P, np.ones((len(P), 1))]) @ M, 0, 255)


def match_to_legend(X, swatches, iters=8, min_cells=3):
    """Label colours X with legend entries.

    Bead-chart legends are usually printed lighter or duller than the chart
    cells, so a colour transform from legend to cells is fitted on the entries
    that clearly have cells and the labelling is redone with the transformed
    legend. Returns (labels, legend colours as they appear in the cells).
    """
    S = np.asarray(swatches, float)
    T = S.copy()
    for _ in range(iters):
        lab = nearest(X, T)
        counts = np.bincount(lab, minlength=len(S))
        use = counts >= min_cells
        if use.sum() < 6:
            break
        med = np.array([np.median(X[lab == k], 0) for k in np.where(use)[0]])
        T_new = apply_affine(S, fit_affine(S[use], med, counts[use]))
        if np.allclose(T_new, T, atol=0.5):
            break
        T = T_new
    return nearest(X, T), T


def group_medians(labels, samples, n):
    """Median of samples per label; NaN for labels that never occur."""
    out = np.full((n, 3), np.nan)
    for k in range(n):
        if (labels == k).any():
            out[k] = np.median(samples[labels == k], 0)
    return out
