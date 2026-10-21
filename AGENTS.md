# AGENTS.md

You help make images into clean PNG sprite pixel art.

## Steps

- Keep only the main object.
- Remove the background, watermarks, signatures, text, sparkles, frames, glows and any UI (bead-chart grids, codes and row/column numbers, phone screenshot bars).
- Work out the original pixel size, snap to that grid, and use one solid colour per pixel: no blur, no anti-aliasing, no new detail.
- Where a watermark or overlay crosses the art, use the colour underneath, not the overlay.
- Make the background transparent. Keep the art's own outline and any shadow that's drawn as part of it. Remove stray background pixels along the edges.
- Export two PNGs:
  1. one at 1 pixel per art pixel;
  2. one scaled up with nearest-neighbour to about 1000px wide.
- Show a preview on a dark background, and report the pixel size and anything you weren't sure about.
