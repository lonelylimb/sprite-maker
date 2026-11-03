"""python -m sprite_tools.fix SPRITE_DIR EDIT [EDIT ...]

Apply hand fixes to a bead-chart sprite made by python -m sprite_tools, then
re-count against the legend, re-trim and re-export the PNGs and the preview.

Edits use the chart's own row and column numbers:
  r31c34=H19     cell at row 31, column 34 becomes bead H19
  r43c19=-       cell becomes empty (transparent)
  r3c33=B11      a cell outside the sprite can be added too
  E15<>F22       swap two codes everywhere (a whole colour group read as the other)

The fixes are recorded in report.json, so they can be checked later.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

from . import beads as beadsmod, export

_CELL = re.compile(r"r(\d+)c(\d+)=(.+)", re.I)
_SWAP = re.compile(r"([^<>=\s]+)<>([^<>=\s]+)")


def load(report):
    """{(row, col): code} in chart numbering, from report.json's cells."""
    c = report.get("cells")
    if not c or "beads" not in report:
        raise SystemExit("report.json has no cell codes; fixes work on bead charts read with --legend "
                         "(rerun python -m sprite_tools on the chart first)")
    grid = {}
    for i, row in enumerate(c["rows"]):
        for j, code in enumerate(row.split()):
            if code != ".":
                grid[(c["first_row"] + i, c["first_col"] + j)] = code
    return grid


def apply(grid, edits, beads, brand="MARD"):
    """Apply edits to grid in place; add unknown codes to beads. Returns a description per edit."""
    done = []
    for e in edits:
        m, s = _CELL.fullmatch(e.strip()), _SWAP.fullmatch(e.strip())
        if m:
            r, c, code = int(m[1]), int(m[2]), m[3].strip()
            old = grid.get((r, c), "-")
            if code == "-":
                grid.pop((r, c), None)
            else:
                code = _known(code, beads, brand)
                grid[(r, c)] = code
            done.append(f"r{r}c{c} {old}->{code}")
        elif s:
            a, b = (_known(x, beads, brand) for x in (s[1], s[2]))
            n = 0
            for k, v in grid.items():
                if v in (a, b):
                    grid[k] = b if v == a else a
                    n += 1
            done.append(f"swapped {a} and {b} ({n} cells)")
        else:
            raise SystemExit(f"cannot read edit {e!r}; use r<row>c<col>=<code>, r<row>c<col>=- or <code><><code>")
    return done


def _known(code, beads, brand):
    code = beadsmod.normalise(code)
    if code not in beads:
        try:
            rgb = beadsmod.colour(code, brand)
        except ValueError as err:
            raise SystemExit(str(err))
        beads[code] = {"count": 0, "legend": None, "rgb": list(rgb)}
    return code


def render(grid, beads):
    """(rgba sprite trimmed to the cells, first row, first col)."""
    rs = [r for r, _ in grid]
    cs = [c for _, c in grid]
    r0, c0 = min(rs), min(cs)
    out = np.zeros((max(rs) - r0 + 1, max(cs) - c0 + 1, 4), np.uint8)
    for (r, c), code in grid.items():
        out[r - r0, c - c0] = (*beads[code]["rgb"], 255)
    return out, r0, c0


def count_notes(grid, beads):
    used = {}
    for code in grid.values():
        used[code] = used.get(code, 0) + 1
    for code, b in beads.items():
        b["count"] = used.get(code, 0)
    off = [f"{c} {b['count']} (legend {b['legend']})" for c, b in beads.items()
           if b["legend"] is not None and b["count"] != b["legend"]]
    total = sum(b["legend"] for b in beads.values() if b["legend"] is not None)
    notes = [f"{len(grid)} beads in the sprite; legend counts total {total}"]
    notes.append("bead counts that differ from the legend: " + ", ".join(off) if off
                 else "every bead count matches the legend")
    return notes


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m sprite_tools.fix", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sprite_dir", help="folder with report.json, e.g. sprites/pink_egg")
    ap.add_argument("edits", nargs="+")
    ap.add_argument("--brand", default="MARD", choices=beadsmod.BRANDS, help="bead range of codes not in the legend")
    ap.add_argument("--width", type=int, default=1000, help="target width of the upscaled PNG")
    a = ap.parse_args(argv)

    d = Path(a.sprite_dir)
    path = d / "report.json"
    if not path.exists():
        raise SystemExit(f"no report.json in {d}")
    report = json.loads(path.read_text())
    grid = load(report)
    beads = report["beads"]
    done = apply(grid, a.edits, beads, a.brand)
    if not grid:
        raise SystemExit("no cells left")
    out, r0, c0 = render(grid, beads)

    old = report.get("files", {})
    name = Path(old.get("sprite", f"{d.name}_x.png")).name.rsplit("_", 1)[0]
    for k in ("sprite", "upscaled"):
        if old.get(k) and Path(old[k]).exists():
            Path(old[k]).unlink()
    files = export.save_outputs(out, d, name, a.width)
    notes = count_notes(grid, beads)
    h, w = out.shape[:2]
    report["cells"] = {"first_row": r0, "first_col": c0, "rows": export.cell_codes(out, beads)}
    report["fixes"] = report.get("fixes", []) + done
    report["notes"].append(f"hand fixes: {'; '.join(done)}")
    report["notes"] += notes
    report.update({"pixel_size": [w, h], "colours": int(len(np.unique(out[out[..., 3] > 0][:, :3], axis=0))),
                   "files": files})
    path.write_text(json.dumps(report, indent=2, default=float))

    print(f"{name}: {w}x{h} pixels after {len(done)} fixes")
    for k in ("sprite", "upscaled", "preview"):
        print(f"  {files[k]}")
    for n in notes:
        print(f"  - {n}")


if __name__ == "__main__":
    main()
