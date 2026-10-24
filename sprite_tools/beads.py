"""Bead colour codes: official colours of the MARD, COCO, 漫漫, 盼盼 and 咪小窝 bead ranges.

data/bead_colours.json is colorSystemMapping.json from https://github.com/Zippland/perler-beads
(AGPL-3.0): 291 bead colours by hex, each with its code in the five ranges.
"""

import json
import re
from pathlib import Path

import numpy as np

BRANDS = ("MARD", "COCO", "漫漫", "盼盼", "咪小窝")
_DATA = json.loads((Path(__file__).parent / "data" / "bead_colours.json").read_text(encoding="utf-8"))


def normalise(code):
    """Codes as charts print them in any style: 'a7', 'A7' and 'A07' are all 'A07'."""
    code = code.strip().upper()
    m = re.fullmatch(r"([A-Z]*)0*(\d+)", code)
    return f"{m[1]}{int(m[2]):02d}" if m and m[1] else (str(int(m[2])) if m else code)


_TABLE = {b: {normalise(v[b]): k for k, v in _DATA.items() if v.get(b)} for b in BRANDS}


def colour(code, brand="MARD"):
    """Official (r, g, b) of a bead code."""
    if brand not in _TABLE:
        raise ValueError(f"unknown bead brand {brand!r}; choose from {', '.join(BRANDS)}")
    hx = _TABLE[brand].get(normalise(code))
    if hx is None:
        raise ValueError(f"{brand} has no bead code {code!r}")
    return tuple(int(hx[i:i + 2], 16) for i in (1, 3, 5))


def parse_legend(text):
    """'A06:132 A10:204 B04' (comma or space separated, counts optional) -> (codes, counts)."""
    codes, counts = [], []
    for item in re.split(r"[\s,]+", text.strip()):
        if not item:
            continue
        code, _, n = item.partition(":")
        codes.append(normalise(code))
        counts.append(int(n) if n else None)
    if len(set(codes)) != len(codes):
        raise ValueError("a bead code appears twice in --legend")
    return codes, counts


def legend_colours(codes, brand="MARD"):
    return np.array([colour(c, brand) for c in codes], float)
