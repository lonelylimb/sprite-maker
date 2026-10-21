"""Optional OCR for bead-chart legend counts (needs the tesseract binary and pytesseract).

Only the legend counts are read this way: they are large black digits on white.
The codes printed inside chart cells are ~5px tall, which tesseract cannot read;
beadchart compares their shapes between cells instead.
"""

import re

import numpy as np
from PIL import Image, ImageOps

try:
    import pytesseract
    pytesseract.get_tesseract_version()
    AVAILABLE = True
except Exception:  # missing module or missing binary
    AVAILABLE = False


def _read(img):
    img = ImageOps.expand(img.resize((img.width * 4, img.height * 4), Image.LANCZOS), 20, 255)
    txt = pytesseract.image_to_string(img, config="--psm 7").strip()
    nums = re.findall(r"\d+", txt.split()[-1]) if txt else []
    return int(nums[-1]) if nums else None


def _crop(rgb, box):
    """Grey, contrast-stretched crop with dark text on light, or None if too small or blank."""
    l, t, r, b = box
    blk = rgb[t + 2:b - 2, l + 2:r - 2]
    if blk.shape[0] < 4 or blk.shape[1] < 4 or np.ptp(blk) < 60:
        return None
    img = ImageOps.autocontrast(Image.fromarray(blk.astype(np.uint8)).convert("L"))
    return ImageOps.invert(img) if np.asarray(img).mean() < 128 else img


def read_count(rgb, box, after):
    """Bead count of a legend entry: in the white box after the swatch, else at the end of the swatch."""
    if not AVAILABLE:
        return None
    for region in (after, box):
        img = _crop(rgb, region)
        if img is not None:
            n = _read(img)
            if n is not None:
                return n
    return None
