"""Optional OCR for bead-chart legend counts (needs the tesseract binary and pytesseract).

Only the legend counts are read this way: they are large black digits on white.
The codes printed inside chart cells are ~5px tall, which tesseract cannot read;
beadchart compares their shapes between cells instead.
"""

import numpy as np
from PIL import Image, ImageOps

try:
    import pytesseract
    pytesseract.get_tesseract_version()
    AVAILABLE = True
except Exception:  # missing module or missing binary
    AVAILABLE = False


def read_number(rgb, box):
    """Integer printed in box (left, top, right, bottom), or None."""
    if not AVAILABLE:
        return None
    l, t, r, b = box
    crop = Image.fromarray(rgb[t:b, l:r].astype(np.uint8)).convert("L")
    if crop.width < 4 or crop.height < 4:
        return None
    crop = ImageOps.autocontrast(crop.resize((crop.width * 4, crop.height * 4), Image.LANCZOS))
    crop = ImageOps.expand(crop, 20, 255)
    txt = pytesseract.image_to_string(crop, config="--psm 7 -c tessedit_char_whitelist=0123456789").strip()
    return int(txt) if txt.isdigit() else None
