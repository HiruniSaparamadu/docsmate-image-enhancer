"""
enhancer/pdf_handler.py  —  Streaming PDF handler, no size limit

Strategy for fixed-time PDF processing:
  - Each page is rasterised at a resolution chosen so the longest edge
    is at most MAX_PAGE_PX pixels.  This caps per-page processing time.
  - Pages are streamed one at a time — no full document is ever held in RAM.
  - MAX_PAGE_PX = 1600 gives excellent print quality at A4/Letter while
    keeping tile count ≤ 40 per page (≈ 15–25 s per page on CPU).

For a 1-page passport scan this means < 20 s total.
For a 10-page birth certificate: pages are processed sequentially;
progress is logged so the user can see activity.
"""

import io
import cv2
import numpy as np
from loguru import logger
from typing import Iterator, List
from PIL import Image

MAX_PAGE_PX   = 1600   # longest edge cap for rasterisation
BASE_DPI      = 150    # starting DPI — adjusted per-page to hit MAX_PAGE_PX


def _dpi_for_page(page_w_pts: float, page_h_pts: float) -> float:
    """
    Choose DPI so the rasterised image's longest edge ≤ MAX_PAGE_PX.
    1 pt = 1/72 inch, so px = pts * dpi / 72.
    """
    longest_pts = max(page_w_pts, page_h_pts)
    dpi = MAX_PAGE_PX * 72 / longest_pts
    return min(dpi, 300)   # never exceed 300 DPI


def pdf_page_count(pdf_bytes: bytes) -> int:
    """Return number of pages without fully loading the document."""
    try:
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        n   = doc.page_count
        doc.close()
        return n
    except Exception:
        return 0


def iter_pdf_pages(pdf_bytes: bytes) -> Iterator[np.ndarray]:
    """
    Generator — yields one BGR numpy array per page.
    Never holds more than one rendered page in memory at once.
    """
    try:
        import fitz
    except ImportError:
        raise RuntimeError("PyMuPDF required: pip install PyMuPDF")

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise ValueError(f"Cannot open PDF: {e}")

    logger.info(f"[PDF] {doc.page_count} page(s)")

    for i, page in enumerate(doc):
        rect = page.rect
        dpi  = _dpi_for_page(rect.width, rect.height)
        scale = dpi / 72
        mat   = fitz.Matrix(scale, scale)
        pix   = page.get_pixmap(matrix=mat, alpha=False)

        arr = np.frombuffer(pix.samples, dtype=np.uint8)
        arr = arr.reshape(pix.height, pix.width, 3)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        logger.info(f"[PDF] page {i+1}/{doc.page_count}: {bgr.shape[1]}×{bgr.shape[0]} px  ({dpi:.0f} DPI)")
        yield bgr
        del pix, arr, bgr   # free memory before next page

    doc.close()


def images_to_pdf(images: List[np.ndarray]) -> bytes:
    """
    Reassemble enhanced BGR pages back into a PDF.
    Images are assumed to be at MAX_PAGE_PX resolution (≈ 150–300 DPI equivalent).
    """
    try:
        import fitz
    except ImportError:
        raise RuntimeError("PyMuPDF required: pip install PyMuPDF")

    doc = fitz.open()

    for i, img in enumerate(images):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        buf = io.BytesIO()
        pil.save(buf, format="PNG", optimize=False)
        img_bytes = buf.getvalue()

        h, w     = img.shape[:2]
        # Store at 150 DPI so output PDF is a reasonable file size
        pts_w    = w * 72 / 150
        pts_h    = h * 72 / 150
        page     = doc.new_page(width=pts_w, height=pts_h)
        page.insert_image(fitz.Rect(0, 0, pts_w, pts_h), stream=img_bytes)
        logger.debug(f"[PDF] assembled page {i+1}")

    pdf_bytes = doc.tobytes(deflate=True)
    doc.close()
    logger.info(f"[PDF] output: {len(pdf_bytes)//1024} KB")
    return pdf_bytes