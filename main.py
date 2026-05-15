"""
main.py — Docsmate Image Enhancer API v6

No file size limits.
No pre-scaling (quality always preserved).
Timeout scales with image area so large images are never cut off.
"""

import os
import cv2
import time
import asyncio
import numpy as np
from pathlib import Path
from loguru import logger
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from enhancer.pipeline import EnhancementPipeline
from enhancer.utils import image_stats

load_dotenv()

app = FastAPI(
    title="Docsmate Image Enhancer",
    version="6.0.0",
    description="Enhance any image or scanned PDF at full quality — no size limits.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
    expose_headers=[
        "X-Enhanced-Size", "X-Image-Type", "X-Has-Face",
        "X-Page-Count", "X-Sharpness-Before", "X-Sharpness-After",
        "X-Processing-Time",
    ],
)

# ── Pipeline singleton ────────────────────────────────────────────────────────
SCALE       = int(os.getenv("ENHANCE_SCALE", "2"))
ENABLE_FACE = os.getenv("ENABLE_FACE", "true").lower() == "true"

pipeline = EnhancementPipeline(scale=SCALE, enable_face=ENABLE_FACE)
logger.info(f"Server ready — scale={SCALE}× face={ENABLE_FACE}")

# ── Format helpers ─────────────────────────────────────────────────────────────
IMAGE_MIMES = {
    "image/jpeg": (".jpg",  cv2.IMWRITE_JPEG_QUALITY),
    "image/jpg":  (".jpg",  cv2.IMWRITE_JPEG_QUALITY),
    "image/png":  (".png",  None),
    "image/webp": (".webp", cv2.IMWRITE_WEBP_QUALITY),
}
ALL_MIMES = {**IMAGE_MIMES, "application/pdf": (".pdf", None)}

EXT_MIME = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png",  ".webp": "image/webp",
    ".pdf": "application/pdf",
}


def _resolve_mime(content_type: str, filename: str) -> str:
    """Infer MIME from content-type header, falling back to file extension."""
    mime = (content_type or "").lower().split(";")[0].strip()
    if mime not in ALL_MIMES and filename:
        mime = EXT_MIME.get(Path(filename).suffix.lower(), mime)
    return mime


def _encode_image(img: np.ndarray, mime: str, quality: int) -> bytes:
    ext, flag = IMAGE_MIMES[mime]
    params    = [flag, quality] if flag else []
    ok, buf   = cv2.imencode(ext, img, params)
    if not ok:
        raise RuntimeError(f"Failed to encode as {ext}")
    return buf.tobytes()


def _timeout_for_image(img: np.ndarray) -> int:
    """
    Compute a generous timeout based on image area.
    Each 256×256 tile takes ~8s on CPU; 4 workers run in parallel.
    We add 60s buffer on top for model overhead.
    """
    from enhancer.upscaler import TILE_SIZE, MAX_WORKERS
    h, w      = img.shape[:2]
    n_tiles   = ((h + TILE_SIZE - 1) // TILE_SIZE) * ((w + TILE_SIZE - 1) // TILE_SIZE)
    batches   = (n_tiles + MAX_WORKERS - 1) // MAX_WORKERS
    estimated = batches * 10   # 10s per batch (conservative)
    timeout   = estimated + 60  # buffer
    logger.info(f"[Timeout] {n_tiles} tiles → {batches} batches → timeout={timeout}s")
    return timeout


async def _in_thread(fn):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, fn)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "scale": SCALE, "face": ENABLE_FACE}


@app.post("/enhance")
async def enhance(
    file: UploadFile = File(...),
    quality: int = Query(default=95, ge=60, le=100),
):
    mime = _resolve_mime(file.content_type or "", file.filename or "")

    if mime not in ALL_MIMES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported type '{mime}'. Accepted: JPEG, PNG, WEBP, PDF.",
        )

    contents = await file.read()
    size_mb   = len(contents) / (1024 * 1024)
    logger.info(f"Upload: {file.filename!r}  mime={mime}  size={size_mb:.1f} MB")

    t_start = time.time()

    # ── PDF ───────────────────────────────────────────────────────────────────
    if mime == "application/pdf":
        # Scale timeout by page count
        try:
            from enhancer.pdf_handler import pdf_page_count
            pages_hint = max(1, pdf_page_count(contents))
        except Exception:
            pages_hint = 1
        pdf_timeout = pages_hint * 120   # 2 min per page (conservative)

        try:
            pdf_out, page_count, profile = await asyncio.wait_for(
                _in_thread(lambda: pipeline.run_pdf(contents)),
                timeout=pdf_timeout,
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail=f"PDF timed out ({pdf_timeout}s). Try splitting into fewer pages.",
            )
        except Exception as exc:
            logger.exception("PDF pipeline error")
            raise HTTPException(status_code=500, detail=f"PDF failed: {exc}")

        elapsed = round(time.time() - t_start, 1)
        return Response(
            content=pdf_out,
            media_type="application/pdf",
            headers={
                "X-Image-Type":      profile.image_type if profile else "document",
                "X-Has-Face":        str(profile.has_face).lower() if profile else "false",
                "X-Page-Count":      str(page_count),
                "X-Processing-Time": f"{elapsed}s",
                "Content-Disposition": f'attachment; filename="enhanced_{file.filename}"',
            },
        )

    # ── Image ─────────────────────────────────────────────────────────────────
    arr = np.frombuffer(contents, np.uint8)
    raw = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if raw is None:
        raise HTTPException(status_code=422, detail="Cannot decode image — file may be corrupt.")

    before  = image_stats(raw)
    timeout = _timeout_for_image(raw)

    try:
        enhanced, profile = await asyncio.wait_for(
            _in_thread(lambda: pipeline.run(contents, quality=quality)),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Enhancement timed out ({timeout}s). Server is under heavy load — try again.",
        )
    except Exception as exc:
        logger.exception("Image pipeline error")
        raise HTTPException(status_code=500, detail=f"Enhancement failed: {exc}")

    try:
        output_bytes = _encode_image(enhanced, mime, quality)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Encoding failed: {exc}")

    after   = image_stats(enhanced)
    elapsed = round(time.time() - t_start, 1)

    return Response(
        content=output_bytes,
        media_type=mime,
        headers={
            "X-Enhanced-Size":    f"{enhanced.shape[1]}x{enhanced.shape[0]}",
            "X-Image-Type":       profile.image_type,
            "X-Has-Face":         str(profile.has_face).lower(),
            "X-Sharpness-Before": str(before.get("sharpness_score", "?")),
            "X-Sharpness-After":  str(after.get("sharpness_score", "?")),
            "X-Processing-Time":  f"{elapsed}s",
            "Content-Disposition": f'attachment; filename="enhanced_{file.filename}"',
        },
    )