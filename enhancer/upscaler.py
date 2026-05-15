"""
enhancer/upscaler.py

Tiled upscaler — processes the image at FULL native resolution.
Speed is controlled purely by tile size and worker count, NOT by
downscaling the input. This means:

  - A 400px image  → upscaled to 800px  (clean, sharp)
  - A 4000px image → upscaled to 8000px (clean, sharp, same tile time)
  - A re-uploaded enhanced image → upscaled again without quality loss

Tile strategy:
  TILE_SIZE = 256px input tiles (→ 512px output per tile at scale=2)
  OVERLAP   = 32px each side    → seamless stitching
  WORKERS   = 4 threads         → 4 tiles processed simultaneously

Wall-clock time per request ≈ ceil(tiles / WORKERS) × time_per_tile
  Example 800×600 image at scale=2: 12 tiles / 4 workers = 3 batches ≈ 30s
  Example 4000×3000 image:          300 tiles / 4 workers = 75 batches ≈ 75s
  (acceptable — quality is never sacrificed for speed)
"""

import cv2
import numpy as np
import concurrent.futures
from pathlib import Path
from loguru import logger
from .detector import ImageProfile
from .utils import bgr_to_pil, safe_clip

TILE_SIZE    = 256
OVERLAP      = 32
MAX_WORKERS  = 4
ESRGAN_MODEL = Path("models/RealESRGAN_x4plus.pth")

# ── Model singletons — loaded once, reused every request ─────────────────────
_edsr_cache   = {}
_loader_ref   = {}
_esrgan_cache = {}


def _get_edsr(scale: int):
    if scale not in _edsr_cache:
        try:
            from super_image import EdsrModel, ImageLoader
            mid = "eugenesiow/edsr-base" if scale == 2 else "eugenesiow/edsr"
            logger.info(f"[Upscaler] loading EDSR scale={scale}")
            _edsr_cache[scale] = EdsrModel.from_pretrained(mid, scale=scale)
            _loader_ref["v"]   = ImageLoader
            logger.info("[Upscaler] EDSR ready")
        except Exception as e:
            logger.warning(f"[Upscaler] EDSR unavailable: {e}")
            _edsr_cache[scale] = None
    return _edsr_cache[scale], _loader_ref.get("v")


def _get_esrgan(scale: int):
    key = f"x{scale}"
    if key not in _esrgan_cache:
        try:
            import torch
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            if not ESRGAN_MODEL.exists():
                raise FileNotFoundError(f"{ESRGAN_MODEL} not found")
            net    = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                             num_block=23, num_grow_ch=32, scale=4)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _esrgan_cache[key] = RealESRGANer(
                scale=4, model_path=str(ESRGAN_MODEL), model=net,
                tile=0, tile_pad=0, pre_pad=0,
                half=(device == "cuda"), device=device,
            )
            logger.info(f"[Upscaler] ESRGAN ready on {device}")
        except Exception as e:
            logger.warning(f"[Upscaler] ESRGAN unavailable: {e}")
            _esrgan_cache[key] = None
    return _esrgan_cache[key]


# ── Per-tile inference ────────────────────────────────────────────────────────

def _run_edsr_tile(tile: np.ndarray, scale: int) -> np.ndarray:
    import torch
    model, loader = _get_edsr(scale)
    if model is None:
        return _lanczos(tile, scale)
    pil = bgr_to_pil(tile)
    inp = loader.load_image(pil)
    with torch.no_grad():
        pred = model(inp)
    arr = (pred.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _run_esrgan_tile(tile: np.ndarray, scale: int) -> np.ndarray:
    model = _get_esrgan(scale)
    if model is None:
        return _run_edsr_tile(tile, scale)
    try:
        out, _ = model.enhance(tile, outscale=4)
        if scale == 2:
            h, w = out.shape[:2]
            out  = cv2.resize(out, (w // 2, h // 2), interpolation=cv2.INTER_LANCZOS4)
        return out
    except Exception as e:
        logger.warning(f"[Upscaler] ESRGAN tile error: {e}")
        return _run_edsr_tile(tile, scale)


def _lanczos(tile: np.ndarray, scale: int) -> np.ndarray:
    h, w = tile.shape[:2]
    return cv2.resize(tile, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)


# ── Tiled upscale (full resolution, no pre-shrink) ───────────────────────────

def _worker(args):
    row, col, tile, scale, use_esrgan = args
    try:
        fn  = _run_esrgan_tile if use_esrgan else _run_edsr_tile
        out = fn(tile, scale)
        return row, col, out
    except Exception as e:
        logger.warning(f"[Upscaler] tile ({row},{col}) failed: {e} — Lanczos fallback")
        return row, col, _lanczos(tile, scale)


def upscale_tiled(img: np.ndarray, scale: int, use_esrgan: bool) -> np.ndarray:
    """
    Upscale img at full resolution using TILE_SIZE×TILE_SIZE tiles with OVERLAP.
    No pre-scaling — input is processed exactly as provided.
    """
    h, w    = img.shape[:2]
    out_h   = h * scale
    out_w   = w * scale
    output  = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    # Build jobs
    jobs    = []
    meta    = []   # (row, col, out_y0, out_x0, out_y1, out_x1, strip_y0, strip_x0)

    for yi, y in enumerate(range(0, h, TILE_SIZE)):
        for xi, x in enumerate(range(0, w, TILE_SIZE)):
            # Read region with overlap
            ry0 = max(0, y - OVERLAP);           rx0 = max(0, x - OVERLAP)
            ry1 = min(h, y + TILE_SIZE + OVERLAP); rx1 = min(w, x + TILE_SIZE + OVERLAP)
            tile = img[ry0:ry1, rx0:rx1].copy()

            pad_top  = (y  - ry0) * scale
            pad_left = (x  - rx0) * scale

            # Destination in output
            oy0 = y * scale;              ox0 = x * scale
            oy1 = min(out_h, (min(h, y + TILE_SIZE)) * scale)
            ox1 = min(out_w, (min(w, x + TILE_SIZE)) * scale)
            src_h = oy1 - oy0;            src_w = ox1 - ox0

            jobs.append((yi, xi, tile, scale, use_esrgan))
            meta.append((yi, xi, oy0, ox0, oy1, ox1, pad_top, pad_left, src_h, src_w))

    logger.info(f"[Upscaler] {len(jobs)} tiles  workers={MAX_WORKERS}  input={w}×{h}")

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for row, col, out_tile in pool.map(_worker, jobs):
            results[(row, col)] = out_tile

    # Stitch
    for yi, xi, oy0, ox0, oy1, ox1, pad_top, pad_left, src_h, src_w in meta:
        tile_out = results[(yi, xi)]
        output[oy0:oy1, ox0:ox1] = tile_out[pad_top:pad_top+src_h, pad_left:pad_left+src_w]

    return output


# ── Public class ──────────────────────────────────────────────────────────────

class Upscaler:
    def __init__(self, scale: int = 2):
        if scale not in (2, 4):
            raise ValueError("scale must be 2 or 4")
        self.scale = scale
        # Pre-load models at startup so first request is not slow
        _get_edsr(scale)
        _get_esrgan(scale)

    def process(self, img: np.ndarray, profile: ImageProfile) -> np.ndarray:
        use_esrgan = (profile.image_type == "photo")
        logger.info(
            f"[Upscaler] {'ESRGAN' if use_esrgan else 'EDSR'} {self.scale}×  "
            f"input={img.shape[1]}×{img.shape[0]}"
        )
        return upscale_tiled(img, self.scale, use_esrgan)