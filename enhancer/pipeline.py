"""
enhancer/pipeline.py

Full-resolution enhancement pipeline — no pre-scaling.

Every image is enhanced at its native resolution. Quality is never
sacrificed for speed. The upscaler handles any size via tiling.

PDF pages are capped at 1600px longest edge by pdf_handler (that is
a rasterisation limit, not a quality reduction — the source is a vector PDF).
"""

import cv2
import numpy as np
from pathlib import Path
from loguru import logger
from typing import Optional, Union

from .detector import analyse, ImageProfile
from .color import ColorRestorer
from .upscaler import Upscaler
from .denoiser import Denoiser
from .sharpener import Sharpener
from .face import FaceRestorer
from .utils import safe_clip, image_stats


class EnhancementPipeline:

    def __init__(self, scale: int = 2, enable_face: bool = True):
        logger.info(f"Pipeline init — scale={scale}× face={enable_face}")
        self.scale     = scale
        self.color     = ColorRestorer()
        self.upscaler  = Upscaler(scale=scale)
        self.denoiser  = Denoiser()
        self.sharpener = Sharpener()
        self.face      = FaceRestorer() if enable_face else None

    # ── IO ───────────────────────────────────────────────────────────────────

    def _load(self, source: Union[str, Path, bytes, np.ndarray]) -> np.ndarray:
        if isinstance(source, (str, Path)):
            img = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Cannot read: {source}")
        elif isinstance(source, bytes):
            arr = np.frombuffer(source, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Cannot decode image bytes — file may be corrupt")
        elif isinstance(source, np.ndarray):
            img = source.copy()
        else:
            raise TypeError(f"Unsupported type: {type(source)}")
        return img

    def save(self, img: np.ndarray, path: str, quality: int = 95) -> str:
        ext = Path(path).suffix.lower()
        if ext in (".jpg", ".jpeg"):
            cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif ext == ".webp":
            cv2.imwrite(path, img, [cv2.IMWRITE_WEBP_QUALITY, quality])
        else:
            cv2.imwrite(path, img)
        logger.info(f"Saved → {path}  ({img.shape[1]}×{img.shape[0]})")
        return path

    # ── Core single-image enhance ─────────────────────────────────────────────

    def _enhance_one(self, img: np.ndarray) -> tuple[np.ndarray, ImageProfile]:
        """
        Run the full 6-stage pipeline on a BGR image.
        Input is processed at its FULL resolution — no downscaling.
        """
        before = image_stats(img)
        logger.info(f"Input: {img.shape[1]}×{img.shape[0]}  sharpness={before['sharpness_score']}")

        # Stage 1: detect profile
        profile = analyse(img)
        logger.info(
            f"Profile: type={profile.image_type}  face={profile.has_face}  "
            f"dark={profile.is_dark}  blurry={profile.is_blurry}"
        )

        # Stage 2: colour & exposure (pre-upscale)
        img = self.color.process(img, profile)

        # Stage 3: AI upscale (full resolution, tiled)
        img = self.upscaler.process(img, profile)
        logger.info(f"After upscale: {img.shape[1]}×{img.shape[0]}")

        # Stage 4: denoise (post-upscale)
        img = self.denoiser.process(img, profile)

        # Stage 5: sharpen
        img = self.sharpener.process(img, profile)

        # Stage 6: face restoration
        if self.face:
            img = self.face.process(img, profile)

        img   = safe_clip(img)
        after = image_stats(img)
        logger.info(
            f"Done — sharpness {before['sharpness_score']} → {after['sharpness_score']}  "
            f"output {img.shape[1]}×{img.shape[0]}"
        )
        return img, profile

    # ── Public: single image ──────────────────────────────────────────────────

    def run(
        self,
        source: Union[str, Path, bytes, np.ndarray],
        output_path: Optional[str] = None,
        quality: int = 95,
    ) -> tuple[np.ndarray, ImageProfile]:
        img = self._load(source)
        result, profile = self._enhance_one(img)
        if output_path:
            self.save(result, output_path, quality)
        return result, profile

    # ── Public: PDF (multi-page) ──────────────────────────────────────────────

    def run_pdf(self, pdf_bytes: bytes) -> tuple[bytes, int, Optional[ImageProfile]]:
        """
        Enhance every page of a scanned PDF.
        Pages are processed one at a time (streaming, no full-doc RAM load).
        Returns (output_pdf_bytes, page_count, first_page_profile).
        """
        from .pdf_handler import iter_pdf_pages, images_to_pdf

        enhanced_pages = []
        first_profile  = None
        page_num       = 0

        for page_bgr in iter_pdf_pages(pdf_bytes):
            page_num += 1
            logger.info(f"[Pipeline] PDF page {page_num}")
            enhanced, profile = self._enhance_one(page_bgr)
            enhanced_pages.append(enhanced)
            if first_profile is None:
                first_profile = profile

        if not enhanced_pages:
            raise ValueError("PDF contained no processable pages")

        output_pdf = images_to_pdf(enhanced_pages)
        return output_pdf, page_num, first_profile