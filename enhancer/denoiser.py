"""
enhancer/denoiser.py

Profile-aware denoiser — applied AFTER upscaling.

Documents:  very light bilateral only — enough to remove JPEG
            compression blocks, never enough to soften text strokes.

Photos:     bilateral + optional light NLM for smooth skin/sky regions,
            blended back so detail is never lost.
"""

import cv2
import numpy as np
from loguru import logger
from .utils import safe_clip, blend, image_stats
from .detector import ImageProfile


class Denoiser:

    def process(self, img: np.ndarray, profile: ImageProfile) -> np.ndarray:
        if profile.image_type == "document":
            return self._denoise_document(img, profile)
        return self._denoise_photo(img, profile)

    # ── Document ────────────────────────────────────────────────────────────

    def _denoise_document(self, img: np.ndarray, profile: ImageProfile) -> np.ndarray:
        """
        Documents need artifact removal (JPEG blocks, scanner noise)
        but MUST NOT soften text.

        Strategy: bilateral with very small d and low sigma.
        Blend at 70% so original edges are preserved.
        """
        logger.debug("[Denoiser] document mode — light bilateral")
        original = img.copy()

        # Very conservative bilateral — removes block artifacts, keeps text edges
        denoised = cv2.bilateralFilter(img, d=3, sigmaColor=15, sigmaSpace=15)

        # Blend lightly — let original text sharpness dominate
        return blend(original, denoised, 0.65)

    # ── Photo ───────────────────────────────────────────────────────────────

    def _denoise_photo(self, img: np.ndarray, profile: ImageProfile) -> np.ndarray:
        """
        Photos can tolerate more denoising, especially in smooth regions.
        Two passes:
          1. Bilateral  — colour noise, edge-preserving
          2. NLM        — luminance grain (only for very grainy/dark images)
        """
        logger.debug("[Denoiser] photo mode")
        original  = img.copy()
        stats     = image_stats(img)

        img = cv2.bilateralFilter(img, d=5, sigmaColor=25, sigmaSpace=25)

        # NLM only for already-dark/noisy images and when sharpness is high
        # (high sharpness means upscaler did its job — noise is safe to remove)
        if profile.is_dark or stats["sharpness_score"] > 120:
            logger.debug("[Denoiser] NLM pass")
            img = cv2.fastNlMeansDenoisingColored(
                img, None,
                h=4.0, hColor=4.0,
                templateWindowSize=5,
                searchWindowSize=15,
            )

        return blend(original, img, 0.82)