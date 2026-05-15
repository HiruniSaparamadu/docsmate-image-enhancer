"""
enhancer/sharpener.py

Profile-aware multi-pass sharpener.

Documents:  aggressive text sharpening — strong USM + high-freq boost
            + optional morphological edge enhancement for very blurry scans.

Photos:     adaptive USM + detail enhance, pulled back if image is already sharp
            to avoid halos on faces and smooth gradients.
"""

import cv2
import numpy as np
from loguru import logger
from .utils import safe_clip, blend, image_stats
from .detector import ImageProfile


class Sharpener:

    # ── Shared primitives ───────────────────────────────────────────────────

    def _usm(self, img: np.ndarray, sigma: float, amount: float, threshold: int) -> np.ndarray:
        """Unsharp mask with pixel-level threshold."""
        blurred   = cv2.GaussianBlur(img, (0, 0), sigma)
        img_f     = img.astype(np.float32)
        blur_f    = blurred.astype(np.float32)
        diff      = img_f - blur_f
        mask      = np.abs(diff).max(axis=2) > threshold
        sharpened = img_f + diff * amount
        sharpened[~mask] = img_f[~mask]
        return safe_clip(sharpened)

    def _high_freq(self, img: np.ndarray, strength: float, sigma: float = 3.0) -> np.ndarray:
        """High-frequency layer boost."""
        base  = cv2.GaussianBlur(img, (0, 0), sigma)
        hf    = img.astype(np.float32) - base.astype(np.float32)
        return safe_clip(img.astype(np.float32) + hf * strength)

    def _detail_enhance(self, img: np.ndarray, sigma_s: float, sigma_r: float, strength: float) -> np.ndarray:
        enhanced = cv2.detailEnhance(img, sigma_s=sigma_s, sigma_r=sigma_r)
        return blend(img, enhanced, strength)

    # ── Document sharpening ─────────────────────────────────────────────────

    def _sharpen_document(self, img: np.ndarray, profile: ImageProfile) -> np.ndarray:
        """
        Three-pass document sharpening:

        Pass 1 — Strong USM  : recovers text strokes from blur/compression
        Pass 2 — HF boost    : restores fine print, serial numbers, microtext
        Pass 3 — Laplacian   : edge crisp-up for very blurry scans only
        """
        logger.debug("[Sharpener] document mode")
        original = img.copy()

        # Pass 1: Strong USM — bigger sigma catches wider blur
        amount = 2.2 if profile.is_blurry else 1.6
        img    = self._usm(img, sigma=1.5, amount=amount, threshold=5)

        # Pass 2: High-frequency boost — fine print recovery
        hf_str = 0.55 if profile.is_blurry else 0.35
        img    = self._high_freq(img, strength=hf_str, sigma=2.5)

        # Pass 3: For severely blurry docs, add a Laplacian edge pass
        if profile.is_blurry:
            logger.debug("[Sharpener] extra Laplacian pass for blurry document")
            img = self._laplacian_sharpen(img, strength=0.4)

        # Blend at 90% — keep 10% of pre-sharpened to avoid halos on photos in doc
        return blend(original, img, 0.90)

    def _laplacian_sharpen(self, img: np.ndarray, strength: float = 0.4) -> np.ndarray:
        """
        Laplacian sharpening: adds the second derivative (edges) back to image.
        Particularly effective at recovering blurred text strokes.
        """
        gray_f   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lap      = cv2.Laplacian(gray_f, cv2.CV_32F, ksize=3)
        # Apply Laplacian boost equally to all channels
        img_f    = img.astype(np.float32)
        for c in range(3):
            img_f[:, :, c] -= lap * strength
        return safe_clip(img_f)

    # ── Photo sharpening ────────────────────────────────────────────────────

    def _sharpen_photo(self, img: np.ndarray, profile: ImageProfile) -> np.ndarray:
        """
        Adaptive photo sharpening:
        - Sharp images get minimal treatment (no halos on faces)
        - Blurry images get stronger USM + detail enhance
        """
        logger.debug("[Sharpener] photo mode")
        stats     = image_stats(img)
        sharpness = stats["sharpness_score"]
        original  = img.copy()

        # Adaptive scale: 1.0 for very blurry, 0.4 for already sharp
        scale = max(0.4, min(1.0, 1.0 - (sharpness - 60) / 900))
        logger.debug(f"[Sharpener] photo adaptive scale={scale:.2f} sharpness={sharpness}")

        amount = 1.3 * scale
        img    = self._usm(img, sigma=1.2, amount=amount, threshold=8)

        if scale > 0.5:
            img = self._high_freq(img, strength=0.20 * scale, sigma=3.0)
            img = self._detail_enhance(img, sigma_s=8, sigma_r=0.12, strength=0.5 * scale)

        return blend(original, img, 0.90)

    # ── Public ──────────────────────────────────────────────────────────────

    def process(self, img: np.ndarray, profile: ImageProfile) -> np.ndarray:
        if profile.image_type == "document":
            return self._sharpen_document(img, profile)
        return self._sharpen_photo(img, profile)