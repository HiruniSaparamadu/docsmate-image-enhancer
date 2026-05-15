import cv2
import numpy as np
from loguru import logger
from .utils import safe_clip, blend
from .detector import ImageProfile


class ColorRestorer:

    def _white_balance(self, img: np.ndarray, strength: float = 0.7) -> np.ndarray:
        """Gray-world WB in LAB — only applied when colour cast is significant."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        avg_a = np.mean(lab[:, :, 1])
        avg_b = np.mean(lab[:, :, 2])
        if abs(avg_a - 128) < 3 and abs(avg_b - 128) < 3:
            return img
        lum = lab[:, :, 0] / 255.0
        lab[:, :, 1] -= (avg_a - 128) * lum * 1.1
        lab[:, :, 2] -= (avg_b - 128) * lum * 1.1
        corrected = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        return blend(img, corrected, strength)

    def _clahe(self, img: np.ndarray, clip: float = 2.0, grid: tuple = (8, 8)) -> np.ndarray:
        """CLAHE on L channel in LAB."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _gamma(self, img: np.ndarray, gamma: float) -> np.ndarray:
        if abs(gamma - 1.0) < 0.01:
            return img
        table = np.array([(i / 255.0) ** (1.0 / gamma) * 255 for i in range(256)], dtype=np.uint8)
        return cv2.LUT(img, table)

    def _saturation(self, img: np.ndarray, boost: float) -> np.ndarray:
        if abs(boost - 1.0) < 0.01:
            return img
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * boost, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _increase_contrast(self, img: np.ndarray, alpha: float = 1.3, beta: int = -20) -> np.ndarray:
        """
        Linear contrast stretch: out = alpha * in + beta
        Used for documents to make text blacker and paper whiter.
        """
        out = img.astype(np.float32) * alpha + beta
        return safe_clip(out)

    def process(self, img: np.ndarray, profile: ImageProfile) -> np.ndarray:
        if profile.image_type == "document":
            return self._process_document(img, profile)
        return self._process_photo(img, profile)

    def _process_document(self, img: np.ndarray, profile: ImageProfile) -> np.ndarray:
        """
        Document profile:
        - Higher CLAHE clip for better text separation
        - Contrast stretch to make text pop
        - Minimal saturation change (documents are mostly greyscale)
        - Stronger gamma lift if dark/underexposed
        """
        logger.debug("[Color] document profile")
        img = self._white_balance(img, strength=0.5)
        clip = 3.5 if profile.is_dark else 2.5
        img  = self._clahe(img, clip=clip, grid=(6, 6))
        img  = self._increase_contrast(img, alpha=1.25, beta=-15)
        gamma = 1.35 if profile.is_dark else 1.08
        img   = self._gamma(img, gamma)
        return safe_clip(img)

    def _process_photo(self, img: np.ndarray, profile: ImageProfile) -> np.ndarray:
        """
        Photo profile:
        - Conservative CLAHE (avoid blowing faces)
        - Gentle saturation lift
        - Gamma based on brightness
        """
        logger.debug("[Color] photo profile")
        img = self._white_balance(img, strength=0.7)
        img = self._clahe(img, clip=2.0, grid=(8, 8))
        img = self._saturation(img, boost=1.15)
        gamma = 1.20 if profile.is_dark else 1.05
        img   = self._gamma(img, gamma)
        return safe_clip(img)