import cv2
import numpy as np
from PIL import Image
from loguru import logger


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL RGB image to OpenCV BGR uint8 array."""
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)


def bgr_to_pil(img: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR uint8 array to PIL RGB image."""
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def safe_clip(img: np.ndarray) -> np.ndarray:
    """Clip and cast to uint8 safely."""
    return np.clip(img, 0, 255).astype(np.uint8)


def blend(original: np.ndarray, processed: np.ndarray, strength: float) -> np.ndarray:
    """
    Blend processed result with original at given strength (0=original, 1=fully processed).
    Useful to dial back any effect that is too aggressive.
    """
    strength = max(0.0, min(1.0, strength))
    return safe_clip(
        original.astype(np.float32) * (1 - strength)
        + processed.astype(np.float32) * strength
    )


def image_stats(img: np.ndarray) -> dict:
    """Return basic image quality stats for logging."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_32F).var()
    return {
        "shape": img.shape,
        "sharpness_score": round(float(laplacian_var), 2),
        "mean_brightness": round(float(gray.mean()), 2),
    }