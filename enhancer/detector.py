import cv2
import numpy as np
from dataclasses import dataclass
from loguru import logger


@dataclass
class ImageProfile:
    image_type: str 
    has_face: bool
    is_dark: bool
    is_blurry: bool
    sharpness: float
    brightness: float
    width: int
    height: int


def analyse(img: np.ndarray) -> ImageProfile:    
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Sharpness checking using Laplacian variance
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())

    # Brightness
    brightness = float(gray.mean())

    # Document detection heuristic    
    image_type = _classify_image_type(img, gray)

    # Face detection
    has_face = _detect_face(img)

    profile = ImageProfile(
        image_type=image_type,
        has_face=has_face,
        is_dark=brightness < 90,
        is_blurry=sharpness < 60,
        sharpness=round(sharpness, 2),
        brightness=round(brightness, 2),
        width=w,
        height=h,
    )
    logger.info(f"[Detector] profile: {profile}")
    return profile


def _classify_image_type(img: np.ndarray, gray: np.ndarray) -> str:    
    h, w = gray.shape

    # 1. White pixel ratio — paper background
    white_pixels = np.sum(gray > 200)
    white_ratio  = white_pixels / (h * w)

    # 2. Saturation — documents are mostly greyscale
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean_sat = float(hsv[:, :, 1].mean())

    # 3. Edge density — documents have lots of text edges
    edges      = cv2.Canny(gray.astype(np.uint8), 50, 150)
    edge_ratio = np.sum(edges > 0) / (h * w)

    # Scoring: higher = more document-like
    score = 0
    if white_ratio > 0.25:    score += 2   # lots of white paper
    if white_ratio > 0.40:    score += 1
    if mean_sat < 40:         score += 2   # low colour saturation
    if mean_sat < 25:         score += 1
    if edge_ratio > 0.05:     score += 1   # dense text edges
    if edge_ratio > 0.10:     score += 1

    image_type = "document" if score >= 4 else "photo"
    logger.debug(
        f"[Detector] type={image_type} white={white_ratio:.2f} "
        f"sat={mean_sat:.1f} edges={edge_ratio:.3f} score={score}"
    )
    return image_type


def _detect_face(img: np.ndarray) -> bool:    
    try:
        gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Use the frontal face cascade that ships with OpenCV
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(20, 20),  
        )
        found = len(faces) > 0
        logger.debug(f"[Detector] faces found: {len(faces)}")
        return found
    except Exception as e:
        logger.warning(f"[Detector] face detection failed: {e}")
        return False