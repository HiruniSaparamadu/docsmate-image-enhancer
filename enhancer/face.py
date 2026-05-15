"""
enhancer/face.py

Face restoration via GFPGAN.

Works for both:
  - Portrait photos (large faces, full frame)
  - Document photos (small passport-size face thumbnails in NICs/passports/licences)

For document faces (small):
  The face region is extracted, upscaled temporarily, restored,
  then pasted back at original position to avoid distorting surrounding text.
"""

import cv2
import numpy as np
from pathlib import Path
from loguru import logger
from .utils import safe_clip, blend
from .detector import ImageProfile


GFPGAN_MODEL = "models/GFPGANv1.4.pth"


class FaceRestorer:

    def __init__(self, blend_strength: float = 0.88):
        self.blend_strength = blend_strength
        self._restorer      = None
        self._loaded        = False

    def _load(self):
        if self._loaded:
            return
        model_path = Path(GFPGAN_MODEL)
        try:
            from gfpgan import GFPGANer
            if not model_path.exists():
                raise FileNotFoundError(
                    f"GFPGAN model not found at {model_path}. "
                    f"Download from: https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
                )
            self._restorer = GFPGANer(
                model_path=str(model_path),
                upscale=1,        # upscaling already done — only restore faces
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
            )
            logger.info("[Face] GFPGAN loaded")
        except ImportError:
            logger.info("[Face] gfpgan not installed — face stage skipped (pip install gfpgan)")
            self._restorer = "skip"
        except Exception as e:
            logger.warning(f"[Face] could not load GFPGAN: {e} — skipping")
            self._restorer = "skip"
        self._loaded = True

    def process(self, img: np.ndarray, profile: ImageProfile) -> np.ndarray:
        if not profile.has_face:
            logger.debug("[Face] no face detected — skipping")
            return img

        self._load()
        if self._restorer == "skip":
            return img

        if profile.image_type == "document":
            return self._restore_document_face(img)
        return self._restore_photo_face(img)

    # ── Photo: full-image GFPGAN ────────────────────────────────────────────

    def _restore_photo_face(self, img: np.ndarray) -> np.ndarray:
        """Standard GFPGAN on full image — best for portrait photos."""
        try:
            _, _, restored = self._restorer.enhance(
                img,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
            )
            if restored is None:
                logger.debug("[Face] GFPGAN returned None — skipping")
                return img
            result = blend(img, restored, self.blend_strength)
            logger.info("[Face] photo face restoration applied")
            return result
        except Exception as e:
            logger.warning(f"[Face] restoration failed: {e}")
            return img

    # ── Document: targeted face patch restoration ───────────────────────────

    def _restore_document_face(self, img: np.ndarray) -> np.ndarray:
        """
        For document images (NIC, passport, licence):
        1. Detect face bounding boxes with Haar cascade
        2. Extract + temporarily scale up each face patch (min 256px)
        3. Run GFPGAN on the enlarged patch
        4. Scale result back down and paste seamlessly into original
        This avoids GFPGAN distorting the surrounding text/borders.
        """
        try:
            gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            faces = cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20)
            )
            if len(faces) == 0:
                logger.debug("[Face] document: no face box found — running full image")
                return self._restore_photo_face(img)

            result = img.copy()
            h_img, w_img = img.shape[:2]

            for (fx, fy, fw, fh) in faces:
                # Add padding around face (15%) for better GFPGAN context
                pad_x = int(fw * 0.15);  pad_y = int(fh * 0.15)
                x1 = max(0, fx - pad_x); y1 = max(0, fy - pad_y)
                x2 = min(w_img, fx + fw + pad_x)
                y2 = min(h_img, fy + fh + pad_y)
                patch = img[y1:y2, x1:x2].copy()

                # GFPGAN needs at least 256px — upscale small passport photos
                ph, pw = patch.shape[:2]
                min_dim = 256
                if min(ph, pw) < min_dim:
                    scale_up = min_dim / min(ph, pw)
                    patch_big = cv2.resize(
                        patch,
                        (int(pw * scale_up), int(ph * scale_up)),
                        interpolation=cv2.INTER_LANCZOS4,
                    )
                else:
                    patch_big  = patch
                    scale_up   = 1.0

                # Restore the upscaled face patch
                _, _, restored_big = self._restorer.enhance(
                    patch_big,
                    has_aligned=False,
                    only_center_face=True,
                    paste_back=True,
                )
                if restored_big is None:
                    continue

                # Scale back to original patch size
                restored_patch = cv2.resize(
                    restored_big, (pw, ph), interpolation=cv2.INTER_LANCZOS4
                )

                # Blend and paste back
                blended = blend(patch, restored_patch, self.blend_strength)
                result[y1:y2, x1:x2] = blended
                logger.info(f"[Face] document face restored at ({x1},{y1})-({x2},{y2})")

            return result

        except Exception as e:
            logger.warning(f"[Face] document face restoration failed: {e}")
            return img