import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from enhancer.utils import safe_clip, blend, image_stats
from enhancer.color import ColorRestorer
from enhancer.denoiser import Denoiser
from enhancer.sharpener import Sharpener


def make_test_image(h=64, w=64) -> np.ndarray:
    """Create a synthetic BGR test image with gradients and edges."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        img[i, :, 0] = int(i / h * 200) + 30
        img[i, :, 1] = 100
        img[i, :, 2] = int((h - i) / h * 200) + 30
    # Add a hard edge
    img[h // 2 :, w // 2 :, :] = 200
    return img


class TestUtils:
    def test_safe_clip(self):
        arr = np.array([-10, 0, 128, 255, 300], dtype=np.float32)
        out = safe_clip(arr)
        assert out.min() >= 0 and out.max() <= 255
        assert out.dtype == np.uint8

    def test_blend_zero(self):
        a = np.full((4, 4, 3), 100, dtype=np.uint8)
        b = np.full((4, 4, 3), 200, dtype=np.uint8)
        result = blend(a, b, 0.0)
        np.testing.assert_array_equal(result, a)

    def test_blend_full(self):
        a = np.full((4, 4, 3), 100, dtype=np.uint8)
        b = np.full((4, 4, 3), 200, dtype=np.uint8)
        result = blend(a, b, 1.0)
        np.testing.assert_array_equal(result, b)

    def test_image_stats_keys(self):
        img = make_test_image()
        stats = image_stats(img)
        assert "sharpness_score" in stats
        assert "mean_brightness" in stats
        assert "shape" in stats


class TestColorRestorer:
    def setup_method(self):
        self.cr = ColorRestorer()
        self.img = make_test_image()

    def test_output_shape_preserved(self):
        out = self.cr.process(self.img)
        assert out.shape == self.img.shape

    def test_output_dtype(self):
        out = self.cr.process(self.img)
        assert out.dtype == np.uint8

    def test_output_range(self):
        out = self.cr.process(self.img)
        assert out.min() >= 0 and out.max() <= 255


class TestDenoiser:
    def setup_method(self):
        self.dn = Denoiser()
        self.img = make_test_image()

    def test_output_shape(self):
        out = self.dn.process(self.img)
        assert out.shape == self.img.shape

    def test_not_all_black(self):
        out = self.dn.process(self.img)
        assert out.sum() > 0


class TestSharpener:
    def setup_method(self):
        self.sh = Sharpener()
        self.img = make_test_image()

    def test_sharpness_improves_or_holds(self):
        before = image_stats(self.img)["sharpness_score"]
        out    = self.sh.process(self.img)
        after  = image_stats(out)["sharpness_score"]
        # Sharpness should not decrease significantly
        assert after >= before * 0.8, f"Sharpness dropped: {before} → {after}"

    def test_output_shape(self):
        out = self.sh.process(self.img)
        assert out.shape == self.img.shape