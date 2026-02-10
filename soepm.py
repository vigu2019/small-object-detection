# soepm.py
# SOEPM = Small Optional Enhancement Preprocessing Module
# Features: CLAHE, Sharpening, optional 2x Upscale
# Intended use: INFERENCE ONLY

from typing import Tuple, Optional
import numpy as np
import cv2

class SOEPM:
    def __init__(
        self,
        use_clahe: bool = True,
        use_sharpen: bool = True,
        upscale_x2: bool = False,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8),
        sharpen_strength: float = 1.0,  # 0.0 = off, 1.0 = default, >1 = stronger
    ):
        """
        use_clahe: apply contrast enhancement
        use_sharpen: apply sharpening
        upscale_x2: if True, upscale image by 2× (bicubic)
        clahe_clip_limit, clahe_tile_grid_size: CLAHE settings
        sharpen_strength: how strong the sharpening should be
        """
        self.use_clahe = use_clahe
        self.use_sharpen = use_sharpen
        self.upscale_x2 = upscale_x2
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self.sharpen_strength = max(0.0, float(sharpen_strength))

        # Prepare CLAHE instance once
        self._clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_grid_size,
        )

    def _apply_clahe(self, img_bgr: np.ndarray) -> np.ndarray:
        # Convert to LAB → apply CLAHE on L channel → back to BGR
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = self._clahe.apply(l)
        lab_eq = cv2.merge((l_eq, a, b))
        bgr_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
        return bgr_eq

    def _apply_sharpen(self, img_bgr: np.ndarray) -> np.ndarray:
        if self.sharpen_strength <= 0.0:
            return img_bgr

        # Unsharp masking style: img + s*(img - blur(img))
        blurred = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=1.0)
        high_freq = cv2.addWeighted(img_bgr, 1.0, blurred, -1.0, 0)
        sharpened = cv2.addWeighted(img_bgr, 1.0, high_freq, self.sharpen_strength, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def _upscale_2x(self, img_bgr: np.ndarray) -> np.ndarray:
        h, w = img_bgr.shape[:2]
        return cv2.resize(img_bgr, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    def __call__(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        img_bgr: HxWx3 uint8 BGR image (OpenCV format)
        returns: processed image (same dtype)
        """
        out = img_bgr

        if self.use_clahe:
            out = self._apply_clahe(out)

        if self.use_sharpen:
            out = self._apply_sharpen(out)

        if self.upscale_x2:
            out = self._upscale_2x(out)

        return out


def load_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def save_bgr(path: str, img_bgr: np.ndarray) -> None:
    ok = cv2.imwrite(path, img_bgr)
    if not ok:
        raise IOError(f"Failed to write image to: {path}")
