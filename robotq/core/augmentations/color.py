"""ColorJitter: temporally consistent brightness/contrast/saturation/hue augmentation."""

from __future__ import annotations

import random

import cv2
import numpy as np

from robotq.core.episode import Episode
from robotq.core.transform import SequenceTransform


class ColorJitter(SequenceTransform):
    """Apply random brightness, contrast, saturation, and hue shifts.

    Params are sampled ONCE per episode for temporal consistency (no flickering).
    """

    def __init__(
        self,
        brightness: float = 0.3,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.05,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        for name, val in [
            ("brightness", brightness),
            ("contrast", contrast),
            ("saturation", saturation),
        ]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in [0.0, 1.0], got {val!r}")
        if not (0.0 <= hue <= 0.5):
            raise ValueError(f"hue must be in [0.0, 0.5], got {hue!r}")
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def get_params(self, episode: Episode) -> dict:
        return {
            "brightness_factor": random.uniform(1 - self.brightness, 1 + self.brightness),
            "contrast_factor": random.uniform(1 - self.contrast, 1 + self.contrast),
            "saturation_factor": random.uniform(1 - self.saturation, 1 + self.saturation),
            "hue_shift": random.uniform(-self.hue, self.hue),
        }

    def apply_to_frame(self, frame: np.ndarray, params: dict) -> np.ndarray:
        # Work in float32 for precision
        img = frame.astype(np.float32)

        # Brightness
        img = img * params["brightness_factor"]

        # Contrast: shift toward mean, then scale
        mean = img.mean()
        img = (img - mean) * params["contrast_factor"] + mean

        # Saturation and hue: convert to HSV
        img_clipped = np.clip(img, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(img_clipped, cv2.COLOR_RGB2HSV).astype(np.float32)

        # Saturation
        hsv[:, :, 1] = hsv[:, :, 1] * params["saturation_factor"]

        # Hue shift (OpenCV H is 0-180)
        hsv[:, :, 0] = (hsv[:, :, 0] + params["hue_shift"] * 180) % 180

        hsv = np.clip(hsv, 0, [180, 255, 255]).astype(np.uint8)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return result
