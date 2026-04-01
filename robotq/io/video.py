"""Pure video decoder for robotq.

This module decodes MP4 files into lists of numpy frames.
It knows nothing about datasets, episodes, or HuggingFace.
LeRobot handles encoding; this module only decodes.
"""

from pathlib import Path

import cv2
import numpy as np


def decode_video(path: str | Path) -> list[np.ndarray]:
    """Decode a video file into a list of RGB frames.

    Args:
        path: Path to the video file.

    Returns:
        List of (H, W, C) uint8 numpy arrays in RGB order.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the video contains 0 frames.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(str(path))
    try:
        frames: list[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # OpenCV reads frames in BGR; convert to RGB.
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()

    if len(frames) == 0:
        raise ValueError(f"Video contains 0 frames: {path}")

    return frames
