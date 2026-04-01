"""Pure video decoder for robotq.

This module decodes MP4 files into lists of numpy frames.
It knows nothing about datasets, episodes, or HuggingFace.
LeRobot handles encoding; this module only decodes.
"""

from pathlib import Path

import cv2
import numpy as np


def decode_video(
    path: str | Path,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> list[np.ndarray]:
    """Decode a video file into a list of RGB frames.

    Args:
        path: Path to the video file.
        start_frame: First frame to decode (0-indexed). Frames before this are skipped.
        end_frame: Frame index to stop at (exclusive). None means decode to end.

    Returns:
        List of (H, W, C) uint8 numpy arrays in RGB order.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the video contains 0 frames in the requested range.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(str(path))
    try:
        # Seek to start frame if needed
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames: list[np.ndarray] = []
        frame_idx = start_frame
        while True:
            if end_frame is not None and frame_idx >= end_frame:
                break
            ret, frame = cap.read()
            if not ret:
                break
            # OpenCV reads frames in BGR; convert to RGB.
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_idx += 1
    finally:
        cap.release()

    if len(frames) == 0:
        raise ValueError(f"Video contains 0 frames in range [{start_frame}:{end_frame}]: {path}")

    return frames
