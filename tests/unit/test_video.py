"""Unit tests for robotq.io.video decode_video()."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from robotq.io.video import decode_video


def _write_test_video(path: Path, frames: list[np.ndarray], fps: float = 30.0) -> None:
    """Write a list of BGR frames to a video file using OpenCV VideoWriter."""
    if not frames:
        raise ValueError("frames must be non-empty")
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


class TestDecodeVideoFrameCount:
    """Verify that the correct number of frames is returned."""

    def test_frame_count_matches(self, tmp_path: Path) -> None:
        video_path = tmp_path / "test.mp4"
        num_frames = 10
        bgr_frames = [
            np.full((64, 64, 3), fill_value=i * 10, dtype=np.uint8) for i in range(num_frames)
        ]
        _write_test_video(video_path, bgr_frames)

        decoded = decode_video(video_path)

        assert len(decoded) == num_frames

    def test_frame_shape(self, tmp_path: Path) -> None:
        video_path = tmp_path / "shape.mp4"
        h, w = 48, 64
        bgr_frames = [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(5)]
        _write_test_video(video_path, bgr_frames)

        decoded = decode_video(video_path)

        assert decoded[0].shape == (h, w, 3)
        assert decoded[0].dtype == np.uint8


class TestDecodeVideoErrors:
    """Verify error handling."""

    def test_nonexistent_file_raises_file_not_found(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist.mp4"
        with pytest.raises(FileNotFoundError):
            decode_video(missing)

    def test_nonexistent_file_accepts_str_path(self, tmp_path: Path) -> None:
        missing = str(tmp_path / "nope.mp4")
        with pytest.raises(FileNotFoundError):
            decode_video(missing)


class TestDecodeVideoColorOrder:
    """Verify that frames are returned in RGB order, not BGR."""

    def test_rgb_channel_order(self, tmp_path: Path) -> None:
        """Write a frame whose R, G, B channels have distinct known values.

        OpenCV VideoWriter expects BGR input.  We write a frame where:
          BGR = (10, 20, 200)  →  R=200, G=20, B=10 in RGB space.

        After decoding, the first channel of the returned frame must be 200
        (red), not 10 (blue).
        """
        video_path = tmp_path / "color.mp4"

        # Construct a solid-colour frame in BGR order for the writer.
        blue_bgr, green_bgr, red_bgr = 10, 20, 200
        bgr_frame = np.zeros((64, 64, 3), dtype=np.uint8)
        bgr_frame[:, :, 0] = blue_bgr  # B channel
        bgr_frame[:, :, 1] = green_bgr  # G channel
        bgr_frame[:, :, 2] = red_bgr  # R channel

        _write_test_video(video_path, [bgr_frame])

        decoded = decode_video(video_path)
        assert len(decoded) == 1

        frame = decoded[0]
        # In RGB order: channel 0 = R, channel 1 = G, channel 2 = B.
        # Video compression may shift pixel values slightly, so allow a small
        # tolerance of ±10 to account for lossy codec artefacts.
        tol = 10
        assert abs(int(frame[0, 0, 0]) - red_bgr) <= tol, (
            f"Expected R~{red_bgr}, got {frame[0, 0, 0]} — frame may still be BGR"
        )
        assert abs(int(frame[0, 0, 2]) - blue_bgr) <= tol, (
            f"Expected B~{blue_bgr}, got {frame[0, 0, 2]}"
        )

    def test_rgb_not_bgr(self, tmp_path: Path) -> None:
        """Confirm R channel value is greater than B channel value after decode.

        This acts as a regression guard: if BGR/RGB conversion is missing,
        R and B values would be swapped and the assertion would fail.
        """
        video_path = tmp_path / "rgb_check.mp4"

        # Write a strongly red frame (high R, low B) in BGR order.
        bgr_frame = np.zeros((32, 32, 3), dtype=np.uint8)
        bgr_frame[:, :, 2] = 220  # R channel in BGR notation
        bgr_frame[:, :, 0] = 20  # B channel in BGR notation

        _write_test_video(video_path, [bgr_frame])

        decoded = decode_video(video_path)
        frame = decoded[0]

        r_val = int(frame[0, 0, 0])
        b_val = int(frame[0, 0, 2])
        assert r_val > b_val, (
            f"R ({r_val}) should be greater than B ({b_val}); BGR→RGB conversion may be missing"
        )
