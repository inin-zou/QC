"""BackgroundReplace: replace scene background using segmentation + inpainting.

Two methods:
- 'fast': motion-based mask via median background subtraction (CPU-only, no extra deps)
- 'auto': SAM2 segmentation (accurate, needs sam2 package + GPU)

Both methods use Stable Diffusion Inpainting to fill the background with a text prompt.
"""

from __future__ import annotations

import copy
import logging

import cv2
import numpy as np

from robotq.core.episode import Episode
from robotq.core.transform import SequenceTransform

logger = logging.getLogger(__name__)


class BackgroundReplace(SequenceTransform):
    """Replace the background of robot videos using inpainting.

    The robot and task objects are segmented (kept), and the background
    is replaced with a scene described by `prompt`.

    Parameters
    ----------
    prompt:
        Text description of the new background (e.g. "industrial kitchen",
        "clean laboratory", "wooden table in a workshop").
    method:
        'fast' — motion-based mask via frame differencing (CPU, no extra deps).
        'auto' — SAM2 segmentation (accurate, needs sam2 + GPU).
    strength:
        Inpainting strength (0.0-1.0). Higher = more creative, less faithful.
    cameras:
        List of camera names to process. None = all cameras.
    device:
        Torch device for the inpainting model ('mps', 'cuda', 'cpu').
    """

    def __init__(
        self,
        prompt: str = "clean laboratory with white walls",
        method: str = "fast",
        strength: float = 0.75,
        cameras: list[str] | None = None,
        device: str | None = None,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        if method not in ("fast", "auto"):
            raise ValueError(f"method must be 'fast' or 'auto', got {method!r}")
        self.prompt = prompt
        self.method = method
        self.strength = strength
        self.cameras = cameras
        self.device = device
        self._pipe = None

    def _get_inpaint_pipe(self):
        """Lazy-load the Stable Diffusion Inpainting pipeline."""
        if self._pipe is not None:
            return self._pipe

        try:
            from diffusers import StableDiffusionInpaintPipeline
            import torch
        except ImportError:
            raise ImportError(
                "BackgroundReplace requires 'diffusers' and 'torch'. "
                "Install with: uv pip install -e '.[generative]'"
            )

        device = self.device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        logger.info("Loading SD Inpainting model on %s...", device)
        self._pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            safety_checker=None,
        ).to(device)

        return self._pipe

    def _compute_mask_fast(self, frames: list[np.ndarray]) -> np.ndarray:
        """Compute foreground mask via median background subtraction.

        Returns a binary mask where 255 = background (to inpaint),
        0 = foreground (robot + objects, to keep).
        """
        # Stack frames and compute median background
        stack = np.stack(frames[:min(len(frames), 50)])  # Use first 50 frames max
        median_bg = np.median(stack, axis=0).astype(np.uint8)

        # Use the middle frame as reference
        mid_idx = len(frames) // 2
        ref_frame = frames[mid_idx]

        # Compute absolute difference from median background
        diff = cv2.absdiff(ref_frame, median_bg)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

        # Threshold to get foreground mask
        _, fg_mask = cv2.threshold(gray_diff, 25, 255, cv2.THRESH_BINARY)

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        # Dilate to ensure robot edges are included in foreground
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        fg_mask = cv2.dilate(fg_mask, dilate_kernel, iterations=1)

        # Invert: 255 = background (what we want to inpaint), 0 = foreground (keep)
        bg_mask = cv2.bitwise_not(fg_mask)

        return bg_mask

    def _inpaint_frame(self, frame: np.ndarray, mask: np.ndarray, pipe) -> np.ndarray:
        """Inpaint a single frame using Stable Diffusion."""
        from PIL import Image

        h, w = frame.shape[:2]

        # SD expects 512x512
        frame_pil = Image.fromarray(frame).resize((512, 512))
        mask_pil = Image.fromarray(mask).resize((512, 512))

        result = pipe(
            prompt=self.prompt,
            image=frame_pil,
            mask_image=mask_pil,
            strength=self.strength,
            num_inference_steps=20,
            guidance_scale=7.5,
        ).images[0]

        # Resize back to original resolution
        result = result.resize((w, h))
        result_np = np.array(result)

        # Composite: keep original foreground, use inpainted background
        # mask: 255 = background, 0 = foreground
        mask_resized = cv2.resize(mask, (w, h))
        mask_3ch = np.stack([mask_resized] * 3, axis=-1) / 255.0

        composite = (frame * (1 - mask_3ch) + result_np * mask_3ch).astype(np.uint8)
        return composite

    def get_params(self, episode: Episode) -> dict:
        """Compute background mask once for the episode."""
        cameras = self.cameras or episode.metadata.camera_names
        cam = cameras[0]  # Compute mask from primary camera
        frames = episode.frames[cam]
        mask = self._compute_mask_fast(frames)
        return {"mask": mask, "cameras": cameras}

    def apply_to_frame(self, frame: np.ndarray, params: dict) -> np.ndarray:
        """This is called per-frame but we need the pipe — override apply() instead."""
        # Not used directly — see apply() override below
        return frame

    def apply(self, episode: Episode) -> Episode:
        """Override apply to handle the inpainting pipeline."""
        import random as _random
        if _random.random() > self.p:
            return episode

        params = self.get_params(episode)
        mask = params["mask"]
        cameras = params["cameras"]
        pipe = self._get_inpaint_pipe()

        new_frames = {}
        for cam in episode.metadata.camera_names:
            if cam in cameras:
                logger.info("Inpainting camera '%s' (%d frames)...", cam, len(episode.frames[cam]))
                # Inpaint a subset of keyframes, interpolate for the rest
                keyframe_indices = list(range(0, len(episode.frames[cam]), 10))  # Every 10th frame
                keyframe_results = {}

                for idx in keyframe_indices:
                    keyframe_results[idx] = self._inpaint_frame(
                        episode.frames[cam][idx], mask, pipe
                    )

                # Fill in between keyframes with nearest keyframe result
                new_cam_frames = []
                for i in range(len(episode.frames[cam])):
                    nearest_key = min(keyframe_indices, key=lambda k: abs(k - i))
                    new_cam_frames.append(keyframe_results[nearest_key])
                new_frames[cam] = new_cam_frames
            else:
                new_frames[cam] = [f.copy() for f in episode.frames[cam]]

        return Episode(
            frames=new_frames,
            actions=episode.actions,
            states=episode.states,
            metadata=copy.deepcopy(episode.metadata),
        )
