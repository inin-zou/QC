import numpy as np
import pytest


@pytest.fixture
def sample_frames():
    """10 random frames at 480x640 resolution."""
    return [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(10)]
