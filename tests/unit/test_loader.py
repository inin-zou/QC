"""Tests for the dataset loader.

The loader uses LeRobot's API internally. Unit tests verify the module
can be imported and the function signature is correct. Real data tests
are in tests/integration/.
"""

from robotq.io.loader import load_dataset


def test_loader_is_callable():
    assert callable(load_dataset)


def test_loader_accepts_keyword_args():
    """Verify the function signature accepts expected keyword args."""
    import inspect
    sig = inspect.signature(load_dataset)
    params = list(sig.parameters.keys())
    assert "repo_id" in params
    assert "max_episodes" in params
    assert "local_dir" in params
