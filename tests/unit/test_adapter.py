"""Unit tests for ActionAdapter protocol and AlohaAdapter implementation."""

import numpy as np
import pytest

from robotq.adapters.aloha import AlohaAdapter


@pytest.fixture
def adapter() -> AlohaAdapter:
    return AlohaAdapter()


# ---------------------------------------------------------------------------
# robot_type
# ---------------------------------------------------------------------------


def test_robot_type(adapter: AlohaAdapter) -> None:
    assert adapter.robot_type == "aloha"


# ---------------------------------------------------------------------------
# slice accessors
# ---------------------------------------------------------------------------


def test_get_left_slice(adapter: AlohaAdapter) -> None:
    assert adapter.get_left_slice() == slice(0, 7)


def test_get_right_slice(adapter: AlohaAdapter) -> None:
    assert adapter.get_right_slice() == slice(7, 14)


# ---------------------------------------------------------------------------
# swap_arms (1-D)
# ---------------------------------------------------------------------------


def test_swap_arms_correctness(adapter: AlohaAdapter) -> None:
    """Swapping [1..7 | 8..14] should yield [8..14 | 1..7]."""
    vec = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], dtype=float)
    result = adapter.swap_arms(vec)
    expected = np.array([8, 9, 10, 11, 12, 13, 14, 1, 2, 3, 4, 5, 6, 7], dtype=float)
    np.testing.assert_array_equal(result, expected)


def test_swap_arms_does_not_mutate_input(adapter: AlohaAdapter) -> None:
    vec = np.arange(14, dtype=float)
    original = vec.copy()
    adapter.swap_arms(vec)
    np.testing.assert_array_equal(vec, original)


def test_swap_arms_double_swap_is_identity(adapter: AlohaAdapter) -> None:
    vec = np.arange(14, dtype=float)
    np.testing.assert_array_equal(adapter.swap_arms(adapter.swap_arms(vec)), vec)


# ---------------------------------------------------------------------------
# swap_arms_batch (2-D)
# ---------------------------------------------------------------------------


def test_swap_arms_batch_shape(adapter: AlohaAdapter) -> None:
    vecs = np.zeros((3, 14))
    result = adapter.swap_arms_batch(vecs)
    assert result.shape == (3, 14)


def test_swap_arms_batch_correctness(adapter: AlohaAdapter) -> None:
    """Each row of a (3, 14) array should be swapped independently."""
    vecs = np.tile(
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], dtype=float),
        (3, 1),
    )
    result = adapter.swap_arms_batch(vecs)
    expected_row = np.array([8, 9, 10, 11, 12, 13, 14, 1, 2, 3, 4, 5, 6, 7], dtype=float)
    for i in range(3):
        np.testing.assert_array_equal(result[i], expected_row)


def test_swap_arms_batch_does_not_mutate_input(adapter: AlohaAdapter) -> None:
    vecs = np.random.default_rng(0).random((3, 14))
    original = vecs.copy()
    adapter.swap_arms_batch(vecs)
    np.testing.assert_array_equal(vecs, original)


# ---------------------------------------------------------------------------
# get_flip_signs
# ---------------------------------------------------------------------------


def test_flip_signs_shape(adapter: AlohaAdapter) -> None:
    signs = adapter.get_flip_signs()
    assert signs.shape == (14,)


def test_flip_signs_values_are_plus_or_minus_one(adapter: AlohaAdapter) -> None:
    signs = adapter.get_flip_signs()
    assert np.all(np.abs(signs) == 1.0), "All flip signs must be +1 or -1"


def test_flip_signs_waist_joints_negated(adapter: AlohaAdapter) -> None:
    signs = adapter.get_flip_signs()
    assert signs[0] == -1.0, "Left waist (index 0) should be negated"
    assert signs[7] == -1.0, "Right waist (index 7) should be negated"


def test_flip_signs_non_waist_joints_positive(adapter: AlohaAdapter) -> None:
    signs = adapter.get_flip_signs()
    non_waist = [i for i in range(14) if i not in (0, 7)]
    assert np.all(signs[non_waist] == 1.0), "Non-waist joints should have sign +1"
