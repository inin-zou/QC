"""Pipeline composition operators for the RobotQ augmentation pipeline.

Three operators for composing Transform instances:
- Compose: applies all transforms in sequence
- OneOf: picks exactly one transform at random
- SomeOf: picks N transforms at random (N drawn from a range)
"""

from __future__ import annotations

import random

from robotq.core.episode import Episode
from robotq.core.transform import Transform


class Compose(Transform):
    """Apply a sequence of transforms in order.

    Each child transform's own ``p`` controls whether it fires individually.
    The outer ``p`` controls whether the whole pipeline runs at all.
    """

    def __init__(self, transforms: list[Transform], p: float = 1.0) -> None:
        super().__init__(p=p)
        self.transforms = transforms

    def apply(self, episode: Episode) -> Episode:
        if random.random() > self.p:
            return episode
        for t in self.transforms:
            episode = t(episode)
        return episode

    def __repr__(self) -> str:
        children = ", ".join(repr(t) for t in self.transforms)
        return f"{self.__class__.__name__}(transforms=[{children}], p={self.p!r})"


class OneOf(Transform):
    """Apply exactly one randomly chosen transform.

    The child transform's own ``p`` still applies after selection.
    """

    def __init__(self, transforms: list[Transform], p: float = 1.0) -> None:
        super().__init__(p=p)
        self.transforms = transforms

    def apply(self, episode: Episode) -> Episode:
        if random.random() > self.p:
            return episode
        if not self.transforms:
            return episode
        t = random.choice(self.transforms)
        return t(episode)

    def __repr__(self) -> str:
        children = ", ".join(repr(t) for t in self.transforms)
        return f"{self.__class__.__name__}(transforms=[{children}], p={self.p!r})"


class SomeOf(Transform):
    """Apply a random subset of transforms.

    ``n`` is an inclusive ``(min, max)`` range for how many transforms to pick.
    Selected transforms are applied in the order they appear in the list.
    """

    def __init__(
        self,
        transforms: list[Transform],
        n: tuple[int, int] = (1, 2),
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.transforms = transforms
        self.n = n

    def apply(self, episode: Episode) -> Episode:
        if random.random() > self.p:
            return episode
        if not self.transforms:
            return episode
        n_min, n_max = self.n
        k = random.randint(n_min, min(n_max, len(self.transforms)))
        # Sample without replacement, then sort to preserve original order
        indices = sorted(random.sample(range(len(self.transforms)), k))
        for i in indices:
            episode = self.transforms[i](episode)
        return episode

    def __repr__(self) -> str:
        children = ", ".join(repr(t) for t in self.transforms)
        return (
            f"{self.__class__.__name__}("
            f"transforms=[{children}], n={self.n!r}, p={self.p!r})"
        )
