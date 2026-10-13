"""Deterministic round-robin DPA baseline."""
from __future__ import annotations

from collections.abc import Iterable


def uniform_counts(
    budget: int,
    depths: Iterable[int] = (16, 24, 32),
    max_per_depth: int = 5,
    active_depths: Iterable[int] | None = None,
) -> tuple[int, ...]:
    """Allocate candidates round-robin under a nominal depth-cost budget.

    At each step, choose the least-sampled currently active depth; ties go to
    the shallower depth. If none of the least-sampled depths fits the remaining
    budget, stop. The policy never skips ahead to a more-sampled depth.
    """
    depths = tuple(int(d) for d in depths)
    if budget < 0 or not depths or any(d <= 0 for d in depths):
        raise ValueError("budget must be nonnegative and depths positive/nonempty")
    if tuple(sorted(set(depths))) != depths:
        raise ValueError("depths must be strictly increasing")
    active_depths = set(depths if active_depths is None else map(int, active_depths))
    if not active_depths.issubset(depths):
        raise ValueError("active_depths must be a subset of depths")
    if max_per_depth < 0:
        raise ValueError("max_per_depth must be nonnegative")

    counts = [0] * len(depths)
    spent = 0
    while True:
        active = [i for i, n in enumerate(counts) if depths[i] in active_depths and n < max_per_depth]
        if not active:
            break
        least_n = min(counts[i] for i in active)
        least = [i for i in active if counts[i] == least_n]
        affordable = [i for i in least if spent + depths[i] <= budget]
        if not affordable:
            break
        i = min(affordable, key=lambda j: depths[j])
        counts[i] += 1
        spent += depths[i]
    return tuple(counts)


def nominal_cost(counts: Iterable[int], depths: Iterable[int] = (16, 24, 32)) -> int:
    counts, depths = tuple(counts), tuple(depths)
    if len(counts) != len(depths):
        raise ValueError("counts and depths must have the same length")
    return sum(int(n) * int(d) for n, d in zip(counts, depths))


if __name__ == "__main__":
    for cap in (48, 80, 120, 160, 200, 240, 320):
        allocation = uniform_counts(cap)
        print(f"B={cap:>3}: n16,n24,n32={allocation}, cost={nominal_cost(allocation)}")
