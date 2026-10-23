"""Suzuki et al. (2020) Exponential / Power-of-2 Maximum Likelihood Amplitude Estimation.

Reference:
    Y. Suzuki, S. Uno, R. Raymond, T. Tanaka, T. Onodera, and N. Yamamoto,
    "Amplitude estimation without phase estimation,"
    Quantum Information Processing 19, 75 (2020), arXiv:1904.10246.

Schedule:
    Exponential (power-of-2) Grover depths n_0 = 0, n_k = 2^(k-1) up to scale_nmax,
    with uniform shots per rung and exact global maximum-likelihood estimation.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from leaderboard.schema_and_api import NonAdaptiveAESubmission, ml_estimate_batch


class MLAEPow2SuzukiSubmission(NonAdaptiveAESubmission):
    """Power-of-2 MLAE baseline (Suzuki et al. 2020) with uniform shots + global ML."""

    name = "MLAE (Pow2, Suzuki et al. 2020)"
    track = "non-adaptive"

    def __init__(self, shots_per_rung: int = 12) -> None:
        self.shots_per_rung = int(shots_per_rung)

    def get_schedule(self, scale_nmax: int) -> tuple[list[int], list[int]]:
        """Return power-of-2 depths {0, 1, 2, 4, ..., 2^m <= scale_nmax} and uniform shots."""
        nmax = max(1, int(scale_nmax))
        depths: list[int] = [0]
        d = 1
        while d <= nmax:
            depths.append(int(d))
            d *= 2
        shots: list[int] = [self.shots_per_rung for _ in depths]
        return depths, shots

    def estimate(
        self,
        depths: Sequence[int],
        shots: Sequence[int],
        counts: np.ndarray,
    ) -> np.ndarray | float:
        """Exact global maximum-likelihood estimation with 25-point parabolic refinement."""
        return ml_estimate_batch(depths, shots, counts)
