"""Suzuki et al. (2020) Linearly Incremental Sequence (LIS) Maximum Likelihood AE.

Reference:
    Y. Suzuki, S. Uno, R. Raymond, T. Tanaka, T. Onodera, and N. Yamamoto,
    "Amplitude estimation without phase estimation,"
    Quantum Information Processing 19, 75 (2020), arXiv:1904.10246.

Schedule:
    Linearly incremental Grover depths n_k = k for k = 0, 1, ..., M with uniform
    shots per rung and exact global maximum-likelihood estimation. Because the
    linear sequence has total queries Q ~ s * M^2 / 2 and Fisher information
    I ~ 4 s M^3 / 3 (giving O(Q^{-3/4}) root-mean-square error scaling), the
    cutoff depth M(scale_nmax) = round(1.35 * scale_nmax^(2/3)) matches the
    benchmark scale query budget while keeping every consecutive integer depth
    n_k = k uniformly sampled.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from leaderboard.schema_and_api import NonAdaptiveAESubmission, ml_estimate_batch


class MLAELinearSuzukiSubmission(NonAdaptiveAESubmission):
    """Linear MLAE baseline (Suzuki et al. 2020, n_k = k, uniform shots + global ML)."""

    name = "MLAE (Linear, Suzuki et al. 2020)"
    track = "non-adaptive"

    def __init__(self, shots_per_rung: int = 2) -> None:
        self.shots_per_rung = int(shots_per_rung)

    def get_schedule(self, scale_nmax: int) -> tuple[list[int], list[int]]:
        """Return consecutive linear depths n_k = 0, 1, ..., M and uniform shots."""
        nmax = max(1, int(scale_nmax))
        m_cap = min(nmax, max(6, int(round(1.35 * (float(nmax) ** (2.0 / 3.0))))))
        depths = list(range(m_cap + 1))
        shots = [self.shots_per_rung for _ in depths]
        return depths, shots

    def estimate(
        self,
        depths: Sequence[int],
        shots: Sequence[int],
        counts: np.ndarray,
    ) -> np.ndarray | float:
        """Exact global maximum-likelihood estimation with 25-point parabolic refinement."""
        return ml_estimate_batch(depths, shots, counts)
