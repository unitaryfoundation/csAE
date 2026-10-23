"""Suzuki et al. (2020) Maximum-Likelihood Amplitude Estimation, exponentially incremental sequence (EIS).

Reference:
    Y. Suzuki, S. Uno, R. Raymond, T. Tanaka, T. Onodera, and N. Yamamoto,
    "Amplitude estimation without phase estimation,"
    Quantum Information Processing 19, 75 (2020), arXiv:1904.10246.

This baseline follows the paper exactly (Sec. 3.2 and 3.3):
    * EIS: m_0 = 0 and m_k = 2^(k-1) for k = 1, ..., M;
    * N_k = N_shot = 100 shots for every circuit, as in the paper's simulations;
    * the estimate is the exact global maximum of the likelihood
      prod_k sin^2((2 m_k + 1) theta)^{h_k} cos^2((2 m_k + 1) theta)^{N_k - h_k}.
The benchmark scale fixes the only free parameter, M, through the deepest circuit
m_M = 2^(M-1) <= scale_nmax (the largest power of two not above the scale). Query
cost and error follow the benchmark conventions (cost n per shot at Grover depth
n >= 1, error in a = sin theta).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from leaderboard.schema_and_api import NonAdaptiveAESubmission, ml_estimate_batch


class MLAEPow2SuzukiSubmission(NonAdaptiveAESubmission):
    """MLAE with Suzuki et al.'s EIS schedule: m_k = 2^(k-1), N_shot = 100, global ML."""

    name = "MLAE-EIS (Suzuki et al. 2020)"
    track = "non-adaptive"

    N_SHOT = 100  # paper value; not a tunable

    def get_schedule(self, scale_nmax: int) -> tuple[list[int], list[int]]:
        """Return m_0 = 0, m_k = 2^(k-1) up to the largest power of two <= scale_nmax."""
        nmax = max(1, int(scale_nmax))
        depths: list[int] = [0]
        d = 1
        while d <= nmax:
            depths.append(int(d))
            d *= 2
        shots = [self.N_SHOT] * len(depths)
        return depths, shots

    def estimate(
        self,
        depths: Sequence[int],
        shots: Sequence[int],
        counts: np.ndarray,
    ) -> np.ndarray | float:
        """Exact global maximum-likelihood estimate (grid search plus local refinement)."""
        return ml_estimate_batch(depths, shots, counts)
