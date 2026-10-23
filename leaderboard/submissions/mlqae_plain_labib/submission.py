"""Labib (2026) Plain Geometric Ladder Maximum-Likelihood Amplitude Estimation.

Reference:
    F. Labib, "Dense-ladder maximum-likelihood amplitude estimation,"
    arXiv:2609.02715 (2026).

Schedule:
    Plain geometric Grover depth ladder {0} U {round(1.45^j) <= scale_nmax}
    with canonical linear shot decay [L, L-1, ..., 1] and exact global
    maximum-likelihood estimation.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from leaderboard.schema_and_api import NonAdaptiveAESubmission, ml_estimate_batch
from mlqae.core import canonical_shots, geom_ladder


class MLQAEPlainLabibSubmission(NonAdaptiveAESubmission):
    """Plain geometric ladder (r=1.45) + global ML (arXiv:2609.02715)."""

    name = "mlqae-plain (r=1.45)"
    track = "non-adaptive"

    def __init__(self, ratio: float = 1.45, base: int = 1, slope: float = 1.0) -> None:
        self.ratio = float(ratio)
        self.base = int(base)
        self.slope = float(slope)

    def get_schedule(self, scale_nmax: int) -> tuple[list[int], list[int]]:
        """Return plain r=1.45 geometric ladder depths and canonical linear shots."""
        depths = geom_ladder(int(scale_nmax), self.ratio)
        shots = canonical_shots(len(depths), base=self.base, slope=self.slope)
        return list(depths), list(shots)

    def estimate(
        self,
        depths: Sequence[int],
        shots: Sequence[int],
        counts: np.ndarray,
    ) -> np.ndarray | float:
        """Exact global maximum-likelihood estimation with 25-point parabolic refinement."""
        return ml_estimate_batch(depths, shots, counts)
