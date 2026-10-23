"""Labib (2026) Flagship Geometric Ladder + Cap Rung Maximum-Likelihood AE.

Reference:
    F. Labib, "Dense-ladder maximum-likelihood amplitude estimation,"
    arXiv:2609.02715 (2026).

Schedule:
    Geometric Grover depth ladder (r=1.45) plus one extra degeneracy-breaking
    rung at round(max(lad) / 1.3), with canonical linear shot decay [L, ..., 1]
    and exact global maximum-likelihood estimation.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from leaderboard.schema_and_api import NonAdaptiveAESubmission, ml_estimate_batch
from mlqae.core import flagship_schedule


class MLQAEFlagshipLabibSubmission(NonAdaptiveAESubmission):
    """Flagship geometric ladder (r=1.45 + cap rung at nmax/1.3) + global ML."""

    name = "mlqae-flagship (r=1.45 + cap)"
    track = "non-adaptive"

    def __init__(
        self,
        ratio: float = 1.45,
        cap_divisor: float = 1.3,
        base: int = 1,
        slope: float = 1.0,
    ) -> None:
        self.ratio = float(ratio)
        self.cap_divisor = float(cap_divisor)
        self.base = int(base)
        self.slope = float(slope)

    def get_schedule(self, scale_nmax: int) -> tuple[list[int], list[int]]:
        """Return flagship schedule (r=1.45 geometric ladder + cap rung, linear shots)."""
        depths, shots = flagship_schedule(
            int(scale_nmax),
            r=self.ratio,
            cap_divisor=self.cap_divisor,
            base=self.base,
            slope=self.slope,
        )
        return list(depths), list(shots)

    def estimate(
        self,
        depths: Sequence[int],
        shots: Sequence[int],
        counts: np.ndarray,
    ) -> np.ndarray | float:
        """Exact global maximum-likelihood estimation with 25-point parabolic refinement."""
        return ml_estimate_batch(depths, shots, counts)
