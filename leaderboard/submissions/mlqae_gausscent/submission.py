"""MLQAE-GaussCent: Flagship Non-Adaptive Schedule + Local Gaussian-Windowed Centroid.

Replaces the point-wise global maximum-likelihood terminal readout with a
continuous two-stage estimator:
1. Global coarse-grid maximum likelihood + 25-point parabolic zoom locates the
   winning mode theta_mle.
2. Fine-grid local posterior integration weighted by a Gaussian window of width
   c = 4.5 * sigma_theta restricted to [-0.75 * w_basin, +0.75 * w_basin] around
   theta_mle reports E_loc[sin(theta)]. This is the continuous Bayes rule for a
   smooth local Gaussian tolerance utility: it integrates out few-shot log-cosine
   skewness and averages near-tied mirror modes near deepest-fringe extrema while
   exponentially suppressing distant rival basins.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np

from leaderboard.schema_and_api import NonAdaptiveAESubmission
from mlqae.core import flagship_schedule, grid_spacing


class MLQAEGaussCentSubmission(NonAdaptiveAESubmission):
    """Flagship non-adaptive ladder with Local Gaussian-Windowed Posterior Centroid readout."""

    name: str = "MLQAE-GaussCent (Flagship + Local Gaussian Centroid)"
    track: str = "non-adaptive"

    def __init__(
        self,
        r: float = 1.45,
        cap_divisor: float = 1.3,
        base: int = 1,
        slope: float = 1.0,
        window_sigma: float = 6.0,
        max_basin_frac: float = 0.75,
    ) -> None:
        self.r = float(r)
        self.cap_divisor = float(cap_divisor)
        self.base = int(base)
        self.slope = float(slope)
        self.window_sigma = float(window_sigma)
        self.max_basin_frac = float(max_basin_frac)
        self.schedule_cache: Dict[int, Tuple[List[int], List[int]]] = {}
        self.table_cache: Dict[
            Tuple[Tuple[int, ...], Tuple[int, ...]],
            Tuple[
                np.ndarray,
                float,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
            ],
        ] = {}
        self.zoom_unit = np.linspace(-0.5, 0.5, 25, dtype=np.float64)

    def get_schedule(self, scale_nmax: int) -> Tuple[List[int], List[int]]:
        """Return flagship schedule (r=1.45 + near-cap rung at round(nmax/1.3))."""
        key = int(scale_nmax)
        if key in self.schedule_cache:
            return self.schedule_cache[key]
        depths, shots = flagship_schedule(
            key,
            r=self.r,
            cap_divisor=self.cap_divisor,
            base=self.base,
            slope=self.slope,
        )
        out = ([int(d) for d in depths], [int(s) for s in shots])
        self.schedule_cache[key] = out
        return out

    def get_tables(
        self,
        depths: Sequence[int],
        shots: Sequence[int],
    ) -> Tuple[
        np.ndarray,
        float,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Return cached coarse-grid and local-window tables for (depths, shots)."""
        key = (tuple(int(d) for d in depths), tuple(int(s) for s in shots))
        if key in self.table_cache:
            return self.table_cache[key]

        d_arr = np.asarray(key[0], dtype=np.float64)
        s_arr = np.asarray(key[1], dtype=np.float64)
        d_odd = 2.0 * d_arr + 1.0
        step = float(grid_spacing(d_arr, s_arr))
        grid = np.arange(1e-6, 0.5 * math.pi, step, dtype=np.float64)
        p_grid = np.clip(np.cos(np.outer(grid, d_odd)) ** 2, 1e-12, 1.0 - 1e-12)
        log_p = np.log(p_grid)
        log_1mp = np.log1p(-p_grid)

        sig = float(1.0 / math.sqrt(4.0 * float(np.sum(s_arr * (d_odd ** 2)))))
        w_basin = float(0.5 * math.pi / float(np.max(d_odd)))
        half_span = min(6.5 * sig, self.max_basin_frac * w_basin)
        offs_loc = np.linspace(-half_span, half_span, 81, dtype=np.float64)
        g_win = np.exp(-0.5 * (offs_loc / (self.window_sigma * sig)) ** 2)

        tables = (grid, step, d_odd, s_arr, log_p, log_1mp, offs_loc, g_win)
        self.table_cache[key] = tables
        return tables

    def estimate(
        self,
        depths: Sequence[int],
        shots: Sequence[int],
        counts: np.ndarray,
    ) -> Union[np.ndarray, float]:
        """Compute Local Gaussian-Windowed Posterior Centroid amplitude estimate(s)."""
        grid, step, d_odd, s_arr, log_p, log_1mp, offs_loc, g_win = self.get_tables(depths, shots)
        counts_arr = np.asarray(counts, dtype=np.float64)
        single_input = counts_arr.ndim == 1

        if single_input:
            k_vec = counts_arr
            nll = -(log_p @ k_vec + log_1mp @ (s_arr - k_vec))
            idx = int(np.clip(np.argmin(nll), 1, len(grid) - 2))
            th0 = float(grid[idx])

            # Stage 1: 25-point local zoom + parabolic interpolation
            th_zoom = th0 + step * self.zoom_unit
            p_zoom = np.clip(np.cos(np.outer(th_zoom, d_odd)) ** 2, 1e-12, 1.0 - 1e-12)
            nll_z = -(np.log(p_zoom) @ k_vec + np.log1p(-p_zoom) @ (s_arr - k_vec))
            j = int(np.clip(np.argmin(nll_z), 1, 23))
            ym1, y0, yp1 = float(nll_z[j - 1]), float(nll_z[j]), float(nll_z[j + 1])
            den = ym1 - 2.0 * y0 + yp1
            off = 0.0
            if abs(den) > 1e-12:
                off = 0.5 * (ym1 - yp1) / max(den, 1e-12)
                off = min(max(off, -0.5), 0.5)
            th_mle = float(th_zoom[j]) + off * (step / 24.0)
            th_mle = min(max(th_mle, 1e-6), 0.5 * math.pi - 1e-6)

            # Stage 2: Fine-grid Gaussian-windowed local posterior centroid
            tl = np.clip(th_mle + offs_loc, 1e-6, 0.5 * math.pi - 1e-6)
            p_loc = np.clip(np.cos(np.outer(tl, d_odd)) ** 2, 1e-12, 1.0 - 1e-12)
            ll_loc = np.log(p_loc) @ k_vec + np.log1p(-p_loc) @ (s_arr - k_vec)
            ll_loc = ll_loc - float(np.max(ll_loc))
            w_loc = np.exp(ll_loc) * g_win
            w_sum = float(np.sum(w_loc))
            if w_sum <= 1e-300:
                return float(np.clip(math.sin(th_mle), 0.0, 1.0))
            a_hat = float(np.sum(w_loc * np.sin(tl)) / w_sum)
            return float(np.clip(a_hat, 0.0, 1.0))

        # Vectorized 2D batch evaluation
        ntrials = counts_arr.shape[0]
        out = np.empty(ntrials, dtype=np.float64)
        chunk = 2000
        dstep = step / 24.0
        for lo in range(0, ntrials, chunk):
            hi = min(lo + chunk, ntrials)
            kc = counts_arr[lo:hi]
            nll = -(kc @ log_p.T + (s_arr[None, :] - kc) @ log_1mp.T)
            idx = np.clip(np.argmin(nll, axis=1), 1, len(grid) - 2)
            th0 = grid[idx]

            th_zoom = th0[:, None] + step * self.zoom_unit[None, :]
            p_zoom = np.clip(
                np.cos(th_zoom[:, :, None] * d_odd[None, None, :]) ** 2,
                1e-12,
                1.0 - 1e-12,
            )
            nll_z = -np.sum(
                kc[:, None, :] * np.log(p_zoom)
                + (s_arr[None, None, :] - kc[:, None, :]) * np.log1p(-p_zoom),
                axis=2,
            )
            j = np.clip(np.argmin(nll_z, axis=1), 1, 23)
            ar = np.arange(hi - lo)
            ym1, y0, yp1 = nll_z[ar, j - 1], nll_z[ar, j], nll_z[ar, j + 1]
            den = ym1 - 2.0 * y0 + yp1
            off = np.where(np.abs(den) > 1e-12, 0.5 * (ym1 - yp1) / np.maximum(den, 1e-12), 0.0)
            th_mle = np.clip(
                th_zoom[ar, j] + np.clip(off, -0.5, 0.5) * dstep,
                1e-6,
                0.5 * math.pi - 1e-6,
            )

            tl = np.clip(th_mle[:, None] + offs_loc[None, :], 1e-6, 0.5 * math.pi - 1e-6)
            p_loc = np.clip(
                np.cos(tl[:, :, None] * d_odd[None, None, :]) ** 2,
                1e-12,
                1.0 - 1e-12,
            )
            ll_loc = np.sum(
                kc[:, None, :] * np.log(p_loc)
                + (s_arr[None, None, :] - kc[:, None, :]) * np.log1p(-p_loc),
                axis=2,
            )
            ll_loc = ll_loc - np.max(ll_loc, axis=1, keepdims=True)
            w_loc = np.exp(ll_loc) * g_win[None, :]
            w_sum = np.maximum(np.sum(w_loc, axis=1, keepdims=True), 1e-300)
            w_loc = w_loc / w_sum
            out[lo:hi] = np.clip(np.sum(w_loc * np.sin(tl), axis=1), 0.0, 1.0)

        return out
