"""Labib et al. (2024) Compressed-Sensing / Sparse-Ruler Amplitude Estimation (csAE).

Reference:
    F. Labib, B. D. Clader, N. Stamatopoulos, and W. J. Zeng,
    "Quantum Amplitude Estimation from Classical Signal Processing,"
    Quantum (2024), arXiv:2405.14697.

Schedule & Estimator:
    - Power-of-two sparse-ruler Grover depths {0, 1, 2, 4, ..., 2^m <= scale_nmax}.
    - Linear shot allocation N_i = ceil(C * (L - i)) with doubled depth-0 shots
      N_0 = 2 * ceil(C * L) for C = 4.0 (matching csae/signals.py).
    - Sign recovery from learned/dyadic sign candidates + virtual Uniform Linear
      Array (ULA) coarray reconstruction + Toeplitz ESPRIT rotational invariance
      + 7-point quadrant/octave disambiguation and local likelihood refinement.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.linalg import toeplitz

from leaderboard.schema_and_api import NonAdaptiveAESubmission


def build_virtual_ula_and_esprit(
    signal: np.ndarray,
    max_ula_len: int = 32,
) -> float:
    """Reconstruct contiguous virtual ULA from dyadic depths and run Toeplitz ESPRIT.

    At depth n_0 = 0, z_0 ~ exp(i * 2*theta).
    At depth n_j = 2^(j-1) (j >= 1), z_j ~ exp(i * (4 * 2^(j-1) * theta + 2*theta)),
    so w_j = z_j * conj(z_0) ~ exp(i * 4 * 2^(j-1) * theta).
    Every integer lag u in 0..U_max-1 has binary expansion u = sum_r b_r 2^r,
    giving the contiguous virtual ULA y[u] = prod_{b_r=1} w_{r+1} ~ exp(i * 4 * u * theta).
    """
    num_dyadic = len(signal) - 1
    if num_dyadic <= 0:
        return 0.5
    z0_conj = np.conj(signal[0])
    w_dyadic = signal[1:] * z0_conj
    norms = np.abs(w_dyadic)
    w_unit = np.where(norms > 1e-12, w_dyadic / np.maximum(norms, 1e-12), 1.0 + 0.0j)

    ula_len = min(int(max_ula_len), 1 << min(num_dyadic, 6))
    y_ula = np.ones(ula_len, dtype=np.complex128)
    for bit_idx in range(min(num_dyadic, 6)):
        step_bit = 1 << bit_idx
        if step_bit >= ula_len:
            break
        indices = np.arange(ula_len)
        active = (indices & step_bit) != 0
        y_ula[active] *= w_unit[bit_idx]

    half_m = max(4, ula_len // 2)
    sub_col = y_ula[:half_m]
    cov_mat = toeplitz(sub_col, np.conj(sub_col))

    eig_vals, eig_vecs = np.linalg.eigh(cov_mat)
    top_vec = eig_vecs[:, int(np.argmax(eig_vals)) : int(np.argmax(eig_vals)) + 1]

    s1 = top_vec[:-1, :]
    s2 = top_vec[1:, :]
    denom = float(np.real(np.vdot(s1[:, 0], s1[:, 0])))
    if denom <= 1e-15:
        return 0.5
    phi_scalar = np.vdot(s1[:, 0], s2[:, 0]) / denom
    phase = float(np.angle(phi_scalar))
    if phase < 0.0:
        phase += 2.0 * math.pi
    theta_raw = 0.25 * phase
    return float(np.clip(theta_raw, 1e-6, math.pi / 2.0 - 1e-6))


def disambiguate_quadrant_and_refine(
    theta_est: float,
    depths_f: np.ndarray,
    shots_f: np.ndarray,
    counts_f: np.ndarray,
) -> tuple[float, float]:
    """Apply csAE quadrant/octave correction and 25-point parabolic likelihood polish."""
    d_odd = 2.0 * depths_f + 1.0
    base_cands = np.array(
        [
            theta_est,
            math.pi / 2.0 - theta_est,
            abs(math.pi / 4.0 - theta_est),
            0.5 * theta_est,
            0.25 * theta_est,
            math.pi / 2.0 - 0.5 * theta_est,
            abs(math.pi / 4.0 - 0.5 * theta_est),
        ],
        dtype=np.float64,
    )
    basin_step = math.pi / (2.0 * float(d_odd[-1]))
    shifts = np.arange(-4, 5, dtype=np.float64) * basin_step
    all_cands = np.clip((base_cands[:, None] + shifts[None, :]).ravel(), 1e-6, math.pi / 2.0 - 1e-6)

    p_mat = np.clip(np.cos(np.outer(all_cands, d_odd)) ** 2, 1e-12, 1.0 - 1e-12)
    nll = -(np.log(p_mat) @ counts_f + np.log1p(-p_mat) @ (shots_f - counts_f))
    best_th = float(all_cands[int(np.argmin(nll))])

    offs = np.linspace(-0.5 * basin_step, 0.5 * basin_step, 25)
    th_loc = np.clip(best_th + offs, 1e-6, math.pi / 2.0 - 1e-6)
    p_loc = np.clip(np.cos(np.outer(th_loc, d_odd)) ** 2, 1e-12, 1.0 - 1e-12)
    nll_loc = -(np.log(p_loc) @ counts_f + np.log1p(-p_loc) @ (shots_f - counts_f))
    j = int(np.clip(int(np.argmin(nll_loc)), 1, 23))
    ym1, y0, yp1 = float(nll_loc[j - 1]), float(nll_loc[j]), float(nll_loc[j + 1])
    den = ym1 - 2.0 * y0 + yp1
    off = 0.5 * (ym1 - yp1) / den if abs(den) > 1e-12 else 0.0
    dstep = float(offs[1] - offs[0])
    th_refined = float(np.clip(th_loc[j] + np.clip(off, -0.5, 0.5) * dstep, 1e-6, math.pi / 2.0 - 1e-6))
    p_ref = np.clip(np.cos(d_odd * th_refined) ** 2, 1e-12, 1.0 - 1e-12)
    nll_ref = -float(np.sum(counts_f * np.log(p_ref) + (shots_f - counts_f) * np.log1p(-p_ref)))
    return th_refined, nll_ref


def progressive_dyadic_sign_candidates(
    depths_f: np.ndarray,
    shots_f: np.ndarray,
    counts_f: np.ndarray,
    beam_width: int = 6,
) -> tuple[np.ndarray, float]:
    """Recover top candidate sign vectors and initial angle via multi-scale basin search."""
    d_odd = 2.0 * depths_f + 1.0
    s_basin = math.pi / (2.0 * float(d_odd[-1]))
    step = s_basin / 6.0
    grid = np.arange(1e-5, math.pi / 2.0 - 1e-5, step)
    p_mat = np.clip(np.cos(np.outer(grid, d_odd)) ** 2, 1e-12, 1.0 - 1e-12)
    nll = -(np.log(p_mat) @ counts_f + np.log1p(-p_mat) @ (shots_f - counts_f))

    order = np.argsort(nll)
    seeds: list[float] = []
    for idx in order:
        th_c = float(grid[int(idx)])
        if all(abs(th_c - prev) >= 0.75 * s_basin for prev in seeds):
            seeds.append(th_c)
            if len(seeds) >= beam_width:
                break

    if not seeds:
        seeds = [float(grid[int(order[0])])]

    seeds_arr = np.asarray(seeds, dtype=np.float64)
    signs_mat = np.where(np.sin(np.outer(seeds_arr, 2.0 * d_odd)) >= 0.0, 1.0, -1.0)
    return np.unique(signs_mat, axis=0), float(seeds[0])


class CSAEESPRITSubmission(NonAdaptiveAESubmission):
    """Compressed-Sensing Amplitude Estimation (Labib et al. 2024, arXiv:2405.14697)."""

    name = "csAE (Labib et al. 2024)"
    track = "non-adaptive"

    def __init__(self, shot_constant: float = 4.0) -> None:
        self.shot_constant = float(shot_constant)

    def get_schedule(self, scale_nmax: int) -> tuple[list[int], list[int]]:
        """Return csAE power-of-2 depths and C=4.0 linear shot ramp (with 2x at depth 0)."""
        nmax = max(1, int(scale_nmax))
        depths: list[int] = [0]
        d = 1
        while d <= nmax:
            depths.append(int(d))
            d *= 2
        num_levels = len(depths)
        shots: list[int] = [
            max(1, int(math.ceil(self.shot_constant * (num_levels - i))))
            for i in range(num_levels)
        ]
        shots[0] = shots[0] * 2
        return depths, shots

    def estimate_single_trial(
        self,
        depths_f: np.ndarray,
        shots_f: np.ndarray,
        counts_f: np.ndarray,
    ) -> float:
        """Run csAE sign recovery + virtual ULA Toeplitz ESPRIT + disambiguation on 1 trial."""
        d_odd = 2.0 * depths_f + 1.0
        p_hat = np.clip(counts_f / shots_f, 0.0, 1.0)
        cos_sig = 2.0 * p_hat - 1.0
        abs_sin = 2.0 * np.sqrt(np.maximum(0.0, p_hat * (1.0 - p_hat)))

        unique_signs, th_seed = progressive_dyadic_sign_candidates(depths_f, shots_f, counts_f)

        best_th, best_nll = disambiguate_quadrant_and_refine(th_seed, depths_f, shots_f, counts_f)

        for signs in unique_signs:
            sig = cos_sig + 1.0j * signs * abs_sin
            th_esp = build_virtual_ula_and_esprit(sig)
            th_corr, nll = disambiguate_quadrant_and_refine(th_esp, depths_f, shots_f, counts_f)
            if nll < best_nll:
                best_nll = nll
                best_th = th_corr

        # Single Hamming-1 sweep on the winning sign assignment (Step 5 of csae/estimator.py)
        best_signs = np.where(np.sin(2.0 * d_odd * best_th) >= 0.0, 1.0, -1.0)
        for flip_idx in range(len(depths_f)):
            cand_signs = best_signs.copy()
            cand_signs[flip_idx] = -cand_signs[flip_idx]
            sig = cos_sig + 1.0j * cand_signs * abs_sin
            th_esp = build_virtual_ula_and_esprit(sig)
            th_corr, nll = disambiguate_quadrant_and_refine(th_esp, depths_f, shots_f, counts_f)
            if nll < best_nll:
                best_nll = nll
                best_th = th_corr

        return float(np.clip(math.sin(best_th), 0.0, 1.0))

    def estimate(
        self,
        depths: Sequence[int],
        shots: Sequence[int],
        counts: np.ndarray,
    ) -> np.ndarray | float:
        """Estimate amplitude(s) for single trial (1D counts) or batch (2D counts)."""
        depths_f = np.asarray(depths, dtype=np.float64)
        shots_f = np.asarray(shots, dtype=np.float64)
        counts_arr = np.asarray(counts, dtype=np.float64)

        if counts_arr.ndim == 1:
            return self.estimate_single_trial(depths_f, shots_f, counts_arr)

        ntrials = counts_arr.shape[0]
        out = np.empty(ntrials, dtype=np.float64)
        for i in range(ntrials):
            out[i] = self.estimate_single_trial(depths_f, shots_f, counts_arr[i])
        return out
