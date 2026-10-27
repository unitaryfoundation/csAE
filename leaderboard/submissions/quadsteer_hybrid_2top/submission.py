"""QuadSteer-Hybrid-2Top: Quadrature-Steered Adaptive Geometric Ladder for Quantum AE.

Batches all unsteered shallow anchor rungs (n < 4) into a single parallel round via
`oracle.measure_batch`, then adaptively steers each individual shot on the deeper rungs
(n >= 4) onto opposite-slope quadrature crossings (cos(2(2n+1)theta_est) ≈ 0, p ≈ 0.5)
of the local-basin Boltzmann posterior centroid, with trajectory-tracked query budget
pinning so that Q_max ≈ Q_ave.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from leaderboard.schema_and_api import AdaptiveAESubmission
from mlqae.core import grid_spacing


SCHEDULE_CACHE: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, int, np.ndarray]] = {}
ZOOM_OFFSETS: np.ndarray = np.linspace(-0.5, 0.5, 25, dtype=np.float64)


def build_hybrid_2top_schedule(
    nmax: int,
    r: float = 1.60,
    top_shots: Tuple[int, ...] = (4, 3, 2, 2, 2),
    shallow_slope: float = 0.90,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct top-anchored r=1.60 geometric ladder with 2 top shots and shallow floor."""
    ds: List[int] = [int(nmax)]
    x = float(nmax)
    while True:
        x = x / r
        n_next = int(round(x))
        if n_next >= ds[-1]:
            n_next = ds[-1] - 1
        if n_next <= 0:
            break
        ds.append(n_next)
    for anchor in (0, 1, 2, 3):
        ds.append(anchor)
    unique_ds = np.array(sorted(list(set(ds))), dtype=np.int64)
    num_levels = len(unique_ds)
    shots = np.empty(num_levels, dtype=np.int64)
    ntop = len(top_shots)
    for j in range(num_levels):
        d = int(unique_ds[j])
        rem = num_levels - 1 - j
        if rem < ntop:
            s_val = int(top_shots[ntop - 1 - rem])
        else:
            s_val = min(15, int(round(top_shots[0] + shallow_slope * (rem - ntop + 1))))
        if d == 0:
            s_val = max(s_val, 12)
        elif d == 1:
            s_val = max(s_val, 11)
        elif d == 2:
            s_val = max(s_val, 10)
        elif d == 3:
            s_val = max(s_val, 9)
        elif d <= 5:
            s_val = max(s_val, 8)
        elif d <= 8:
            s_val = max(s_val, 6)
        elif d <= 13 and rem >= 3:
            s_val = max(s_val, 5)
        shots[j] = s_val
    return unique_ds, shots


def get_cached_tables(
    nmax: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, int, np.ndarray]:
    """Return cached schedule, likelihood grid, and nominal remaining budget tables."""
    key = int(nmax)
    cached = SCHEDULE_CACHE.get(key)
    if cached is not None:
        return cached

    base_depths, base_shots = build_hybrid_2top_schedule(key)
    step = float(grid_spacing(base_depths, base_shots))
    grid = np.arange(1e-6, 0.5 * math.pi, step, dtype=np.float64)
    max_possible_n = int(math.ceil(1.45 * int(base_depths[-1]))) + 25
    d_all = 2.0 * np.arange(max_possible_n + 1, dtype=np.float64)[:, None] + 1.0
    probs = np.clip(np.cos(d_all * grid[None, :]) ** 2, 1e-12, 1.0 - 1e-12)
    log_p = np.log(probs)
    log_1mp = np.log1p(-probs)

    q_target = int(np.sum(base_depths * base_shots) + base_shots[0])
    num_levels = len(base_depths)
    q_after = np.zeros(num_levels, dtype=np.int64)
    acc = 0
    for j in range(num_levels - 1, -1, -1):
        q_after[j] = acc
        d_val = int(base_depths[j])
        cost = 1 if d_val == 0 else d_val
        acc += int(base_shots[j]) * cost

    out = (base_depths, base_shots, grid, step, log_p, log_1mp, q_target, q_after)
    SCHEDULE_CACHE[key] = out
    return out


def local_boltzmann_centroid(
    nll: np.ndarray,
    grid: np.ndarray,
    step: float,
    cur_d_max: int,
    nmax: int,
) -> Tuple[float, int]:
    """Compute the Boltzmann-weighted centroid within the local basin around the MAP peak."""
    n_grid = len(grid)
    stride = max(1, int(nmax // (2 * max(4, cur_d_max))))
    if stride > 1:
        coarse_slice = nll[1 : n_grid - 1 : stride]
        coarse_idx = int(np.argmin(coarse_slice))
        best_g = 1 + coarse_idx * stride
        lo_s = max(1, best_g - 2 * stride)
        hi_s = min(n_grid - 2, best_g + 2 * stride)
        fine_slice = nll[lo_s : hi_s + 1]
        best_g = lo_s + int(np.argmin(fine_slice))
    else:
        best_g = 1 + int(np.argmin(nll[1 : n_grid - 1]))

    best_v = float(nll[best_g])
    hw_g = max(4, int(math.ceil((math.pi / (2.0 * (2.0 * max(1, cur_d_max) + 1.0))) / step)))
    lo_g = max(1, best_g - hw_g)
    hi_g = min(n_grid - 2, best_g + hw_g)

    local_diff = nll[lo_g : hi_g + 1] - best_v
    mask = local_diff < 9.0
    if np.any(mask):
        weights = np.exp(-local_diff[mask])
        w_sum = float(np.sum(weights))
        if w_sum > 0.0:
            th_mean = float(np.dot(weights, grid[lo_g : hi_g + 1][mask]) / w_sum)
            return th_mean, best_g
    return float(grid[best_g]), best_g


def zoom_parabolic_refine(
    th0: float,
    step: float,
    depths: np.ndarray,
    shots: np.ndarray,
    counts: np.ndarray,
) -> float:
    """25-point local zoom across [-step/2, +step/2] + parabolic interpolation."""
    th_loc = th0 + step * ZOOM_OFFSETS
    d_odd = 2.0 * depths.astype(np.float64) + 1.0
    p = np.clip(np.cos(np.outer(th_loc, d_odd)) ** 2, 1e-12, 1.0 - 1e-12)
    k_f = counts.astype(np.float64)
    nk_f = (shots - counts).astype(np.float64)
    z_nlls = -(np.log(p) @ k_f + np.log1p(-p) @ nk_f)
    best_z = int(np.clip(np.argmin(z_nlls), 1, 23))
    ym1 = float(z_nlls[best_z - 1])
    y0 = float(z_nlls[best_z])
    yp1 = float(z_nlls[best_z + 1])
    den = ym1 - 2.0 * y0 + yp1
    off = 0.0
    if abs(den) > 1e-12:
        off = 0.5 * (ym1 - yp1) / max(den, 1e-12)
        off = min(max(off, -0.5), 0.5)
    dstep = step / 24.0
    return float(th_loc[best_z]) + off * dstep


class QuadSteerHybrid2TopSubmission(AdaptiveAESubmission):
    """QuadSteer-Hybrid-2Top (r=1.60 Quadrature-Steered Adaptive Geometric Ladder)."""

    name = "QuadSteer-Hybrid-2Top (r=1.60 Quadrature-Steered Ladder)"
    track = "adaptive-fixed-budget"

    def __init__(self, min_steer_depth: int = 4, win_frac: float = 0.15) -> None:
        self.min_steer_depth = int(min_steer_depth)
        self.win_frac = float(win_frac)

    def run_trial(
        self,
        oracle: Any,
        target_scale_or_budget: int,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        _ = rng
        nmax = max(1, int(target_scale_or_budget))
        (
            base_depths,
            base_shots,
            grid,
            step,
            log_p,
            log_1mp,
            q_target,
            q_after,
        ) = get_cached_tables(nmax)

        num_levels = len(base_depths)
        n_grid = len(grid)
        max_n = len(log_p) - 1

        # Respect any hard oracle budget or depth cap if configured
        budget_cap = oracle.max_query_budget
        eff_q_target = q_target if budget_cap is None else min(q_target, int(budget_cap))
        depth_cap = oracle.max_depth_cap
        if depth_cap is not None:
            max_n = min(max_n, int(depth_cap))

        rec_depths: List[int] = []
        rec_shots: List[int] = []
        rec_counts: List[int] = []

        nll = np.zeros(n_grid, dtype=np.float64)
        q_tot = 0
        d_max = 0

        # Stage 1: Batch all unsteered shallow rungs (n < min_steer_depth) in 1 parallel round
        shallow_mask = base_depths < self.min_steer_depth
        n_shallow = int(np.sum(shallow_mask))
        if n_shallow > 0:
            sh_depths = [int(d) for d in base_depths[:n_shallow]]
            sh_shots = [int(s) for s in base_shots[:n_shallow]]
            sh_counts_arr = oracle.measure_batch(sh_depths, sh_shots)
            for idx in range(n_shallow):
                d_val = sh_depths[idx]
                s_val = sh_shots[idx]
                k_val = int(sh_counts_arr[idx])
                rec_depths.append(d_val)
                rec_shots.append(s_val)
                rec_counts.append(k_val)
                q_tot += s_val * (1 if d_val == 0 else d_val)
                if d_val > d_max:
                    d_max = d_val
                nll -= k_val * log_p[d_val] + (s_val - k_val) * log_1mp[d_val]

        # Stage 2: Per-shot quadrature steering on deep rungs (n >= min_steer_depth)
        for j in range(n_shallow, num_levels):
            s = int(base_shots[j])
            n_bar = int(base_depths[j])
            prev_n = -1
            prev_sign = 0.0
            for sh in range(s):
                th_est, _ = local_boltzmann_centroid(nll, grid, step, d_max, nmax)
                is_last_shot = (j == num_levels - 1) and (sh == s - 1)
                if is_last_shot:
                    n_cap = min(max_n, max(1, eff_q_target - q_tot))
                    delta = max(6, int(round(0.06 * n_bar)))
                    n_lo = max(1, n_cap - delta)
                    n_hi = n_cap
                elif j >= num_levels - 2:
                    rem_shots_here = s - sh
                    rem_nom = rem_shots_here * n_bar + int(q_after[j])
                    rem_act = eff_q_target - q_tot
                    tot_rem_sh = float(
                        rem_shots_here + (int(base_shots[num_levels - 1]) if j == num_levels - 2 else 0)
                    )
                    n_cen = int(round(n_bar + 0.6 * (rem_act - rem_nom) / tot_rem_sh))
                    hw = max(3, int(math.ceil(self.win_frac * n_bar)))
                    n_lo = max(1, n_cen - hw)
                    n_hi = min(max_n, n_cen + hw)
                else:
                    hw = max(2, int(math.ceil(self.win_frac * n_bar)))
                    n_lo = max(1, n_bar - hw)
                    n_hi = min(max_n, n_bar + hw)

                if budget_cap is not None:
                    rem_q = int(budget_cap) - q_tot
                    if rem_q < 1:
                        break
                    n_hi = min(n_hi, rem_q)
                    n_lo = min(n_lo, n_hi)

                best_n = n_hi
                best_sc = 1e300
                for cand_n in range(n_lo, n_hi + 1):
                    d_c = 2.0 * cand_n + 1.0
                    phase = 2.0 * d_c * th_est
                    sc = abs(math.cos(phase))
                    if prev_sign != 0.0 and (math.sin(phase) * prev_sign > 0.0):
                        sc += 0.20
                    if cand_n == prev_n:
                        sc += 0.15
                    if sc < best_sc:
                        best_sc = sc
                        best_n = cand_n

                n = int(best_n)
                prev_n = n
                prev_sign = math.sin(2.0 * (2.0 * n + 1.0) * th_est)
                if n > d_max:
                    d_max = n

                k = int(oracle.measure(n, 1))
                rec_depths.append(n)
                rec_shots.append(1)
                rec_counts.append(k)
                q_tot += n
                if k == 1:
                    nll -= log_p[n]
                else:
                    nll -= log_1mp[n]

        best_g = 1 + int(np.argmin(nll[1 : n_grid - 1]))
        th_final = zoom_parabolic_refine(
            float(grid[best_g]),
            step,
            np.asarray(rec_depths, dtype=np.int64),
            np.asarray(rec_shots, dtype=np.int64),
            np.asarray(rec_counts, dtype=np.int64),
        )
        return float(np.clip(math.sin(th_final), 0.0, 1.0))
