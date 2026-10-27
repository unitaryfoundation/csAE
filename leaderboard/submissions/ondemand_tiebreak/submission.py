"""On-Demand Bhattacharyya Tie-Breaker (OnDemand-UltraLean) Submission.

Implements an adaptive variable-query Maximum-Likelihood Amplitude Estimation
algorithm:
1. Executes a lean-mid / strong-shallow baseline schedule on the plain r=1.45
   geometric ladder in 1 parallel round via oracle.measure_batch(base_depths, base_shots),
   eliminating the permanent cap-divisor rung tax.
2. Evaluates the exact log-likelihood l(theta) on a fine grid and identifies
   the primary peak's local basin by walking monotonically left and right to the
   first local minima, capped at the Fisher-adaptive merge width
   w_merge = c_w * sqrt(2 * tau_r) * sigma_theta(N) to catch flattened/merged
   mirror peaks near extrema of d_max.
3. Computes the log-likelihood gap Delta_l = l(theta_1) - l(theta_2) to the
   highest rival mode theta_2 outside the local basin.
4. Only when Delta_l < tau_r, dynamically fires a disambiguating shot at the
   candidate depth n* <= d_max maximizing the boundary-regularized (eps=0.04)
   Bhattacharyya distance per query:
       score(n) = -ln BC_eps(p_n(theta_1), p_n(theta_2)) / shot_cost(n)
   subject to the per-trial remaining query budget.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from leaderboard.schema_and_api import AdaptiveAESubmission, QuantumAEOracle
from mlqae.core import flagship_schedule, grid_spacing

ZOOM_OFFSETS_25 = np.linspace(-0.5, 0.5, 25, dtype=np.float64)
PLAN_CACHE: Dict[Tuple[int, float], Dict[str, Any]] = {}


def build_ultralean_plan(
    nmax: int,
    eta: float = 0.0,
    s_shallow: float = 0.65,
    top_tuple: Tuple[int, ...] = (2, 2, 2, 1, 1),
) -> Dict[str, Any]:
    """Construct and cache the baseline schedule, candidate pool, and log-likelihood tables."""
    key = (int(nmax), round(float(eta), 9))
    cached = PLAN_CACHE.get(key)
    if cached is not None:
        return cached

    lad_p, _ = flagship_schedule(int(nmax), r=1.45, cap_divisor=None)
    dmax = int(max(lad_p))

    d_hi = list(range(max(1, int(0.50 * dmax)), dmax + 1, 1))
    d_mid = list(range(max(1, int(0.15 * dmax)), max(1, int(0.50 * dmax)), max(1, dmax // 80)))
    cand_list = sorted(set(int(d) for d in (lad_p + d_mid + d_hi)))
    cand_depths = np.asarray(cand_list, dtype=np.int64)

    num_lad = len(lad_p)
    k_top = len(top_tuple)
    s_cand = np.zeros(len(cand_list), dtype=np.int64)
    base_depths_list: List[int] = []
    base_shots_list: List[int] = []
    base_cand_indices: List[int] = []

    for j, d_val in enumerate(lad_p):
        from_top = num_lad - 1 - j
        if from_top < k_top:
            sh = int(top_tuple[k_top - 1 - from_top])
        else:
            sh = max(
                int(top_tuple[0]),
                int(round(top_tuple[0] + s_shallow * (from_top - (k_top - 1)))),
            )
        c_idx = cand_list.index(int(d_val))
        s_cand[c_idx] = sh
        base_depths_list.append(int(d_val))
        base_shots_list.append(sh)
        base_cand_indices.append(c_idx)

    base_depths = np.asarray(base_depths_list, dtype=np.int64)
    base_shots = np.asarray(base_shots_list, dtype=np.int64)
    base_shots_f = base_shots.astype(np.float64)
    base_idx_arr = np.asarray(base_cand_indices, dtype=np.int64)

    step = float(grid_spacing(cand_depths, s_cand) * 0.65)
    grid = np.arange(1e-6, 0.5 * math.pi, step, dtype=np.float64)

    cand_d = 2.0 * cand_depths.astype(np.float64) + 1.0
    cand_d2 = 4.0 * cand_d * cand_d
    cand_costs = np.where(cand_depths == 0, 1.0, cand_depths.astype(np.float64))

    cos2_mat = np.cos(np.outer(cand_d, grid)) ** 2
    if eta > 0.0:
        vis = ((1.0 - float(eta)) ** cand_d)[:, None]
        p_mat = np.clip(vis * cos2_mat + 0.5 * (1.0 - vis), 1e-12, 1.0 - 1e-12)
    else:
        p_mat = np.clip(cos2_mat, 1e-12, 1.0 - 1e-12)

    log_p = np.log(p_mat)
    log_1mp = np.log1p(-p_mat)

    base_log_p = log_p[base_idx_arr]
    base_log_1mp = log_1mp[base_idx_arr]
    base_logit_p = np.ascontiguousarray(base_log_p - base_log_1mp)
    base_const_ll = np.asarray(base_shots_f @ base_log_1mp, dtype=np.float64)

    base_fisher = float(np.sum(s_cand.astype(np.float64) * cand_d2))
    base_queries = float(np.sum(np.where(base_depths == 0, 1.0, base_depths.astype(np.float64)) * base_shots_f))

    plan = {
        "nmax": int(nmax),
        "dmax": dmax,
        "eta": float(eta),
        "base_depths": base_depths,
        "base_shots": base_shots,
        "base_shots_f": base_shots_f,
        "base_idx_arr": base_idx_arr,
        "cand_depths": cand_depths,
        "cand_d": cand_d,
        "cand_d2": cand_d2,
        "cand_costs": cand_costs,
        "grid": grid,
        "step": step,
        "log_p": log_p,
        "log_1mp": log_1mp,
        "base_logit_p": base_logit_p,
        "base_const_ll": base_const_ll,
        "base_fisher": base_fisher,
        "base_queries": base_queries,
    }
    PLAN_CACHE[key] = plan
    return plan


def find_rival_and_gap_fast(
    ll: np.ndarray,
    step: float,
    sigma_th: float,
    tau: float,
    c_w: float,
) -> Tuple[int, int, float]:
    """Identify global MLE index j1 and highest rival mode j2 outside the local basin."""
    num_grid = int(len(ll))
    j1 = int(np.argmax(ll))
    best_ll = float(ll[j1])

    w_merge = c_w * math.sqrt(2.0 * tau) * sigma_th
    w_merge_idx = max(2, int(round(w_merge / step)))

    j_left = j1
    min_left = max(0, j1 - w_merge_idx)
    while j_left > min_left and ll[j_left - 1] <= ll[j_left]:
        j_left -= 1

    j_right = j1
    max_right = min(num_grid - 1, j1 + w_merge_idx)
    while j_right < max_right and ll[j_right + 1] <= ll[j_right]:
        j_right += 1

    best_riv_ll = -1e300
    j2 = j1
    if j_left > 0:
        jl = int(np.argmax(ll[:j_left]))
        val_l = float(ll[jl])
        if val_l > best_riv_ll:
            best_riv_ll = val_l
            j2 = jl
    if j_right + 1 < num_grid:
        jr = int(np.argmax(ll[j_right + 1 :])) + j_right + 1
        val_r = float(ll[jr])
        if val_r > best_riv_ll:
            best_riv_ll = val_r
            j2 = jr

    return j1, j2, best_ll - best_riv_ll


def pick_best_depth_bc_fast(
    th1: float,
    th2: float,
    cand_d: np.ndarray,
    cand_costs: np.ndarray,
    cost_exp: float,
    eps_reg: float,
    q_rem: float,
) -> int:
    """Select candidate depth index maximizing regularized -ln BC / cost^cost_exp."""
    mask = cand_costs <= q_rem
    if not np.any(mask):
        return -1

    valid_indices = np.flatnonzero(mask)
    d_sub = cand_d[valid_indices]
    costs_sub = cand_costs[valid_indices]

    c1 = np.cos(d_sub * th1)
    c2 = np.cos(d_sub * th2)
    scale = 1.0 - 2.0 * eps_reg
    p1 = eps_reg + scale * (c1 * c1)
    p2 = eps_reg + scale * (c2 * c2)

    bc = np.sqrt(p1 * p2) + np.sqrt((1.0 - p1) * (1.0 - p2))
    bc = np.clip(bc, 1e-15, 1.0)
    scores = -np.log(bc) / (costs_sub if cost_exp == 1.0 else (costs_sub ** cost_exp))
    best_local = int(np.argmax(scores))
    if float(scores[best_local]) <= 0.0:
        return -1
    return int(valid_indices[best_local])


def local_refine_zoom25(
    th0: float,
    step: float,
    active_d: np.ndarray,
    active_k: np.ndarray,
    active_n: np.ndarray,
    eta: float = 0.0,
) -> float:
    """25-point local zoom around th0 + parabolic interpolation matching mlqae/core.py."""
    th_loc = th0 + step * ZOOM_OFFSETS_25
    cos2 = np.cos(np.outer(th_loc, active_d)) ** 2
    if eta > 0.0:
        vis = ((1.0 - eta) ** active_d)[None, :]
        p = np.clip(vis * cos2 + 0.5 * (1.0 - vis), 1e-12, 1.0 - 1e-12)
    else:
        p = np.clip(cos2, 1e-12, 1.0 - 1e-12)

    nll = -(np.log(p) @ active_k + np.log1p(-p) @ (active_n - active_k))
    best_j = int(np.argmin(nll))
    if best_j <= 0:
        best_j = 1
    elif best_j >= 24:
        best_j = 23

    ym1 = float(nll[best_j - 1])
    y0 = float(nll[best_j])
    yp1 = float(nll[best_j + 1])
    den = ym1 - 2.0 * y0 + yp1
    off = 0.0
    if abs(den) > 1e-12:
        off = 0.5 * (ym1 - yp1) / max(den, 1e-12)
        if off < -0.5:
            off = -0.5
        elif off > 0.5:
            off = 0.5
    dstep = step / 24.0
    th_refined = float(th_loc[best_j]) + off * dstep
    return min(max(th_refined, 0.0), 0.5 * math.pi)


class OnDemandTieBreakSubmission(AdaptiveAESubmission):
    """OnDemand-UltraLean: Lean-Mid Baseline + Boundary-Regularized Bhattacharyya Tie-Breaker."""

    name = "OnDemand-UltraLean (Bhattacharyya Tie-Breaker)"
    track = "adaptive-variable"

    def __init__(
        self,
        s_shallow: float = 0.65,
        top_tuple: Tuple[int, ...] = (2, 2, 2, 1, 1),
        tau_schedule: Sequence[float] = (5.8, 4.4, 3.7, 3.1, 2.5, 2.0),
        q_mult: float = 3.5,
        c_w: float = 1.20,
        cost_exp: float = 1.00,
        eps_reg: float = 0.04,
    ) -> None:
        self.s_shallow = float(s_shallow)
        self.top_tuple = tuple(int(x) for x in top_tuple)
        self.tau_arr = np.asarray(tau_schedule, dtype=np.float64)
        self.q_mult = float(q_mult)
        self.c_w = float(c_w)
        self.cost_exp = float(cost_exp)
        self.eps_reg = float(eps_reg)

    def run_trial(
        self,
        oracle: QuantumAEOracle,
        target_scale_or_budget: int,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """Execute 1 baseline batch round followed by on-demand Bhattacharyya tie-breaking."""
        scale_nmax = max(1, int(target_scale_or_budget))
        depth_cap = oracle.max_depth_cap
        if depth_cap is not None and int(depth_cap) > 0:
            scale_nmax = min(scale_nmax, int(depth_cap))

        eta_val = float(oracle.eta) if oracle.eta is not None else 0.0
        plan = build_ultralean_plan(
            scale_nmax,
            eta=eta_val,
            s_shallow=self.s_shallow,
            top_tuple=self.top_tuple,
        )

        base_depths = plan["base_depths"]
        base_shots = plan["base_shots"]
        base_counts = oracle.measure_batch(base_depths, base_shots)
        counts_f = np.asarray(base_counts, dtype=np.float64)

        ll = plan["base_const_ll"] + counts_f @ plan["base_logit_p"]

        cand_depths = plan["cand_depths"]
        cand_d = plan["cand_d"]
        cand_d2 = plan["cand_d2"]
        cand_costs = plan["cand_costs"]
        grid = plan["grid"]
        step = plan["step"]
        log_p = plan["log_p"]
        log_1mp = plan["log_1mp"]

        cur_fisher = float(plan["base_fisher"])
        q_rem = self.q_mult * float(plan["dmax"])
        rem_oracle = oracle.remaining_budget
        if rem_oracle is not None:
            q_rem = min(q_rem, float(rem_oracle))

        extra_indices: List[int] = []
        extra_k: List[float] = []

        max_rounds = int(len(self.tau_arr))
        rounds = 0
        while rounds < max_rounds and q_rem >= 1.0:
            tau_cur = float(self.tau_arr[rounds])
            sigma_th = 1.0 / math.sqrt(cur_fisher)
            j1, j2, gap = find_rival_and_gap_fast(ll, step, sigma_th, tau_cur, self.c_w)
            if gap >= tau_cur:
                break

            th1 = float(grid[j1])
            th2 = float(grid[j2])
            best_k = pick_best_depth_bc_fast(
                th1,
                th2,
                cand_d,
                cand_costs,
                self.cost_exp,
                self.eps_reg,
                q_rem,
            )
            if best_k < 0:
                break

            depth_star = int(cand_depths[best_k])
            k_obs = int(oracle.measure(depth_star, 1))
            cost_k = float(cand_costs[best_k])
            q_rem -= cost_k
            cur_fisher += float(cand_d2[best_k])

            if k_obs == 1:
                ll = ll + log_p[best_k]
            else:
                ll = ll + log_1mp[best_k]

            extra_indices.append(best_k)
            extra_k.append(float(k_obs))
            rounds += 1

        num_grid = int(len(grid))
        j_best = int(np.argmax(ll))
        if j_best < 1:
            j_best = 1
        elif j_best > num_grid - 2:
            j_best = num_grid - 2

        if not extra_indices:
            active_d = 2.0 * base_depths.astype(np.float64) + 1.0
            active_k = counts_f
            active_n = plan["base_shots_f"]
        else:
            extra_idx_arr = np.asarray(extra_indices, dtype=np.int64)
            active_d = np.concatenate(
                [2.0 * base_depths.astype(np.float64) + 1.0, cand_d[extra_idx_arr]]
            )
            active_k = np.concatenate([counts_f, np.asarray(extra_k, dtype=np.float64)])
            active_n = np.concatenate(
                [plan["base_shots_f"], np.ones(len(extra_indices), dtype=np.float64)]
            )

        th_est = local_refine_zoom25(
            float(grid[j_best]),
            step,
            active_d,
            active_k,
            active_n,
            eta=eta_val,
        )
        return float(min(max(math.sin(th_est), 0.0), 1.0))

    def run_batch(
        self,
        oracles: Sequence[QuantumAEOracle],
        target_scale_or_budget: int,
        rngs: Optional[Sequence[np.random.Generator]] = None,
    ) -> np.ndarray:
        """Vectorized batch evaluation across a sequence of sandboxed QuantumAEOracles."""
        num_trials = len(oracles)
        if num_trials == 0:
            return np.empty(0, dtype=np.float64)

        estimates = np.empty(num_trials, dtype=np.float64)
        for i in range(num_trials):
            trial_rng = rngs[i] if rngs is not None else None
            estimates[i] = self.run_trial(oracles[i], target_scale_or_budget, trial_rng)
        return np.clip(estimates, 0.0, 1.0)
