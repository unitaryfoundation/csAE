"""Iterative Quantum Amplitude Estimation (IQAE) — Grinko et al. (2021).

Reference:
    D. Grinko, J. Gacon, C. Zoufal, and S. Woerner,
    "Iterative Quantum Amplitude Estimation,"
    npj Quantum Information 7, 52 (2021), arXiv:1912.05559.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
from scipy.stats import beta as beta_dist

from leaderboard.schema_and_api import AdaptiveAESubmission, QuantumAEOracle

CP_CACHE: Dict[Tuple[int, int, int], Tuple[float, float]] = {}
LMAX_CACHE: Dict[Tuple[int, int], float] = {}


def clopper_pearson_confint(counts: int, shots: int, alpha: float) -> Tuple[float, float]:
    """Compute exact Clopper-Pearson (Beta) confidence interval [p_min, p_max]."""
    alpha_key = int(round(alpha * 1e12))
    key = (int(counts), int(shots), alpha_key)
    cached = CP_CACHE.get(key)
    if cached is not None:
        return cached

    if counts <= 0:
        p_lo = 0.0
    else:
        p_lo = float(beta_dist.ppf(0.5 * alpha, counts, shots - counts + 1))

    if counts >= shots:
        p_hi = 1.0
    else:
        p_hi = float(beta_dist.ppf(1.0 - 0.5 * alpha, counts + 1, shots - counts))

    p_lo = min(max(p_lo, 0.0), 1.0)
    p_hi = min(max(p_hi, 0.0), 1.0)
    res = (p_lo, p_hi)
    CP_CACHE[key] = res
    return res


def compute_lmax_iqae(alpha_stage: float, n_shots: int) -> float:
    """Compute maximum angular half-width L_max for IQAE shot-reduction rule."""
    alpha_key = int(round(alpha_stage * 1e12))
    key = (alpha_key, int(n_shots))
    cached = LMAX_CACHE.get(key)
    if cached is not None:
        return cached

    best = 0.0
    for k in range(n_shots + 1):
        p_min, p_max = clopper_pearson_confint(k, n_shots, alpha_stage)
        c_min = min(max(2.0 * p_min - 1.0, -1.0), 1.0)
        c_max = min(max(2.0 * p_max - 1.0, -1.0), 1.0)
        val = 0.5 * abs(math.acos(c_min) - math.acos(c_max))
        if val > best:
            best = val
    LMAX_CACHE[key] = best
    return best


def find_next_k(
    k_cur: int,
    theta_min: float,
    theta_max: float,
    r_min: float,
    max_k_cap: int,
) -> Tuple[int, bool]:
    """Find the largest Grover depth k <= max_k_cap such that cos(2*(2k+1)*theta) is monotonic on [theta_min, theta_max]."""
    width = max(1e-12, theta_max - theta_min)
    k_upper = int(math.floor((math.pi / (2.0 * width) - 1.0) / 2.0))
    k_upper = min(k_upper, max_k_cap)
    k_min_req = int(math.ceil((r_min * (2 * k_cur + 1) - 1.0) / 2.0))

    cand_k = k_upper
    while cand_k >= max(1, k_min_req):
        d_odd = 2 * cand_k + 1
        branch_lo = int(math.floor(2.0 * d_odd * theta_min / math.pi))
        branch_hi = int(math.floor(2.0 * d_odd * theta_max / math.pi))
        if branch_lo == branch_hi:
            return cand_k, True
        cand_k -= 1
    return k_cur, False


def invert_cosine_branch(theta_mid: float, k_depth: int, p_val: float) -> float:
    """Invert p = cos^2((2k+1)*theta) on the monotonic half-period containing theta_mid."""
    d_odd = 2.0 * k_depth + 1.0
    c_val = min(max(2.0 * p_val - 1.0, -1.0), 1.0)
    acos_val = math.acos(c_val)
    branch = int(math.floor(2.0 * d_odd * theta_mid / math.pi))
    if branch % 2 == 0:
        theta = (branch * math.pi + acos_val) / (2.0 * d_odd)
    else:
        theta = ((branch + 1) * math.pi - acos_val) / (2.0 * d_odd)
    return min(max(theta, 0.0), 0.5 * math.pi)


class IQAEGrinkoSubmission(AdaptiveAESubmission):
    """Grinko et al. (2021) Iterative Quantum Amplitude Estimation."""

    name = "IQAE (Grinko et al. 2021)"
    track = "adaptive-variable"

    def run_trial(
        self,
        oracle: QuantumAEOracle,
        target_scale_or_budget: int,
        rng: np.random.Generator,
    ) -> float:
        scale_nmax = max(4, int(target_scale_or_budget))
        eps_target = 0.22 / float(scale_nmax)
        delta_conf = 0.05
        n_shots_default = 28
        r_ratio = 2.0

        t_stages = max(1, int(math.ceil(math.log(1.0 / (2.0 * eps_target)) / math.log(r_ratio))))
        alpha_stage = delta_conf / float(t_stages)
        l_max = compute_lmax_iqae(alpha_stage, n_shots_default)

        theta_min = 0.0
        theta_max = 0.5 * math.pi
        k_cur = 0
        num_heads = 0
        num_flips = 0

        budget_cap = oracle.max_query_budget
        depth_cap = oracle.max_depth_cap
        max_k_allowed = scale_nmax if depth_cap is None else min(scale_nmax, int(depth_cap))
        hard_query_limit = (25 * scale_nmax) if budget_cap is None else int(budget_cap)

        rounds_executed = 0
        max_rounds = 48

        while (math.sin(theta_max) - math.sin(theta_min)) > 2.0 * eps_target and rounds_executed < max_rounds:
            next_k, advanced = find_next_k(k_cur, theta_min, theta_max, r_ratio, max_k_allowed)
            if advanced:
                k_cur = next_k
                num_heads = 0
                num_flips = 0

            k_i = 4 * k_cur + 2
            eps_theta = eps_target / math.pi
            cutoff_k = int(l_max / max(eps_theta, 1e-12))
            if k_i > cutoff_k:
                n_shots_i = max(1, int((l_max / max(eps_theta, 1e-12)) * n_shots_default / float(k_i) / 8.0))
            else:
                n_shots_i = n_shots_default

            unit_cost = 1 if k_cur == 0 else k_cur
            rem_q = hard_query_limit - int(oracle.total_queries)
            if rem_q < unit_cost:
                break
            n_shots_i = min(n_shots_i, max(1, rem_q // unit_cost))

            heads = int(oracle.measure(k_cur, n_shots_i))
            num_heads += heads
            num_flips += n_shots_i
            rounds_executed += 1

            p_min, p_max = clopper_pearson_confint(num_heads, num_flips, alpha_stage)
            theta_mid = 0.5 * (theta_min + theta_max)
            th_1 = invert_cosine_branch(theta_mid, k_cur, p_min)
            th_2 = invert_cosine_branch(theta_mid, k_cur, p_max)
            if th_1 > th_2:
                th_1, th_2 = th_2, th_1

            theta_min = max(theta_min, th_1 - 1e-12)
            theta_max = min(theta_max, th_2 + 1e-12)

            if k_cur == max_k_allowed and num_flips >= n_shots_default:
                break

        if int(oracle.total_queries) == 0:
            k0 = int(oracle.measure(0, n_shots_default))
            p0 = min(max(k0 / float(n_shots_default), 0.0), 1.0)
            return float(math.sqrt(max(0.0, 1.0 - p0)))

        theta_hat = 0.5 * (theta_min + theta_max)
        a_hat = math.sin(theta_hat)
        return float(min(max(a_hat, 0.0), 1.0))
