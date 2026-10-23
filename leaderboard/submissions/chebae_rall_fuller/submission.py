"""Chebyshev Amplitude Estimation (chebAE) — Rall & Fuller (2023).

Reference:
    P. Rall and B. Fuller,
    "Amplitude Estimation from Quantum Signal Processing,"
    Quantum 7, 937 (2023), arXiv:2207.08628.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
from scipy.stats import beta as beta_dist

from leaderboard.schema_and_api import AdaptiveAESubmission, QuantumAEOracle

CP_CACHE: Dict[Tuple[int, int, int], Tuple[float, float]] = {}
ERR_MAX_CACHE: Dict[Tuple[int, int], float] = {}


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


def max_error_cp(delta_stage: float, n_shots: int) -> float:
    """Maximum Clopper-Pearson probability half-width across all count outcomes 0..n_shots."""
    alpha_key = int(round(delta_stage * 1e12))
    key = (alpha_key, int(n_shots))
    cached = ERR_MAX_CACHE.get(key)
    if cached is not None:
        return cached

    best = 0.0
    for c in range(n_shots + 1):
        lo, hi = clopper_pearson_confint(c, n_shots, delta_stage)
        half = 0.5 * (hi - lo)
        if half > best:
            best = half
    ERR_MAX_CACHE[key] = best
    return best


def find_next_cheb_depth(
    theta_min: float,
    theta_max: float,
    min_deg: int,
    max_depth_cap: int,
) -> int | None:
    """Find highest odd Chebyshev degree d = 2k+1 > min_deg with no extrema on [theta_min, theta_max]."""
    span = max(1e-12, theta_max - theta_min)
    deg = int((0.5 * math.pi) / span)
    if deg % 2 == 0:
        deg += 1
    max_deg = 2 * max_depth_cap + 1
    if deg > max_deg:
        deg = max_deg

    while deg > min_deg:
        b_lo = int(2.0 * deg * theta_min / math.pi)
        b_hi = int(2.0 * deg * theta_max / math.pi)
        if b_lo == b_hi:
            return deg
        deg -= 2
    return None


def invert_cheb_theta(theta_int: float, deg: int, p_val: float) -> float:
    """Invert cos^2(deg * theta) = p_val on the monotonic half-period containing theta_int."""
    c_half = math.pi / (2.0 * deg)
    t = math.floor(theta_int / c_half)
    arg = min(max(2.0 * p_val - 1.0, -1.0), 1.0)
    acos_val = math.acos(arg)
    if int(t) % 2 == 0:
        theta = acos_val / (2.0 * deg)
    else:
        theta = 2.0 * c_half - acos_val / (2.0 * deg)
    k_per = int(t) // 2
    theta += math.pi * k_per / float(deg)
    return min(max(theta, 0.0), 0.5 * math.pi)


class ChebAERallFullerSubmission(AdaptiveAESubmission):
    """Rall & Fuller (2023) Chebyshev Amplitude Estimation with late-stage 1-shot stepping."""

    name = "chebAE (Rall & Fuller 2023)"
    track = "adaptive-variable"

    def run_trial(
        self,
        oracle: QuantumAEOracle,
        target_scale_or_budget: int,
        rng: np.random.Generator,
    ) -> float:
        scale_nmax = max(4, int(target_scale_or_budget))
        eps_target = 0.21 / float(scale_nmax)
        delta_conf = 0.05
        nu_param = 8.0
        r_ratio = 2
        n_shots_early = 36

        t_stages = max(1, int(math.ceil(math.log(1.0 / (2.0 * eps_target)) / math.log(float(r_ratio)))))
        delta_stage = delta_conf / float(t_stages)
        err_max = max_error_cp(delta_stage, n_shots_early)

        theta_min = 0.0
        theta_max = 0.5 * math.pi
        deg = 1
        num_flips = 0
        num_heads = 0

        budget_cap = oracle.max_query_budget
        depth_cap = oracle.max_depth_cap
        max_k_allowed = (2 * scale_nmax) if depth_cap is None else int(depth_cap)
        hard_query_limit = (25 * scale_nmax) if budget_cap is None else int(budget_cap)

        steps = 0
        max_steps = 220

        while (math.sin(theta_max) - math.sin(theta_min)) > 2.0 * eps_target and steps < max_steps:
            new_deg = find_next_cheb_depth(theta_min, theta_max, min_deg=deg * r_ratio, max_depth_cap=max_k_allowed)
            if new_deg is not None:
                deg = new_deg
                num_flips = 0
                num_heads = 0

            k_depth = deg // 2
            a_Span = math.sin(theta_max) - math.sin(theta_min)
            p_hi = math.cos(deg * theta_max) ** 2
            p_lo = math.cos(deg * theta_min) ** 2
            gap = max(1e-12, abs(p_hi - p_lo))

            if err_max * (a_Span / gap) < nu_param * eps_target:
                n_shots_i = 1
            else:
                n_shots_i = n_shots_early

            unit_cost = 1 if k_depth == 0 else k_depth
            rem_q = hard_query_limit - int(oracle.total_queries)
            if rem_q < unit_cost:
                break
            n_shots_i = min(n_shots_i, max(1, rem_q // unit_cost))

            heads = int(oracle.measure(k_depth, n_shots_i))
            num_heads += heads
            num_flips += n_shots_i
            steps += 1

            p_min, p_max = clopper_pearson_confint(num_heads, num_flips, delta_stage)
            theta_mid = 0.5 * (theta_min + theta_max)
            th_1 = invert_cheb_theta(theta_mid, deg, p_min)
            th_2 = invert_cheb_theta(theta_mid, deg, p_max)
            if th_1 > th_2:
                th_1, th_2 = th_2, th_1

            theta_min = max(theta_min, th_1 - 1e-15)
            theta_max = min(theta_max, th_2 + 1e-15)

        if int(oracle.total_queries) == 0:
            k0 = int(oracle.measure(0, n_shots_early))
            p0 = min(max(k0 / float(n_shots_early), 0.0), 1.0)
            return float(math.sqrt(max(0.0, 1.0 - p0)))

        a_min_final = math.sin(theta_min)
        a_max_final = math.sin(theta_max)
        return float(min(max(0.5 * (a_min_final + a_max_final), 0.0), 1.0))
