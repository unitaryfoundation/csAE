"""Bayesian Amplitude Estimation (BAE-Adaptive) — Ramôa & Santos (2024).

Reference:
    G. Ramôa and L. Santos,
    "Bayesian Quantum Amplitude Estimation,"
    arXiv:2401.15072 (2024).

Implements sequential Bayesian experimental design for Quantum Amplitude
Estimation where the posterior distribution over theta = arcsin(a) in (0, pi/2)
is updated via Bayes' rule after each measurement round, and the next Grover
depth k is selected from the posterior standard deviation sigma_theta and
Bayesian credible region [theta_min, theta_max], followed by full multi-round
Bayesian posterior estimation.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import beta as beta_dist

from leaderboard.schema_and_api import AdaptiveAESubmission, QuantumAEOracle

BAE_CRED_CACHE: Dict[Tuple[int, int, int], Tuple[float, float]] = {}


def uniform_beta_credible_interval(counts: int, shots: int, alpha_cred: float) -> Tuple[float, float]:
    """Compute Bayes-Laplace Beta(1 + counts, 1 + shots - counts) credible interval on p = cos^2((2k+1)*theta)."""
    alpha_key = int(round(alpha_cred * 1e12))
    key = (int(counts), int(shots), alpha_key)
    cached = BAE_CRED_CACHE.get(key)
    if cached is not None:
        return cached

    a_post = 1.0 + float(counts)
    b_post = 1.0 + float(shots - counts)
    p_lo = float(beta_dist.ppf(0.5 * alpha_cred, a_post, b_post))
    p_hi = float(beta_dist.ppf(1.0 - 0.5 * alpha_cred, a_post, b_post))
    p_lo = min(max(p_lo, 0.0), 1.0)
    p_hi = min(max(p_hi, 0.0), 1.0)
    res = (p_lo, p_hi)
    BAE_CRED_CACHE[key] = res
    return res


def invert_branch_angle(theta_mid: float, deg: int, p_val: float) -> float:
    """Invert cos^2(deg * theta) = p_val on the monotonic half-period containing theta_mid."""
    c_half = math.pi / (2.0 * deg)
    t = math.floor(theta_mid / c_half)
    arg = min(max(2.0 * p_val - 1.0, -1.0), 1.0)
    acos_val = math.acos(arg)
    if int(t) % 2 == 0:
        theta = acos_val / (2.0 * deg)
    else:
        theta = 2.0 * c_half - acos_val / (2.0 * deg)
    k_per = int(t) // 2
    theta += math.pi * k_per / float(deg)
    return min(max(theta, 0.0), 0.5 * math.pi)


def compute_posterior_moments_on_interval(
    theta_min: float,
    theta_max: float,
    records: List[Tuple[int, int, int]],
) -> Tuple[float, float, float]:
    """Compute full multi-round Bayesian posterior mean, std (sigma_theta), and MAP on [theta_min, theta_max]."""
    lo = max(1e-6, min(theta_min, 0.5 * math.pi - 2e-6))
    hi = min(0.5 * math.pi - 1e-6, max(theta_max, lo + 1e-6))
    grid = np.linspace(lo, hi, 73, dtype=np.float64)
    # Uniform prior on amplitude a = sin(theta) in (0, 1) induces prior density p(theta) = cos(theta)
    log_post = np.log(np.clip(np.cos(grid), 1e-12, 1.0))
    for k_d, n_sh, k_cnt in records:
        d_odd = 2.0 * k_d + 1.0
        p = np.clip(np.cos(d_odd * grid) ** 2, 1e-12, 1.0 - 1e-12)
        log_post += k_cnt * np.log(p) + (n_sh - k_cnt) * np.log1p(-p)

    j_map = int(np.argmax(log_post))
    max_lp = float(log_post[j_map])
    weights = np.exp(np.maximum(log_post - max_lp, -50.0))
    w_sum = float(np.sum(weights))
    step = float(grid[1] - grid[0])

    if w_sum <= 1e-30:
        return float(grid[j_map]), max(step, 0.25 * (hi - lo)), float(grid[j_map])

    mean_th = float(np.sum(weights * grid)) / w_sum
    var_th = float(np.sum(weights * ((grid - mean_th) ** 2))) / w_sum
    sigma_th = math.sqrt(max(var_th, (0.25 * step) ** 2))

    theta_map = float(grid[j_map])
    if 0 < j_map < len(grid) - 1:
        ym1, y0, yp1 = float(log_post[j_map - 1]), float(log_post[j_map]), float(log_post[j_map + 1])
        den = 2.0 * y0 - ym1 - yp1
        if den > 1e-12:
            off = min(max(0.5 * (yp1 - ym1) / den, -0.5), 0.5)
            theta_map = float(grid[j_map]) + off * step

    return mean_th, sigma_th, theta_map


def select_bae_next_degree(
    theta_min: float,
    theta_max: float,
    sigma_th: float,
    min_deg: int,
    max_depth_cap: int,
    c_bae: float = 0.55,
) -> int | None:
    """Select next odd Grover degree d = 2k+1 via Ramoa & Santos sigma_theta utility scaling."""
    span = max(1e-12, theta_max - theta_min)
    deg_branch = int((0.5 * math.pi) / span)
    deg_sigma = int(round(c_bae / max(sigma_th, 1e-12)))
    deg = max(deg_branch, deg_sigma)
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


class BAERamoaSantosSubmission(AdaptiveAESubmission):
    """Ramôa & Santos (2024) Bayesian Amplitude Estimation (`BAE-Adaptive`)."""

    name = "BAE-Adaptive (Ramoa & Santos 2024)"
    track = "adaptive-variable"

    def run_trial(
        self,
        oracle: QuantumAEOracle,
        target_scale_or_budget: int,
        rng: np.random.Generator,
    ) -> float:
        scale_nmax = max(4, int(target_scale_or_budget))
        eps_target = 0.19 / float(scale_nmax)
        alpha_cred = 0.0030
        n_shots_batch = 12

        theta_min = 0.0
        theta_max = 0.5 * math.pi
        deg = 1
        num_flips = 0
        num_heads = 0

        budget_cap = oracle.max_query_budget
        depth_cap = oracle.max_depth_cap
        max_k_allowed = scale_nmax if depth_cap is None else min(scale_nmax, int(depth_cap))
        hard_query_limit = (20 * scale_nmax) if budget_cap is None else int(budget_cap)

        records: List[Tuple[int, int, int]] = []
        steps = 0
        max_steps = 80

        while (math.sin(theta_max) - math.sin(theta_min)) > 2.0 * eps_target and steps < max_steps:
            if num_flips > 0:
                sigma_fisher = 1.0 / (2.0 * float(deg) * math.sqrt(float(num_flips) + 3.0))
                sigma_cred = (theta_max - theta_min) / 5.92
                sigma_th = max(sigma_fisher, sigma_cred)
            else:
                sigma_th = 0.45

            min_next_deg = max(deg + 2, int(round(1.65 * deg)))
            new_deg = select_bae_next_degree(
                theta_min,
                theta_max,
                sigma_th,
                min_deg=min_next_deg,
                max_depth_cap=max_k_allowed,
            )
            if new_deg is not None:
                deg = new_deg
                num_flips = 0
                num_heads = 0

            k_depth = deg // 2
            if k_depth >= max(8, max_k_allowed // 2):
                n_shots_i = 4
            elif k_depth >= max(4, max_k_allowed // 5):
                n_shots_i = 8
            else:
                n_shots_i = n_shots_batch

            unit_cost = 1 if k_depth == 0 else k_depth
            rem_q = hard_query_limit - int(oracle.total_queries)
            if rem_q < unit_cost:
                break
            n_shots_i = min(n_shots_i, max(1, rem_q // unit_cost))

            heads = int(oracle.measure(k_depth, n_shots_i))
            num_heads += heads
            num_flips += n_shots_i
            records.append((k_depth, n_shots_i, heads))
            steps += 1

            p_min, p_max = uniform_beta_credible_interval(num_heads, num_flips, alpha_cred)
            theta_mid = 0.5 * (theta_min + theta_max)
            th_1 = invert_branch_angle(theta_mid, deg, p_min)
            th_2 = invert_branch_angle(theta_mid, deg, p_max)
            if th_1 > th_2:
                th_1, th_2 = th_2, th_1

            theta_min = max(theta_min, th_1 - 1e-12)
            theta_max = min(theta_max, th_2 + 1e-12)

            if k_depth == max_k_allowed and num_flips >= 8:
                break

        if not records:
            k0 = int(oracle.measure(0, n_shots_batch))
            records.append((0, n_shots_batch, k0))

        mean_th, _, map_th = compute_posterior_moments_on_interval(theta_min, theta_max, records)
        # Bayesian posterior blend of posterior mean and MAP inside terminal credible region
        theta_hat = 0.5 * (mean_th + map_th)
        return float(min(max(math.sin(theta_hat), 0.0), 1.0))
