"""Iterative Quantum Amplitude Estimation with Chernoff-Hoeffding Bounds — Grinko et al. (2021).

Reference:
    D. Grinko, J. Gacon, C. Zoufal, and S. Woerner,
    "Iterative Quantum Amplitude Estimation,"
    npj Quantum Information 7, 52 (2021), arXiv:1912.05559.

Implements the Chernoff-Hoeffding variant of IQAE analyzed in Grinko et al. (2021),
where each stage bounds the Bernoulli success probability p(k, theta) = cos^2((2k+1)*theta)
via the analytical Hoeffding inequality
    eps_p = sqrt(ln(2 * T_max / alpha) / (2 * N_i))
and advances Grover depth k via `find_next_k`.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from leaderboard.schema_and_api import AdaptiveAESubmission, QuantumAEOracle


def chernoff_hoeffding_confint(counts: int, shots: int, alpha_stage: float) -> Tuple[float, float]:
    """Compute Chernoff-Hoeffding confidence interval [p_min, p_max] at level 1 - alpha_stage."""
    if shots <= 0:
        return 0.0, 1.0
    p_hat = counts / float(shots)
    rad = math.sqrt(max(0.0, math.log(2.0 / max(alpha_stage, 1e-15)) / (2.0 * float(shots))))
    p_lo = min(max(p_hat - rad, 0.0), 1.0)
    p_hi = min(max(p_hat + rad, 0.0), 1.0)
    return p_lo, p_hi


def compute_lmax_chernoff(alpha_stage: float, n_shots: int) -> float:
    """Compute maximum angular half-width L_max under Chernoff-Hoeffding confidence intervals."""
    rad = math.sqrt(max(0.0, math.log(2.0 / max(alpha_stage, 1e-15)) / (2.0 * float(max(1, n_shots)))))
    best = 0.0
    for k in range(n_shots + 1):
        p_hat = k / float(n_shots)
        p_min = min(max(p_hat - rad, 0.0), 1.0)
        p_max = min(max(p_hat + rad, 0.0), 1.0)
        c_min = min(max(2.0 * p_min - 1.0, -1.0), 1.0)
        c_max = min(max(2.0 * p_max - 1.0, -1.0), 1.0)
        val = 0.5 * abs(math.acos(c_min) - math.acos(c_max))
        if val > best:
            best = val
    return best


def find_next_k_chernoff(
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


def polish_within_chernoff_interval(
    theta_min: float,
    theta_max: float,
    records: List[Tuple[int, int, int]],
) -> float:
    """Refine the angle estimate within the guaranteed Chernoff confidence interval [theta_min, theta_max]."""
    lo = max(1e-6, min(theta_min, 0.5 * math.pi - 2e-6))
    hi = min(0.5 * math.pi - 1e-6, max(theta_max, lo + 1e-6))
    grid = np.linspace(lo, hi, 49, dtype=np.float64)
    ll = np.zeros_like(grid)
    for k_d, n_sh, k_cnt in records:
        d_odd = 2.0 * k_d + 1.0
        p = np.clip(np.cos(d_odd * grid) ** 2, 1e-12, 1.0 - 1e-12)
        ll += k_cnt * np.log(p) + (n_sh - k_cnt) * np.log1p(-p)
    j = int(np.argmax(ll))
    if 0 < j < len(grid) - 1:
        ym1, y0, yp1 = float(ll[j - 1]), float(ll[j]), float(ll[j + 1])
        den = 2.0 * y0 - ym1 - yp1
        if den > 1e-12:
            off = min(max(0.5 * (yp1 - ym1) / den, -0.5), 0.5)
            return float(grid[j]) + off * float(grid[1] - grid[0])
    return float(grid[j])


class IQAEChernoffGrinkoSubmission(AdaptiveAESubmission):
    """Grinko et al. (2021) Iterative Quantum Amplitude Estimation (Chernoff-Hoeffding variant)."""

    name = "IQAE-Chernoff (Grinko et al. 2021)"
    track = "adaptive-variable"

    def run_trial(
        self,
        oracle: QuantumAEOracle,
        target_scale_or_budget: int,
        rng: np.random.Generator,
    ) -> float:
        scale_nmax = max(4, int(target_scale_or_budget))
        eps_target = 0.24 / float(scale_nmax)
        delta_conf = 0.05
        n_shots_default = 32
        r_ratio = 2.0

        t_stages = max(1, int(math.ceil(math.log(1.0 / (2.0 * eps_target)) / math.log(r_ratio))))
        alpha_stage = delta_conf / float(t_stages)
        l_max = compute_lmax_chernoff(alpha_stage, n_shots_default)

        theta_min = 0.0
        theta_max = 0.5 * math.pi
        k_cur = 0
        num_heads = 0
        num_flips = 0

        budget_cap = oracle.max_query_budget
        depth_cap = oracle.max_depth_cap
        max_k_allowed = scale_nmax if depth_cap is None else min(scale_nmax, int(depth_cap))
        hard_query_limit = (28 * scale_nmax) if budget_cap is None else int(budget_cap)

        records: List[Tuple[int, int, int]] = []
        rounds_executed = 0
        max_rounds = 52

        while (math.sin(theta_max) - math.sin(theta_min)) > 2.0 * eps_target and rounds_executed < max_rounds:
            next_k, advanced = find_next_k_chernoff(k_cur, theta_min, theta_max, r_ratio, max_k_allowed)
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
            records.append((k_cur, n_shots_i, heads))
            rounds_executed += 1

            p_min, p_max = chernoff_hoeffding_confint(num_heads, num_flips, alpha_stage)
            theta_mid = 0.5 * (theta_min + theta_max)
            th_1 = invert_cosine_branch(theta_mid, k_cur, p_min)
            th_2 = invert_cosine_branch(theta_mid, k_cur, p_max)
            if th_1 > th_2:
                th_1, th_2 = th_2, th_1

            theta_min = max(theta_min, th_1 - 1e-12)
            theta_max = min(theta_max, th_2 + 1e-12)

            if k_cur == max_k_allowed and num_flips >= n_shots_default:
                break

        if not records:
            k0 = int(oracle.measure(0, n_shots_default))
            records.append((0, n_shots_default, k0))

        theta_hat = polish_within_chernoff_interval(theta_min, theta_max, records)
        return float(min(max(math.sin(theta_hat), 0.0), 1.0))
