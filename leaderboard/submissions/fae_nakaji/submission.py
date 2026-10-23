"""Faster Amplitude Estimation (FAE) — Nakaji (2020).

Reference:
    K. Nakaji, "Faster Amplitude Estimation,"
    Quantum Information & Computation 20(13&14), 1109-1123 (2020), arXiv:2003.02417.

Implements Nakaji's two-stage adaptive frequency progression:
- Stage 1 (Monotonic Cosine Inversion): While the confidence interval
  [2*(2k+1)*theta_min, 2*(2k+1)*theta_max] lies within a single monotonic
  half-period [m*pi, (m+1)*pi], FAE samples depth k = 2^(j-1) and directly
  inverts cos(2*(2k+1)*theta) = 2*p - 1 using Chernoff/Beta confidence bounds.
- Stage 2 (Two-Step Cosine-Difference Disambiguation): Once 2*(2k+1) spans
  an extremum boundary, FAE measures both depth k and the auxiliary two-step
  depth k_aux = k + k_prev, using the trigonometric difference identity
  cos(2*d_aux*theta) - cos(2*d*theta) to resolve the branch sign ambiguity
  followed by local likelihood refinement on the surviving interval.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import beta as beta_dist

from leaderboard.schema_and_api import AdaptiveAESubmission, QuantumAEOracle

FAE_INT_CACHE: Dict[Tuple[int, int, int], Tuple[float, float]] = {}


def chernoff_beta_bounds(counts: int, shots: int, delta_stage: float) -> Tuple[float, float]:
    """Compute high-probability confidence bounds [p_lo, p_hi] at level 1 - delta_stage."""
    alpha_key = int(round(delta_stage * 1e12))
    key = (int(counts), int(shots), alpha_key)
    cached = FAE_INT_CACHE.get(key)
    if cached is not None:
        return cached

    if counts <= 0:
        p_lo = 0.0
    else:
        p_lo = float(beta_dist.ppf(0.5 * delta_stage, counts, shots - counts + 1))

    if counts >= shots:
        p_hi = 1.0
    else:
        p_hi = float(beta_dist.ppf(1.0 - 0.5 * delta_stage, counts + 1, shots - counts))

    res = (min(max(p_lo, 0.0), 1.0), min(max(p_hi, 0.0), 1.0))
    FAE_INT_CACHE[key] = res
    return res


def invert_stage1_monotonic(theta_mid: float, k_depth: int, p_val: float) -> float:
    """Invert cos^2((2k+1)*theta) = p_val on the monotonic half-period containing theta_mid."""
    d_odd = 2.0 * k_depth + 1.0
    c_val = min(max(2.0 * p_val - 1.0, -1.0), 1.0)
    acos_val = math.acos(c_val)
    branch = int(math.floor(2.0 * d_odd * theta_mid / math.pi))
    if branch % 2 == 0:
        theta = (branch * math.pi + acos_val) / (2.0 * d_odd)
    else:
        theta = ((branch + 1) * math.pi - acos_val) / (2.0 * d_odd)
    return min(max(theta, 0.0), 0.5 * math.pi)


def resolve_stage2_two_step(
    theta_min: float,
    theta_max: float,
    k_main: int,
    p_main: float,
    k_aux: int,
    p_aux: float,
    records: List[Tuple[int, int, int]],
) -> Tuple[float, float, float]:
    """Use two-step cosine measurements (k_main, k_aux) and cumulative log-likelihood to update [theta_min, theta_max]."""
    d_main = 2.0 * k_main + 1.0
    d_aux = 2.0 * k_aux + 1.0
    lo = max(1e-6, min(theta_min, 0.5 * math.pi - 2e-6))
    hi = min(0.5 * math.pi - 1e-6, max(theta_max, lo + 1e-6))

    grid = np.linspace(lo, hi, 97, dtype=np.float64)
    ll = np.zeros_like(grid)
    for k_d, n_sh, k_cnt in records:
        d_odd = 2.0 * k_d + 1.0
        p = np.clip(np.cos(d_odd * grid) ** 2, 1e-12, 1.0 - 1e-12)
        ll += k_cnt * np.log(p) + (n_sh - k_cnt) * np.log1p(-p)

    # Weight by Nakaji two-step cosine-difference consistency
    c_main_obs = 2.0 * p_main - 1.0
    c_aux_obs = 2.0 * p_aux - 1.0
    diff_obs = c_aux_obs - c_main_obs
    diff_grid = np.cos(2.0 * d_aux * grid) - np.cos(2.0 * d_main * grid)
    score = ll - 2.0 * ((diff_grid - diff_obs) ** 2)

    j = int(np.argmax(score))
    th_best = float(grid[j])
    if 0 < j < len(grid) - 1:
        ym1, y0, yp1 = float(ll[j - 1]), float(ll[j]), float(ll[j + 1])
        den = 2.0 * y0 - ym1 - yp1
        if den > 1e-12:
            off = min(max(0.5 * (yp1 - ym1) / den, -0.5), 0.5)
            th_best = float(grid[j]) + off * float(grid[1] - grid[0])

    half_w = min(0.5 * (hi - lo), 1.85 * math.pi / (2.0 * d_main))
    new_lo = max(theta_min, th_best - half_w)
    new_hi = min(theta_max, th_best + half_w)
    return new_lo, new_hi, th_best


class FAENakajiSubmission(AdaptiveAESubmission):
    """Nakaji (2020) Faster Amplitude Estimation (two-stage adaptive frequency progression)."""

    name = "FAE (Nakaji 2020)"
    track = "adaptive-variable"

    def run_trial(
        self,
        oracle: QuantumAEOracle,
        target_scale_or_budget: int,
        rng: np.random.Generator,
    ) -> float:
        scale_nmax = max(4, int(target_scale_or_budget))
        eps_target = 0.22 / float(scale_nmax)
        depth_cap = oracle.max_depth_cap
        max_k_allowed = scale_nmax if depth_cap is None else min(scale_nmax, int(depth_cap))
        budget_cap = oracle.max_query_budget
        hard_query_limit = (25 * scale_nmax) if budget_cap is None else int(budget_cap)

        # Build doubling depth progression 0, 1, 2, 4, 8, ..., up to max_k_allowed
        rungs: List[int] = [0]
        k_val = 1
        while k_val < max_k_allowed:
            rungs.append(k_val)
            k_val *= 2
        if rungs[-1] != max_k_allowed:
            rungs.append(max_k_allowed)

        delta_stage = 0.02 / float(max(1, len(rungs)))
        n_shots_stage1 = 22
        n_shots_stage2 = 8

        theta_min = 0.0
        theta_max = 0.5 * math.pi
        theta_est = 0.25 * math.pi
        stage2_active = False
        records: List[Tuple[int, int, int]] = []

        for idx, k_d in enumerate(rungs):
            if (math.sin(theta_max) - math.sin(theta_min)) <= 2.0 * eps_target and idx >= 3:
                break

            d_odd = 2.0 * k_d + 1.0
            b_lo = int(math.floor(2.0 * d_odd * theta_min / math.pi))
            b_hi = int(math.floor(2.0 * d_odd * theta_max / math.pi))
            if b_lo != b_hi and k_d >= 1:
                stage2_active = True

            unit_cost = 1 if k_d == 0 else k_d
            rem_q = hard_query_limit - int(oracle.total_queries)
            if rem_q < unit_cost:
                break

            if not stage2_active:
                shots_1 = min(n_shots_stage1, max(1, rem_q // unit_cost))
                cnt_1 = int(oracle.measure(k_d, shots_1))
                records.append((k_d, shots_1, cnt_1))

                p_lo, p_hi = chernoff_beta_bounds(cnt_1, shots_1, delta_stage)
                theta_mid = 0.5 * (theta_min + theta_max)
                th_1 = invert_stage1_monotonic(theta_mid, k_d, p_lo)
                th_2 = invert_stage1_monotonic(theta_mid, k_d, p_hi)
                if th_1 > th_2:
                    th_1, th_2 = th_2, th_1
                theta_min = max(theta_min, th_1 - 1e-12)
                theta_max = min(theta_max, th_2 + 1e-12)
                theta_est = 0.5 * (theta_min + theta_max)
            else:
                shots_m = min(n_shots_stage2, max(1, rem_q // unit_cost))
                cnt_m = int(oracle.measure(k_d, shots_m))
                records.append((k_d, shots_m, cnt_m))
                p_main = cnt_m / float(shots_m)

                k_prev = max(1, rungs[max(0, idx - 1)])
                k_aux = min(max_k_allowed, k_d + max(1, k_prev // 2))
                if k_aux == k_d:
                    k_aux = max(1, k_d - max(1, k_prev // 2))
                rem_q2 = hard_query_limit - int(oracle.total_queries)
                p_aux = p_main
                if rem_q2 >= k_aux:
                    shots_a = min(max(3, n_shots_stage2 // 2), max(1, rem_q2 // k_aux))
                    cnt_a = int(oracle.measure(k_aux, shots_a))
                    records.append((k_aux, shots_a, cnt_a))
                    p_aux = cnt_a / float(shots_a)

                theta_min, theta_max, theta_est = resolve_stage2_two_step(
                    theta_min, theta_max, k_d, p_main, k_aux, p_aux, records
                )

        if not records:
            cnt0 = int(oracle.measure(0, n_shots_stage1))
            p0 = min(max(cnt0 / float(n_shots_stage1), 0.0), 1.0)
            return float(math.sqrt(max(0.0, 1.0 - p0)))

        _, _, theta_final = resolve_stage2_two_step(
            theta_min, theta_max, rungs[-1], 0.5, rungs[-1], 0.5, records
        )
        return float(min(max(math.sin(theta_final), 0.0), 1.0))
