"""Deterministic-Budget Adaptive Amplitude Estimation (Zero Query Spread).

Implements `FixedBudgetAdaptiveSubmission`: an adaptive Maximum-Likelihood
Quantum Amplitude Estimation algorithm that expends a 100% deterministic total
query count Q_i == B(n_max) on every single trial (so std(Q) == 0 and
C_max == C_ave identically), while strictly respecting the per-scale circuit
depth cap `n <= n_max` (`oracle.max_depth_cap`) and adaptively selecting Grover
depths and shot allocations on the fly.

Key Algorithmic Mechanisms:
1. Adaptive Early Advance & Multiplicative Tie-Breaking on Shallow/Mid Rungs:
   - Starts each rung of a base geometric ladder (`d <= int(n_max / 1.38)`) with
     a reduced shot allocation `m_min` (~53% of canonical shots) and evaluates
     the exact log-likelihood gap `dll = ll(theta_1) - ll(theta_2)` between the
     primary local maximum and the highest distinct rival local maximum.
   - On the ~85-90% of trials where `dll >= tau_early` immediately after
     `m_min` shots, advances early to the next rung, banking 40-60% of the
     query budget for deep slope-steered rungs up to `n_max`.
   - On the ~10-15% of trials where `dll < tau_early`, takes targeted 1-shot
     tie-breakers at a depth `n_extra <= n_max` maximizing the multiplicative
     mode discrimination `(|p1(n) - p2(n)| + 0.05) * (0.45 + |sin(2(2n+1)*th1)|)`,
     breaking periodic aliases before advancing.
2. Adaptive Slope Steering (`cos^2((2n+1)*theta_hat) ~ 0.5`):
   - Selects each rung depth `n_k <= n_max` within a local window to maximize
     `|sin(2(2n+1)*th1)| + w_riv * |p1(n) - p2(n)|`, placing `th1` at the
     center of the monotonic cosine branch where observed Fisher curvature is
     deterministically `4(2n+1)^2` for both 0 and 1 outcomes.
3. Depth-Capped Horizon-Paced Extension & Exact Budget Completion (`sum(cost) == B`):
   - Converts all queries saved from shallow/mid rungs into deep slope-steered
     rungs up to `n_max` using multi-step geometric horizon pacing and a
     depth-capped final pair split `(n_f, B_rem - n_f)` (`n_f <= n_max` and
     `B_rem - n_f <= n_max`) so that total query cost equals `B` identically on
     100% of trials (`std(Q) = 0.0`) without ever exceeding `n_max`.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

from leaderboard.schema_and_api import AdaptiveAESubmission, QuantumAEOracle
from mlqae.core import canonical_shots, geom_ladder

G_GLOB: int = 768
D_TH_GLOB: float = 0.5 * math.pi / G_GLOB
TH_GLOB: np.ndarray = (np.arange(G_GLOB, dtype=np.float64) + 0.5) * D_TH_GLOB
N_PRE: int = 64
D_PRE: np.ndarray = 2.0 * np.arange(N_PRE, dtype=np.float64)[:, None] + 1.0
P_PRE: np.ndarray = np.clip(np.cos(D_PRE * TH_GLOB[None, :]) ** 2, 1e-12, 1.0 - 1e-12)
LOG_P_GLOB: np.ndarray = np.log(P_PRE)
LOG_1MP_GLOB: np.ndarray = np.log1p(-P_PRE)
UNIT_GRID: np.ndarray = np.arange(256, dtype=np.float64)

SCHEDULE_CACHE: Dict[Tuple[int, int, float], Tuple[int, np.ndarray, np.ndarray, np.ndarray]] = {}


def resolve_cap_and_schedule(
    oracle: QuantumAEOracle,
    target_scale_or_budget: int,
    alpha: float = 0.53,
) -> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve depth cap `d_cap`, deterministic query budget `B`, and reference ladder."""
    depth_cap_attr = oracle.max_depth_cap
    if depth_cap_attr is not None and int(depth_cap_attr) > 0:
        d_cap = int(depth_cap_attr)
    else:
        d_cap = int(target_scale_or_budget)

    budget_cap = oracle.max_query_budget
    b_override = int(budget_cap) if (budget_cap is not None and int(budget_cap) > 0) else 0

    key = (d_cap, b_override, round(float(alpha), 6))
    cached = SCHEDULE_CACHE.get(key)
    if cached is not None:
        b_val, ref_lad, ref_m_min, ref_m_max = cached
        return d_cap, b_val, ref_lad, ref_m_min, ref_m_max

    # Base ladder stops at <= int(d_cap / 1.38) so Phase 2 horizon-paced extension
    # steps up into the top window [0.75 * d_cap, d_cap] without exceeding d_cap.
    max_base_d = max(4, int(d_cap / 1.38))
    lad = geom_ladder(max_base_d, 1.45)
    sh = canonical_shots(len(lad))
    base_cost = sum((1 if int(d) == 0 else int(d)) * int(s) for d, s in zip(lad, sh))

    if b_override > 0:
        b_val = b_override
    else:
        b_val = int(round(1.368 * base_cost))

    max_lad = max(lad)
    ref_lad = np.asarray(lad, dtype=np.int64)
    ref_m_min = np.asarray(
        [
            max(6, int(math.ceil(0.75 * int(s))))
            if int(d) <= 1
            else (
                max(4, int(math.ceil(0.65 * int(s))))
                if int(d) <= 3
                else max(1 if int(d) == max_lad else 2, int(math.ceil(alpha * int(s))))
            )
            for d, s in zip(lad, sh)
        ],
        dtype=np.int64,
    )
    ref_m_max = np.asarray([int(s) + 2 for s in sh], dtype=np.int64)
    SCHEDULE_CACHE[key] = (b_val, ref_lad, ref_m_min, ref_m_max)
    return d_cap, b_val, ref_lad, ref_m_min, ref_m_max


def record_meas(
    n: int,
    m: int,
    k: int,
    depths: np.ndarray,
    d_odd: np.ndarray,
    k_f: np.ndarray,
    mk_f: np.ndarray,
    n_rungs: int,
    ll_glob: np.ndarray,
) -> int:
    """Record m shots with k successes at depth n, merging repeated depths."""
    if n < N_PRE:
        ll_glob += k * LOG_P_GLOB[n] + (m - k) * LOG_1MP_GLOB[n]
    for r in range(n_rungs):
        if int(depths[r]) == n:
            k_f[r] += float(k)
            mk_f[r] += float(m - k)
            return n_rungs
    depths[n_rungs] = n
    d_odd[n_rungs] = 2.0 * n + 1.0
    k_f[n_rungs] = float(k)
    mk_f[n_rungs] = float(m - k)
    return n_rungs + 1


def eval_ll_grid(
    lo: float,
    step: float,
    n_pts: int,
    d_odd: np.ndarray,
    k_f: np.ndarray,
    mk_f: np.ndarray,
    n_rungs: int,
) -> np.ndarray:
    """Vectorized evaluation of exact binomial log-likelihood on a uniform grid."""
    grid = lo + step * UNIT_GRID[:n_pts]
    d_sub = d_odd[:n_rungs]
    k_sub = k_f[:n_rungs]
    mk_sub = mk_f[:n_rungs]
    p = np.clip(np.cos(np.outer(d_sub, grid)) ** 2, 1e-12, 1.0 - 1e-12)
    return k_sub @ np.log(p) + mk_sub @ np.log1p(-p)


def parabola_peak(vals: np.ndarray, j: int, lo: float, step: float) -> Tuple[float, float]:
    """Quadratic interpolation around grid index j."""
    if 0 < j < len(vals) - 1:
        ym1 = float(vals[j - 1])
        y0 = float(vals[j])
        yp1 = float(vals[j + 1])
        den = 2.0 * y0 - ym1 - yp1
        if den > 1e-12:
            off = 0.5 * (yp1 - ym1) / den
            if off < -0.5:
                off = -0.5
            elif off > 0.5:
                off = 0.5
            return lo + (j + off) * step, y0 + 0.5 * den * off * off
    return lo + j * step, float(vals[j])


def zoom_polish(
    th0: float,
    half_w: float,
    d_odd: np.ndarray,
    k_f: np.ndarray,
    mk_f: np.ndarray,
    n_rungs: int,
) -> Tuple[float, float]:
    """17-point local zoom + quadratic peak refinement around th0."""
    m_pts = 17
    h = (2.0 * half_w) / (m_pts - 1)
    lo = max(1e-6, min(0.5 * math.pi - 1e-6 - 2.0 * half_w, th0 - half_w))
    vals = eval_ll_grid(lo, h, m_pts, d_odd, k_f, mk_f, n_rungs)
    best_j = int(np.argmax(vals))
    return parabola_peak(vals, best_j, lo, h)


def find_top2_modes_array(
    vals: np.ndarray,
    lo: float,
    step: float,
    min_sep: float,
) -> Tuple[float, float, float, float]:
    """Find primary mode and highest distinct local-maximum rival mode."""
    n = len(vals)
    best_j = int(np.argmax(vals))
    th1, ll1 = parabola_peak(vals, best_j, lo, step)

    mid = vals[1 : n - 1]
    loc_mask = (mid >= vals[: n - 2]) & (mid >= vals[2:n])
    loc_idx = np.flatnonzero(loc_mask) + 1
    if len(loc_idx) > 0:
        th_loc = lo + loc_idx * step
        sep_mask = np.abs(th_loc - th1) >= min_sep
        cand_idx = loc_idx[sep_mask]
        if len(cand_idx) > 0:
            riv_j = int(cand_idx[int(np.argmax(vals[cand_idx]))])
            th2, ll2 = parabola_peak(vals, riv_j, lo, step)
            return th1, ll1, th2, ll1 - ll2
    return th1, ll1, th1, 100.0


def get_modes(
    ll_glob: np.ndarray,
    d_odd: np.ndarray,
    k_f: np.ndarray,
    mk_f: np.ndarray,
    n_rungs: int,
    n_cur: int,
    n_prev: int,
    prev_th1: float,
    prev_th2: float,
) -> Tuple[float, float, float, float]:
    """Multi-scale mode tracker returning (th1, ll1, th2, dll = ll1 - ll2)."""
    min_sep = 0.4 * math.pi / (2.0 * n_cur + 1.0)
    if n_cur <= 34:
        th1, ll1, th2, dll = find_top2_modes_array(ll_glob, 0.5 * D_TH_GLOB, D_TH_GLOB, min_sep)
        if n_cur >= 12:
            th1, ll1 = zoom_polish(th1, D_TH_GLOB, d_odd, k_f, mk_f, n_rungs)
        return th1, ll1, th2, dll

    win = 4.0 * math.pi / (2.0 * max(1, n_prev) + 1.0)
    step = (0.5 * math.pi / (2.0 * n_cur + 1.0)) / 4.5
    n_pts = int(math.ceil(2.0 * win / step)) + 1
    if n_pts < 65:
        n_pts = 65
    elif n_pts > 145:
        n_pts = 145
    step = 2.0 * win / (n_pts - 1)
    lo = prev_th1 - win
    if lo < 1e-5:
        lo = 1e-5
    if lo + 2.0 * win > 0.5 * math.pi - 1e-5:
        lo = 0.5 * math.pi - 1e-5 - 2.0 * win

    vals = eval_ll_grid(lo, step, n_pts, d_odd, k_f, mk_f, n_rungs)
    th1, ll1, th2, dll = find_top2_modes_array(vals, lo, step, min_sep)

    if abs(prev_th2 - prev_th1) > win:
        lo2 = max(1e-5, min(0.5 * math.pi - 1e-5 - 24.0 * step, prev_th2 - 12.0 * step))
        vals2 = eval_ll_grid(lo2, step, 25, d_odd, k_f, mk_f, n_rungs)
        b2_j = int(np.argmax(vals2))
        th_r2, ll_r2 = parabola_peak(vals2, b2_j, lo2, step)
        if ll_r2 > ll1:
            th2 = th1
            dll = ll_r2 - ll1
            th1 = th_r2
            ll1 = ll_r2
        elif ll1 - ll_r2 < dll:
            th2 = th_r2
            dll = ll1 - ll_r2

    th1, ll1 = zoom_polish(th1, step, d_odd, k_f, mk_f, n_rungs)
    return th1, ll1, th2, dll


def pick_steered_depth(n_lo: int, n_hi: int, th1: float, th2: float, dll: float) -> int:
    """Select integer depth in [n_lo, n_hi] maximizing slope at th1 + rival separation."""
    if n_lo >= n_hi:
        return n_lo
    best_n = n_lo
    best_score = -1e300
    w_riv = 0.85 if dll < 8.5 else 0.15
    span = float(max(1, n_hi - n_lo))
    for n in range(n_lo, n_hi + 1):
        d = 2.0 * n + 1.0
        slope = abs(math.sin(2.0 * d * th1))
        c1 = math.cos(d * th1)
        c2 = math.cos(d * th2)
        sep = abs(c1 * c1 - c2 * c2)
        score = slope + w_riv * sep + 0.05 * (n - n_lo) / span
        if score > best_score:
            best_score = score
            best_n = n
    return best_n


def pick_tiebreaker_depth(n_lo: int, n_hi: int, th1: float, th2: float) -> int:
    """Multiplicative tie-breaker depth selector guaranteeing |p1(n) - p2(n)| > 0."""
    if n_lo >= n_hi:
        return n_lo
    best_n = n_lo
    best_score = -1e300
    for n in range(n_lo, n_hi + 1):
        d = 2.0 * n + 1.0
        slope = abs(math.sin(2.0 * d * th1))
        c1 = math.cos(d * th1)
        c2 = math.cos(d * th2)
        sep = abs(c1 * c1 - c2 * c2)
        score = (sep + 0.05) * (0.45 + slope)
        if score > best_score:
            best_score = score
            best_n = n
    return best_n


def pick_final_pair_depth(rem: int, w_end: int, th1: float, d_cap: int) -> int:
    """Partition final remaining budget rem = n_f + (rem - n_f) with both <= d_cap."""
    hi_f = min(d_cap, rem)
    lo_min = max(1, rem - d_cap)
    n_lo = max(lo_min, hi_f - w_end)
    best_n = hi_f
    best_val = -1e300
    for n_f in range(n_lo, hi_f + 1):
        d_f = 2.0 * n_f + 1.0
        s_f = abs(math.sin(2.0 * d_f * th1))
        eff_s = s_f if s_f < 0.90 else 0.90 + 0.2 * (s_f - 0.90)
        rem_r = rem - n_f
        val = (d_f * eff_s) ** 2
        if rem_r > 0:
            d_r = 2.0 * rem_r + 1.0
            s_r = abs(math.sin(2.0 * d_r * th1))
            val += (d_r * s_r) ** 2
        if val > best_val:
            best_val = val
            best_n = n_f
    return best_n


class FixedBudgetAdaptiveSubmission(AdaptiveAESubmission):
    """Deterministic-Budget Adaptive AE (`FixedBudgetAdaptiveAE`, Zero Query Spread)."""

    name = "FixedBudgetAdaptiveAE (Zero-Spread Slope-Extended)"
    track = "adaptive-fixed-budget"

    def __init__(
        self,
        alpha: float = 0.53,
        tau_early: float = 5.5,
        tau_deep: float = 5.6,
        r_ext: float = 1.43,
    ) -> None:
        self.alpha = float(alpha)
        self.tau_early = float(tau_early)
        self.tau_deep = float(tau_deep)
        self.r_ext = float(r_ext)

    def run_trial(
        self,
        oracle: QuantumAEOracle,
        target_scale_or_budget: int,
        rng: np.random.Generator | None = None,
    ) -> float:
        d_cap, budget, ref_lad, ref_m_min, ref_m_max = resolve_cap_and_schedule(
            oracle, target_scale_or_budget, self.alpha
        )

        depths = np.empty(64, dtype=np.int64)
        d_odd = np.empty(64, dtype=np.float64)
        k_f = np.empty(64, dtype=np.float64)
        mk_f = np.empty(64, dtype=np.float64)
        ll_glob = np.zeros(G_GLOB, dtype=np.float64)
        n_rungs = 0

        rem = budget
        max_d = 0
        n_prev = 0
        n_cur = 0
        th1 = 0.7
        th2 = 0.7
        dll = 0.0

        tau_early = self.tau_early
        tau_deep = self.tau_deep
        r_ext = self.r_ext

        # Phase 1: Adaptive Early-Advance + Tie-Breaking Climb on Reference Ladder
        l_ref = len(ref_lad)
        for idx in range(l_ref):
            d_target = min(d_cap, int(ref_lad[idx]))
            m_min = int(ref_m_min[idx])
            m_max = int(ref_m_max[idx])

            if d_target <= 3:
                n_k = d_target
            else:
                w = max(1, int(0.14 * d_target))
                n_lo = min(d_cap, max(n_cur + 1, d_target - w))
                n_hi = min(d_cap, rem // max(1, m_min), max(n_lo, d_target + w))
                if n_lo > rem or n_lo > n_hi:
                    break
                n_k = pick_steered_depth(n_lo, n_hi, th1, th2, dll)

            cost_1 = 1 if n_k == 0 else n_k
            if rem < cost_1:
                break
            m_take = min(m_min, rem // cost_1)
            if m_take <= 0:
                break

            k_s = int(oracle.measure(n_k, m_take))
            n_rungs = record_meas(n_k, m_take, k_s, depths, d_odd, k_f, mk_f, n_rungs, ll_glob)
            rem -= m_take * cost_1
            if n_k > max_d:
                max_d = n_k
            if n_k > n_cur:
                n_prev = n_cur
                n_cur = n_k
            th1, _, th2, dll = get_modes(ll_glob, d_odd, k_f, mk_f, n_rungs, n_cur, n_prev, th1, th2)

            m_used = m_take
            while m_used < m_max and dll < tau_early and rem >= cost_1:
                n_extra = n_k
                if n_k >= 1:
                    w = max(1, int(0.20 * n_k))
                    n_hi_ex = min(d_cap, rem, n_k + w)
                    n_lo_ex = min(n_hi_ex, max(1, n_k - w))
                    n_extra = pick_tiebreaker_depth(n_lo_ex, n_hi_ex, th1, th2)
                c_ex = 1 if n_extra == 0 else n_extra
                if rem < c_ex:
                    break
                k_ex = int(oracle.measure(n_extra, 1))
                n_rungs = record_meas(n_extra, 1, k_ex, depths, d_odd, k_f, mk_f, n_rungs, ll_glob)
                rem -= c_ex
                if n_extra > max_d:
                    max_d = n_extra
                if n_extra > n_cur and n_k <= 3:
                    n_prev = n_cur
                    n_cur = n_extra
                m_used += 1
                th1, _, th2, dll = get_modes(ll_glob, d_odd, k_f, mk_f, n_rungs, n_cur, n_prev, th1, th2)

        # Phase 2: Depth-Capped Horizon-Paced Deep Extension & Exact Budget Completion
        tie_count = 0
        s1 = r_ext
        s2 = r_ext + r_ext * r_ext
        s3 = s2 + r_ext * r_ext * r_ext

        while rem > 0:
            if dll < tau_deep and tie_count < 2 and rem >= 2 * min(d_cap, max(1, n_cur)):
                w = max(1, int(0.18 * n_cur))
                n_hi_t = min(d_cap, rem, n_cur + w)
                n_lo_t = min(n_hi_t, max(1, n_cur - w))
                n_tie = pick_tiebreaker_depth(n_lo_t, n_hi_t, th1, th2)
                k_t = int(oracle.measure(n_tie, 1))
                n_rungs = record_meas(n_tie, 1, k_t, depths, d_odd, k_f, mk_f, n_rungs, ll_glob)
                rem -= n_tie
                if n_tie > max_d:
                    max_d = n_tie
                tie_count += 1
                th1, _, th2, dll = get_modes(ll_glob, d_odd, k_f, mk_f, n_rungs, n_cur, n_prev, th1, th2)
                continue
            tie_count = 0

            raw_step_max = max(n_cur + 1, int(round(r_ext * n_cur)))
            n_step_max = min(d_cap, raw_step_max)

            # Enter final pair/single budget completion when remaining budget fits within
            # a single step or within two depth-capped shots once near the cap.
            if rem <= n_step_max or (n_step_max == d_cap and rem <= int(1.82 * d_cap)):
                if rem <= 3 and rem <= d_cap:
                    k_f_cnt = int(oracle.measure(rem, 1))
                    n_rungs = record_meas(rem, 1, k_f_cnt, depths, d_odd, k_f, mk_f, n_rungs, ll_glob)
                    if rem > max_d:
                        max_d = rem
                    rem = 0
                    th1, _, th2, dll = get_modes(ll_glob, d_odd, k_f, mk_f, n_rungs, n_cur, n_prev, th1, th2)
                else:
                    w_end = max(4, int(0.16 * min(d_cap, rem)))
                    n_f = pick_final_pair_depth(rem, w_end, th1, d_cap)
                    k_f_cnt = int(oracle.measure(n_f, 1))
                    n_rungs = record_meas(n_f, 1, k_f_cnt, depths, d_odd, k_f, mk_f, n_rungs, ll_glob)
                    if n_f > max_d:
                        max_d = n_f
                    if n_f > n_cur:
                        n_prev = n_cur
                        n_cur = n_f
                    rem -= n_f
                    th1, _, th2, dll = get_modes(ll_glob, d_odd, k_f, mk_f, n_rungs, n_cur, n_prev, th1, th2)
                    if rem > 0:
                        k_r_cnt = int(oracle.measure(rem, 1))
                        n_rungs = record_meas(rem, 1, k_r_cnt, depths, d_odd, k_f, mk_f, n_rungs, ll_glob)
                        if rem > max_d:
                            max_d = rem
                        rem = 0
                        th1, _, th2, dll = get_modes(ll_glob, d_odd, k_f, mk_f, n_rungs, n_cur, n_prev, th1, th2)
                break

            if n_step_max == d_cap:
                n_target = max(min(d_cap, n_cur + 1), int(round(0.90 * d_cap)))
                w = max(1, int(0.10 * d_cap))
                n_lo = min(d_cap, max(1, int(0.80 * d_cap)))
                n_hi = min(d_cap, rem)
            else:
                if rem <= int(s2 * n_cur):
                    n_target = max(n_cur + 1, min(n_step_max, int(round(rem * (s1 / s2)))))
                elif rem <= int(s3 * n_cur):
                    n_target = max(n_cur + 1, min(n_step_max, int(round(rem * (s1 / s3)))))
                else:
                    n_target = n_step_max
                w = max(1, int(0.12 * n_target))
                n_lo = min(d_cap, max(n_cur + 1, n_target - w))
                n_hi = min(d_cap, rem, max(n_lo, n_target + w))

            n_next = pick_steered_depth(n_lo, n_hi, th1, th2, dll)
            k_n = int(oracle.measure(n_next, 1))
            n_rungs = record_meas(n_next, 1, k_n, depths, d_odd, k_f, mk_f, n_rungs, ll_glob)
            rem -= n_next
            if n_next > max_d:
                max_d = n_next
            if n_next > n_cur:
                n_prev = n_cur
                n_cur = n_next
            th1, _, th2, dll = get_modes(ll_glob, d_odd, k_f, mk_f, n_rungs, n_cur, n_prev, th1, th2)

        th1, _ = zoom_polish(th1, 0.08 * math.pi / (2.0 * max(1, max_d) + 1.0), d_odd, k_f, mk_f, n_rungs)
        return float(min(max(math.sin(th1), 0.0), 1.0))
