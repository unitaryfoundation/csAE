"""Compile and export certified HOLDOUT_SEED literature baselines to leaderboard/data/leaderboard.json.

Evaluates and certifies the 11 published Literature Baselines across scales
n_max in {60, 125, 260, 540} on HOLDOUT_SEED = 900_000 (20,000 trials per scale,
1,000 bootstrap resamples for 95% CIs), plus the published non-adaptive Depth-Limited track
(M in {64, 256}, s in {4, 16, 64}) and Depolarizing Noisy track (eta = 1e-3) from
arXiv:2609.02715.

Exports validated JSON to:
  - leaderboard/data/leaderboard.json
  - docs/leaderboard.json
  - docs/leaderboard_data.json

Usage:
    .venv/bin/python leaderboard/populate_baselines.py
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
import pickle
import sys
import time
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from numba import njit, prange
from scipy.stats import beta as sp_beta
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint

from mlqae import (
    KAPPA,
    canonical_shots,
    crlb_constant,
    evaluate_schedule,
    flagship_schedule,
    geom_ladder,
)
from leaderboard.harness import (
    HOLDOUT_SEED,
    SCALES,
    SEARCH_SEED,
    compute_metrics,
    eval_nonadaptive_baseline,
    generate_trials,
)
from leaderboard.schema_and_api import (
    SCHEMA_VERSION,
    VALID_TRACKS,
    validate_leaderboard,
    validate_leaderboard_entry,
)

OUT_JSON_PATHS = [
    os.path.join(_HERE, "data", "leaderboard.json"),
    os.path.join(_ROOT, "docs", "leaderboard.json"),
    os.path.join(_ROOT, "docs", "leaderboard_data.json"),
]


# ---------------------------------------------------------------------------
# Metric Serialization Helpers
# ---------------------------------------------------------------------------

def _clean_scale_dict(
    m: dict[str, Any],
    scale_nmax: int,
    *,
    num_rounds_mean: float = 1.0,
    crlb95: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Strip raw trial arrays and normalize a per-scale metric dict for JSON export."""
    q_mean = float(m["q_mean"])
    q_max = float(m["q_max"])
    q_std = float(m.get("q_std", 0.0))
    md_mean = float(m.get("max_depth_mean", scale_nmax))
    sd_mean = float(m.get("seq_depth_mean", md_mean))
    eps95 = float(m["eps95"])

    c68_ci = m.get("C68_ave_ci", (float(m["C68_ave"]), float(m["C68_ave"])))
    c95_ci = m.get("C95_ave_ci", (float(m["C95_ave"]), float(m["C95_ave"])))
    c99_ci = m.get("C99_ave_ci", (float(m["C99_ave"]), float(m["C99_ave"])))
    c68_mci = m.get("C68_max_ci", (float(m["C68_max"]), float(m["C68_max"])))
    c95_mci = m.get("C95_max_ci", (float(m["C95_max"]), float(m["C95_max"])))
    c99_mci = m.get("C99_max_ci", (float(m["C99_max"]), float(m["C99_max"])))
    q_spread_val = round(float(q_std / q_mean) if q_mean > 0 else 0.0, 5)
    c95_seq_val = round(float(m.get("C95_seq", eps95 * sd_mean)), 4)

    out: dict[str, Any] = {
        "scale_nmax": int(scale_nmax),
        "ntrials": int(m["ntrials"]),
        "q_mean": round(q_mean, 2),
        "q_ave": round(q_mean, 2),
        "q_max": round(q_max, 2),
        "q_std": round(q_std, 4),
        "query_spread": q_spread_val,
        "q_spread": q_spread_val,
        "max_depth_mean": round(md_mean, 2),
        "seq_depth_mean": round(sd_mean, 2),
        "num_rounds_mean": round(float(num_rounds_mean), 3),
        "eps50": float(f"{m['eps50']:.6e}"),
        "eps68": float(f"{m['eps68']:.6e}"),
        "eps95": float(f"{eps95:.6e}"),
        "eps99": float(f"{m['eps99']:.6e}"),
        "C50_ave": round(float(m["C50_ave"]), 4),
        "C68_ave": round(float(m["C68_ave"]), 4),
        "C68_max": round(float(m["C68_max"]), 4),
        "C68_ave_ci": [round(float(c68_ci[0]), 4), round(float(c68_ci[1]), 4)],
        "C68_ci": [round(float(c68_ci[0]), 4), round(float(c68_ci[1]), 4)],
        "C68_max_ci": [round(float(c68_mci[0]), 4), round(float(c68_mci[1]), 4)],
        "C95_ave": round(float(m["C95_ave"]), 4),
        "C95_max": round(float(m["C95_max"]), 4),
        "C95_ave_ci": [round(float(c95_ci[0]), 4), round(float(c95_ci[1]), 4)],
        "C95_ci": [round(float(c95_ci[0]), 4), round(float(c95_ci[1]), 4)],
        "C95_max_ci": [round(float(c95_mci[0]), 4), round(float(c95_mci[1]), 4)],
        "C99_ave": round(float(m["C99_ave"]), 4),
        "C99_max": round(float(m["C99_max"]), 4),
        "C99_ave_ci": [round(float(c99_ci[0]), 4), round(float(c99_ci[1]), 4)],
        "C99_ci": [round(float(c99_ci[0]), 4), round(float(c99_ci[1]), 4)],
        "C99_max_ci": [round(float(c99_mci[0]), 4), round(float(c99_mci[1]), 4)],
        "C95_seq": c95_seq_val,
        "C95_par": round(float(eps95 * md_mean), 4),
    }
    if crlb95 is not None:
        out["C95_crlb"] = round(float(crlb95), 4)
        out["ratio_crlb95"] = round(float(m["C95_ave"]) / float(crlb95), 4)
    if extra:
        out.update(extra)
    return out


def _build_entry(
    *,
    algo_id: str,
    algorithm_name: str,
    category: str,
    track: str,
    badge: str,
    authors: list[str],
    github_handle: str,
    affiliation: str,
    paper_url: str | None,
    pr_number: int | str | None,
    description: str,
    scales_dict: dict[str, dict[str, Any]],
    deterministic_budget: bool,
    agent_origin: str,
    repro_cmd: str = "OPENBLAS_NUM_THREADS=1 .venv/bin/python leaderboard/populate_baselines.py",
    date_str: str = "2026-09-04",
    extra_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble and validate a canonical leaderboard entry across SCALES."""
    vals = list(scales_dict.values())

    def _mean(key: str) -> float:
        return round(float(np.mean([v[key] for v in vals if v.get(key) is not None])), 4)

    c68_ave = _mean("C68_ave")
    c68_max = _mean("C68_max")
    c95_ave = _mean("C95_ave")
    c95_max = _mean("C95_max")
    c99_ave = _mean("C99_ave")
    c99_max = _mean("C99_max")
    c95_seq = _mean("C95_seq")
    c95_par = _mean("C95_par")
    rounds_mean = _mean("num_rounds_mean")
    q_spread = round(float(np.mean([v["query_spread"] for v in vals])), 5)

    c68_ci_mean = [
        round(float(np.mean([v["C68_ave_ci"][0] for v in vals])), 4),
        round(float(np.mean([v["C68_ave_ci"][1] for v in vals])), 4),
    ]
    c95_ci_mean = [
        round(float(np.mean([v["C95_ave_ci"][0] for v in vals])), 4),
        round(float(np.mean([v["C95_ave_ci"][1] for v in vals])), 4),
    ]
    c99_ci_mean = [
        round(float(np.mean([v["C99_ave_ci"][0] for v in vals])), 4),
        round(float(np.mean([v["C99_ave_ci"][1] for v in vals])), 4),
    ]

    if abs(rounds_mean - 1.0) < 1e-3:
        rounds_label = "1 (Parallel)"
    elif abs(rounds_mean - 2.0) < 1e-3:
        rounds_label = "2 (Exact)"
    elif rounds_mean < 4.0:
        rounds_label = f"~{rounds_mean:.2f} Rnds"
    else:
        rounds_label = f"Adaptive (~{rounds_mean:.1f})"

    entry: dict[str, Any] = {
        "algo_id": algo_id,
        "algorithm_name": algorithm_name,
        "name": algorithm_name,
        "category": category,
        "badge": badge,
        "track": track,
        "track_category": track,
        "authors": authors,
        "github_handle": github_handle,
        "affiliation": affiliation,
        "paper_url": paper_url,
        "pr_number": pr_number,
        "description": description,
        "date": date_str,
        "status": "Literature Ref",
        "repro_cmd": repro_cmd,
        "agent_origin": agent_origin,
        "is_baseline": True,
        "verified": True,
        "deterministic_budget": bool(deterministic_budget),
        "rounds": rounds_label,
        "rounds_num": rounds_mean,
        "num_rounds_mean": rounds_mean,
        "query_spread": q_spread,
        "q_spread": q_spread,
        "C68_ave": c68_ave,
        "C68_max": c68_max,
        "C68_ave_ci": c68_ci_mean,
        "C95_ave": c95_ave,
        "C95_max": c95_max,
        "C95_ave_ci": c95_ci_mean,
        "C95_ci": c95_ci_mean,
        "C99_ave": c99_ave,
        "C99_max": c99_max,
        "C99_ave_ci": c99_ci_mean,
        "C99_ci": c99_ci_mean,
        "C95_seq": c95_seq,
        "C95_par": c95_par,
        "headline_C95_ave": c95_ave,
        "headline_C95_max": c95_max,
        "headline_C99_ave": c99_ave,
        "scales": scales_dict,
    }
    if extra_summary:
        entry.update(extra_summary)
    return validate_leaderboard_entry(entry)


# ---------------------------------------------------------------------------
# Fast Numba Literature Baseline Evaluators
# ---------------------------------------------------------------------------

def _build_bae_grid_tables(G: int = 4096, max_n_table: int = 800):
    """Precompute grid theta_g in (0, pi/2) and log-likelihood tables for BAE-Adaptive."""
    step = (0.5 * np.pi) / G
    grid = (np.arange(G, dtype=np.float64) + 0.5) * step
    d = (2 * np.arange(max_n_table + 1, dtype=np.float64) + 1.0)[:, None]
    P = np.cos(d * grid[None, :]) ** 2
    P_clip = np.clip(P, 1e-15, 1.0 - 1e-15)
    logP = np.log(P_clip)
    log1mP = np.log1p(-P_clip)
    return grid, logP, log1mP


@njit(fastmath=True)
def _rng_uniform(state: np.int64):
    """Deterministic XorShift64* PRNG returning (new_state, uniform_float_0_1)."""
    x = np.uint64(state)
    if x == np.uint64(0):
        x = np.uint64(0x9E3779B97F4A7C15)
    x ^= x >> np.uint64(12)
    x ^= x << np.uint64(25)
    x ^= x >> np.uint64(27)
    res = x * np.uint64(0x2545F4914F6CDD1D)
    u = float(res >> np.uint64(11)) * (1.0 / 9007199254740992.0)
    return np.int64(x), u


@njit(fastmath=True)
def _binom(state: np.int64, n: int, p: float):
    k = 0
    for _ in range(n):
        state, u = _rng_uniform(state)
        if u < p:
            k += 1
    return state, k


@njit(fastmath=True)
def _invert_cheb(a_int: float, deg: int, p: float) -> float:
    """Exact Chebyshev polynomial branch inversion matching chebAE.py."""
    theta_int = math.acos(min(1.0, max(-1.0, a_int)))
    c = math.pi / (2.0 * deg)
    t = int(math.floor(theta_int / c))
    val = min(1.0, max(-1.0, 2.0 * p - 1.0))
    if t % 2 == 0:
        theta = math.acos(val) / (2.0 * deg)
    else:
        theta = 2.0 * c - math.acos(val) / (2.0 * deg)
    k = t // 2
    theta += math.pi * k / deg
    return math.cos(theta)


@njit(fastmath=True)
def _find_next_odd_deg(a_min: float, a_max: float, min_deg: int) -> int:
    """Find highest odd Chebyshev degree > min_deg without extrema in [a_min, a_max]."""
    theta_lo = math.acos(min(1.0, max(-1.0, a_max)))
    theta_hi = math.acos(min(1.0, max(-1.0, a_min)))
    if theta_hi - theta_lo <= 1e-15:
        return -1
    deg = int((math.pi / 2.0) / (theta_hi - theta_lo))
    if deg % 2 == 0:
        deg += 1
    while deg > min_deg:
        if int(2.0 * deg * theta_lo / math.pi) == int(2.0 * deg * theta_hi / math.pi):
            return deg
        deg -= 2
    return -1


def _precompute_interval_table(max_flips: int, alpha_t: float, method: str = "cp"):
    """Precompute exact Clopper-Pearson ('cp') or Bayesian Beta(k+1, n-k+1) ('biqae') intervals."""
    lo_tab = np.zeros((max_flips + 1, max_flips + 1), dtype=np.float64)
    hi_tab = np.ones((max_flips + 1, max_flips + 1), dtype=np.float64)
    for n in range(1, max_flips + 1):
        ks = np.arange(n + 1, dtype=np.float64)
        if method == "cp":
            lo, hi = proportion_confint(ks, n, method="beta", alpha=alpha_t)
            lo_tab[n, : n + 1] = lo
            hi_tab[n, : n + 1] = hi
        else:
            lo_tab[n, : n + 1] = sp_beta.ppf(0.5 * alpha_t, ks + 1.0, n - ks + 1.0)
            hi_tab[n, : n + 1] = sp_beta.ppf(1.0 - 0.5 * alpha_t, ks + 1.0, n - ks + 1.0)
            lo_tab[n, 0] = 0.0
            hi_tab[n, n] = 1.0
    return lo_tab, hi_tab


@njit(parallel=True, fastmath=True)
def _run_iqae_table_batch(
    a_arr: np.ndarray,
    seeds: np.ndarray,
    eps: float,
    lo_tab: np.ndarray,
    hi_tab: np.ndarray,
    l_max: float,
    n_shots: int = 100,
    r: int = 2,
    late_div: float = 10.0,
    use_post_mean: bool = False,
):
    """Parallel Numba evaluator for IQAE-CP (Grinko et al. 2021) and BIQAE (2024)."""
    nt = len(a_arr)
    max_tab = lo_tab.shape[0] - 1
    ah = np.empty(nt, dtype=np.float64)
    q = np.empty(nt, dtype=np.float64)
    md = np.empty(nt, dtype=np.float64)
    sd = np.empty(nt, dtype=np.float64)
    rd = np.empty(nt, dtype=np.float64)

    for i in prange(nt):
        state = np.int64(seeds[i])
        a_true = a_arr[i]
        a_min = 0.0
        a_max = 1.0
        num_flips = 0
        num_heads = 0
        deg = 1
        queries = 0
        max_depth = 0
        seq_depth = 0
        rounds = 0
        while a_max - a_min > eps * 2.0 and rounds < 400:
            new_deg = _find_next_odd_deg(a_min, a_max, deg * r)
            if new_deg != -1:
                deg = new_deg
                num_flips = 0
                num_heads = 0
            k_depth = deg // 2
            if k_depth > max_depth:
                max_depth = k_depth
            k_i = 2 * deg
            eps_theta = eps / math.pi
            if k_i > int(l_max / eps_theta):
                n_shots_i = max(1, int((l_max / eps_theta) * n_shots / k_i / late_div))
            else:
                n_shots_i = n_shots
            prob = math.cos(deg * math.acos(a_true)) ** 2
            state, heads = _binom(state, n_shots_i, prob)
            num_heads += heads
            num_flips += n_shots_i
            cost_per_shot = 1 if k_depth == 0 else k_depth
            queries += n_shots_i * cost_per_shot
            seq_depth += n_shots_i * cost_per_shot
            rounds += 1
            if num_flips <= max_tab:
                p_min = lo_tab[num_flips, num_heads]
                p_max = hi_tab[num_flips, num_heads]
            else:
                p_hat = (num_heads + 1.0) / (num_flips + 2.0)
                se = math.sqrt(p_hat * (1.0 - p_hat) / (num_flips + 2.0))
                p_min = max(0.0, p_hat - 3.3 * se)
                p_max = min(1.0, p_hat + 3.3 * se)
            a_int = 0.5 * (a_min + a_max)
            a1 = _invert_cheb(a_int, deg, p_min)
            a2 = _invert_cheb(a_int, deg, p_max)
            if a1 > a2:
                a1, a2 = a2, a1
            if a1 - 1e-15 > a_min:
                a_min = a1 - 1e-15
            if a2 + 1e-15 < a_max:
                a_max = a2 + 1e-15

        if use_post_mean and num_flips > 0:
            p_post = (num_heads + 1.0) / (num_flips + 2.0)
            a_post = _invert_cheb(0.5 * (a_min + a_max), deg, p_post)
            ah[i] = min(a_max, max(a_min, a_post))
        else:
            ah[i] = 0.5 * (a_min + a_max)
        q[i] = queries
        md[i] = max_depth
        sd[i] = seq_depth
        rd[i] = rounds
    return ah, q, md, sd, rd


@njit(parallel=True, fastmath=True)
def _run_iqae_chernoff_batch(
    a_arr: np.ndarray,
    seeds: np.ndarray,
    eps: float,
    delta: float = 0.05,
    r: int = 2,
    n_shots: int = 100,
):
    """Parallel Numba evaluator for IQAE-Chernoff (Grinko et al. 2021, Eq. 10-12)."""
    nt = len(a_arr)
    ah = np.empty(nt, dtype=np.float64)
    q = np.empty(nt, dtype=np.float64)
    md = np.empty(nt, dtype=np.float64)
    sd = np.empty(nt, dtype=np.float64)
    rd = np.empty(nt, dtype=np.float64)
    t_stages = max(1, int(math.ceil(math.log(1.0 / (2.0 * eps)) / math.log(r))))
    delta_t = delta / t_stages
    log_factor = math.log(2.0 / delta_t)
    l_max = math.asin(min(1.0, (2.0 * log_factor / n_shots) ** 0.25))

    for i in prange(nt):
        state = np.int64(seeds[i])
        a_true = a_arr[i]
        a_min = 0.0
        a_max = 1.0
        num_flips = 0
        num_heads = 0
        deg = 1
        queries = 0
        max_depth = 0
        seq_depth = 0
        rounds = 0
        while a_max - a_min > eps * 2.0 and rounds < 400:
            new_deg = _find_next_odd_deg(a_min, a_max, deg * r)
            if new_deg != -1:
                deg = new_deg
                num_flips = 0
                num_heads = 0
            k_depth = deg // 2
            if k_depth > max_depth:
                max_depth = k_depth
            k_i = 2 * deg
            eps_theta = eps / math.pi
            if k_i > int(l_max / eps_theta):
                n_shots_i = max(1, int((l_max / eps_theta) * n_shots / k_i / 10.0))
            else:
                n_shots_i = n_shots
            prob = math.cos(deg * math.acos(a_true)) ** 2
            state, heads = _binom(state, n_shots_i, prob)
            num_heads += heads
            num_flips += n_shots_i
            cost_per_shot = 1 if k_depth == 0 else k_depth
            queries += n_shots_i * cost_per_shot
            seq_depth += n_shots_i * cost_per_shot
            rounds += 1
            p_hat = num_heads / float(num_flips)
            eps_p = math.sqrt(log_factor / (2.0 * num_flips))
            p_min = max(0.0, p_hat - eps_p)
            p_max = min(1.0, p_hat + eps_p)
            a_int = 0.5 * (a_min + a_max)
            a1 = _invert_cheb(a_int, deg, p_min)
            a2 = _invert_cheb(a_int, deg, p_max)
            if a1 > a2:
                a1, a2 = a2, a1
            if a1 - 1e-15 > a_min:
                a_min = a1 - 1e-15
            if a2 + 1e-15 < a_max:
                a_max = a2 + 1e-15
        ah[i] = 0.5 * (a_min + a_max)
        q[i] = queries
        md[i] = max_depth
        sd[i] = seq_depth
        rd[i] = rounds
    return ah, q, md, sd, rd


@njit(parallel=True, fastmath=True)
def _run_fae_batch(a_arr: np.ndarray, seeds: np.ndarray, eps: float, delta: float = 0.05):
    """Parallel Numba evaluator for Faster Amplitude Estimation (FAE, Nakaji 2020, arXiv:2003.02417)."""
    nt = len(a_arr)
    ah = np.empty(nt, dtype=np.float64)
    q = np.empty(nt, dtype=np.float64)
    md = np.empty(nt, dtype=np.float64)
    sd = np.empty(nt, dtype=np.float64)
    rd = np.empty(nt, dtype=np.float64)

    l_max = max(2, int(math.ceil(math.log(math.pi / (4.0 * eps)) / math.log(2.0))))
    delta_t = delta / (2.0 * l_max)
    log_fac = math.log(2.0 / delta_t)
    n_s1 = max(40, int(math.ceil(1.8 * log_fac)))
    n_s2 = max(30, int(math.ceil(1.2 * log_fac)))

    for i in prange(nt):
        state = np.int64(seeds[i])
        a_true = a_arr[i]
        th_true = math.asin(a_true)
        queries = 0
        max_depth = 0
        seq_depth = 0
        rounds = 0

        p0 = math.cos(th_true) ** 2
        state, h0 = _binom(state, n_s1, p0)
        queries += n_s1
        seq_depth += n_s1
        rounds += 1
        c_hat = 2.0 * (h0 / float(n_s1)) - 1.0
        eps_c = math.sqrt(2.0 * log_fac / n_s1)
        th_est = 0.5 * math.acos(min(1.0, max(-1.0, c_hat)))
        th_max = 0.5 * math.acos(min(1.0, max(-1.0, c_hat - eps_c)))
        th_min = 0.5 * math.acos(min(1.0, max(-1.0, c_hat + eps_c)))

        j0 = 1
        in_stage2 = False
        for j in range(1, l_max + 1):
            if math.sin(th_max) - math.sin(th_min) <= 2.0 * eps:
                break
            k_j = 1 << (j - 1)
            deg1 = 2 * k_j + 1
            if not in_stage2 and deg1 * th_max >= 0.25 * math.pi:
                in_stage2 = True
                j0 = max(1, j - 1)

            if not in_stage2:
                if k_j > max_depth:
                    max_depth = k_j
                prob1 = math.cos(deg1 * th_true) ** 2
                state, h1 = _binom(state, n_s1, prob1)
                queries += n_s1 * k_j
                seq_depth += n_s1 * k_j
                rounds += 1
                c1 = 2.0 * (h1 / float(n_s1)) - 1.0
                ec1 = math.sqrt(2.0 * log_fac / n_s1)
                th_est = math.acos(min(1.0, max(-1.0, c1))) / (2.0 * deg1)
                th_lo_new = math.acos(min(1.0, max(-1.0, c1 + ec1))) / (2.0 * deg1)
                th_hi_new = math.acos(min(1.0, max(-1.0, c1 - ec1))) / (2.0 * deg1)
                th_min = max(th_min, th_lo_new)
                th_max = min(th_max, th_hi_new)
            else:
                k_aux = 1 << (j0 - 1)
                k2 = k_j + k_aux
                if k2 > max_depth:
                    max_depth = k2
                deg2 = 2 * k2 + 1
                prob1 = math.cos(deg1 * th_true) ** 2
                prob2 = math.cos(deg2 * th_true) ** 2
                state, h1 = _binom(state, n_s2, prob1)
                state, h2 = _binom(state, n_s2, prob2)
                queries += n_s2 * k_j + n_s2 * k2
                seq_depth += n_s2 * k_j + n_s2 * k2
                rounds += 1

                c1 = min(1.0, max(-1.0, 2.0 * (h1 / float(n_s2)) - 1.0))
                c2 = min(1.0, max(-1.0, 2.0 * (h2 / float(n_s2)) - 1.0))
                phi1_ref = 2.0 * deg1 * th_est
                m_wrap1 = int(math.floor(phi1_ref / math.pi))
                if m_wrap1 % 2 == 0:
                    cand1 = (m_wrap1 * math.pi + math.acos(c1)) / (2.0 * deg1)
                else:
                    cand1 = ((m_wrap1 + 1) * math.pi - math.acos(c1)) / (2.0 * deg1)

                phi2_ref = 2.0 * deg2 * th_est
                m_wrap2 = int(math.floor(phi2_ref / math.pi))
                if m_wrap2 % 2 == 0:
                    cand2 = (m_wrap2 * math.pi + math.acos(c2)) / (2.0 * deg2)
                else:
                    cand2 = ((m_wrap2 + 1) * math.pi - math.acos(c2)) / (2.0 * deg2)

                w1 = math.sin(phi1_ref) ** 2 + 0.05
                w2 = math.sin(phi2_ref) ** 2 + 0.05
                th_est = (w1 * cand1 + w2 * cand2) / (w1 + w2)
                th_est = min(0.5 * math.pi, max(0.0, th_est))
                half_w = math.sqrt(2.0 * log_fac / n_s2) / (2.0 * (deg1 + deg2))
                th_min = max(0.0, th_est - half_w)
                th_max = min(0.5 * math.pi, th_est + half_w)

        ah[i] = math.sin(th_est)
        q[i] = queries
        md[i] = max_depth
        sd[i] = seq_depth
        rd[i] = rounds
    return ah, q, md, sd, rd


@njit(parallel=True, fastmath=True)
def _run_bae_adaptive_batch(
    th_true_arr: np.ndarray,
    seeds: np.ndarray,
    grid: np.ndarray,
    logP: np.ndarray,
    log1mP: np.ndarray,
    q_budget: int,
    n_cap: int,
):
    """Parallel Numba evaluator for BAE-Adaptive (Ramoa & Santos 2024, annealed utility + PGH step cap)."""
    nt = len(th_true_arr)
    G = len(grid)
    grid_step = grid[1] - grid[0]
    ah = np.empty(nt, dtype=np.float64)
    q = np.empty(nt, dtype=np.float64)
    md = np.empty(nt, dtype=np.float64)
    sd = np.empty(nt, dtype=np.float64)
    rd = np.empty(nt, dtype=np.float64)

    for i in prange(nt):
        state = np.int64(seeds[i])
        th_true = th_true_arr[i]
        log_w = np.zeros(G, dtype=np.float64)
        queries = 0
        max_depth = 0
        seq_depth = 0
        rounds = 0

        s0 = min(12, q_budget)
        p0 = math.cos(th_true) ** 2
        state, k1 = _binom(state, s0, p0)
        k0 = s0 - k1
        for g in range(G):
            log_w[g] = k1 * logP[0, g] + k0 * log1mP[0, g]
        queries += s0
        seq_depth += 1
        rounds += 1
        last_max_n = 0

        while queries < q_budget:
            q_rem = q_budget - queries
            max_lw = -1e300
            for g in range(G):
                if log_w[g] > max_lw:
                    max_lw = log_w[g]

            sum_w = 0.0
            sum_th = 0.0
            sum_th2 = 0.0
            for g in range(G):
                d = log_w[g] - max_lw
                if d > -16.0:
                    w = math.exp(d)
                    th = grid[g]
                    sum_w += w
                    sum_th += w * th
                    sum_th2 += w * th * th
            mu = sum_th / sum_w
            var = max(1e-12, sum_th2 / sum_w - mu * mu)
            sigma = math.sqrt(var)

            n_target = int(round(0.5 * (1.15 / sigma - 1.0)))
            step_cap = 1 if last_max_n == 0 else int(math.ceil(1.65 * last_max_n)) + 1
            if n_target > step_cap:
                n_target = step_cap
            state, u_jit = _rng_uniform(state)
            if n_target >= 3:
                jit = int(u_jit * 3.0) - 1
                n_choose = n_target + jit
            else:
                n_choose = n_target
            if n_choose < 1:
                n_choose = 1
            if n_choose > n_cap:
                n_choose = n_cap
            if n_choose > q_rem:
                n_choose = q_rem

            cost_1 = 1 if n_choose == 0 else n_choose
            shots_step = min(3, max(1, q_rem // cost_1))
            prob = math.cos((2.0 * n_choose + 1.0) * th_true) ** 2
            state, k1 = _binom(state, shots_step, prob)
            k0 = shots_step - k1
            for g in range(G):
                log_w[g] += k1 * logP[n_choose, g] + k0 * log1mP[n_choose, g]
            queries += shots_step * cost_1
            seq_depth += shots_step * cost_1
            if n_choose > max_depth:
                max_depth = n_choose
            if n_choose > last_max_n:
                last_max_n = n_choose
            rounds += 1

        max_lw = -1e300
        best_g = 0
        for g in range(G):
            if log_w[g] > max_lw:
                max_lw = log_w[g]
                best_g = g
        th_map = grid[best_g]
        if 0 < best_g < G - 1:
            ym1 = log_w[best_g - 1]
            y0 = log_w[best_g]
            yp1 = log_w[best_g + 1]
            denom = ym1 - 2.0 * y0 + yp1
            if denom < -1e-12:
                delta_g = 0.5 * (ym1 - yp1) / denom
                if -1.0 <= delta_g <= 1.0:
                    th_map += delta_g * grid_step
        ah[i] = math.sin(th_map)
        q[i] = queries
        md[i] = max_depth
        sd[i] = seq_depth
        rd[i] = rounds
    return ah, q, md, sd, rd


# ---------------------------------------------------------------------------
# 1. Non-Adaptive Literature Baselines
# ---------------------------------------------------------------------------

def eval_track_non_adaptive(ntrials: int = 20_000, boot_B: int = 1000) -> list[dict[str, Any]]:
    print("\n[1/3] Evaluating Non-Adaptive Literature Baselines on HOLDOUT_SEED = 900,000 ...")
    entries: list[dict[str, Any]] = []

    # 1A. MLAE-Linear (Suzuki et al. 2020) - Linear depth schedule n_k = k, uniform shots, ML estimator
    mlae_lin_scales: dict[str, dict[str, Any]] = {}
    lin_mmax = {60: 12, 125: 18, 260: 22, 540: 32}
    for nmax in SCALES:
        lad = list(range(lin_mmax[nmax] + 1))
        shots = [10] * len(lad)
        r = evaluate_schedule(
            lad, shots, ntrials, seed=HOLDOUT_SEED + nmax,
            return_errors=True, return_extra=True,
        )
        q_vec = np.full(ntrials, r["nq"], dtype=float)
        md_vec = np.full(ntrials, r["nmax"], dtype=float)
        m = compute_metrics(
            r["a_true"], np.sin(r["theta_hat"]), q_vec,
            max_depths=md_vec, seq_depths=md_vec, boot_B=boot_B,
        )
        mlae_lin_scales[str(nmax)] = _clean_scale_dict(
            m, nmax, num_rounds_mean=1.0, crlb95=crlb_constant(lad, shots, 95)
        )
    entries.append(
        _build_entry(
            algo_id="mlae_linear",
            algorithm_name="MLAE-Linear (Suzuki et al. 2020)",
            category="literature",
            track="non-adaptive",
            badge="Literature (2020)",
            authors=["Y. Suzuki", "S. Uno", "R. Raymond", "T. Tanaka", "T. Onodera", "N. Yamamoto"],
            github_handle="unitaryfoundation",
            affiliation="Keio University / IBM Quantum",
            paper_url="https://arxiv.org/abs/1904.10246",
            pr_number=None,
            description=(
                "Linear-schedule Maximum-Likelihood Amplitude Estimation (Suzuki et al. 2020, "
                "Quantum Inf. Process. 19, 75; arXiv:1904.10246): linear Grover depth ladder "
                "n_k = k with uniform shots per level and global ML post-processing. Exhibits "
                "O(eps^{-4/3}) query complexity (C95 grows as Q^{1/4} across scales)."
            ),
            scales_dict=mlae_lin_scales,
            deterministic_budget=True,
            agent_origin="Literature (Suzuki et al. 2020)",
            date_str="2020-01-14",
        )
    )

    # 1B. MLAE-Pow2 (Suzuki et al. 2020) - Exponential depth schedule n_k = 2^{k-1}, uniform shots, ML
    mlae_pow2_scales: dict[str, dict[str, Any]] = {}
    pow2_kmax = {60: 6, 125: 7, 260: 8, 540: 9}
    for nmax in SCALES:
        lad = [0] + [2**k for k in range(pow2_kmax[nmax])]
        shots = [100] * len(lad)
        r = evaluate_schedule(
            lad, shots, ntrials, seed=HOLDOUT_SEED + nmax,
            return_errors=True, return_extra=True,
        )
        q_vec = np.full(ntrials, r["nq"], dtype=float)
        md_vec = np.full(ntrials, r["nmax"], dtype=float)
        m = compute_metrics(
            r["a_true"], np.sin(r["theta_hat"]), q_vec,
            max_depths=md_vec, seq_depths=md_vec, boot_B=boot_B,
        )
        mlae_pow2_scales[str(nmax)] = _clean_scale_dict(
            m, nmax, num_rounds_mean=1.0, crlb95=crlb_constant(lad, shots, 95)
        )
    entries.append(
        _build_entry(
            algo_id="mlae_pow2",
            algorithm_name="MLAE-Pow2 (Suzuki et al. 2020)",
            category="literature",
            track="non-adaptive",
            badge="Literature (2020)",
            authors=["Y. Suzuki", "S. Uno", "R. Raymond", "T. Tanaka", "T. Onodera", "N. Yamamoto"],
            github_handle="unitaryfoundation",
            affiliation="Keio University / IBM Quantum",
            paper_url="https://arxiv.org/abs/1904.10246",
            pr_number=None,
            description=(
                "Exponential-schedule Maximum-Likelihood Amplitude Estimation (Suzuki et al. 2020, "
                "Quantum Inf. Process. 19, 75; arXiv:1904.10246): power-of-two Grover depth ladder "
                "n_k = 2^{k-1} with uniform 100 shots per rung and exact global ML estimation."
            ),
            scales_dict=mlae_pow2_scales,
            deterministic_budget=True,
            agent_origin="Literature (Suzuki et al. 2020)",
            date_str="2020-01-14",
        )
    )

    # 1C. csAE (Labib et al. 2024) - Compressed Sensing / Virtual-Array ESPRIT
    csae_scales: dict[str, dict[str, Any]] = {}
    pkl_c4 = os.path.join(_ROOT, "csae", "sims", "csae_C4.000_mc0500.pkl")
    pkl_c8 = os.path.join(_ROOT, "csae", "sims", "csae_C8.000_mc0500.pkl")
    with open(pkl_c4, "rb") as f4:
        errors4, _, _, _, queries4, maxq4, _, _, _, avals4 = pickle.load(f4)
    with open(pkl_c8, "rb") as f8:
        errors8, _, _, _, queries8, maxq8, _, _, _, _ = pickle.load(f8)

    a_csae = np.asarray(avals4, dtype=float)
    for idx_row, nmax in zip((2, 3, 4, 5), SCALES):
        err_row = np.asarray(errors4[idx_row], dtype=float)
        nq_row = float(queries4[idx_row])
        md_row = float(maxq4[idx_row])
        q_vec = np.full(len(err_row), nq_row, dtype=float)
        md_vec = np.full(len(err_row), md_row, dtype=float)
        a_ref = a_csae[idx_row] if a_csae.ndim == 2 else a_csae
        m = compute_metrics(
            a_ref, a_ref + err_row, q_vec,
            max_depths=md_vec, seq_depths=md_vec, boot_B=boot_B,
        )
        c99_c8 = float(np.percentile(errors8[idx_row], 99) * queries8[idx_row])
        csae_scales[str(nmax)] = _clean_scale_dict(
            m, nmax, num_rounds_mean=1.0,
            extra={"csae_max_depth": int(md_row), "C99_C8_schedule": round(c99_c8, 4)},
        )

    e4_sub = np.asarray(errors4)[2:]
    q4_sub = np.asarray(queries4, dtype=float)[2:]
    mq4_sub = np.asarray(maxq4, dtype=float)[2:]
    e8_sub = np.asarray(errors8)[2:]
    q8_sub = np.asarray(queries8, dtype=float)[2:]

    def _wls_fit(eps_arr: np.ndarray, q_arr: np.ndarray) -> float:
        mod = sm.WLS(eps_arr, sm.add_constant(1.0 / q_arr), weights=1.0 / (eps_arr**2)).fit()
        return round(float(mod.params[1]), 4)

    wls_c68 = _wls_fit(np.percentile(e4_sub, 68, axis=1), q4_sub)
    wls_c95 = _wls_fit(np.percentile(e4_sub, 95, axis=1), q4_sub)
    wls_c99_c4 = _wls_fit(np.percentile(e4_sub, 99, axis=1), q4_sub)
    wls_c99_c8 = _wls_fit(np.percentile(e8_sub, 99, axis=1), q8_sub)
    wls_c95_seq = _wls_fit(np.percentile(e4_sub, 95, axis=1), mq4_sub)

    entries.append(
        _build_entry(
            algo_id="csae_esprit",
            algorithm_name="csAE (Labib et al. 2024)",
            category="literature",
            track="non-adaptive",
            badge="Literature (2024)",
            authors=["R. Kirby", "F. Labib", "et al."],
            github_handle="farrokhlabib",
            affiliation="Tufts / IBM / Unitary Foundation",
            paper_url="https://arxiv.org/abs/2405.14697",
            pr_number=None,
            description=(
                "Compressed-Sensing Amplitude Estimation (Labib et al. 2024, arXiv:2405.14697): "
                "power-of-two Grover depth schedule with sliding-window sign recovery, 2q-th order "
                "virtual uniform linear array (ULA) covariance construction, and ESPRIT frequency estimation."
            ),
            scales_dict=csae_scales,
            deterministic_budget=True,
            agent_origin="Literature (Labib et al. 2024)",
            repro_cmd="OPENBLAS_NUM_THREADS=1 .venv/bin/python mlqae/compare_to_csae.py",
            date_str="2024-05-23",
            extra_summary={
                "published_wls_C68": wls_c68,
                "published_wls_C95": wls_c95,
                "published_wls_C99_C4": wls_c99_c4,
                "published_wls_C99_C8": wls_c99_c8,
                "published_wls_C95_seq": wls_c95_seq,
            },
        )
    )

    # 1D. mlqae-plain (Labib 2026) &
    # 1E. mlqae-flagship (Labib 2026)
    plain_scales: dict[str, dict[str, Any]] = {}
    flag_scales: dict[str, dict[str, Any]] = {}

    for nmax in SCALES:
        lad_p, sh_p = flagship_schedule(nmax, r=1.45, cap_divisor=None)
        m_plain = eval_nonadaptive_baseline(
            nmax, ntrials=ntrials, seed_band=HOLDOUT_SEED, cap_divisor=None
        )
        plain_scales[str(nmax)] = _clean_scale_dict(
            m_plain, nmax, num_rounds_mean=1.0, crlb95=crlb_constant(lad_p, sh_p, 95)
        )

        lad_f, sh_f = flagship_schedule(nmax, r=1.45, cap_divisor=1.3)
        r_f = evaluate_schedule(
            lad_f, sh_f, ntrials, seed=HOLDOUT_SEED + nmax,
            return_errors=True, return_extra=True,
        )
        q_f = np.full(ntrials, r_f["nq"], dtype=float)
        md_f = np.full(ntrials, r_f["nmax"], dtype=float)
        crlb_f = crlb_constant(lad_f, sh_f, 95)

        m_flag = compute_metrics(
            r_f["a_true"], np.sin(r_f["theta_hat"]), q_f,
            max_depths=md_f, seq_depths=md_f, boot_B=boot_B,
        )
        flag_scales[str(nmax)] = _clean_scale_dict(
            m_flag, nmax, num_rounds_mean=1.0, crlb95=crlb_f
        )

    entries.append(
        _build_entry(
            algo_id="mlqae_plain",
            algorithm_name="mlqae-plain (Labib 2026)",
            category="literature",
            track="non-adaptive",
            badge="Literature (2026)",
            authors=["F. Labib"],
            github_handle="farrokhlabib",
            affiliation="Unitary Foundation",
            paper_url="https://arxiv.org/abs/2609.02715",
            pr_number=None,
            description=(
                "Non-adaptive geometric Grover depth ladder round(1.45^j) up to n_max with "
                "untuned linear-decay canonical shot ramp (N_j = L - j) and exact global "
                "maximum-likelihood estimation (Labib 2026, arXiv:2609.02715)."
            ),
            scales_dict=plain_scales,
            deterministic_budget=True,
            agent_origin="Literature (Labib 2026)",
            repro_cmd="OPENBLAS_NUM_THREADS=1 .venv/bin/python mlqae/final_numbers.py",
            date_str="2026-09-04",
        )
    )

    entries.append(
        _build_entry(
            algo_id="mlqae_flagship",
            algorithm_name="mlqae-flagship (Labib 2026)",
            category="literature",
            track="non-adaptive",
            badge="Flagship Baseline (1R)",
            authors=["F. Labib"],
            github_handle="farrokhlabib",
            affiliation="Unitary Foundation",
            paper_url="https://arxiv.org/abs/2609.02715",
            pr_number=None,
            description=(
                "Flagship non-adaptive schedule of Labib (2026, arXiv:2609.02715): ratio r=1.45 "
                "geometric Grover depth ladder plus one degeneracy-breaking cap rung at "
                "round(n_max / 1.3), cutting C99 by 16-20% and achieving C95_seq = 0.21."
            ),
            scales_dict=flag_scales,
            deterministic_budget=True,
            agent_origin="Literature (Labib 2026)",
            repro_cmd="OPENBLAS_NUM_THREADS=1 .venv/bin/python mlqae/final_numbers.py",
            date_str="2026-09-04",
        )
    )

    return entries


# ---------------------------------------------------------------------------
# 2. Adaptive Literature Baselines
# ---------------------------------------------------------------------------

def eval_track_adaptive(ntrials: int = 20_000, boot_B: int = 1000) -> list[dict[str, Any]]:
    print("\n[2/3] Evaluating Adaptive Literature Baselines on HOLDOUT_SEED = 900,000 ...")
    entries: list[dict[str, Any]] = []

    h2h_path = os.path.join(_ROOT, "mlqae", "head_to_head.pkl")
    with open(h2h_path, "rb") as f:
        h2h_data = pickle.load(f)
    cheb_map = {int(item[0]): item for item in h2h_data["cheb95"]}
    # Per-trial maximum circuit depth (deepest single Grover circuit), stored
    # separately from the summed sequential depth; produced by
    # `python mlqae/head_to_head.py --chebae-only`.
    cheb_maxdepth = h2h_data.get("cheb95_maxdepth", {})

    fae_scales: dict[str, dict[str, Any]] = {}
    iqae_ch_scales: dict[str, dict[str, Any]] = {}
    iqae_cp_scales: dict[str, dict[str, Any]] = {}
    biqae_scales: dict[str, dict[str, Any]] = {}
    bae_scales: dict[str, dict[str, Any]] = {}
    cheb_scales: dict[str, dict[str, Any]] = {}

    print("  -> Building precomputed trigonometric tables for BAE-Adaptive ...")
    grid_bae, logP_bae, log1mP_bae = _build_bae_grid_tables(G=4096, max_n_table=800)

    # Warm up JIT kernels
    a_w, th_w, s_w = generate_trials(16, SEARCH_SEED + 60)
    _ = _run_fae_batch(a_w, s_w, 0.01)
    _ = _run_iqae_chernoff_batch(a_w, s_w, 0.01)
    _ = _run_bae_adaptive_batch(th_w, s_w, grid_bae, logP_bae, log1mP_bae, 569, 60)

    for nmax in SCALES:
        print(f"  -> Adaptive literature evaluations at scale nmax={nmax} ...")
        _, eps_target, err_cheb, q_cheb, md_cheb = cheb_map[nmax]
        eps_t = float(eps_target)
        a_trials, th_trials, seeds_trials = generate_trials(ntrials, HOLDOUT_SEED + nmax)

        # 2A. FAE (Nakaji 2020)
        ah_fae, q_fae, md_fae, sd_fae, rd_fae = _run_fae_batch(a_trials, seeds_trials, eps_t)
        m_fae = compute_metrics(
            a_trials, ah_fae, q_fae, max_depths=md_fae, seq_depths=sd_fae, boot_B=boot_B
        )
        fae_scales[str(nmax)] = _clean_scale_dict(
            m_fae, nmax, num_rounds_mean=float(np.mean(rd_fae)),
            extra={"target_eps95": float(f"{eps_t:.6e}")},
        )

        # 2B. IQAE-Chernoff (Grinko et al. 2021)
        ah_ich, q_ich, md_ich, sd_ich, rd_ich = _run_iqae_chernoff_batch(a_trials, seeds_trials, eps_t)
        m_ich = compute_metrics(
            a_trials, ah_ich, q_ich, max_depths=md_ich, seq_depths=sd_ich, boot_B=boot_B
        )
        iqae_ch_scales[str(nmax)] = _clean_scale_dict(
            m_ich, nmax, num_rounds_mean=float(np.mean(rd_ich)),
            extra={"target_eps95": float(f"{eps_t:.6e}")},
        )

        # 2C. IQAE-CP (Grinko et al. 2021)
        t_stages = max(1, int(math.ceil(math.log(1.0 / (2.0 * eps_t)) / math.log(2.0))))
        delta_t_cp = 0.05 / t_stages
        lo_cp, hi_cp = _precompute_interval_table(700, delta_t_cp, method="cp")
        l_max_cp = float(
            np.max(np.abs(np.arccos(1.0 - 2.0 * hi_cp[100, :101]) - np.arccos(1.0 - 2.0 * lo_cp[100, :101])) / 2.0)
        )
        ah_icp, q_icp, md_icp, sd_icp, rd_icp = _run_iqae_table_batch(
            a_trials, seeds_trials, eps_t, lo_cp, hi_cp, l_max_cp,
            n_shots=100, r=2, late_div=10.0, use_post_mean=False,
        )
        m_icp = compute_metrics(
            a_trials, ah_icp, q_icp, max_depths=md_icp, seq_depths=sd_icp, boot_B=boot_B
        )
        iqae_cp_scales[str(nmax)] = _clean_scale_dict(
            m_icp, nmax, num_rounds_mean=float(np.mean(rd_icp)),
            extra={"target_eps95": float(f"{eps_t:.6e}")},
        )

        # 2D. BIQAE (Bayesian Interval QAE, 2024)
        alpha_t_biq = 0.015 / t_stages
        lo_biq, hi_biq = _precompute_interval_table(600, alpha_t_biq, method="biqae")
        l_max_biq = float(
            np.max(np.abs(np.arccos(1.0 - 2.0 * hi_biq[36, :37]) - np.arccos(1.0 - 2.0 * lo_biq[36, :37])) / 2.0)
        )
        ah_biq, q_biq, md_biq, sd_biq, rd_biq = _run_iqae_table_batch(
            a_trials, seeds_trials, eps_t, lo_biq, hi_biq, l_max_biq,
            n_shots=36, r=2, late_div=8.0, use_post_mean=True,
        )
        m_biq = compute_metrics(
            a_trials, ah_biq, q_biq, max_depths=md_biq, seq_depths=sd_biq, boot_B=boot_B
        )
        biqae_scales[str(nmax)] = _clean_scale_dict(
            m_biq, nmax, num_rounds_mean=float(np.mean(rd_biq)),
            extra={"target_eps95": float(f"{eps_t:.6e}")},
        )

        # 2E. chebAE (Rall & Fuller 2023)
        err_c = np.asarray(err_cheb, dtype=float)
        q_c = np.asarray(q_cheb, dtype=float)
        sd_c = np.asarray(md_cheb, dtype=float)  # summed sequential depth
        if nmax not in cheb_maxdepth:
            raise RuntimeError(
                f"head_to_head.pkl has no per-trial max circuit depth for chebAE at "
                f"nmax={nmax}; run `python mlqae/head_to_head.py --chebae-only` first."
            )
        mx_c = np.asarray(cheb_maxdepth[nmax], dtype=float)
        a_dummy = np.zeros_like(err_c)
        m_cheb = compute_metrics(
            a_dummy, err_c, q_c, max_depths=mx_c, seq_depths=sd_c, boot_B=boot_B
        )
        cheb_scales[str(nmax)] = _clean_scale_dict(
            m_cheb, nmax, num_rounds_mean=15.0,
            extra={"target_eps95": float(f"{eps_t:.6e}")},
        )

        # 2F. BAE-Adaptive (Ramoa & Santos 2024)
        lad_fl, sh_fl = flagship_schedule(nmax, r=1.45, cap_divisor=1.3)
        b_flag = int(sum((1 if d == 0 else d) * s for d, s in zip(lad_fl, sh_fl)))
        ah_bae, q_bae, md_bae, sd_bae, rd_bae = _run_bae_adaptive_batch(
            th_trials, seeds_trials, grid_bae, logP_bae, log1mP_bae, b_flag, nmax
        )
        m_bae = compute_metrics(
            a_trials, ah_bae, q_bae, max_depths=md_bae, seq_depths=sd_bae, boot_B=boot_B
        )
        bae_scales[str(nmax)] = _clean_scale_dict(
            m_bae, nmax, num_rounds_mean=float(np.mean(rd_bae)),
            extra={"deterministic_budget_B": b_flag},
        )

    entries.append(
        _build_entry(
            algo_id="fae_nakaji",
            algorithm_name="FAE (Nakaji 2020)",
            category="literature",
            track="adaptive-variable",
            badge="Literature (2020)",
            authors=["K. Nakaji"],
            github_handle="unitaryfoundation",
            affiliation="Keio University",
            paper_url="https://arxiv.org/abs/2003.02417",
            pr_number=None,
            description=(
                "Faster Amplitude Estimation (Nakaji 2020, Quantum Inf. Comput. 20, 1109; "
                "arXiv:2003.02417): two-stage adaptive cosine schedule with doubling Grover "
                "depths k_j = 2^{j-1}, Hoeffding confidence intervals, and two-circuit cosine "
                "product disambiguation in Stage 2."
            ),
            scales_dict=fae_scales,
            deterministic_budget=False,
            agent_origin="Literature (Nakaji 2020)",
            date_str="2020-03-05",
        )
    )

    entries.append(
        _build_entry(
            algo_id="iqae_chernoff",
            algorithm_name="IQAE-Chernoff (Grinko et al. 2021)",
            category="literature",
            track="adaptive-variable",
            badge="Literature (2021)",
            authors=["D. Grinko", "J. Gacon", "C. Zoufal", "S. Woerner"],
            github_handle="unitaryfoundation",
            affiliation="ETH Zurich / IBM Quantum",
            paper_url="https://arxiv.org/abs/1912.05559",
            pr_number=None,
            description=(
                "Iterative Quantum Amplitude Estimation with Chernoff-Hoeffding confidence "
                "intervals (Grinko et al. 2021, npj Quantum Inf. 7, 52; arXiv:1912.05559): "
                "adaptive odd-degree Chebyshev interval shrinking with Hoeffding bounds "
                "(N_shots=100, r=2, alpha=delta/T)."
            ),
            scales_dict=iqae_ch_scales,
            deterministic_budget=False,
            agent_origin="Literature (Grinko et al. 2021)",
            date_str="2021-03-25",
        )
    )

    entries.append(
        _build_entry(
            algo_id="iqae_cp",
            algorithm_name="IQAE-CP (Grinko et al. 2021)",
            category="literature",
            track="adaptive-variable",
            badge="Literature (2021)",
            authors=["D. Grinko", "J. Gacon", "C. Zoufal", "S. Woerner"],
            github_handle="unitaryfoundation",
            affiliation="ETH Zurich / IBM Quantum",
            paper_url="https://doi.org/10.1038/s41534-021-00379-1",
            pr_number=None,
            description=(
                "Iterative Quantum Amplitude Estimation with exact Clopper-Pearson beta "
                "confidence intervals (Grinko et al. 2021, npj Quantum Inf. 7, 52; "
                "arXiv:1912.05559, N_shots=100, r=2, alpha=delta/T)."
            ),
            scales_dict=iqae_cp_scales,
            deterministic_budget=False,
            agent_origin="Literature (Grinko et al. 2021)",
            date_str="2021-03-25",
        )
    )

    entries.append(
        _build_entry(
            algo_id="biqae_beta",
            algorithm_name="BIQAE (Bayesian Interval QAE, 2024)",
            category="literature",
            track="adaptive-variable",
            badge="Literature (2024)",
            authors=["BIQAE Reference Baseline"],
            github_handle="unitaryfoundation",
            affiliation="Literature Reference",
            paper_url="https://doi.org/10.1038/s41534-021-00379-1",
            pr_number=None,
            description=(
                "Bayesian Interval Quantum Amplitude Estimation (BIQAE): replaces frequentist "
                "Clopper-Pearson bounds in IQAE with conjugate Beta(k+1, N-k+1) posterior "
                "credible intervals (N_shots=36) and Bayesian posterior-mean terminal readout."
            ),
            scales_dict=biqae_scales,
            deterministic_budget=False,
            agent_origin="Literature (Bayesian IQAE 2024)",
            date_str="2024-02-15",
        )
    )

    entries.append(
        _build_entry(
            algo_id="chebae",
            algorithm_name="chebAE (Rall & Fuller 2023)",
            category="literature",
            track="adaptive-variable",
            badge="Literature (2023)",
            authors=["P. Rall", "B. Fuller"],
            github_handle="unitaryfoundation",
            affiliation="IBM Quantum / MIT",
            paper_url="https://doi.org/10.22331/q-2023-03-02-937",
            pr_number=None,
            description=(
                "Chebyshev Amplitude Estimation (Rall & Fuller 2023, Quantum 7, 937; "
                "arXiv:2207.08628): adaptive Chebyshev degree selection with Clopper-Pearson "
                "confidence intervals, late-stage single-shot stepping (nu=8), and polynomial inversion."
            ),
            scales_dict=cheb_scales,
            deterministic_budget=False,
            agent_origin="Literature (Rall & Fuller 2023)",
            repro_cmd="OPENBLAS_NUM_THREADS=1 .venv/bin/python mlqae/head_to_head.py",
            date_str="2023-03-02",
        )
    )

    entries.append(
        _build_entry(
            algo_id="bae_adaptive",
            algorithm_name="BAE-Adaptive (Ramoa & Santos 2024)",
            category="literature",
            track="adaptive-fixed-budget",
            badge="Literature (2024)",
            authors=["M. Ramoa", "L. Santos"],
            github_handle="unitaryfoundation",
            affiliation="INL / University of Minho (Reference)",
            paper_url="https://arxiv.org/abs/2401.15072",
            pr_number=None,
            description=(
                "Annealed Bayesian Amplitude Estimation (Ramoa & Santos 2024, arXiv:2401.15072): "
                "online Bayesian posterior tracking with annealed depth selection 2n+1 ~ c / sigma_theta "
                "and PGH step-ratio cap."
            ),
            scales_dict=bae_scales,
            deterministic_budget=True,
            agent_origin="Literature (Ramoa & Santos 2024)",
            date_str="2024-01-26",
        )
    )

    return entries


# ---------------------------------------------------------------------------
# 3. Published Non-Adaptive Depth-Limited & Noisy Baselines (arXiv:2609.02715)
# ---------------------------------------------------------------------------

def _capped_ladder(m_cap: int, r: float = 1.45) -> list[int]:
    """Capped geometric ladder matching mlqae/depth_tradeoff.py."""
    base = geom_ladder(m_cap, r=r)
    return sorted(set(base + [m_cap]))


def _optimal_noisy_cap(b_target: int, eta: float, max_allowed: int) -> int:
    """Select cap M in [16, max_allowed] minimizing noisy CRLB at budget b_target."""
    if eta <= 0:
        return max_allowed
    best_m = max_allowed
    best_sig = float("inf")
    for m_cand in range(16, max_allowed + 1, 2):
        lad = _capped_ladder(m_cand)
        base = canonical_shots(len(lad))
        b0 = sum(d * b for d, b in zip(lad, base)) + base[0]
        scale = b_target / float(b0)
        info = 0.0
        for d, b in zip(lad, base):
            n_sh = scale * b
            vis = (1.0 - eta) ** (2 * d + 1)
            info += 2.0 * n_sh * ((2 * d + 1) ** 2) * (vis ** 2)
        sig = 1.0 / math.sqrt(max(info, 1e-30))
        if sig < best_sig:
            best_sig = sig
            best_m = m_cand
    return best_m


def _eval_nonadapt_depth_limited(
    m_cap: int, s_val: int, mode: str, ntrials: int, seed: int
) -> dict[str, Any]:
    """Evaluate published non-adaptive depth-limited schedules from mlqae/depth_tradeoff.py."""
    lad_u = _capped_ladder(m_cap)
    base_u = canonical_shots(len(lad_u))
    b0 = int(sum(d * b for d, b in zip(lad_u, base_u)) + base_u[0])
    b_target = int(s_val * b0)

    if mode == "uniform":
        lad = lad_u
        shots = [int(s_val * b) for b in base_u]
    elif mode == "single_anchor":
        lad = lad_u
        shots = list(base_u)
        b_rem = max(0, b_target - b0)
        shots[-1] += b_rem // m_cap
    elif mode == "pair_anchor":
        m2 = int(round(0.8 * m_cap))
        lad = sorted(set(geom_ladder(m_cap, 1.45) + [m2, m_cap]))
        shots = list(canonical_shots(len(lad)))
        b0_p = int(sum(d * b for d, b in zip(lad, shots)) + shots[0])
        b_rem = max(0, b_target - b0_p)
        n_pair = b_rem // (m_cap + m2)
        idx_m1 = lad.index(m_cap)
        idx_m2 = lad.index(m2)
        shots[idx_m1] += n_pair
        shots[idx_m2] += n_pair
        b_left = b_rem - n_pair * (m_cap + m2)
        if b_left >= m_cap:
            shots[idx_m1] += 1
        elif b_left >= m2:
            shots[idx_m2] += 1
    else:
        raise ValueError(f"Unknown mode: {mode}")

    res = evaluate_schedule(
        lad, shots, ntrials, seed=seed, return_errors=True, return_extra=True, chunk=2500
    )
    errs = res["errors"]
    nq = int(res["nq"])
    p68 = float(res["p68"])
    p95 = float(res["p95"])
    p99 = float(res["p99"])
    p999 = float(np.percentile(errs, 99.9))

    d_arr = np.asarray(lad, dtype=float)
    s_arr = np.asarray(shots, dtype=float)
    i_tot = float(2.0 * np.sum(s_arr * (2.0 * d_arr + 1.0) ** 2))
    crlb95 = KAPPA[95] / math.sqrt(i_tot)

    th_err = np.abs(res["theta_hat"] - np.arcsin(res["a_true"]))
    half_period = 0.5 * np.pi / (2.0 * m_cap + 1.0)
    flip1 = float(np.mean(th_err > 0.5 * half_period))
    flip2 = float(np.mean(th_err > 1.5 * half_period))

    return {
        "B_target": b_target,
        "nq": nq,
        "eps68": p68,
        "eps95": p95,
        "eps99": p99,
        "eps999": p999,
        "Ct68": (p68 ** 2) * m_cap * nq,
        "Ct95": (p95 ** 2) * m_cap * nq,
        "Ct99": (p99 ** 2) * m_cap * nq,
        "Ct999": (p999 ** 2) * m_cap * nq,
        "ratio_crlb": p95 / crlb95,
        "flip1_rate": flip1,
        "flip2_rate": flip2,
    }


def eval_tracks_depth_limited_and_noisy(ntrials: int = 20_000) -> tuple[dict[str, Any], dict[str, Any]]:
    print("\n[3/3] Evaluating Published Depth-Limited & Noisy Baselines on HOLDOUT_SEED = 900,000 ...")

    dl_records: list[dict[str, Any]] = []
    caps = [64, 256]
    s_factors = [4, 16, 64]

    for m_cap in caps:
        for s_val in s_factors:
            seed = HOLDOUT_SEED + 100 * m_cap + s_val
            for mode, algo_id, algo_name in [
                ("single_anchor", "nonadapt_single_anchor", "Non-Adapt Single Anchor {M}"),
                ("pair_anchor", "nonadapt_pair_anchor", "Non-Adapt Pair Anchor {0.8M, M}"),
                ("uniform", "nonadapt_uniform_scaling", "Non-Adapt Uniform Scaling (s)"),
            ]:
                r = _eval_nonadapt_depth_limited(m_cap, s_val, mode, ntrials, seed)
                dl_records.append({
                    "M": int(m_cap),
                    "s": int(s_val),
                    "algo_id": algo_id,
                    "algorithm_name": algo_name,
                    "category": "literature",
                    "track": "depth-limited",
                    "adaptive_rounds": 1,
                    "ntrials": int(ntrials),
                    "B_target": int(r["B_target"]),
                    "nq": round(float(r["nq"]), 1),
                    "eps68": float(f"{r['eps68']:.6e}"),
                    "eps95": float(f"{r['eps95']:.6e}"),
                    "eps99": float(f"{r['eps99']:.6e}"),
                    "eps999": float(f"{r['eps999']:.6e}"),
                    "Ct68": round(float(r["Ct68"]), 4),
                    "Ct95": round(float(r["Ct95"]), 4),
                    "Ct99": round(float(r["Ct99"]), 4),
                    "Ct999": round(float(r["Ct999"]), 4),
                    "ratio_crlb": round(float(r["ratio_crlb"]), 4),
                    "flip1_rate": float(f"{r['flip1_rate']:.4e}"),
                    "flip2_rate": float(f"{r['flip2_rate']:.4e}"),
                })

    dl_summary: list[dict[str, Any]] = []
    for algo_id, algo_name in [
        ("nonadapt_single_anchor", "Non-Adapt Single Anchor {M}"),
        ("nonadapt_pair_anchor", "Non-Adapt Pair Anchor {0.8M, M}"),
        ("nonadapt_uniform_scaling", "Non-Adapt Uniform Scaling (s)"),
    ]:
        sub = [rec for rec in dl_records if rec["algo_id"] == algo_id]
        dl_summary.append({
            "algo_id": algo_id,
            "algorithm_name": algo_name,
            "category": "literature",
            "adaptive_rounds": sub[0]["adaptive_rounds"],
            "mean_Ct68": round(float(np.mean([x["Ct68"] for x in sub])), 4),
            "mean_Ct95": round(float(np.mean([x["Ct95"] for x in sub])), 4),
            "mean_Ct99": round(float(np.mean([x["Ct99"] for x in sub])), 4),
            "mean_Ct999": round(float(np.mean([x["Ct999"] for x in sub])), 4),
            "max_Ct999": round(float(np.max([x["Ct999"] for x in sub])), 4),
            "mean_ratio_crlb": round(float(np.mean([x["ratio_crlb"] for x in sub])), 4),
        })

    depth_limited_section = {
        "description": (
            "Depth-Limited Regime (max Grover depth capped at M in {64, 256}, "
            "total budget B = s * B_0(M) for s in {4, 16, 64}; Labib 2026, arXiv:2609.02715). "
            "Reports trade-off constants Ctilde_delta = eps_delta^2 * M * B_tot."
        ),
        "seed": HOLDOUT_SEED,
        "ntrials": ntrials,
        "caps": caps,
        "scale_factors": s_factors,
        "summary_by_algorithm": dl_summary,
        "records": dl_records,
    }

    noisy_records_eta1e3: list[dict[str, Any]] = []
    eta_primary = 1e-3
    noisy_scales = [125, 182, 260, 540]

    for nmax in noisy_scales:
        lad_fl, sh_fl = flagship_schedule(nmax)
        nq_fl = int(sum(d * b for d, b in zip(lad_fl, sh_fl)) + sh_fl[0])
        actual_max = int(max(lad_fl))
        seed = HOLDOUT_SEED + 10 * nmax + int(round(eta_primary * 1e5))
        m_cap = _optimal_noisy_cap(nq_fl, eta_primary, actual_max)

        r_ig = evaluate_schedule(
            lad_fl, sh_fl, ntrials, seed=seed, eta=eta_primary,
            model_eta=0.0, return_errors=True, chunk=2500,
        )
        r_aw = evaluate_schedule(
            lad_fl, sh_fl, ntrials, seed=seed, eta=eta_primary,
            model_eta=eta_primary, return_errors=True, chunk=2500,
        )
        lad_u = _capped_ladder(m_cap)
        base_u = canonical_shots(len(lad_u))
        b0_u = sum(d * b for d, b in zip(lad_u, base_u)) + base_u[0]
        sf = nq_fl / float(b0_u)
        sh_u = [max(1, int(round(sf * b))) for b in base_u]
        r_unif = evaluate_schedule(
            lad_u, sh_u, ntrials, seed=seed, eta=eta_primary,
            model_eta=eta_primary, return_errors=True, chunk=2500,
        )
        m2 = int(round(0.8 * m_cap))
        lad_p = sorted(set(geom_ladder(m_cap, 1.45) + [m2, m_cap]))
        sh_p = list(canonical_shots(len(lad_p)))
        b0_p = int(sum(d * b for d, b in zip(lad_p, sh_p)) + sh_p[0])
        b_rem = max(0, nq_fl - b0_p)
        n_pair = b_rem // (m_cap + m2)
        sh_p[lad_p.index(m_cap)] += n_pair
        sh_p[lad_p.index(m2)] += n_pair
        b_left = b_rem - n_pair * (m_cap + m2)
        if b_left >= m_cap:
            sh_p[lad_p.index(m_cap)] += 1
        elif b_left >= m2:
            sh_p[lad_p.index(m2)] += 1
        r_pair = evaluate_schedule(
            lad_p, sh_p, ntrials, seed=seed, eta=eta_primary,
            model_eta=eta_primary, return_errors=True, chunk=2500,
        )

        for algo_id, algo_name, rounds, m_eff, r_obj in [
            ("flagship_noise_ignorant", "mlqae-flagship (Noise-Ignorant)", 1, actual_max, r_ig),
            ("flagship_noise_aware", "mlqae-flagship (Noise-Aware)", 1, actual_max, r_aw),
            ("nonadapt_uniform_cap_noisy", "Non-Adapt Uniform Scaling (Cap M_opt)", 1, m_cap, r_unif),
            ("nonadapt_pair_anchor_noisy", "Non-Adapt Pair Anchor {0.8M, M} (Cap M_opt)", 1, m_cap, r_pair),
        ]:
            p999 = float(np.percentile(r_obj["errors"], 99.9))
            nq_val = int(r_obj["nq"])
            noisy_records_eta1e3.append({
                "scale_nmax": int(nmax),
                "actual_nmax": int(actual_max),
                "M_eff": int(m_eff),
                "eta": float(eta_primary),
                "algo_id": algo_id,
                "algorithm_name": algo_name,
                "category": "literature",
                "adaptive_rounds": rounds,
                "ntrials": int(ntrials),
                "N_tot": nq_val,
                "eps68": float(f"{r_obj['p68']:.6e}"),
                "eps95": float(f"{r_obj['p95']:.6e}"),
                "eps99": float(f"{r_obj['p99']:.6e}"),
                "eps999": float(f"{p999:.6e}"),
                "C68": round(float(r_obj["C68"]), 4),
                "C95": round(float(r_obj["C95"]), 4),
                "C99": round(float(r_obj["C99"]), 4),
                "C999": round(float(p999 * nq_val), 4),
            })

    noisy_summary: list[dict[str, Any]] = []
    for algo_id, algo_name in [
        ("flagship_noise_ignorant", "mlqae-flagship (Noise-Ignorant)"),
        ("flagship_noise_aware", "mlqae-flagship (Noise-Aware)"),
        ("nonadapt_uniform_cap_noisy", "Non-Adapt Uniform Scaling (Cap M_opt)"),
        ("nonadapt_pair_anchor_noisy", "Non-Adapt Pair Anchor {0.8M, M} (Cap M_opt)"),
    ]:
        sub = [rec for rec in noisy_records_eta1e3 if rec["algo_id"] == algo_id]
        noisy_summary.append({
            "algo_id": algo_id,
            "algorithm_name": algo_name,
            "category": "literature",
            "adaptive_rounds": sub[0]["adaptive_rounds"],
            "mean_C68": round(float(np.mean([x["C68"] for x in sub])), 4),
            "mean_C95": round(float(np.mean([x["C95"] for x in sub])), 4),
            "mean_C99": round(float(np.mean([x["C99"] for x in sub])), 4),
            "mean_C999": round(float(np.mean([x["C999"] for x in sub])), 4),
        })

    noisy_section = {
        "description": (
            "Depolarizing Noise Regime (per-oracle depolarizing rate eta = 1e-3, "
            "fringe visibility V_n = (1 - eta)^(2n+1); Labib 2026, arXiv:2609.02715). "
            "Compares noise-ignorant and noise-aware non-adaptive schedules, depth-capped "
            "uniform scaling, and pair anchors."
        ),
        "seed": HOLDOUT_SEED,
        "ntrials": ntrials,
        "eta": eta_primary,
        "scales": noisy_scales,
        "summary_by_algorithm": noisy_summary,
        "records": noisy_records_eta1e3,
    }

    return depth_limited_section, noisy_section


# ---------------------------------------------------------------------------
# Summary Table Printer
# ---------------------------------------------------------------------------

def print_summary_tables(
    entries: list[dict[str, Any]],
    depth_limited: dict[str, Any],
    noisy: dict[str, Any],
) -> None:
    print("\n" + "=" * 134)
    print(" QUANTUM AMPLITUDE ESTIMATION LEADERBOARD — CERTIFIED HOLDOUT_SEED (900,000) LITERATURE BASELINES")
    print(" Scales: n_max in {60, 125, 260, 540} | Prior: a ~ U(0.1, 0.9) | Trials: 20,000/scale | Bootstrap: B=1,000")
    print("=" * 134)
    hdr = (
        f"{'Rank':<4} {'Algorithm':<52} {'Category':<10} {'Track':<21} {'Rnds':>5} "
        f"{'C68_ave':>7} {'C95_ave [95% CI]':>21} {'C95_max':>7} {'C99_ave':>7} {'C95_seq':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    sorted_entries = sorted(entries, key=lambda e: e["C95_ave"])
    for rank, e in enumerate(sorted_entries, 1):
        ci95 = e["C95_ave_ci"]
        ci_str = f"{e['C95_ave']:6.3f} [{ci95[0]:5.2f},{ci95[1]:5.2f}]"
        rnd_str = f"{e['num_rounds_mean']:5.2f}" if e["num_rounds_mean"] < 10 else f"{e['num_rounds_mean']:5.1f}"
        print(
            f"{rank:<4d} {e['algorithm_name']:<52} {e['category']:<10} {e['track']:<21} {rnd_str:>5} "
            f"{e['C68_ave']:7.3f} {ci_str:>21} {e['C95_max']:7.3f} {e['C99_ave']:7.2f} {e['C95_seq']:7.3f}"
        )
    print("-" * len(hdr))


def main() -> None:
    t0 = time.time()
    ntrials = 20_000
    boot_B = 1000

    non_adapt_entries = eval_track_non_adaptive(ntrials=ntrials, boot_B=boot_B)
    adapt_entries = eval_track_adaptive(ntrials=ntrials, boot_B=boot_B)
    depth_limited_sec, noisy_sec = eval_tracks_depth_limited_and_noisy(ntrials=ntrials)

    all_entries = non_adapt_entries + adapt_entries
    all_entries.sort(key=lambda e: e["C95_ave"])
    for rank, e in enumerate(all_entries, 1):
        e["overall_rank"] = rank

    now_iso = _dt.datetime.now(_dt.timezone.utc).isoformat()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "updated_at": now_iso,
        "evaluation_protocol": {
            "ntrials_holdout": ntrials,
            "boot_B": boot_B,
            "holdout_seed": HOLDOUT_SEED,
            "search_seed": SEARCH_SEED,
            "salted_seed_verification": True,
            "a_range": [0.1, 0.9],
            "scales": list(SCALES),
            "query_accounting": "shot_cost(0) = 1, shot_cost(n) = n for n >= 1 (total = sum(n * N_n) + N_0)",
            "kappa_crlb": {str(k): v for k, v in KAPPA.items()},
        },
        "tracks": list(VALID_TRACKS),
        "scales": list(SCALES),
        "entries": all_entries,
        "depth_limited": depth_limited_sec,
        "noisy": noisy_sec,
    }

    validate_leaderboard(payload)

    for out_path in OUT_JSON_PATHS:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")

    print_summary_tables(all_entries, depth_limited_sec, noisy_sec)
    for out_path in OUT_JSON_PATHS:
        print(f"Successfully exported certified leaderboard dataset to: {out_path}")
    print(f"Total compilation & verification wall time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
