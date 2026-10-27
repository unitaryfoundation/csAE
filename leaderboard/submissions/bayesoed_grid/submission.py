"""BayesOED-Grid (Fractional-Moment OED) submission for the Adaptive AE track."""

import math
from typing import Any
import numpy as np

from leaderboard.schema_and_api import AdaptiveAESubmission, QuantumAEOracle, shot_cost
from mlqae.core import canonical_shots, geom_ladder


G_GRID: int = 6144
MAX_N_TABLE: int = 1600


def build_oed_tables(g_size: int = G_GRID, max_n: int = MAX_N_TABLE):
    """Precompute grid theta_g in (0, pi/2) and likelihood tables for n in [0, max_n]."""
    step = (0.5 * math.pi) / float(g_size)
    grid = (np.arange(g_size, dtype=np.float64) + 0.5) * step
    d = (2.0 * np.arange(max_n + 1, dtype=np.float64) + 1.0)[:, None]
    p_table = np.cos(d * grid[None, :]) ** 2
    p_clip = np.clip(p_table, 1e-15, 1.0 - 1e-15)
    log_p = np.log(p_clip)
    log_1mp = np.log1p(-p_clip)
    p_table_t = np.ascontiguousarray(p_table.T)
    return grid, step, p_table_t, log_p, log_1mp


def compute_plain_budget(scale_nmax: int) -> int:
    """Compute canonical plain-ladder query budget for scale_nmax."""
    l_arr = geom_ladder(int(scale_nmax), 1.45)
    s_arr = canonical_shots(len(l_arr))
    return int(sum(shot_cost(int(d)) * int(s) for d, s in zip(l_arr, s_arr)))


GRID, GRID_STEP, P_TABLE_T, LOGP_TABLE, LOG1MP_TABLE = build_oed_tables()

SCALE_BUDGETS: dict[int, int] = {
    60: compute_plain_budget(60),
    125: compute_plain_budget(125),
    260: compute_plain_budget(260),
    540: compute_plain_budget(540),
}


def pow075_scalar(x: float) -> float:
    """Compute x^0.75 = sqrt(x * sqrt(x)) for scalar x >= 0."""
    if x <= 0.0:
        return 0.0
    return math.sqrt(x * math.sqrt(x))


def local_refine_single(
    th0: float,
    depths: list[int],
    k1s: list[int],
    shots: list[int],
    grid_step: float,
) -> float:
    """Exact likelihood 25-point zoom + parabolic interpolation around th0."""
    if len(depths) == 0:
        return th0
    half = 0.75 * grid_step
    dstep = (2.0 * half) / 24.0
    offs = np.linspace(-half, half, 25)
    th_vec = th0 + offs
    valid_th = (th_vec > 1e-9) & (th_vec < 0.5 * math.pi - 1e-9)

    d_vec = 2.0 * np.asarray(depths, dtype=np.float64)[:, None] + 1.0
    k1_vec = np.asarray(k1s, dtype=np.float64)[:, None]
    s_vec = np.asarray(shots, dtype=np.float64)[:, None]
    k0_vec = s_vec - k1_vec

    p_mat = np.clip(np.cos(d_vec * th_vec[None, :]) ** 2, 1e-15, 1.0 - 1e-15)
    nll = -np.sum(k1_vec * np.log(p_mat) + k0_vec * np.log1p(-p_mat), axis=0)
    nll = np.where(valid_th, nll, 1e300)

    best_j = int(np.argmin(nll))
    if best_j <= 0:
        best_j = 1
    elif best_j >= 24:
        best_j = 23

    ym1 = float(nll[best_j - 1])
    y0 = float(nll[best_j])
    yp1 = float(nll[best_j + 1])
    den = ym1 - 2.0 * y0 + yp1
    if abs(den) > 1e-12 and den > 0.0:
        frac = 0.5 * (ym1 - yp1) / den
        if frac < -0.5:
            frac = -0.5
        elif frac > 0.5:
            frac = 0.5
    else:
        frac = 0.0
    return float(th0 + float(offs[best_j]) + frac * dstep)


class BayesOEDGridSubmission(AdaptiveAESubmission):
    """Exact Grid Bayesian OED with Fractional-Moment (p=1.5) Utility and 2-Stage Endgame."""

    algorithm_name = "BayesOED-Grid (Fractional-Moment OED)"
    track = "adaptive-fixed-budget"

    def run_trial(
        self,
        oracle: QuantumAEOracle,
        target_scale_or_budget: int,
        rng: np.random.Generator | None = None,
    ) -> float:
        """Execute a single adaptive AE trial using Exact Grid Bayesian OED."""
        scale_val = int(target_scale_or_budget)
        if scale_val in SCALE_BUDGETS:
            q_budget = SCALE_BUDGETS[scale_val]
        elif scale_val <= 600:
            q_budget = compute_plain_budget(scale_val)
        else:
            q_budget = scale_val

        if oracle.max_query_budget is not None:
            q_budget = min(q_budget, int(oracle.max_query_budget))

        n_cap = MAX_N_TABLE
        if oracle.max_depth_cap is not None:
            n_cap = min(n_cap, int(oracle.max_depth_cap))

        alpha_horizon = 1.35
        eta_tail = 0.35
        lam_uni = -0.7
        lam_multi = -1.0
        rho_thresh = 1.25
        n0_init = 4
        max_ratio_step = 1.8
        band_lo_frac = 0.45
        min_rem_frac = 0.65

        live_idx = np.arange(G_GRID, dtype=np.int64)

        hist_d: list[int] = []
        hist_k1: list[int] = []
        hist_s: list[int] = []

        queries_used = 0
        last_max_n = 0

        if n0_init > 0 and q_budget > 0:
            s0 = min(n0_init, q_budget)
            k1 = int(oracle.measure(0, s0))
            k0 = s0 - k1
            lw_live = k1 * LOGP_TABLE[0] + k0 * LOG1MP_TABLE[0]
            hist_d.append(0)
            hist_k1.append(k1)
            hist_s.append(s0)
            queries_used += s0
        else:
            lw_live = np.zeros(G_GRID, dtype=np.float64)

        while queries_used < q_budget:
            q_rem = q_budget - queries_used

            max_lw = float(np.max(lw_live))
            diff = lw_live - max_lw

            keep_live = diff > -32.0
            if np.any(keep_live):
                live_idx = live_idx[keep_live]
                lw_live = lw_live[keep_live]
                diff = diff[keep_live]

            act_mask = diff > -14.0
            if np.any(act_mask):
                act_idx = live_idx[act_mask]
                diff_act = diff[act_mask]
            else:
                act_idx = live_idx
                diff_act = diff

            th_act = GRID[act_idx]
            we = np.exp(diff_act)
            wt = np.sqrt(we * np.sqrt(we))
            sum_we = float(np.sum(we))
            sum_wt = float(np.sum(wt))
            we_n = we * (1.0 / sum_we)
            wt_n = wt * (1.0 / sum_wt)

            w_act = (1.0 - eta_tail) * we_n + eta_tail * wt_n
            th_act2 = th_act * th_act
            w_th = w_act * th_act
            w_th2 = w_act * th_act2

            mu = float(np.sum(w_th))
            m2 = float(np.sum(w_th2))
            mu_ex = float(np.sum(we_n * th_act))
            m2_ex = float(np.sum(we_n * th_act2))

            var_curr = max(1e-16, m2 - mu * mu)
            var_ex = max(1e-16, m2_ex - mu_ex * mu_ex)
            sigma_eff = math.sqrt(var_curr)
            sigma_ex = math.sqrt(var_ex)
            vp_curr = pow075_scalar(var_curr)

            is_multimodal = sigma_eff > rho_thresh * sigma_ex
            lam_step = lam_multi if is_multimodal else lam_uni

            n_safe = int(0.5 * (alpha_horizon / sigma_eff - 1.0))
            if n_safe < 0:
                n_safe = 0
            if max_ratio_step > 0.0 and last_max_n >= 1:
                n_step_cap = int(math.ceil(max_ratio_step * last_max_n)) + 2
                if n_safe > n_step_cap:
                    n_safe = n_step_cap
            elif max_ratio_step > 0.0 and last_max_n == 0:
                if n_safe > 3:
                    n_safe = 3

            n_target = min(n_safe, min(n_cap, MAX_N_TABLE))

            if min_rem_frac > 0.0 and n_target >= 12 and q_rem < int(min_rem_frac * n_target):
                break

            is_final_shot = False
            if n_target >= 15 and not is_multimodal:
                if q_rem <= int(1.30 * n_target):
                    is_final_shot = True
                    n_target = min(q_rem, min(int(1.15 * n_target), min(n_cap, MAX_N_TABLE)))
                elif q_rem < int(1.85 * n_target):
                    split_cap = int(0.46 * q_rem)
                    if split_cap >= 12 and split_cap < n_target:
                        n_target = split_cap

            n_hi = min(n_target, q_rem)

            if n_hi <= 0:
                chosen_n = 0
            else:
                if is_final_shot:
                    top_lo = max(1, n_hi - 14)
                else:
                    blo = 0.35 if is_multimodal else band_lo_frac
                    top_lo = max(1, min(n_hi - 24, int(blo * n_hi)))

                p_act_hi = P_TABLE_T[act_idx, : n_hi + 1]
                if top_lo > 1:
                    step_low = max(1, (top_lo - 1) // 10)
                    cands = np.concatenate((
                        np.array([0], dtype=np.int64),
                        np.arange(1, top_lo, step_low, dtype=np.int64),
                        np.arange(top_lo, n_hi + 1, dtype=np.int64),
                    ))
                    p_sub_t = p_act_hi[:, cands]
                else:
                    cands = np.arange(0, n_hi + 1, dtype=np.int64)
                    p_sub_t = p_act_hi

                w_mat = np.vstack((w_act, w_th, w_th2))
                s_mat = w_mat @ p_sub_t
                s0_all = s_mat[0]
                s1_all = s_mat[1]
                s2_all = s_mat[2]

                chosen_n = n_hi
                lam_eff = max(0.0, lam_step) if is_final_shot else lam_step

                valid = (s0_all > 1e-9) & (s0_all < 1.0 - 1e-9)
                if np.any(valid):
                    c_v = cands[valid]
                    s0 = s0_all[valid]
                    s1 = s1_all[valid]
                    s2 = s2_all[valid]
                    one_m_s0 = 1.0 - s0

                    mu1 = s1 / s0
                    v1 = np.maximum(0.0, s2 / s0 - mu1 * mu1)
                    mu0 = (mu - s1) / one_m_s0
                    v0 = np.maximum(0.0, (m2 - s2) / one_m_s0 - mu0 * mu0)

                    vp1 = np.sqrt(v1 * np.sqrt(v1))
                    vp0 = np.sqrt(v0 * np.sqrt(v0))
                    evp = s0 * vp1 + one_m_s0 * vp0

                    lev = (2.0 * c_v.astype(np.float64) + 1.0) ** lam_eff
                    u_val = (vp_curr - evp) * lev

                    best_idx = int(np.argmax(u_val))
                    if float(u_val[best_idx]) > -1e300:
                        chosen_n = int(c_v[best_idx])

            c_cost = 1 if chosen_n == 0 else chosen_n
            y = int(oracle.measure(chosen_n, 1))

            if y == 1:
                lw_live = lw_live + LOGP_TABLE[chosen_n, live_idx]
            else:
                lw_live = lw_live + LOG1MP_TABLE[chosen_n, live_idx]

            if len(hist_d) > 0 and hist_d[-1] == chosen_n:
                hist_k1[-1] += y
                hist_s[-1] += 1
            else:
                hist_d.append(chosen_n)
                hist_k1.append(y)
                hist_s.append(1)

            queries_used += c_cost
            if chosen_n > last_max_n:
                last_max_n = chosen_n

        best_g = int(live_idx[int(np.argmax(lw_live))])
        th_map = local_refine_single(
            float(GRID[best_g]),
            hist_d,
            hist_k1,
            hist_s,
            GRID_STEP,
        )
        a_hat = math.sin(th_map)
        return float(min(1.0, max(0.0, a_hat)))
