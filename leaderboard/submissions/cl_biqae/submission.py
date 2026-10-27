"""Standalone leaderboard submission for CL-BIQAE (Cumulative-Likelihood BIQAE).

Modernizes Chebyshev / Bayesian Interval Quantum Amplitude Estimation
(chebAE, Rall & Fuller 2023; BIQAE, Li et al. 2023 / Kiss et al. 2024)
under the strict odd-degree Grover cost model:
  1. Strictly odd Grover degrees d = 2n + 1 with cost(n) = max(1, n) per shot.
  2. Exact cumulative multi-round binomial log-likelihood evaluated on a compact
     129-point moving grid around the evolving posterior mean (replacing
     memoryless per-round Jeffreys Beta posteriors and union-bound scheduling).
  3. Single-shot deep adaptive jumps (N_shots = 8 at d=1, 2 at d<=7, 1 for d>7)
     with r_min = 1.0 and gap-relaxation to avoid dead-zone stalls.
  4. Two-stage credible interval multipliers (z_early = 3.0, z_late = 2.08,
     z_stop = 1.96) and an explicit worst-case query safety cap.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from leaderboard.schema_and_api import AdaptiveAESubmission, shot_cost


def find_next_cheb_odd(
    phi_lo: float,
    phi_hi: float,
    min_deg: int = 0,
    max_deg: Optional[int] = None,
) -> Optional[int]:
    """Find the highest odd Chebyshev degree deg = 2n + 1 > min_deg (and <= max_deg)
    such that [phi_lo, phi_hi] lies strictly inside a single monotone half-period
    [t * pi / (2 * deg), (t + 1) * pi / (2 * deg)] of cos^2(deg * phi).
    """
    d_phi = phi_hi - phi_lo
    if d_phi <= 1e-12:
        deg = max_deg if max_deg is not None else max(min_deg + 2, 1)
    else:
        deg = int((np.pi / 2.0) / d_phi)
    if max_deg is not None and deg > max_deg:
        deg = int(max_deg)
    if deg % 2 == 0:
        deg -= 1

    inv_pi2 = 2.0 / np.pi
    while deg > min_deg:
        if int(deg * phi_lo * inv_pi2) == int(deg * phi_hi * inv_pi2):
            return deg
        deg -= 2
    return None


def eval_moving_posterior(
    degs: np.ndarray,
    ks: np.ndarray,
    ns: np.ndarray,
    center: float,
    half_width: float,
    n_pts: int = 129,
) -> Tuple[float, float, float]:
    """Evaluate exact cumulative binomial log-likelihood over [center - hw, center + hw].

    Returns
    -------
    (phi_hat_mle, phi_mean, phi_std)
    """
    lo = max(1e-6, center - half_width)
    hi = min(np.pi / 2.0 - 1e-6, center + half_width)
    grid = np.linspace(lo, hi, n_pts)
    c2 = np.cos(np.outer(grid, degs)) ** 2
    c2 = np.clip(c2, 1e-12, 1.0 - 1e-12)
    ll = np.sum(
        ks[None, :] * np.log(c2) + (ns[None, :] - ks[None, :]) * np.log1p(-c2),
        axis=1,
    )
    ll_max = float(np.max(ll))
    w = np.exp(ll - ll_max)
    w /= np.sum(w)

    phi_mean = float(np.sum(w * grid))
    phi_var = float(np.sum(w * (grid - phi_mean) ** 2))
    phi_std = float(np.sqrt(max(phi_var, 1e-16)))

    j = int(np.clip(np.argmax(ll), 1, n_pts - 2))
    ym1, y0, yp1 = float(ll[j - 1]), float(ll[j]), float(ll[j + 1])
    den = ym1 - 2.0 * y0 + yp1
    off = 0.5 * (ym1 - yp1) / den if den < -1e-12 else 0.0
    dstep = float(grid[1] - grid[0])
    phi_hat = float(grid[j] + np.clip(off, -0.5, 0.5) * dstep)

    return phi_hat, phi_mean, phi_std


class CLBIQAESubmission(AdaptiveAESubmission):
    """Cumulative-Likelihood Bayesian Interval Adaptive AE (CL-BIQAE-Ave)."""

    name = "CL-BIQAE (Cumulative-Likelihood BIQAE)"
    track = "adaptive-variable"
    authors = ["Farrokh Labib"]
    github_handle = "FarLab"
    affiliation = "Unitary Foundation"
    description = (
        "Modernized Chebyshev / Bayesian Interval QAE (CL-BIQAE) using odd Grover "
        "degrees d=2n+1, exact multi-round cumulative log-likelihood on a 129-point "
        "moving grid, single-shot adaptive jumps, two-stage credible intervals, and "
        "a worst-case query cap."
    )

    def run_trial(
        self,
        oracle: Any,
        target_scale_or_budget: float,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """Execute a single adaptive amplitude estimation trial at scale `nmax`."""
        nmax = max(1, int(round(float(target_scale_or_budget))))
        # Calibrate target accuracy eps to match flagship non-adaptive eps95 at scale nmax
        eps = 0.212 / float(nmax)

        # CL-BIQAE-Ave hyperparameters (tuned on SEARCH_SEED = 100_000)
        z_early = 3.0
        z_late = 2.08
        z_stop = 1.96
        batch_d1 = 8
        batch_early = 2
        batch_mid = 1
        r_min = 1.0
        jacobian_exp = 1.0
        cap_factor = 1.15
        win_mult = 6.0
        gap_relax = True
        q_cap_const = 1.85
        q_cap_offset = 35

        q_cap = int(round(q_cap_const / eps + q_cap_offset))
        oracle_budget = oracle.max_query_budget
        if oracle_budget is not None:
            q_cap = min(q_cap, int(oracle_budget))

        oracle_depth_cap = oracle.max_depth_cap

        deg_to_idx: Dict[int, int] = {}
        degs_list: List[int] = []
        ks_list: List[int] = []
        ns_list: List[int] = []

        deg = 1
        queries = 0
        fisher_info = 0.0

        center = np.pi / 4.0
        half_win = np.pi / 4.0
        nom_target_fisher = (z_stop * 0.8169 / eps) ** 2

        while True:
            n_depth = deg // 2
            if oracle_depth_cap is not None and n_depth > int(oracle_depth_cap):
                n_depth = int(oracle_depth_cap)
                deg = 2 * n_depth + 1

            cost_per_shot = shot_cost(n_depth)

            if deg == 1 and fisher_info == 0.0:
                n_shots = batch_d1
            elif deg <= 7:
                n_shots = batch_early
            elif batch_mid > 1 and 4.0 * batch_mid * (deg ** 2) < 0.25 * nom_target_fisher:
                n_shots = batch_mid
            else:
                n_shots = 1

            rem_q = q_cap - queries
            if rem_q <= 0:
                break
            if n_shots * cost_per_shot > rem_q:
                n_shots = max(1, rem_q // cost_per_shot)
                if n_shots * cost_per_shot > rem_q:
                    break

            # Oracle measures with p(n, theta) = cos^2((2n+1)*theta) where a = sin(theta).
            # In Chebyshev angle phi = arccos(a) = pi/2 - theta, T_{2n+1}(a)^2 = cos^2((2n+1)*phi)
            # = sin^2((2n+1)*theta) = 1 - cos^2((2n+1)*theta). Thus `heads = n_shots - k`
            # follows Binomial(n_shots, cos^2(deg * phi)).
            k = int(oracle.measure(n_depth, n_shots))
            heads = n_shots - k

            queries += n_shots * cost_per_shot
            fisher_info += 4.0 * n_shots * (deg ** 2)

            if deg in deg_to_idx:
                idx = deg_to_idx[deg]
                ks_list[idx] += heads
                ns_list[idx] += n_shots
            else:
                deg_to_idx[deg] = len(degs_list)
                degs_list.append(deg)
                ks_list.append(heads)
                ns_list.append(n_shots)

            degs_arr = np.asarray(degs_list, dtype=np.float64)
            ks_arr = np.asarray(ks_list, dtype=np.float64)
            ns_arr = np.asarray(ns_list, dtype=np.float64)

            phi_hat, phi_mean, phi_std_post = eval_moving_posterior(
                degs_arr, ks_arr, ns_arr, center, half_win, n_pts=129
            )
            phi_std_fisher = 1.0 / float(np.sqrt(fisher_info))

            # Re-center window if posterior shifted near boundary
            if deg > 1 and abs(phi_mean - center) > 0.55 * half_win:
                center = phi_mean
                half_win = min(np.pi / 4.0, max(half_win, win_mult * phi_std_fisher))
                phi_hat, phi_mean, phi_std_post = eval_moving_posterior(
                    degs_arr, ks_arr, ns_arr, center, half_win, n_pts=129
                )

            phi_std = max(phi_std_post, 0.90 * phi_std_fisher)
            center = phi_mean
            half_win = min(np.pi / 4.0, win_mult * phi_std)

            z_sel = z_early if deg <= 9 else z_late
            phi_lo = max(1e-6, center - z_sel * phi_std)
            phi_hi = min(np.pi / 2.0 - 1e-6, center + z_sel * phi_std)

            sin_scale = (
                (float(np.sin(center)) / 0.8169) ** jacobian_exp
                if jacobian_exp > 0.0
                else 1.0
            )
            eff_half_width = z_stop * 0.8169 * sin_scale * phi_std
            if eff_half_width <= eps:
                break

            target_fisher = (z_stop * 0.8169 * sin_scale / eps) ** 2
            rem_fisher = target_fisher - fisher_info
            if rem_fisher <= 0.0:
                break

            max_deg: Optional[int] = None
            if cap_factor is not None:
                max_deg = max(1, int(cap_factor * 0.5 * float(np.sqrt(rem_fisher))))
                if max_deg % 2 == 0:
                    max_deg += 1

            rem_q = q_cap - queries
            if rem_q <= 0:
                break
            max_deg_q = 2 * rem_q + 1
            max_deg = max_deg_q if max_deg is None else min(max_deg, max_deg_q)

            if oracle_depth_cap is not None:
                max_deg_cap = 2 * int(oracle_depth_cap) + 1
                max_deg = min(max_deg, max_deg_cap)

            min_deg = max(deg, int(deg * r_min)) if r_min > 1.0 else 0
            best_monotone = find_next_cheb_odd(
                phi_lo, phi_hi, min_deg=min_deg, max_deg=max_deg
            )
            if best_monotone is None and min_deg > 0:
                best_monotone = find_next_cheb_odd(
                    phi_lo, phi_hi, min_deg=0, max_deg=max_deg
                )

            if gap_relax and deg > 9:
                d_ceil = int((np.pi / 2.0) / max(1e-12, phi_hi - phi_lo))
                if max_deg is not None:
                    d_ceil = min(d_ceil, max_deg)
                if best_monotone is None or best_monotone < 0.72 * d_ceil:
                    phi_lo_r = max(1e-6, center - 0.90 * z_sel * phi_std)
                    phi_hi_r = min(
                        np.pi / 2.0 - 1e-6, center + 0.90 * z_sel * phi_std
                    )
                    alt = find_next_cheb_odd(
                        phi_lo_r, phi_hi_r, min_deg=0, max_deg=max_deg
                    )
                    if alt is not None and (
                        best_monotone is None or alt > best_monotone
                    ):
                        best_monotone = alt

            if best_monotone is not None:
                deg = best_monotone

        # Safeguard if q_cap was exhausted before any shot could be taken
        if len(degs_list) == 0:
            k = int(oracle.measure(0, 1))
            return float(np.clip(np.sqrt(1.0 - float(k)), 0.0, 1.0))

        # Final continuous MLE refinement around cumulative posterior mode
        phi_hat, _, _ = eval_moving_posterior(
            degs_arr, ks_arr, ns_arr, center, 3.5 * phi_std_fisher, n_pts=129
        )
        a_hat = float(np.clip(np.cos(phi_hat), 0.0, 1.0))
        return a_hat
