"""Two/Three-Round Conditional Semi-Adaptive Parallel AE (3R-Cond, r=1.50).

Implements `SemiAdapt3RCondSubmission` for the `semi-adaptive` track of the
Quantum Amplitude Estimation Leaderboard Challenge:

1. **Round 1 (Lean Top-Down Geometric Ladder):**
   Runs a single parallel batch `oracle.measure_batch(r1_depths, r1_shots)`
   on a top-down `r=1.50` geometric ladder anchored at `nmax_cap` with the top
   `k_hold=2` rungs held back for Round 2 and a lean linear shot ramp
   (`slope=0.85`), up to `n_1 ~ 0.44 * n_max`.
2. **Round 2 (Opposite-Slope Phase Steering + Conditional Rival-Killer Shots):**
   Extracts the primary MLE mode `th1` and top two outside-basin rival modes
   `th_riv1, th_riv2` from the Round-1 log-likelihood `LL1`. For the top two
   held-back rungs (`r2_shots = (2, 2)`, `phi0_fracs = (0.20, 0.25)`), steers
   the 2 shots to complementary opposite-slope phases `+-phi_0` within an 18%
   depth window `[round(0.82 * nom_d), nom_d]`, augmented by the Bhattacharyya
   separation exponent against `th_riv1` and `th_riv2`. When `gap1 < 5.5`
   (or `< 2.2`), appends 1 (or 2) conditional rival-killer shots at depth
   `n_kill <= 0.55 * nmax_cap` in the same Round-2 parallel batch.
3. **Round 3 (Conditional Tie-Breaker, ~4% of trials):**
   Only when a sub-basin rival still survives after Round 2 (`gap2_r2 < 4.5`),
   executes a 1-shot (or 2-shot if `gap2_r2 < 2.0`) disambiguating batch at
   depth `n_r3 <= 0.72 * nmax_cap`, keeping the average round count at ~2.04
   and average sequential depth at ~1.5 * n_max.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from leaderboard.schema_and_api import AdaptiveAESubmission, QuantumAEOracle
from mlqae.core import canonical_shots, geom_ladder, grid_spacing


def topdown_geom_ladder(nmax_cap: int, r: float) -> List[int]:
    """Geometric ladder anchored at `nmax_cap`: `round(nmax_cap / r^k)` down to 0."""
    rungs: List[int] = []
    x = float(nmax_cap)
    while round(x) >= 1:
        v = int(round(x))
        if v not in rungs:
            rungs.append(v)
        x /= r
    return sorted([0] + rungs)


def bhattacharyya_exp_vec(
    th_a: float,
    th_b: float,
    d_cand: np.ndarray,
    eta: float = 0.0,
) -> np.ndarray:
    """Chernoff/Bhattacharyya exponent -ln(sqrt(p1*p2) + sqrt((1-p1)*(1-p2)))."""
    p1 = np.cos(th_a * d_cand) ** 2
    p2 = np.cos(th_b * d_cand) ** 2
    if eta > 0.0:
        vt = (1.0 - eta) ** d_cand
        p1 = vt * p1 + 0.5 * (1.0 - vt)
        p2 = vt * p2 + 0.5 * (1.0 - vt)
    p1 = np.clip(p1, 1e-12, 1.0 - 1e-12)
    p2 = np.clip(p2, 1e-12, 1.0 - 1e-12)
    bc = np.sqrt(p1 * p2) + np.sqrt((1.0 - p1) * (1.0 - p2))
    return -np.log(np.clip(bc, 1e-12, 1.0))


def refine_single_mode(
    th0: float,
    step: float,
    zoom_offs: np.ndarray,
    d_arr: np.ndarray,
    k_arr: np.ndarray,
    s_arr: np.ndarray,
    eta: float = 0.0,
) -> float:
    """Exact 25-point zoom + quadratic interpolation around coarse peak `th0`."""
    th_c = th0 + zoom_offs
    p = np.cos(np.outer(th_c, d_arr)) ** 2
    if eta > 0.0:
        vt = (1.0 - eta) ** d_arr[None, :]
        p = vt * p + 0.5 * (1.0 - vt)
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    nll = -(np.log(p) @ k_arr + np.log1p(-p) @ (s_arr - k_arr))
    j = int(np.clip(np.argmin(nll), 1, 23))
    ym1 = float(nll[j - 1])
    y0 = float(nll[j])
    yp1 = float(nll[j + 1])
    den = ym1 - 2.0 * y0 + yp1
    off = 0.0
    if abs(den) > 1e-12:
        off = 0.5 * (ym1 - yp1) / max(den, 1e-12)
        off = float(np.clip(off, -0.5, 0.5))
    dstep = float(zoom_offs[1] - zoom_offs[0])
    return float(th_c[j]) + off * dstep


def extract_modes_single_trial(
    ll_vec: np.ndarray,
    grid: np.ndarray,
    step: float,
    zoom_offs: np.ndarray,
    excl_pts: int,
    d_arr: np.ndarray,
    k_arr: np.ndarray,
    s_arr: np.ndarray,
    eta: float = 0.0,
    extract_rivals: bool = True,
) -> Tuple[float, float, float, float, float]:
    """Extract primary MLE mode and top 2 rival modes outside the primary basin."""
    g_len = int(len(grid))
    ib = int(np.clip(np.argmax(ll_vec), 1, g_len - 2))
    th_hat = refine_single_mode(
        float(grid[ib]), step, zoom_offs, d_arr, k_arr, s_arr, eta=eta
    )
    if not extract_rivals:
        return th_hat, th_hat, 0.0, th_hat, 0.0

    ll_max = float(ll_vec[ib])
    ll_work = ll_vec.copy()

    lo1 = max(0, ib - excl_pts)
    hi1 = min(g_len, ib + excl_pts + 1)
    ll_work[lo1:hi1] = -np.inf
    ir1 = int(np.clip(np.argmax(ll_work), 1, g_len - 2))
    gap1 = ll_max - float(ll_vec[ir1])
    th_riv1 = refine_single_mode(
        float(grid[ir1]), step, zoom_offs, d_arr, k_arr, s_arr, eta=eta
    )

    lo2 = max(0, ir1 - excl_pts)
    hi2 = min(g_len, ir1 + excl_pts + 1)
    ll_work[lo2:hi2] = -np.inf
    ir2 = int(np.clip(np.argmax(ll_work), 1, g_len - 2))
    gap2 = ll_max - float(ll_vec[ir2])
    th_riv2 = refine_single_mode(
        float(grid[ir2]), step, zoom_offs, d_arr, k_arr, s_arr, eta=eta
    )

    return th_hat, th_riv1, gap1, th_riv2, gap2


def merge_batch_schedule(
    depths: Sequence[int],
    shots: Sequence[int],
) -> Tuple[List[int], List[int]]:
    """Combine any duplicate depths in a single parallel round while preserving order."""
    merged_shots: Dict[int, int] = {}
    order: List[int] = []
    for d_val, s_val in zip(depths, shots):
        d_int = int(d_val)
        s_int = int(s_val)
        if s_int <= 0:
            continue
        if d_int not in merged_shots:
            merged_shots[d_int] = s_int
            order.append(d_int)
        else:
            merged_shots[d_int] += s_int
    return order, [merged_shots[d_int] for d_int in order]


class SemiAdapt3RCondSubmission(AdaptiveAESubmission):
    """3R-Cond (r=1.50 Opposite-Slope + Tie-Breaker) semi-adaptive AE submission."""

    name = "3R-Cond (r=1.50 Opposite-Slope + Tie-Breaker)"
    track = "semi-adaptive"

    def __init__(
        self,
        r: float = 1.50,
        k_hold: int = 2,
        shallow_slope: float = 0.85,
        r2_shots: Tuple[int, int] = (2, 2),
        phi0_fracs: Tuple[float, float] = (0.20, 0.25),
        win_frac: float = 0.18,
        rival_weight: float = 0.80,
        kill_thresh_1: float = 5.5,
        kill_thresh_2: float = 2.2,
        kill_max_frac: float = 0.55,
        r3_thresh_1: float = 4.5,
        r3_thresh_2: float = 2.0,
        r3_max_frac: float = 0.72,
    ) -> None:
        self.r = float(r)
        self.k_hold = int(k_hold)
        self.shallow_slope = float(shallow_slope)
        self.r2_shots = (int(r2_shots[0]), int(r2_shots[1]))
        self.phi0_fracs = (float(phi0_fracs[0]), float(phi0_fracs[1]))
        self.win_frac = float(win_frac)
        self.rival_weight = float(rival_weight)
        self.kill_thresh_1 = float(kill_thresh_1)
        self.kill_thresh_2 = float(kill_thresh_2)
        self.kill_max_frac = float(kill_max_frac)
        self.r3_thresh_1 = float(r3_thresh_1)
        self.r3_thresh_2 = float(r3_thresh_2)
        self.r3_max_frac = float(r3_max_frac)
        self.cache: Dict[Tuple[int, float], Dict[str, Any]] = {}

    def get_scale_data(self, scale_nmax: int, eta: float) -> Dict[str, Any]:
        """Precompute and cache grid tables and candidate depth windows for `scale_nmax`."""
        key = (int(scale_nmax), round(float(eta), 10))
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        base_lad = geom_ladder(int(scale_nmax), 1.45)
        nmax_cap = int(max(base_lad))
        lad_full = topdown_geom_ladder(nmax_cap, self.r)
        shots_full = canonical_shots(len(lad_full), base=1, slope=self.shallow_slope)

        r1_depths = [int(d) for d in lad_full[: -self.k_hold]]
        r1_shots = [int(s) for s in shots_full[: -self.k_hold]]
        r2_nom_depths = [int(d) for d in lad_full[-self.k_hold :]]
        r2_nom_shots = list(self.r2_shots)

        n1_max = int(max(r1_depths))
        step = float(grid_spacing(lad_full, shots_full))
        grid = np.arange(1e-6, 0.5 * math.pi, step, dtype=np.float64)
        zoom_offs = np.linspace(-0.5 * step, 0.5 * step, 25, dtype=np.float64)

        depths_all = np.arange(nmax_cap + 1, dtype=np.float64)
        d_all = 2.0 * depths_all + 1.0
        p_all = np.cos(np.outer(d_all, grid)) ** 2
        if eta > 0.0:
            vt = (1.0 - eta) ** d_all[:, None]
            p_all = vt * p_all + 0.5 * (1.0 - vt)
        p_all = np.clip(p_all, 1e-12, 1.0 - 1e-12)
        log_p_all = np.log(p_all)
        log_1mp_all = np.log1p(-p_all)

        r1_d_idx = np.asarray(r1_depths, dtype=np.int64)
        r1_d_odd = 2.0 * r1_d_idx.astype(np.float64) + 1.0
        r1_s_arr = np.asarray(r1_shots, dtype=np.float64)

        w_r1 = math.pi / (2.0 * (2.0 * n1_max + 1.0))
        excl_r1 = max(2, int(round(0.70 * w_r1 / step)))
        w_r2 = math.pi / (2.0 * (2.0 * nmax_cap + 1.0))
        excl_r2 = max(2, int(round(0.70 * w_r2 / step)))

        r2_windows: List[Tuple[np.ndarray, np.ndarray, float, int]] = []
        for idx_k, (nom_d, nom_s) in enumerate(zip(r2_nom_depths, r2_nom_shots)):
            pf = self.phi0_fracs[min(idx_k, len(self.phi0_fracs) - 1)]
            phi0 = pf * math.pi
            lo_d = max(n1_max + 1, int(round(nom_d * (1.0 - self.win_frac))))
            hi_d = min(nmax_cap, nom_d)
            if hi_d < lo_d:
                lo_d = hi_d = nom_d
            cand_n = np.arange(lo_d, hi_d + 1, dtype=np.int64)
            cand_d = 2.0 * cand_n.astype(np.float64) + 1.0
            r2_windows.append((cand_n, cand_d, phi0, int(nom_s)))

        hi_kill = max(2, int(round(nmax_cap * self.kill_max_frac)))
        cand_kill_n = np.arange(1, hi_kill + 1, dtype=np.int64)
        cand_kill_d = 2.0 * cand_kill_n.astype(np.float64) + 1.0
        denom_kill = (cand_kill_n.astype(np.float64) / float(hi_kill)) ** 0.35 + 0.25

        hi_r3 = max(2, int(round(nmax_cap * self.r3_max_frac)))
        cand_r3_n = np.arange(1, hi_r3 + 1, dtype=np.int64)
        cand_r3_d = 2.0 * cand_r3_n.astype(np.float64) + 1.0
        denom_r3 = (cand_r3_n.astype(np.float64) / float(hi_r3)) ** 0.35 + 0.20

        data: Dict[str, Any] = {
            "nmax_cap": nmax_cap,
            "r1_depths": r1_depths,
            "r1_shots": r1_shots,
            "r1_d_idx": r1_d_idx,
            "r1_d_odd": r1_d_odd,
            "r1_s_arr": r1_s_arr,
            "n1_max": n1_max,
            "step": step,
            "grid": grid,
            "zoom_offs": zoom_offs,
            "log_p_all": log_p_all,
            "log_1mp_all": log_1mp_all,
            "log_p_r1": log_p_all[r1_d_idx, :],
            "log_1mp_r1": log_1mp_all[r1_d_idx, :],
            "excl_r1": excl_r1,
            "excl_r2": excl_r2,
            "r2_windows": r2_windows,
            "cand_kill_n": cand_kill_n,
            "cand_kill_d": cand_kill_d,
            "denom_kill": denom_kill,
            "cand_r3_n": cand_r3_n,
            "cand_r3_d": cand_r3_d,
            "denom_r3": denom_r3,
        }
        self.cache[key] = data
        return data

    def run_trial(
        self,
        oracle: QuantumAEOracle,
        target_scale_or_budget: int = 60,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """Execute one 2-round (or conditional 3-round) trial against `oracle`."""
        _ = rng
        scale_nmax = max(1, int(target_scale_or_budget))
        eta = float(oracle.eta)
        sdata = self.get_scale_data(scale_nmax, eta)

        r1_depths: List[int] = sdata["r1_depths"]
        r1_shots: List[int] = sdata["r1_shots"]
        r1_d_odd: np.ndarray = sdata["r1_d_odd"]
        r1_s_arr: np.ndarray = sdata["r1_s_arr"]
        grid: np.ndarray = sdata["grid"]
        step: float = sdata["step"]
        zoom_offs: np.ndarray = sdata["zoom_offs"]
        log_p_all: np.ndarray = sdata["log_p_all"]
        log_1mp_all: np.ndarray = sdata["log_1mp_all"]

        # ------------------------------------------------------------------
        # Round 1: Parallel Geometric Ladder up to n_{L-2}
        # ------------------------------------------------------------------
        k1_raw = oracle.measure_batch(r1_depths, r1_shots)
        k1_arr = np.asarray(k1_raw, dtype=np.float64)

        ll1 = k1_arr @ sdata["log_p_r1"] + (r1_s_arr - k1_arr) @ sdata["log_1mp_r1"]
        th1, th_riv1, gap1, th_riv2, gap2 = extract_modes_single_trial(
            ll1,
            grid,
            step,
            zoom_offs,
            sdata["excl_r1"],
            r1_d_odd,
            k1_arr,
            r1_s_arr,
            eta=eta,
            extract_rivals=True,
        )

        # ------------------------------------------------------------------
        # Round 2: Opposite-Slope Phase Steering + Conditional Rival Killers
        # ------------------------------------------------------------------
        w_riv1 = math.exp(-min(max(gap1, 0.0), 20.0) / 3.0)
        w_riv2 = 0.5 * math.exp(-min(max(gap2, 0.0), 20.0) / 3.0)

        r2_raw_depths: List[int] = []
        r2_raw_shots: List[int] = []

        for cand_n, cand_d, phi0, nom_s in sdata["r2_windows"]:
            e1 = bhattacharyya_exp_vec(th1, th_riv1, cand_d, eta=eta)
            e2 = bhattacharyya_exp_vec(th1, th_riv2, cand_d, eta=eta)
            e_tot = self.rival_weight * (w_riv1 * e1 + w_riv2 * e2)
            psi = 2.0 * th1 * cand_d

            s_pos = nom_s // 2
            s_neg = nom_s - s_pos
            if s_pos > 0:
                score_pos = np.cos(psi - 2.0 * phi0) + e_tot
                n_pos = int(cand_n[int(np.argmax(score_pos))])
                r2_raw_depths.append(n_pos)
                r2_raw_shots.append(s_pos)
            if s_neg > 0:
                score_neg = np.cos(psi + 2.0 * phi0) + e_tot
                n_neg = int(cand_n[int(np.argmax(score_neg))])
                r2_raw_depths.append(n_neg)
                r2_raw_shots.append(s_neg)

        if self.kill_thresh_1 > 0.0 and gap1 < self.kill_thresh_1:
            cand_kill_n: np.ndarray = sdata["cand_kill_n"]
            cand_kill_d: np.ndarray = sdata["cand_kill_d"]
            denom_kill: np.ndarray = sdata["denom_kill"]
            e1_k = bhattacharyya_exp_vec(th1, th_riv1, cand_kill_d, eta=eta)
            e2_k = bhattacharyya_exp_vec(th1, th_riv2, cand_kill_d, eta=eta)
            eff1 = (e1_k + 0.3 * math.exp(-gap2 / 3.0) * e2_k) / denom_kill
            idx_k1 = int(np.argmax(eff1))
            nk1 = int(cand_kill_n[idx_k1])
            r2_raw_depths.append(nk1)
            r2_raw_shots.append(1)

            if self.kill_thresh_2 > 0.0 and gap1 < self.kill_thresh_2:
                eff2 = (0.5 * e1_k + 0.8 * e2_k) / denom_kill
                eff2[idx_k1] = -np.inf
                nk2 = int(cand_kill_n[int(np.argmax(eff2))])
                r2_raw_depths.append(nk2)
                r2_raw_shots.append(1)

        r2_depths, r2_shots = merge_batch_schedule(r2_raw_depths, r2_raw_shots)
        k2_raw = oracle.measure_batch(r2_depths, r2_shots)
        k2_arr = np.asarray(k2_raw, dtype=np.float64)
        r2_d_idx = np.asarray(r2_depths, dtype=np.int64)
        r2_d_odd = 2.0 * r2_d_idx.astype(np.float64) + 1.0
        r2_s_arr = np.asarray(r2_shots, dtype=np.float64)

        ll_tot = ll1 + (
            k2_arr @ log_p_all[r2_d_idx, :]
            + (r2_s_arr - k2_arr) @ log_1mp_all[r2_d_idx, :]
        )
        d_12 = np.concatenate([r1_d_odd, r2_d_odd])
        k_12 = np.concatenate([k1_arr, k2_arr])
        s_12 = np.concatenate([r1_s_arr, r2_s_arr])

        # ------------------------------------------------------------------
        # Round 3: Conditional Tie-Breaker (~4% of trials when gap2_r2 < 4.5)
        # ------------------------------------------------------------------
        if self.r3_thresh_1 > 0.0:
            th2, th2_riv1, gap2_r2, th2_riv2, gap2_r2_2 = extract_modes_single_trial(
                ll_tot,
                grid,
                step,
                zoom_offs,
                sdata["excl_r2"],
                d_12,
                k_12,
                s_12,
                eta=eta,
                extract_rivals=True,
            )
            if gap2_r2 < self.r3_thresh_1:
                cand_r3_n: np.ndarray = sdata["cand_r3_n"]
                cand_r3_d: np.ndarray = sdata["cand_r3_d"]
                denom_r3: np.ndarray = sdata["denom_r3"]
                e1_r3 = bhattacharyya_exp_vec(th2, th2_riv1, cand_r3_d, eta=eta)
                e2_r3 = bhattacharyya_exp_vec(th2, th2_riv2, cand_r3_d, eta=eta)
                eff_r3 = (e1_r3 + 0.3 * math.exp(-gap2_r2_2 / 3.0) * e2_r3) / denom_r3
                idx_r3_1 = int(np.argmax(eff_r3))
                nk3 = int(cand_r3_n[idx_r3_1])

                r3_raw_depths: List[int] = [nk3]
                r3_raw_shots: List[int] = [1]

                if self.r3_thresh_2 > 0.0 and gap2_r2 < self.r3_thresh_2:
                    eff_r3b = (0.5 * e1_r3 + 0.8 * e2_r3) / denom_r3
                    eff_r3b[idx_r3_1] = -np.inf
                    nk3b = int(cand_r3_n[int(np.argmax(eff_r3b))])
                    r3_raw_depths.append(nk3b)
                    r3_raw_shots.append(1)

                r3_depths, r3_shots = merge_batch_schedule(r3_raw_depths, r3_raw_shots)
                k3_raw = oracle.measure_batch(r3_depths, r3_shots)
                k3_arr = np.asarray(k3_raw, dtype=np.float64)
                r3_d_idx = np.asarray(r3_depths, dtype=np.int64)
                r3_d_odd = 2.0 * r3_d_idx.astype(np.float64) + 1.0
                r3_s_arr = np.asarray(r3_shots, dtype=np.float64)

                ll_tot = ll_tot + (
                    k3_arr @ log_p_all[r3_d_idx, :]
                    + (r3_s_arr - k3_arr) @ log_1mp_all[r3_d_idx, :]
                )
                d_12 = np.concatenate([d_12, r3_d_odd])
                k_12 = np.concatenate([k_12, k3_arr])
                s_12 = np.concatenate([s_12, r3_s_arr])

                th_final, _, _, _, _ = extract_modes_single_trial(
                    ll_tot,
                    grid,
                    step,
                    zoom_offs,
                    sdata["excl_r2"],
                    d_12,
                    k_12,
                    s_12,
                    eta=eta,
                    extract_rivals=False,
                )
                return float(np.clip(math.sin(th_final), 0.0, 1.0))

            return float(np.clip(math.sin(th2), 0.0, 1.0))

        th_final, _, _, _, _ = extract_modes_single_trial(
            ll_tot,
            grid,
            step,
            zoom_offs,
            sdata["excl_r2"],
            d_12,
            k_12,
            s_12,
            eta=eta,
            extract_rivals=False,
        )
        return float(np.clip(math.sin(th_final), 0.0, 1.0))

    def estimate(
        self,
        oracle: QuantumAEOracle,
        nmax: int = 60,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """Alias for `run_trial` supporting direct `estimate(oracle, nmax)` invocation."""
        return self.run_trial(oracle, int(nmax), rng)


Submission = SemiAdapt3RCondSubmission
