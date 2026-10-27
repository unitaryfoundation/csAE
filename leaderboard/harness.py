"""Shared evaluation harness for the Quantum Amplitude Estimation Challenge Leaderboard.

Provides:
- Standardized SEARCH_SEED and HOLDOUT_SEED trial generators (matching mlqae/head_to_head.py).
- Strict query accounting matching mlqae/core.py:
    shot at depth n=0 costs 1 query; shot at depth n>=1 costs n queries.
- Reference baselines (mlqae plain ladder, mlqae flagship + cap rung).
- Fast bootstrap CI computation for C68, C95, C99 (both average-query and max-query).
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from mlqae import (geom_ladder, canonical_shots, flagship_schedule,
                   evaluate_schedule, boot_ci_constant, crlb_constant)

SEARCH_SEED = 100_000
HOLDOUT_SEED = 900_000
SCALES = (125, 260, 540, 1000)


def shot_cost(n: int) -> int:
    """Query cost of 1 shot at Grover depth n (matching mlqae/core.py)."""
    return 1 if n == 0 else int(n)


def generate_trials(ntrials: int, seed: int, a_range=(0.1, 0.9)):
    """Generate (a_true, theta_true, per_trial_seeds) deterministically."""
    rng = np.random.default_rng(seed)
    a_true = rng.uniform(a_range[0], a_range[1], ntrials)
    th_true = np.arcsin(a_true)
    trial_seeds = rng.integers(0, 2**31 - 1, size=ntrials, dtype=np.int64)
    return a_true, th_true, trial_seeds


def compute_metrics(a_true, a_hat, queries, max_depths=None, seq_depths=None,
                    boot_B=1000, seed=1):
    """Compute standardized C68, C95, C99 (ave and max) + bootstrap 95% CIs."""
    err = np.abs(np.asarray(a_hat, dtype=float) - np.asarray(a_true, dtype=float))
    q = np.asarray(queries, dtype=float)
    q_mean = float(np.mean(q))
    q_max = float(np.max(q))
    q_std = float(np.std(q))

    out = {
        'ntrials': len(err),
        'q_mean': q_mean,
        'q_max': q_max,
        'q_std': q_std,
        'errors': err,
        'queries': q,
    }
    if max_depths is not None:
        out['max_depth_mean'] = float(np.mean(max_depths))
    if seq_depths is not None:
        out['seq_depth_mean'] = float(np.mean(seq_depths))

    for pct in (50, 68, 95, 99):
        ep = float(np.percentile(err, pct))
        out[f'eps{pct}'] = ep
        out[f'C{pct}_ave'] = ep * q_mean
        out[f'C{pct}_max'] = ep * q_max
        if seq_depths is not None:
            out[f'C{pct}_seq'] = ep * float(np.mean(seq_depths))

    if boot_B and boot_B > 0:
        rng = np.random.default_rng(seed)
        n = len(err)
        for pct in (68, 95, 99):
            c_ave_b = np.empty(boot_B)
            c_max_b = np.empty(boot_B)
            for b in range(boot_B):
                idx = rng.integers(0, n, n)
                ep = np.percentile(err[idx], pct)
                c_ave_b[b] = ep * np.mean(q[idx])
                c_max_b[b] = ep * np.max(q[idx])
            out[f'C{pct}_ave_ci'] = (float(np.percentile(c_ave_b, 2.5)),
                                     float(np.percentile(c_ave_b, 97.5)))
            out[f'C{pct}_max_ci'] = (float(np.percentile(c_max_b, 2.5)),
                                     float(np.percentile(c_max_b, 97.5)))
    return out


def format_metrics(name: str, m: dict) -> str:
    ci95 = m.get('C95_ave_ci', (np.nan, np.nan))
    ci99 = m.get('C99_ave_ci', (np.nan, np.nan))
    seq_str = f" seq={m['C95_seq']:.2f}" if 'C95_seq' in m else ""
    return (f"{name:<34} q_ave={m['q_mean']:7.1f} (max={m['q_max']:6.0f}) "
            f"eps95={m['eps95']:.2e} | "
            f"C68={m['C68_ave']:.3f}  "
            f"C95={m['C95_ave']:.3f} [{ci95[0]:.2f},{ci95[1]:.2f}] "
            f"(C95max={m['C95_max']:.2f})  "
            f"C99={m['C99_ave']:.2f} [{ci99[0]:.2f},{ci99[1]:.2f}]{seq_str}")


def eval_nonadaptive_baseline(nmax: int, ntrials=20000, seed_band=HOLDOUT_SEED,
                              cap_divisor=1.3):
    """Run mlqae flagship (or plain ladder if cap_divisor=None)."""
    lad, shots = flagship_schedule(nmax, r=1.45, cap_divisor=cap_divisor)
    r = evaluate_schedule(lad, shots, ntrials, seed=seed_band + nmax,
                          return_errors=True, return_extra=True)
    q = np.full(ntrials, r['nq'], dtype=float)
    md = np.full(ntrials, r['nmax'], dtype=float)
    return compute_metrics(r['a_true'], np.sin(r['theta_hat']), q,
                           max_depths=md, seq_depths=md)


if __name__ == '__main__':
    print("=== Reference Baselines on HOLDOUT_SEED (ntrials=20000) ===")
    for nmax in (60, 125, 260):
        m_plain = eval_nonadaptive_baseline(nmax, cap_divisor=None)
        m_flag = eval_nonadaptive_baseline(nmax, cap_divisor=1.3)
        print(format_metrics(f"mlqae-plain (nmax={nmax})", m_plain))
        print(format_metrics(f"mlqae-flagship (nmax={nmax})", m_flag))
