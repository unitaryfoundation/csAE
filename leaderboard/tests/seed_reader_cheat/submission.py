"""Demo cheat (test fixture, NOT a leaderboard submission): scanner-clean seed reader.

Reproduces REVIEW_2026-11-07.md section 5, item 1. The static AST scanner only matches
the literal names `sys`, `os` and `np`, so aliased imports pass. The cheat harvests every
integer it can find on `sys.argv`, in `os.environ` and on `__main__` (candidate public
seed bands and trial counts) plus every string that could be a certification secret,
rebuilds the trial tables the harness would generate for each candidate, and keys them
by the first draw of the per-trial algorithm RNG the harness hands to `run_trial`. On a
hit it returns the memorized `a_true` (after one token oracle query); on a miss it falls
back to the honest MLQAE flagship estimator.

With the process-isolated harness the worker holds no seed band, no secret and an
algorithm seed that is cryptographically independent of the table, so every lookup
misses and the cheat certifies at the flagship's honest constant.
"""
import sys as s
import os as o_
import __main__ as m_
from operator import attrgetter

import numpy as np

from leaderboard.harness import SCALES, generate_trial_table
from leaderboard.schema_and_api import MLQAEFlagshipSubmission

try:  # the blind-table derivation is public code; only the key is secret
    from leaderboard.verifier_and_runner import derive_blind_trial_table
except Exception:  # pragma: no cover
    derive_blind_trial_table = None

_VERIFY_OFFSETS = (11, 29, 125, 540)
_VERIFY_NTRIALS = (20, 80, 250)
_DEFAULT_NTRIALS = (2000, 20000)
# Names a cheat would try on the harness `__main__` module (review section 5, item 2).
_MAIN_ATTRS = (
    "_STARTUP_SECRET_SEED", "_STARTUP_SALT_SEED", "_STARTUP_CI_SALT",
    "active_secret", "secret", "secret_seed", "seed", "seed_band", "args",
    "HOLDOUT_SEED", "SEARCH_SEED", "BLIND_TEST_SEED", "SALTED_CONCORDANCE_SEED",
)


_SEED_ENV_MARKERS = ("SEED", "SECRET", "SALT", "LEADERBOARD")


def _harvest():
    """Collect candidate seed bands / trial counts (ints) and candidate secrets (strings)."""
    ints, strings, ntrials = set(), set(), set()

    def _take(val, *, is_ntrials=False):
        if isinstance(val, bool):
            return
        if isinstance(val, int):
            (ntrials if is_ntrials else ints).add(int(val))
        elif isinstance(val, str):
            tok = val.split("=", 1)[-1].strip()
            if tok.lstrip("-").isdigit():
                (ntrials if is_ntrials else ints).add(int(tok))
            elif 4 <= len(tok) <= 256:
                strings.add(tok)

    argv = list(s.argv)
    for i, arg in enumerate(argv):
        low = arg.lower()
        nxt = argv[i + 1] if i + 1 < len(argv) else None
        if low.startswith("--ntrials"):
            _take(arg if "=" in arg else (nxt or ""), is_ntrials=True)
        elif low.startswith("--seed") or low.startswith("--salted") or "secret" in low or "salt" in low:
            _take(arg if "=" in arg else (nxt or ""))
    for key, val in o_.environ.items():
        if any(mk in key.upper() for mk in _SEED_ENV_MARKERS):
            _take(val)
    for name in _MAIN_ATTRS:
        try:
            _take(attrgetter(name)(m_))
        except Exception:
            pass
    return ints, strings, ntrials


def _key(alg_seed):
    return round(float(np.random.default_rng(int(alg_seed)).random()), 15)


def _build_lookup():
    ints, strings, n_cli = _harvest()
    bands = {v for v in ints if 1_000 <= v <= 4_000_000_000}
    ntrials = {v for v in n_cli if 10 <= v <= 100_000}
    ntrials |= {max(500, n // 2) for n in list(ntrials)} | set(_VERIFY_NTRIALS)
    lookup = {}
    for band in bands:
        combos = [(band + n, n) for n in SCALES] + [(band + 50_000 + n, n) for n in SCALES]
        combos += [(band + off, 540 if off == 540 else 125) for off in _VERIFY_OFFSETS]
        for seed, nmax in combos:
            for n_t in ntrials:
                try:
                    a_true, _, _, alg_seeds = generate_trial_table(n_t, seed)
                except Exception:
                    continue
                for a_val, ts in zip(a_true, alg_seeds):
                    lookup[_key(ts)] = float(a_val)
                if derive_blind_trial_table is not None:
                    for secret in strings:
                        try:
                            a_b, _, _, alg_b = derive_blind_trial_table(secret, seed, nmax, n_t)
                        except Exception:
                            continue
                        for a_val, ts in zip(a_b, alg_b):
                            lookup[_key(ts)] = float(a_val)
    return lookup


class SeedReaderCheat:
    """Adaptive-track entry point: `run_trial(oracle, nmax, rng)`."""

    track = "adaptive-variable"

    def __init__(self):
        self._honest = MLQAEFlagshipSubmission()
        self._lookup = None

    def run_trial(self, oracle, nmax, rng):
        if self._lookup is None:
            self._lookup = _build_lookup()
        hit = self._lookup.get(_key_from_rng(rng))
        if hit is not None:
            oracle.measure(1, 2)  # token query so queries_used > 0
            return hit
        depths, shots = self._honest.get_schedule(int(nmax))
        counts = oracle.measure_batch(depths, shots)
        est = self._honest.estimate(depths, shots, counts)
        return float(np.asarray(est, dtype=float).reshape(-1)[0])


def _key_from_rng(rng):
    return round(float(rng.random()), 15)


Submission = SeedReaderCheat
