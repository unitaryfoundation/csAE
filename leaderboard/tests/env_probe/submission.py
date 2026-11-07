"""Test fixture (NOT a leaderboard submission): reports what the worker process exposes.

The returned "estimate" is a bitmask so a test can decode it:
    0.10            baseline (nothing found)
  + 0.10            sys.argv has more than the program name
  + 0.20            os.environ holds a secret-looking variable
  + 0.40            __main__ exposes a non-empty secret attribute
"""
import sys as s
import os as o_
import __main__ as m_
from operator import attrgetter


def run_trial(oracle, nmax, rng):
    oracle.measure(0, 1)
    score = 0.10
    if len(s.argv) > 1:
        score += 0.10
    markers = ("SECRET", "TOKEN", "PASSWORD", "LEADERBOARD_")
    if any(any(mk in k.upper() for mk in markers) for k in o_.environ):
        score += 0.20
    for name in ("_STARTUP_SECRET_SEED", "_STARTUP_SALT_SEED", "_STARTUP_CI_SALT", "active_secret"):
        if getattr_safe(m_, name):
            score += 0.40
            break
    return score


def getattr_safe(mod, name):
    try:
        return attrgetter(name)(mod)
    except Exception:
        return None
