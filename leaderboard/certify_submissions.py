"""Discover and certify all AE submissions under `leaderboard/submissions/*`.

Runs `validate_metadata()` (from `leaderboard/schema_and_api.py`) and the
5-stage `verify_submission()` pipeline (from `leaderboard/verifier_and_runner.py`:
AST security, input/output validation, bit-identical reproducibility,
zero-information dummy oracle test, and asymptotic scaling) on every submission
folder in `leaderboard/submissions/*`, then prints a complete certification
report table.

Usage:
    .venv/bin/python leaderboard/certify_submissions.py
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from leaderboard.schema_and_api import validate_metadata
from leaderboard.verifier_and_runner import (
    _load_submission_from_dir,
    verify_submission,
)

SUBMISSIONS_DIR = os.path.join(_HERE, "submissions")


def discover_submission_dirs(base_dir: str = SUBMISSIONS_DIR) -> List[str]:
    """Return sorted list of submission directories containing `submission.py` and `metadata.json`."""
    if not os.path.isdir(base_dir):
        return []
    found: List[str] = []
    for entry in sorted(os.listdir(base_dir)):
        sub_dir = os.path.join(base_dir, entry)
        if not os.path.isdir(sub_dir) or entry.startswith(".") or entry.startswith("__"):
            continue
        if os.path.isfile(os.path.join(sub_dir, "submission.py")) and os.path.isfile(
            os.path.join(sub_dir, "metadata.json")
        ):
            found.append(sub_dir)
    return found


def certify_all_submissions(base_dir: str = SUBMISSIONS_DIR) -> List[Dict[str, Any]]:
    """Run metadata schema validation and 5-stage `verify_submission()` on all discovered submissions."""
    sub_dirs = discover_submission_dirs(base_dir)
    if not sub_dirs:
        raise RuntimeError(f"No submission directories found under {base_dir}")

    results: List[Dict[str, Any]] = []
    print("=" * 136)
    print(f"DISCOVERED {len(sub_dirs)} SUBMISSIONS IN {base_dir}")
    print("=" * 136)

    for idx, sub_dir in enumerate(sub_dirs, start=1):
        slug = os.path.basename(sub_dir)
        meta_path = os.path.join(sub_dir, "metadata.json")
        t0 = time.perf_counter()

        meta = validate_metadata(meta_path)
        sub_obj, algo_name, py_path = _load_submission_from_dir(sub_dir)
        track = meta.get("track", getattr(sub_obj, "track", "unknown"))

        rep = verify_submission(
            sub_obj,
            name=algo_name,
            submission_path=py_path,
            allowed_dir=sub_dir,
        )
        elapsed_s = time.perf_counter() - t0
        sc_diag = rep.diagnostics.get("scaling", {})

        results.append({
            "slug": slug,
            "algorithm_name": algo_name,
            "track": track,
            "report": rep,
            "elapsed_s": elapsed_s,
            "q_60": sc_diag.get("q_ave_small", float("nan")),
            "q_260": sc_diag.get("q_ave_large", float("nan")),
            "eps95_60": sc_diag.get("eps95_small", float("nan")),
            "eps95_260": sc_diag.get("eps95_large", float("nan")),
            "C95_60": sc_diag.get("C95_small", float("nan")),
            "C95_260": sc_diag.get("C95_large", float("nan")),
        })
        status_str = "PASS" if rep.passed else "FAIL"
        print(
            f"[{idx:2d}/{len(sub_dirs):2d}] {slug:<24} | {track:<21} | {status_str:<4} | "
            f"C95(60)={sc_diag.get('C95_small', float('nan')):6.3f} | "
            f"C95(260)={sc_diag.get('C95_large', float('nan')):6.3f} | "
            f"({elapsed_s:.2f}s)"
        )
        if not rep.passed:
            for v in rep.violations:
                print(f"       -> VIOLATION: {v}")

    print("\n" + "=" * 136)
    print("LEADERBOARD SUBMISSION CERTIFICATION SUMMARY TABLE")
    print("=" * 136)
    header = (
        f"{'Slug':<24} | {'Track':<21} | {'AST':^4} | {'I/O':^4} | {'Repr':^4} | "
        f"{'0-Inf':^5} | {'Scale':^5} | {'eps95(60)':>9} | {'eps95(260)':>10} | "
        f"{'C95(60)':>7} | {'C95(260)':>8} | {'Status':^8}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        rep = r["report"]
        print(
            f"{r['slug']:<24} | {r['track']:<21} | "
            f"{'OK' if rep.ast_passed else 'NO':^4} | "
            f"{'OK' if rep.input_output_passed else 'NO':^4} | "
            f"{'OK' if rep.reproducibility_passed else 'NO':^4} | "
            f"{'OK' if rep.zero_info_passed else 'NO':^5} | "
            f"{'OK' if rep.scaling_passed else 'NO':^5} | "
            f"{r['eps95_60']:9.2e} | {r['eps95_260']:10.2e} | "
            f"{r['C95_60']:7.3f} | {r['C95_260']:8.3f} | "
            f"{'CERTIFIED' if rep.passed else 'REJECTED':^8}"
        )
    print("=" * len(header))

    n_pass = sum(1 for r in results if r["report"].passed)
    print(f"\nCertified {n_pass}/{len(results)} submissions.")
    if n_pass != len(results):
        sys.exit(1)
    return results


if __name__ == "__main__":
    certify_all_submissions()
