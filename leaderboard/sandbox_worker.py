"""Sandbox worker: runs one untrusted submission for `leaderboard/verifier_and_runner.py`.

Launched by `IsolatedSubmission` as

    python -I sandbox_worker.py --submission-file <path> --entry <name> --fd-in <r> --fd-out <w>

The worker never receives the certification secret, the trial amplitudes or the
measurement RNG: it only gets `nmax`, an independent algorithm-RNG seed and the
budget/depth caps per trial, and asks the parent over the RPC pipe for every
oracle sample. Before the submission module is imported, `sys.argv` is truncated
and every secret-looking environment variable is removed, so nothing reachable
from `sys`, `os` or `__main__` identifies the trial table.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


def _parse(argv):
    opts = {"--submission-file": None, "--entry": None, "--fd-in": None, "--fd-out": None}
    i = 1
    while i < len(argv):
        key = argv[i]
        if key in opts and i + 1 < len(argv):
            opts[key] = argv[i + 1]
            i += 2
        else:
            i += 1
    return opts


def main() -> int:
    opts = _parse(sys.argv)
    submission_file = opts["--submission-file"]
    entry = opts["--entry"] or None
    fd_in = int(opts["--fd-in"])
    fd_out = int(opts["--fd-out"])

    # Scrub BEFORE importing anything that could be reached by the submission.
    sys.argv[:] = sys.argv[:1]
    for key in list(os.environ.keys()):
        ku = key.upper()
        if ku.startswith(("LEADERBOARD_", "CI_SECRET")) or any(
            m in ku for m in ("SECRET", "SEED", "SALT", "TOKEN", "PASSWORD", "PASSWD", "CREDENTIAL", "PRIVATE_KEY", "API_KEY")
        ):
            os.environ.pop(key, None)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from leaderboard import verifier_and_runner as vr

    vr.scrub_process_for_untrusted_code()
    return vr._worker_main(submission_file, entry, fd_in, fd_out)


if __name__ == "__main__":
    sys.exit(main())
