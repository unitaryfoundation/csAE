"""CI Verification, Anti-Cheat & Statistical Benchmark Engine for the AE Leaderboard.

Implements:
1. Blind Secret Seed Scrubbing & HMAC Trial Permutation:
   - Automatically pops and scrubs `LEADERBOARD_SECRET_SEED` and `LEADERBOARD_SALT_SEED`
     (or any custom `--secret-seed-env` variable) from `os.environ` at startup BEFORE
     importing or executing any user submission module.
   - Uses `hmac.new(secret.encode(), ..., hashlib.sha256)` to derive both the blind
     evaluation seed and a cryptographic trial permutation order so submitters cannot
     predict or memorize test amplitudes.
2. Static & Runtime Anti-Cheat Verification (`verify_submission`):
   - AST inspection blocking forbidden modules (`inspect`, `sys._getframe`, `ctypes`,
     `gc`, `subprocess`, `socket`, `pickle`, `importlib`, `builtins`, etc.),
     forbidden builtins (`globals()`, `locals()`, `vars()`, `dir()`, `eval()`,
     `exec()`, `compile()`, `getattr()`, `__import__()`), dunder frame/closure
     introspection (`__closure__`, `cell_contents`, `f_locals`, `f_globals`,
     `f_back`, `__dict__`), and network/file I/O outside the submission directory.
   - Guarded Black-Box Quantum Oracle (`GuardedOracle`) with strict attribute
     lockdown, immutability checks, and exact query/depth/round accounting.
   - Oracle Tamper & Zero-Information Test: executes the submission against a
     constant-0.5 Bernoulli coin-flip oracle while keeping `a_true` as a honeypot
     in the caller frame, verifying statistically that estimates carry zero
     mutual information / correlation with `a_true`.
   - Reproducibility Check: verifies that running twice with the same seed
     produces bit-identical amplitude estimates, query counts, and round histories.
   - Input & Output Validation: rejects negative or non-integer Grover depths,
     zero/negative/non-integer shots, and non-finite or out-of-bounds estimates
     outside [0, 1].
   - Asymptotic Scaling Check: verifies across 2 scales that eps95 decreases as
     query budget increases, achieves quantum depth scaling (eps95 * nmax <= 8.0),
     and that C95 beats both random guessing and depth-0 classical sampling.
3. Full Statistical Evaluation (`benchmark_submission`):
   - Evaluates across standard scales nmax in {125, 260, 540, 1000} on
     HOLDOUT_SEED = 900_000 (or blind HMAC secret seed when active) for the
     standard prior a ~ U(0.1, 0.9) AND tests prior robustness on a ~ U(0.01, 0.99).
   - Computes C68_ave, C95_ave (+ 95% bootstrap CI), C95_max (+ 95% bootstrap CI),
     C99_ave (+ 95% bootstrap CI), C99_max (+ 95% bootstrap CI), sequential depth
     constant C95_seq, number of adaptive rounds, query spread std(Q)/Q_ave, and
     wall-clock runtime per trial.
"""
from __future__ import annotations

import ast
import hashlib
import hmac
import inspect as _host_inspect
import math
import os
import sys
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# Immediately pop and scrub secret seeds from os.environ at module load time
# BEFORE any user submission module can be imported or executed.
_STARTUP_SECRET_SEED: Optional[str] = os.environ.pop("LEADERBOARD_SECRET_SEED", None)
_STARTUP_SALT_SEED: Optional[str] = os.environ.pop("LEADERBOARD_SALT_SEED", None)
_STARTUP_CI_SALT: Optional[str] = os.environ.pop("CI_SECRET_SALT", None)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from mlqae.core import (
    KAPPA,
    canonical_shots,
    crlb_constant,
    flagship_schedule,
    geom_ladder,
    grid_spacing,
)
from leaderboard.harness import (
    HOLDOUT_SEED,
    SCALES,
    SEARCH_SEED,
    compute_metrics,
    generate_trials,
    shot_cost,
)

__all__ = [
    "SecurityViolationError",
    "OracleInputError",
    "OracleTamperError",
    "OutputValidationError",
    "GuardedOracle",
    "VerificationReport",
    "scrub_secret_seed_env",
    "derive_blind_seed_and_permutation",
    "check_ast_security",
    "verify_submission",
    "benchmark_submission",
    "format_benchmark_report",
    "MLQAEFlagshipSubmission",
    "MLQAEPlainSubmission",
    "ReferenceBayesianAdaptiveSubmission",
    "AdaptiveTwoRoundSubmission",
]


# ============================================================================
# Exceptions & Secret Seed HMAC Derivation
# ============================================================================

class SecurityViolationError(Exception):
    """Raised when a submission violates static AST or runtime anti-cheat rules."""


class OracleInputError(ValueError):
    """Raised when a submission passes invalid depth or shot parameters to the oracle."""


class OracleTamperError(SecurityViolationError):
    """Raised when a submission attempts to inspect or mutate private oracle state."""


class OutputValidationError(ValueError):
    """Raised when a submission returns an invalid amplitude estimate outside [0, 1]."""


def scrub_secret_seed_env(env_var_name: str = "LEADERBOARD_SECRET_SEED") -> Optional[str]:
    """Pop and scrub `env_var_name` (plus standard secret env vars) from `os.environ`.

    Returns the secret string if present in `os.environ` or captured at startup,
    while guaranteeing that `os.environ` no longer holds any secret seed keys.
    """
    val = os.environ.pop(env_var_name, None)
    os.environ.pop("LEADERBOARD_SECRET_SEED", None)
    os.environ.pop("LEADERBOARD_SALT_SEED", None)
    os.environ.pop("CI_SECRET_SALT", None)
    if val:
        return val
    if _STARTUP_SECRET_SEED:
        return _STARTUP_SECRET_SEED
    if _STARTUP_SALT_SEED:
        return _STARTUP_SALT_SEED
    if _STARTUP_CI_SALT:
        return _STARTUP_CI_SALT
    return None


def derive_blind_seed_and_permutation(
    secret: str,
    base_seed: int,
    nmax: int,
    ntrials: int,
) -> Tuple[int, np.ndarray]:
    """Derive a blind evaluation seed and cryptographic trial permutation via HMAC-SHA256."""
    secret_bytes = secret.encode("utf-8")
    seed_msg = f"ae-leaderboard-blind-seed:base={int(base_seed)}:nmax={int(nmax)}".encode("utf-8")
    seed_digest = hmac.new(secret_bytes, seed_msg, hashlib.sha256).digest()
    blind_seed = 10_000_000 + (int.from_bytes(seed_digest[:8], "big") % 1_990_000_000)

    perm_msg = f"ae-leaderboard-trial-perm:base={int(base_seed)}:nmax={int(nmax)}:n={int(ntrials)}".encode("utf-8")
    perm_digest = hmac.new(secret_bytes, perm_msg, hashlib.sha256).digest()
    perm_seed = int.from_bytes(perm_digest[:8], "big")
    perm_rng = np.random.default_rng(perm_seed)
    perm = perm_rng.permutation(int(ntrials))
    return int(blind_seed), perm


# ============================================================================
# 1. Static AST Anti-Cheat Inspector
# ============================================================================

FORBIDDEN_MODULES = frozenset({
    "inspect",
    "ctypes",
    "gc",
    "subprocess",
    "socket",
    "ssl",
    "urllib",
    "http",
    "requests",
    "ftplib",
    "smtplib",
    "telnetlib",
    "xmlrpc",
    "shutil",
    "pty",
    "fcntl",
    "mmap",
    "resource",
    "signal",
    "pickle",
    "cPickle",
    "_pickle",
    "marshal",
    "shelve",
    "dbm",
    "sqlite3",
    "importlib",
    "imp",
    "pkgutil",
    "zipimport",
    "runpy",
    "builtins",
    "__builtin__",
    "code",
    "codeop",
    "compileall",
    "traceback",
    "linecache",
    "tokenize",
    "pdb",
    "bdb",
    "trace",
    "tracemalloc",
    "multiprocessing",
    "threading",
    "concurrent",
    "asyncio",
    "tempfile",
    "glob",
})

FORBIDDEN_BUILTINS = frozenset({
    "globals",
    "locals",
    "vars",
    "dir",
    "eval",
    "exec",
    "compile",
    "__import__",
    "getattr",
    "setattr",
    "delattr",
    "breakpoint",
    "input",
    "memoryview",
    "_getframe",
})

FORBIDDEN_ATTRIBUTES = frozenset({
    "_getframe",
    "f_locals",
    "f_globals",
    "f_back",
    "f_builtins",
    "f_code",
    "f_lasti",
    "gi_frame",
    "gi_code",
    "cr_frame",
    "cr_code",
    "ag_frame",
    "tb_frame",
    "tb_next",
    "__closure__",
    "cell_contents",
    "__globals__",
    "__code__",
    "__dict__",
    "__subclasses__",
    "__bases__",
    "__mro__",
    "__builtins__",
    "__getattribute__",
    "__setattr__",
    "__delattr__",
    "__reduce__",
    "__reduce_ex__",
    "_a_true",
    "a_true",
    "theta_true",
    "_theta_true",
    "_secret",
    "exc_info",
    "last_traceback",
})

FORBIDDEN_OS_CALLS = frozenset({
    "system",
    "popen",
    "fork",
    "forkpty",
    "kill",
    "killpg",
    "spawnl",
    "spawnle",
    "spawnlp",
    "spawnlpe",
    "spawnv",
    "spawnve",
    "spawnvp",
    "spawnvpe",
    "execl",
    "execle",
    "execlp",
    "execlpe",
    "execv",
    "execve",
    "execvp",
    "execvpe",
    "remove",
    "unlink",
    "rmdir",
    "removedirs",
    "rename",
    "replace",
    "chmod",
    "chown",
    "link",
    "symlink",
    "listdir",
    "scandir",
    "walk",
    "fwalk",
    "open",
    "fdopen",
    "urandom",
    "putenv",
    "unsetenv",
    "getenv",
    "environ",
})

FORBIDDEN_NP_IO_CALLS = frozenset({
    "load",
    "save",
    "savez",
    "savez_compressed",
    "loadtxt",
    "savetxt",
    "genfromtxt",
    "fromfile",
    "tofile",
    "memmap",
    "DataSource",
})


class _AntiCheatASTVisitor(ast.NodeVisitor):
    """AST visitor enforcing static security rules on submitted code."""

    def __init__(self, allowed_dir: Optional[str] = None):
        self.allowed_dir = os.path.realpath(allowed_dir) if allowed_dir else None
        self.violations: List[str] = []

    def _record(self, node: ast.AST, msg: str) -> None:
        lineno = getattr(node, "lineno", "?")
        self.violations.append(f"Line {lineno}: {msg}")

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root_mod = alias.name.split(".")[0]
            if root_mod in FORBIDDEN_MODULES or alias.name in FORBIDDEN_MODULES:
                self._record(node, f"Forbidden import '{alias.name}'")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        root_mod = mod.split(".")[0]
        if root_mod in FORBIDDEN_MODULES or mod in FORBIDDEN_MODULES:
            self._record(node, f"Forbidden import from '{mod}'")
        for alias in node.names:
            if alias.name in FORBIDDEN_MODULES or alias.name in FORBIDDEN_ATTRIBUTES or alias.name in FORBIDDEN_BUILTINS:
                self._record(node, f"Forbidden symbol import '{alias.name}' from '{mod}'")
            if root_mod == "os" and alias.name in FORBIDDEN_OS_CALLS:
                self._record(node, f"Forbidden os function import 'os.{alias.name}'")
            if root_mod == "sys" and alias.name in ("_getframe", "modules", "exc_info", "settrace", "setprofile"):
                self._record(node, f"Forbidden sys symbol import 'sys.{alias.name}'")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        attr = node.attr
        if attr in FORBIDDEN_ATTRIBUTES:
            self._record(node, f"Forbidden attribute access '.{attr}'")
        if isinstance(node.value, ast.Name):
            base = node.value.id
            if base == "sys" and attr in ("_getframe", "modules", "exc_info", "settrace", "setprofile", "audit"):
                self._record(node, f"Forbidden sys attribute 'sys.{attr}'")
            elif base == "os" and attr in FORBIDDEN_OS_CALLS:
                self._record(node, f"Forbidden os operation 'os.{attr}'")
            elif base in ("np", "numpy") and attr in FORBIDDEN_NP_IO_CALLS:
                self._record(node, f"Forbidden numpy file I/O '{base}.{attr}'")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in ("__builtins__", "__loader__", "__spec__"):
            self._record(node, f"Forbidden special identifier '{node.id}'")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name):
            fname = func.id
            if fname in FORBIDDEN_BUILTINS:
                self._record(node, f"Forbidden builtin call '{fname}()'")
            elif fname == "open":
                self._check_open_call(node)
        elif isinstance(func, ast.Attribute):
            attr = func.attr
            if attr in FORBIDDEN_BUILTINS:
                self._record(node, f"Forbidden method/builtin call '.{attr}()'")
            elif attr == "open":
                self._check_open_call(node)
        self.generic_visit(node)

    def _check_open_call(self, node: ast.Call) -> None:
        """Allow read-only open() only on static literal paths strictly inside allowed_dir."""
        if self.allowed_dir is None:
            self._record(node, "Forbidden file I/O call 'open()' (no local directory access permitted)")
            return
        if not node.args or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
            self._record(node, "Forbidden dynamic file I/O 'open()': path must be a string literal inside submission directory")
            return
        raw_path = node.args[0].value
        if "://" in raw_path:
            self._record(node, f"Forbidden network/URI open('{raw_path}')")
            return
        mode = "r"
        if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
            mode = node.args[1].value
        for kw in node.keywords:
            if kw.arg == "mode" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                mode = kw.value.value
        if any(ch in mode for ch in ("w", "a", "x", "+")):
            self._record(node, f"Forbidden file write mode '{mode}' in open()")
            return
        cand_path = raw_path if os.path.isabs(raw_path) else os.path.join(self.allowed_dir, raw_path)
        real_cand = os.path.realpath(cand_path)
        if not (real_cand == self.allowed_dir or real_cand.startswith(self.allowed_dir + os.sep)):
            self._record(node, f"Forbidden file I/O outside submission directory: '{raw_path}'")


def _extract_source_and_dir(
    submission: Any,
    source_code: Optional[str] = None,
    submission_path: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """Extract Python source code and allowed directory for AST inspection."""
    if source_code is not None:
        allowed_dir = os.path.dirname(os.path.abspath(submission_path)) if submission_path else _HERE
        return textwrap.dedent(source_code), allowed_dir

    if submission_path is not None and os.path.isfile(submission_path):
        abs_path = os.path.abspath(submission_path)
        with open(abs_path, "r", encoding="utf-8") as fh:
            return fh.read(), os.path.dirname(abs_path)

    if isinstance(submission, str) and os.path.isfile(submission):
        abs_path = os.path.abspath(submission)
        with open(abs_path, "r", encoding="utf-8") as fh:
            return fh.read(), os.path.dirname(abs_path)

    target = submission
    if not _host_inspect.isfunction(target) and not _host_inspect.isclass(target) and not _host_inspect.ismethod(target):
        target = type(submission)

    try:
        src = _host_inspect.getsource(target)
    except (OSError, TypeError) as exc:
        raise SecurityViolationError(
            f"Cannot retrieve source code for AST inspection of {submission!r}: {exc}"
        ) from exc

    try:
        src_file = _host_inspect.getsourcefile(target)
        allowed_dir = os.path.dirname(os.path.abspath(src_file)) if src_file else _HERE
    except (OSError, TypeError):
        allowed_dir = _HERE

    return textwrap.dedent(src), allowed_dir


def check_ast_security(
    submission: Any,
    source_code: Optional[str] = None,
    submission_path: Optional[str] = None,
    allowed_dir: Optional[str] = None,
) -> Tuple[bool, List[str]]:
    """Perform static AST inspection on submitted code."""
    try:
        src, inferred_dir = _extract_source_and_dir(submission, source_code, submission_path)
    except SecurityViolationError as exc:
        return False, [str(exc)]

    eff_dir = allowed_dir if allowed_dir is not None else inferred_dir
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return False, [f"SyntaxError during AST parse: {exc}"]

    visitor = _AntiCheatASTVisitor(allowed_dir=eff_dir)
    visitor.visit(tree)
    return len(visitor.violations) == 0, visitor.violations


# ============================================================================
# 2. Guarded Black-Box Quantum Oracle
# ============================================================================

# The algorithm RNG handed to a submission is derived from the trial seed with this
# XOR so it never shares a stream with the measurement RNG. The trial seed itself is
# never exposed to submissions.
ALG_RNG_XOR = 0x5DEECE66D

_ALLOWED_ORACLE_ATTRS = frozenset({
    "measure",
    "measure_batch",
    "nmax",
    "budget",
    "max_queries",
    "max_query_budget",
    "max_depth_cap",
    "eta",
    "queries_used",
    "total_queries",
    "total_shots",
    "remaining_queries",
    "remaining_budget",
    "rounds_used",
    "num_rounds",
    "max_depth_used",
    "max_depth",
    "seq_depth_used",
    "seq_depth",
    "rng",
    "history",
    "accounting",
})


def _validate_depth_and_shots(depth: Any, shots: Any, max_depth_limit: int = 1_000_000) -> Tuple[int, int]:
    """Validate that Grover depth is an int >= 0 and shots is an int >= 1."""
    if isinstance(depth, bool) or not isinstance(depth, (int, np.integer)):
        raise OracleInputError(f"Grover depth must be an integer, got {depth!r} ({type(depth).__name__})")
    if isinstance(shots, bool) or not isinstance(shots, (int, np.integer)):
        raise OracleInputError(f"Shot count must be an integer, got {shots!r} ({type(shots).__name__})")
    d_int = int(depth)
    s_int = int(shots)
    if d_int < 0:
        raise OracleInputError(f"Grover depth must be non-negative (>= 0), got {d_int}")
    if d_int > max_depth_limit:
        raise OracleInputError(f"Grover depth {d_int} exceeds maximum allowed depth {max_depth_limit}")
    if s_int <= 0:
        raise OracleInputError(f"Shot count must be strictly positive (>= 1), got {s_int}")
    return d_int, s_int


class GuardedOracle:
    """Tamper-evident black-box quantum amplitude estimation oracle.

    The true amplitude `a_true` and measurement RNG are NEVER stored as
    instance attributes on `GuardedOracle`. All state lives inside an opaque
    lexical closure created by `_create_guarded_oracle`, and `__getattribute__`
    blocks access to any attribute not in `_ALLOWED_ORACLE_ATTRS`.
    """

    __slots__ = ("_dispatch",)

    def __init__(self, dispatch_fn: Callable[[str, Any], Any]):
        object.__setattr__(self, "_dispatch", dispatch_fn)

    def __getattribute__(self, name: str) -> Any:
        if name not in _ALLOWED_ORACLE_ATTRS:
            raise OracleTamperError(f"Forbidden access to private or unknown oracle attribute '{name}'")
        dispatch = object.__getattribute__(self, "_dispatch")
        return dispatch(name, None)

    def __setattr__(self, name: str, value: Any) -> None:
        raise OracleTamperError(f"Oracle is immutable; cannot set attribute '{name}'")

    def __delattr__(self, name: str) -> None:
        raise OracleTamperError(f"Oracle is immutable; cannot delete attribute '{name}'")

    def __call__(self, depth: Union[int, Sequence[int]], shots: Union[int, Sequence[int]] = 1) -> Union[int, np.ndarray]:
        dispatch = object.__getattribute__(self, "_dispatch")
        if isinstance(depth, (list, tuple, np.ndarray)):
            return dispatch("measure_batch", (depth, shots))
        return dispatch("measure", (depth, shots))

    def __repr__(self) -> str:
        dispatch = object.__getattribute__(self, "_dispatch")
        return (
            f"<GuardedOracle nmax={dispatch('nmax', None)} "
            f"queries_used={dispatch('queries_used', None)} "
            f"rounds_used={dispatch('rounds_used', None)}>"
        )


def _create_guarded_oracle(
    th_true: float,
    nmax: int,
    trial_seed: int,
    *,
    zero_info_mode: bool = False,
    eta: float = 0.0,
    max_queries: Optional[int] = None,
    max_depth_limit: int = 1_000_000,
) -> Tuple[GuardedOracle, Callable[[], Dict[str, Any]]]:
    """Create a (GuardedOracle, get_private_stats) pair with lexical isolation."""
    meas_rng = np.random.default_rng(int(trial_seed))
    alg_rng = np.random.default_rng(int(trial_seed) ^ ALG_RNG_XOR)

    queries_used = 0
    total_shots = 0
    rounds_used = 0
    max_depth_used = 0
    seq_depth_used = 0
    tamper_attempts = 0
    history: List[Tuple[int, int, int]] = []

    secret_theta = float("nan") if zero_info_mode else float(th_true)

    def _sample_single(d_int: int, s_int: int) -> int:
        if zero_info_mode:
            return int(meas_rng.binomial(s_int, 0.5))
        d_factor = 2.0 * d_int + 1.0
        c = math.cos(d_factor * secret_theta)
        p = c * c
        if eta > 0.0:
            v = (1.0 - eta) ** d_factor
            p = v * p + 0.5 * (1.0 - v)
        p = min(max(p, 0.0), 1.0)
        return int(meas_rng.binomial(s_int, p))

    def _measure(depth: Any, shots: Any = 1) -> int:
        nonlocal queries_used, total_shots, rounds_used, max_depth_used, seq_depth_used
        d_int, s_int = _validate_depth_and_shots(depth, shots, max_depth_limit)
        cost = s_int * shot_cost(d_int)
        if max_queries is not None and queries_used + cost > max_queries:
            raise OracleInputError(
                f"Query budget exceeded: requested {cost} queries at depth {d_int}, "
                f"but only {max_queries - queries_used} remain of budget {max_queries}"
            )
        k = _sample_single(d_int, s_int)
        queries_used += cost
        total_shots += s_int
        rounds_used += 1
        if d_int > max_depth_used:
            max_depth_used = d_int
        seq_depth_used += max(1, d_int) if d_int == 0 else d_int
        history.append((d_int, s_int, k))
        return k

    def _measure_batch(depths: Sequence[Any], shots: Sequence[Any]) -> np.ndarray:
        nonlocal queries_used, total_shots, rounds_used, max_depth_used, seq_depth_used
        if not isinstance(depths, (list, tuple, np.ndarray)) or not isinstance(shots, (list, tuple, np.ndarray)):
            raise OracleInputError("measure_batch requires sequences (list, tuple, or ndarray) for depths and shots")
        if len(depths) == 0 or len(depths) != len(shots):
            raise OracleInputError(
                f"measure_batch requires non-empty equal-length sequences, got len(depths)={len(depths)}, len(shots)={len(shots)}"
            )

        validated_pairs: List[Tuple[int, int]] = []
        batch_cost = 0
        batch_shots = 0
        round_max_depth = 0
        for d_raw, s_raw in zip(depths, shots):
            d_int, s_int = _validate_depth_and_shots(d_raw, s_raw, max_depth_limit)
            validated_pairs.append((d_int, s_int))
            batch_cost += s_int * shot_cost(d_int)
            batch_shots += s_int
            if d_int > round_max_depth:
                round_max_depth = d_int

        if max_queries is not None and queries_used + batch_cost > max_queries:
            raise OracleInputError(
                f"Query budget exceeded in measure_batch: requested {batch_cost} queries, "
                f"but only {max_queries - queries_used} remain of budget {max_queries}"
            )

        counts = np.empty(len(validated_pairs), dtype=np.int64)
        for idx, (d_int, s_int) in enumerate(validated_pairs):
            k = _sample_single(d_int, s_int)
            counts[idx] = k
            history.append((d_int, s_int, k))
            if d_int > max_depth_used:
                max_depth_used = d_int

        queries_used += batch_cost
        total_shots += batch_shots
        rounds_used += 1
        seq_depth_used += max(1, round_max_depth) if round_max_depth == 0 else round_max_depth
        return counts

    def _dispatch(attr: str, payload: Any) -> Any:
        nonlocal tamper_attempts
        if payload is not None:
            if attr == "measure":
                return _measure(*payload)
            if attr == "measure_batch":
                return _measure_batch(*payload)
        if attr == "measure":
            return _measure
        if attr == "measure_batch":
            return _measure_batch
        if attr == "nmax":
            return int(nmax)
        if attr in ("budget", "max_queries", "max_query_budget"):
            return max_queries
        if attr == "max_depth_cap":
            return max_depth_limit
        if attr == "eta":
            return float(eta)
        if attr in ("queries_used", "total_queries"):
            return int(queries_used)
        if attr == "total_shots":
            return int(total_shots)
        if attr in ("remaining_queries", "remaining_budget"):
            return None if max_queries is None else max(0, int(max_queries - queries_used))
        if attr in ("rounds_used", "num_rounds"):
            return int(rounds_used)
        if attr in ("max_depth_used", "max_depth"):
            return int(max_depth_used)
        if attr in ("seq_depth_used", "seq_depth"):
            return int(seq_depth_used)
        if attr == "rng":
            return alg_rng
        if attr == "history":
            return list(history)
        if attr == "accounting":
            return {
                "total_queries": int(queries_used),
                "total_shots": int(total_shots),
                "max_depth": int(max_depth_used),
                "seq_depth": int(seq_depth_used),
                "num_rounds": int(rounds_used),
                "eta": float(eta),
                "max_query_budget": max_queries,
                "max_depth_cap": max_depth_limit,
            }
        tamper_attempts += 1
        raise OracleTamperError(f"Invalid dispatch request '{attr}'")

    def _get_private_stats() -> Dict[str, Any]:
        return {
            "queries_used": queries_used,
            "rounds_used": rounds_used,
            "max_depth_used": max_depth_used,
            "seq_depth_used": seq_depth_used,
            "tamper_attempts": tamper_attempts,
            "history": tuple(history),
        }

    return GuardedOracle(_dispatch), _get_private_stats


# ============================================================================
# 3. Submission Invocation & Output Validation
# ============================================================================

def _validate_estimate(a_hat: Any) -> float:
    """Validate that the returned amplitude estimate is a finite float in [0, 1]."""
    if isinstance(a_hat, bool) or a_hat is None:
        raise OutputValidationError(f"Amplitude estimate must be a real float in [0, 1], got {a_hat!r}")
    if isinstance(a_hat, np.ndarray):
        if a_hat.ndim != 0 and a_hat.size != 1:
            raise OutputValidationError(f"Amplitude estimate must be a scalar, got array of shape {a_hat.shape}")
        a_hat = float(a_hat.item())
    if isinstance(a_hat, complex):
        raise OutputValidationError(f"Amplitude estimate must be real, got complex {a_hat!r}")
    try:
        val = float(a_hat)
    except (TypeError, ValueError) as exc:
        raise OutputValidationError(f"Amplitude estimate could not be converted to float: {a_hat!r}") from exc
    if not math.isfinite(val):
        raise OutputValidationError(f"Amplitude estimate must be finite, got {val}")
    if val < 0.0 or val > 1.0:
        raise OutputValidationError(f"Amplitude estimate {val:.6f} is outside valid interval [0, 1]")
    return val


def _resolve_callable(submission: Any) -> Callable[..., Any]:
    """Resolve a submission (function, class, or object instance) to its entry callable."""
    if _host_inspect.isclass(submission):
        instance = submission()
        return _resolve_callable(instance)
    if hasattr(submission, "get_schedule") and callable(getattr(submission, "get_schedule")) and hasattr(submission, "estimate"):
        def _nonadaptive_trial_adapter(oracle: GuardedOracle, nmax: int) -> float:
            depths, shots = submission.get_schedule(int(nmax))
            counts = oracle.measure_batch(depths, shots)
            est = submission.estimate(depths, shots, counts)
            arr = np.asarray(est, dtype=float)
            return float(arr.reshape(-1)[0])
        return _nonadaptive_trial_adapter
    if hasattr(submission, "run_trial") and callable(getattr(submission, "run_trial")):
        return getattr(submission, "run_trial")
    if callable(submission):
        return submission
    for method_name in ("estimate", "run", "solve", "__call__"):
        if hasattr(submission, method_name):
            fn = getattr(submission, method_name)
            if callable(fn):
                return fn
    raise TypeError(
        f"Submission {submission!r} must be a callable or expose .run_trial() / .estimate() / .get_schedule()"
    )


def _invoke_single_trial(
    fn: Callable[..., Any],
    oracle: GuardedOracle,
    nmax: int,
    trial_seed: int,
) -> float:
    """Invoke `fn` with `(oracle)`, `(oracle, nmax)`, or `(oracle, nmax, rng/seed)`."""
    sig = _host_inspect.signature(fn)
    params = [
        p for p in sig.parameters.values()
        if p.kind in (_host_inspect.Parameter.POSITIONAL_ONLY, _host_inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_varargs = any(p.kind == _host_inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())

    if len(params) == 1:
        raw_out = fn(oracle)
    elif len(params) == 2:
        raw_out = fn(oracle, int(nmax))
    elif len(params) >= 3 or has_varargs:
        # Always hand over the algorithm RNG, never the trial seed: the measurement
        # RNG is derived from the trial seed, so exposing it would let a submission
        # replay the binomial draws and recover theta without any information.
        raw_out = fn(oracle, int(nmax), oracle.rng)
    else:
        raw_out = fn(oracle)

    a_hat = _validate_estimate(raw_out)
    if oracle.queries_used <= 0:
        raise OutputValidationError("Submission returned an estimate without making any oracle queries (queries_used == 0)")
    return a_hat


def _run_trials_guarded(
    submission: Any,
    nmax: int,
    ntrials: int,
    seed: int,
    *,
    a_range: Tuple[float, float] = (0.1, 0.9),
    zero_info_mode: bool = False,
    eta: float = 0.0,
    max_queries: Optional[int] = None,
    secret_seed: Optional[str] = None,
) -> Dict[str, Any]:
    """Run `ntrials` guarded trials of `submission` at scale `nmax`.

    If `secret_seed` is provided (or popped from `LEADERBOARD_SECRET_SEED`), the
    effective seed and trial permutation order are derived via HMAC-SHA256 so
    the test amplitudes and their ordering are 100% blind.
    """
    fn = _resolve_callable(submission)
    if secret_seed:
        eff_seed, perm = derive_blind_seed_and_permutation(secret_seed, seed, nmax, ntrials)
        a_raw, th_raw, seeds_raw = generate_trials(ntrials, eff_seed, a_range=a_range)
        a_true_arr = a_raw[perm]
        th_true_arr = th_raw[perm]
        trial_seeds = seeds_raw[perm]
    else:
        a_true_arr, th_true_arr, trial_seeds = generate_trials(ntrials, seed, a_range=a_range)

    a_hat_arr = np.empty(ntrials, dtype=np.float64)
    queries_arr = np.empty(ntrials, dtype=np.float64)
    rounds_arr = np.empty(ntrials, dtype=np.float64)
    max_depths_arr = np.empty(ntrials, dtype=np.float64)
    seq_depths_arr = np.empty(ntrials, dtype=np.float64)
    histories: List[Tuple[Tuple[int, int, int], ...]] = []

    t0 = time.perf_counter()
    for i in range(ntrials):
        # Honeypot local variables in caller frame for zero-info side-channel test:
        a_true = float(a_true_arr[i])
        theta_true = float(th_true_arr[i])
        t_seed = int(trial_seeds[i])

        oracle, get_stats = _create_guarded_oracle(
            theta_true,
            nmax=nmax,
            trial_seed=t_seed,
            zero_info_mode=zero_info_mode,
            eta=eta,
            max_queries=max_queries,
            max_depth_limit=int(nmax),  # the benchmark scale is a hard depth cap
        )
        a_hat = _invoke_single_trial(fn, oracle, nmax, t_seed)
        stats = get_stats()
        if stats["tamper_attempts"] > 0:
            raise OracleTamperError(f"Detected {stats['tamper_attempts']} oracle tamper attempts on trial {i}")
        if object.__getattribute__(oracle, "_dispatch") is None:
            raise OracleTamperError("Oracle dispatch table was modified")

        a_hat_arr[i] = a_hat
        queries_arr[i] = stats["queries_used"]
        rounds_arr[i] = stats["rounds_used"]
        max_depths_arr[i] = stats["max_depth_used"]
        seq_depths_arr[i] = stats["seq_depth_used"]
        histories.append(stats["history"])

    elapsed = time.perf_counter() - t0
    return {
        "a_true": a_true_arr,
        "a_hat": a_hat_arr,
        "errors": np.abs(a_hat_arr - a_true_arr),
        "queries": queries_arr,
        "rounds": rounds_arr,
        "max_depths": max_depths_arr,
        "seq_depths": seq_depths_arr,
        "histories": histories,
        "elapsed_sec": elapsed,
        "runtime_per_trial_sec": elapsed / max(1, ntrials),
        "runtime_per_trial_ms": 1000.0 * elapsed / max(1, ntrials),
    }


# ============================================================================
# 4. Five-Stage Verification Pipeline (`verify_submission`)
# ============================================================================

@dataclass
class VerificationReport:
    """Detailed report produced by `verify_submission`."""

    name: str
    passed: bool
    ast_passed: bool = False
    input_output_passed: bool = False
    reproducibility_passed: bool = False
    zero_info_passed: bool = False
    scaling_passed: bool = False
    violations: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            f"=== Verification Report: {self.name} [{status}] ===",
            f"  1. Static AST Security Check      : {'PASS' if self.ast_passed else 'FAIL'}",
            f"  2. Input/Output Validation Check  : {'PASS' if self.input_output_passed else 'FAIL'}",
            f"  3. Reproducibility Check          : {'PASS' if self.reproducibility_passed else 'FAIL'}",
            f"  4. Zero-Info / Oracle Tamper Test : {'PASS' if self.zero_info_passed else 'FAIL'}",
            f"  5. Asymptotic Scaling Check       : {'PASS' if self.scaling_passed else 'FAIL'}",
        ]
        if self.violations:
            lines.append("  Violations:")
            for v in self.violations:
                lines.append(f"    - {v}")
        return "\n".join(lines)


def _depth0_classical_c95(q_ave: float, ntrials: int = 2000, seed: int = 777_777) -> Tuple[float, float]:
    """Compute reference (eps95, C95) for a trivial depth-0 classical sampler with round(q_ave) shots."""
    n_shots = max(1, int(round(q_ave)))
    rng = np.random.default_rng(seed)
    a_true = rng.uniform(0.1, 0.9, ntrials)
    p0 = 1.0 - a_true * a_true
    k = rng.binomial(n_shots, p0)
    a_hat = np.sqrt(np.clip(1.0 - k / float(n_shots), 0.0, 1.0))
    err = np.abs(a_hat - a_true)
    eps95 = float(np.percentile(err, 95))
    return eps95, eps95 * n_shots


def verify_submission(
    submission: Any,
    *,
    name: str = "Submission",
    source_code: Optional[str] = None,
    submission_path: Optional[str] = None,
    allowed_dir: Optional[str] = None,
    scales: Tuple[int, int] = (125, 540),
    ntrials_repro: int = 20,
    ntrials_zero_info: int = 80,
    ntrials_scaling: int = 250,
    seed: int = SEARCH_SEED,
    secret_seed: Optional[str] = None,
) -> VerificationReport:
    """Run all 5 CI verification and anti-cheat checks on a submitted AE algorithm."""
    scrub_secret_seed_env()
    report = VerificationReport(name=name, passed=False)

    # ------------------------------------------------------------------
    # Stage 1: Static AST Security Check
    # ------------------------------------------------------------------
    ast_ok, ast_violations = check_ast_security(
        submission,
        source_code=source_code,
        submission_path=submission_path,
        allowed_dir=allowed_dir,
    )
    report.ast_passed = ast_ok
    for v in ast_violations:
        report.violations.append(f"[AST] {v}")

    # ------------------------------------------------------------------
    # Stage 2 & 3: Input/Output Validation + Reproducibility Check
    # ------------------------------------------------------------------
    nmax_small, nmax_large = int(scales[0]), int(scales[1])
    try:
        run1 = _run_trials_guarded(
            submission, nmax=nmax_small, ntrials=ntrials_repro, seed=seed + 11, secret_seed=secret_seed
        )
        run2 = _run_trials_guarded(
            submission, nmax=nmax_small, ntrials=ntrials_repro, seed=seed + 11, secret_seed=secret_seed
        )
        report.input_output_passed = True

        est_identical = bool(np.array_equal(run1["a_hat"], run2["a_hat"]))
        q_identical = bool(np.array_equal(run1["queries"], run2["queries"]))
        seq_identical = bool(np.array_equal(run1["seq_depths"], run2["seq_depths"]))
        hist_identical = run1["histories"] == run2["histories"]

        if est_identical and q_identical and seq_identical and hist_identical:
            report.reproducibility_passed = True
        else:
            max_diff = float(np.max(np.abs(run1["a_hat"] - run2["a_hat"])))
            q_diff = float(np.max(np.abs(run1["queries"] - run2["queries"])))
            report.violations.append(
                f"[Reproducibility] Non-deterministic execution detected: "
                f"max |a_hat_1 - a_hat_2| = {max_diff:.3e}, max |Q_1 - Q_2| = {q_diff:.1f}, "
                f"histories_equal={hist_identical}"
            )
    except (OracleInputError, OutputValidationError) as exc:
        report.input_output_passed = False
        report.violations.append(f"[Input/Output Validation] {exc}")
        return report
    except (OracleTamperError, SecurityViolationError) as exc:
        report.input_output_passed = False
        report.violations.append(f"[Security/Tamper] {exc}")
        return report
    except Exception as exc:
        report.input_output_passed = False
        report.violations.append(f"[Runtime Error] {type(exc).__name__}: {exc}")
        return report

    # ------------------------------------------------------------------
    # Stage 4: Oracle Tamper & Zero-Information Test
    # ------------------------------------------------------------------
    try:
        zi_run = _run_trials_guarded(
            submission,
            nmax=nmax_small,
            ntrials=ntrials_zero_info,
            seed=seed + 29,
            zero_info_mode=True,
            secret_seed=secret_seed,
        )
        zi_err = zi_run["errors"]
        zi_mae = float(np.mean(zi_err))
        zi_eps95 = float(np.percentile(zi_err, 95))

        a_t = zi_run["a_true"]
        a_h = zi_run["a_hat"]
        if np.std(a_h) > 1e-12 and np.std(a_t) > 1e-12:
            corr = float(np.corrcoef(a_t, a_h)[0, 1])
        else:
            corr = 0.0

        report.diagnostics["zero_info_mae"] = zi_mae
        report.diagnostics["zero_info_eps95"] = zi_eps95
        report.diagnostics["zero_info_corr"] = corr

        corr_thresh = 4.2 / math.sqrt(max(4, ntrials_zero_info - 1))
        if zi_mae < 0.12 or zi_eps95 < 0.22 or abs(corr) > corr_thresh:
            report.zero_info_passed = False
            report.violations.append(
                f"[Zero-Info Test] Side-channel leakage of a_true detected under constant-0.5 dummy oracle! "
                f"MAE={zi_mae:.4f} (min legal >= 0.12), eps95={zi_eps95:.4f} (min legal >= 0.22), "
                f"corr(a_hat, a_true)={corr:.4f} (threshold +- {corr_thresh:.4f})"
            )
        else:
            report.zero_info_passed = True
    except (OracleInputError, OutputValidationError, OracleTamperError, SecurityViolationError) as exc:
        report.zero_info_passed = False
        report.violations.append(f"[Zero-Info / Oracle Tamper] {exc}")
        return report
    except Exception as exc:
        report.zero_info_passed = False
        report.violations.append(f"[Zero-Info Runtime Error] {type(exc).__name__}: {exc}")
        return report

    # ------------------------------------------------------------------
    # Stage 5: Asymptotic Scaling Check across 2 scales
    # ------------------------------------------------------------------
    try:
        res_small = _run_trials_guarded(
            submission, nmax=nmax_small, ntrials=ntrials_scaling, seed=seed + nmax_small, secret_seed=secret_seed
        )
        res_large = _run_trials_guarded(
            submission, nmax=nmax_large, ntrials=ntrials_scaling, seed=seed + nmax_large, secret_seed=secret_seed
        )
        m_small = compute_metrics(
            res_small["a_true"], res_small["a_hat"], res_small["queries"],
            max_depths=res_small["max_depths"], seq_depths=res_small["seq_depths"], boot_B=0
        )
        m_large = compute_metrics(
            res_large["a_true"], res_large["a_hat"], res_large["queries"],
            max_depths=res_large["max_depths"], seq_depths=res_large["seq_depths"], boot_B=0
        )

        q1, q2 = m_small["q_mean"], m_large["q_mean"]
        eps1, eps2 = m_small["eps95"], m_large["eps95"]
        c95_1, c95_2 = m_small["C95_ave"], m_large["C95_ave"]
        c95_par_2 = eps2 * nmax_large
        eps_d0, c95_d0 = _depth0_classical_c95(q2)

        report.diagnostics["scaling"] = {
            "nmax_small": nmax_small,
            "nmax_large": nmax_large,
            "q_ave_small": q1,
            "q_ave_large": q2,
            "eps95_small": eps1,
            "eps95_large": eps2,
            "C95_small": c95_1,
            "C95_large": c95_2,
            "C95_par_large": c95_par_2,
            "depth0_eps95_at_q2": eps_d0,
            "depth0_C95_at_q2": c95_d0,
        }

        scaling_errors: List[str] = []
        if not (q2 > q1):
            scaling_errors.append(
                f"Query budget did not grow with scale: Q_ave({nmax_large})={q2:.1f} <= Q_ave({nmax_small})={q1:.1f}"
            )
        if not (eps2 < 0.70 * eps1):
            scaling_errors.append(
                f"eps95 did not decrease sufficiently with scale: "
                f"eps95({nmax_large})={eps2:.4e} vs eps95({nmax_small})={eps1:.4e}"
            )
        if not (c95_par_2 <= 8.0):
            scaling_errors.append(
                f"Failed quantum depth scaling at nmax={nmax_large}: "
                f"eps95 * nmax = {c95_par_2:.2f} > 8.0 (eps95={eps2:.4e} is too large for scale {nmax_large})"
            )
        # Relative criterion only: the submission must beat depth-0 classical sampling at
        # the same query budget by at least 2x. No absolute cap on C95, because published
        # sub-Heisenberg baselines run as specified (e.g. Suzuki et al.'s LIS, O(Q^-3/4))
        # legitimately have large constants at these scales while still being quantum.
        if not (c95_2 < 0.50 * c95_d0):
            scaling_errors.append(
                f"C95_ave({nmax_large})={c95_2:.3f} failed to beat trivial/depth-0 baseline "
                f"(depth-0 C95={c95_d0:.2f} at the same query budget; need < {0.5 * c95_d0:.2f})"
            )

        if scaling_errors:
            report.scaling_passed = False
            for err_msg in scaling_errors:
                report.violations.append(f"[Asymptotic Scaling] {err_msg}")
        else:
            report.scaling_passed = True

    except (OracleInputError, OutputValidationError, OracleTamperError, SecurityViolationError) as exc:
        report.scaling_passed = False
        report.violations.append(f"[Scaling Check Error] {exc}")
        return report
    except Exception as exc:
        report.scaling_passed = False
        report.violations.append(f"[Scaling Runtime Error] {type(exc).__name__}: {exc}")
        return report

    report.passed = (
        report.ast_passed
        and report.input_output_passed
        and report.reproducibility_passed
        and report.zero_info_passed
        and report.scaling_passed
    )
    return report


# ============================================================================
# 5. Full Statistical Benchmark Engine (`benchmark_submission`)
# ============================================================================

def _evaluate_submission_at_scale(
    submission: Any,
    nmax: int,
    ntrials: int,
    seed: int,
    *,
    a_range: Tuple[float, float] = (0.1, 0.9),
    boot_B: int = 1000,
    boot_seed: int = 1,
    secret_seed: Optional[str] = None,
) -> Dict[str, Any]:
    """Run guarded evaluation at a single scale `nmax` and compute full leaderboard metrics."""
    raw = _run_trials_guarded(
        submission, nmax=nmax, ntrials=ntrials, seed=seed, a_range=a_range, secret_seed=secret_seed
    )
    m = compute_metrics(
        raw["a_true"],
        raw["a_hat"],
        raw["queries"],
        max_depths=raw["max_depths"],
        seq_depths=raw["seq_depths"],
        boot_B=boot_B,
        seed=boot_seed,
    )
    q_mean = m["q_mean"]
    q_std = m["q_std"]
    m["nmax"] = int(nmax)
    m["a_range"] = tuple(a_range)
    m["query_spread"] = float(q_std / q_mean) if q_mean > 0 else 0.0
    m["rounds_mean"] = float(np.mean(raw["rounds"]))
    m["rounds_max"] = int(np.max(raw["rounds"]))
    m["rounds_min"] = int(np.min(raw["rounds"]))
    m["C95_par"] = float(m["eps95"] * m["max_depth_mean"])
    m["runtime_total_sec"] = float(raw["elapsed_sec"])
    m["runtime_per_trial_sec"] = float(raw["runtime_per_trial_sec"])
    m["runtime_per_trial_ms"] = 1000.0 * elapsed / max(1, ntrials) if (elapsed := raw["elapsed_sec"]) >= 0 else 0.0
    return m


def _aggregate_scale_metrics(per_scale: Dict[int, Dict[str, Any]]) -> Dict[str, float]:
    """Compute scale-averaged summary metrics across all evaluated scales."""
    scales_list = sorted(per_scale.keys())
    keys = [
        "C68_ave",
        "C95_ave",
        "C95_max",
        "C99_ave",
        "C99_max",
        "C95_seq",
        "C95_par",
        "rounds_mean",
        "query_spread",
        "runtime_per_trial_sec",
        "runtime_per_trial_ms",
    ]
    summary: Dict[str, float] = {}
    for k in keys:
        vals = [per_scale[s][k] for s in scales_list if k in per_scale[s]]
        if vals:
            summary[k] = float(np.mean(vals))
    return summary


def benchmark_submission(
    submission: Any,
    *,
    name: str = "Submission",
    scales: Sequence[int] = SCALES,
    ntrials: int = 2000,
    ntrials_wide_prior: Optional[int] = None,
    seed_band: int = HOLDOUT_SEED,
    boot_B: int = 1000,
    verify_first: bool = True,
    source_code: Optional[str] = None,
    submission_path: Optional[str] = None,
    secret_seed: Optional[str] = None,
) -> Dict[str, Any]:
    """Run full statistical evaluation across standard scales {125, 260, 540, 1000}."""
    active_secret = secret_seed if secret_seed is not None else scrub_secret_seed_env()

    verification: Optional[VerificationReport] = None
    if verify_first:
        v_scales = (min(scales), max(scales)) if len(scales) >= 2 else (125, 540)
        verification = verify_submission(
            submission,
            name=name,
            source_code=source_code,
            submission_path=submission_path,
            scales=v_scales,
            secret_seed=active_secret,
        )
        if not verification.passed:
            raise SecurityViolationError(
                f"Submission '{name}' failed anti-cheat verification:\n{verification.summary()}"
            )

    if ntrials_wide_prior is None:
        ntrials_wide_prior = max(500, ntrials // 2)

    standard_by_scale: Dict[int, Dict[str, Any]] = {}
    wide_prior_by_scale: Dict[int, Dict[str, Any]] = {}

    for nmax in scales:
        nmax_int = int(nmax)
        standard_by_scale[nmax_int] = _evaluate_submission_at_scale(
            submission,
            nmax=nmax_int,
            ntrials=ntrials,
            seed=seed_band + nmax_int,
            a_range=(0.1, 0.9),
            boot_B=boot_B,
            secret_seed=active_secret,
        )
        wide_prior_by_scale[nmax_int] = _evaluate_submission_at_scale(
            submission,
            nmax=nmax_int,
            ntrials=ntrials_wide_prior,
            seed=seed_band + 50_000 + nmax_int,
            a_range=(0.01, 0.99),
            boot_B=max(200, boot_B // 2),
            secret_seed=active_secret,
        )

    return {
        "name": name,
        "seed_band": seed_band,
        "blind_secret_active": bool(active_secret),
        "scales": tuple(int(s) for s in scales),
        "ntrials": ntrials,
        "ntrials_wide_prior": ntrials_wide_prior,
        "verification": verification,
        "standard_prior": {
            "a_range": (0.1, 0.9),
            "per_scale": standard_by_scale,
            "summary": _aggregate_scale_metrics(standard_by_scale),
        },
        "wide_prior": {
            "a_range": (0.01, 0.99),
            "per_scale": wide_prior_by_scale,
            "summary": _aggregate_scale_metrics(wide_prior_by_scale),
        },
    }


def format_benchmark_report(bench: Dict[str, Any]) -> str:
    """Format the output of `benchmark_submission` as a publication-quality table."""
    lines: List[str] = []
    name = bench["name"]
    blind_tag = " + BLIND_HMAC_SECRET" if bench.get("blind_secret_active") else ""
    lines.append("=" * 124)
    lines.append(
        f"LEADERBOARD BENCHMARK REPORT: {name} "
        f"(SEED_BAND={bench['seed_band']}{blind_tag}, ntrials={bench['ntrials']})"
    )
    lines.append("=" * 124)

    for section_key, label in [
        ("standard_prior", "Standard Prior a ~ U(0.10, 0.90)"),
        ("wide_prior", "Prior Robustness a ~ U(0.01, 0.99)"),
    ]:
        sec = bench[section_key]
        lines.append(f"\n--- {label} ---")
        header = (
            f"{'nmax':>5} | {'Q_ave':>7} | {'Q_std/Q':>7} | {'Rounds':>6} | "
            f"{'C68_ave':>7} | {'C95_ave [95% CI]':>21} | {'C95_max':>7} | "
            f"{'C99_ave [95% CI]':>21} | {'C99_max':>7} | {'C95_seq':>7} | {'ms/trial':>8}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for nmax in bench["scales"]:
            m = sec["per_scale"][nmax]
            ci95 = m.get("C95_ave_ci", (float("nan"), float("nan")))
            ci99 = m.get("C99_ave_ci", (float("nan"), float("nan")))
            lines.append(
                f"{nmax:5d} | {m['q_mean']:7.1f} | {m['query_spread']:7.4f} | {m['rounds_mean']:6.2f} | "
                f"{m['C68_ave']:7.3f} | {m['C95_ave']:6.3f} [{ci95[0]:5.2f}, {ci95[1]:5.2f}] | {m['C95_max']:7.2f} | "
                f"{m['C99_ave']:6.2f} [{ci99[0]:5.2f}, {ci99[1]:5.2f}] | {m['C99_max']:7.2f} | "
                f"{m['C95_seq']:7.3f} | {m['runtime_per_trial_ms']:8.3f}"
            )
        s = sec["summary"]
        lines.append("-" * len(header))
        lines.append(
            f"{'MEAN':>5} | {'-':>7} | {s['query_spread']:7.4f} | {s['rounds_mean']:6.2f} | "
            f"{s['C68_ave']:7.3f} | {s['C95_ave']:6.3f} {'':>14} | {s['C95_max']:7.2f} | "
            f"{s['C99_ave']:6.2f} {'':>14} | {s['C99_max']:7.2f} | "
            f"{s['C95_seq']:7.3f} | {s['runtime_per_trial_ms']:8.3f}"
        )
    lines.append("=" * 124)
    return "\n".join(lines)


# ============================================================================
# 6. Reference Literature Baselines (Non-Adaptive & Adaptive)
# ============================================================================

_MLE_TABLE_CACHE: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]] = {}
_ZOOM_OFFS_25 = np.linspace(-0.5, 0.5, 25, dtype=np.float64)


def _get_cached_mle_table(depths: Sequence[int], shots: Sequence[int]):
    """Cache coarse grid log-probability tables per (depths, shots) schedule."""
    key = (tuple(int(d) for d in depths), tuple(int(s) for s in shots))
    cached = _MLE_TABLE_CACHE.get(key)
    if cached is not None:
        return cached
    d_arr = np.asarray(key[0], dtype=np.float64)
    s_arr = np.asarray(key[1], dtype=np.float64)
    d_odd = 2.0 * d_arr + 1.0
    step = grid_spacing(d_arr, s_arr)
    grid = np.arange(1e-6, 0.5 * math.pi, step, dtype=np.float64)
    p = np.clip(np.cos(np.outer(grid, d_odd)) ** 2, 1e-12, 1.0 - 1e-12)
    log_p = np.log(p)
    log_1mp = np.log1p(-p)
    out = (grid, step, d_odd, log_p, log_1mp)
    _MLE_TABLE_CACHE[key] = out
    return out


def _local_refine_single_trial(th0: float, step: float, d_odd: np.ndarray, k_arr: np.ndarray, s_arr: np.ndarray) -> float:
    """Exact 25-point zoom + quadratic interpolation around coarse peak `th0`."""
    th_loc = th0 + step * _ZOOM_OFFS_25
    p = np.clip(np.cos(np.outer(th_loc, d_odd)) ** 2, 1e-12, 1.0 - 1e-12)
    nll = -(np.log(p) @ k_arr + np.log1p(-p) @ (s_arr - k_arr))
    j = int(np.argmin(nll))
    if j <= 0:
        j = 1
    elif j >= 24:
        j = 23
    ym1, y0, yp1 = float(nll[j - 1]), float(nll[j]), float(nll[j + 1])
    den = ym1 - 2.0 * y0 + yp1
    off = 0.0
    if abs(den) > 1e-12:
        off = 0.5 * (ym1 - yp1) / max(den, 1e-12)
        off = min(max(off, -0.5), 0.5)
    dstep = step / 24.0
    th_refined = float(th_loc[j]) + off * dstep
    return min(max(th_refined, 0.0), 0.5 * math.pi)


class MLQAEFlagshipSubmission:
    """Reference non-adaptive MLQAE flagship baseline (geometric ladder + cap rung, Labib et al.)."""

    def __init__(self, r: float = 1.45, cap_divisor: Optional[float] = 1.3):
        self.r = r
        self.cap_divisor = cap_divisor

    def estimate(self, oracle: GuardedOracle, nmax: int) -> float:
        depths, shots = flagship_schedule(int(nmax), r=self.r, cap_divisor=self.cap_divisor)
        counts = oracle.measure_batch(depths, shots)
        grid, step, d_odd, log_p, log_1mp = _get_cached_mle_table(depths, shots)
        k_arr = counts.astype(np.float64)
        s_arr = np.asarray(shots, dtype=np.float64)
        nll = -(log_p @ k_arr + log_1mp @ (s_arr - k_arr))
        idx = int(np.clip(np.argmin(nll), 1, len(grid) - 2))
        th_hat = _local_refine_single_trial(float(grid[idx]), step, d_odd, k_arr, s_arr)
        return float(math.sin(th_hat))


class MLQAEPlainSubmission(MLQAEFlagshipSubmission):
    """Reference non-adaptive plain geometric ladder MLQAE baseline."""

    def __init__(self, r: float = 1.45):
        super().__init__(r=r, cap_divisor=None)


_ADAPT_GRID_CACHE: Dict[int, Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]] = {}


def _get_adaptive_full_table(nmax: int):
    """Precompute dense grid and all integer depths 0..nmax for Bayesian MAP estimation."""
    cached = _ADAPT_GRID_CACHE.get(int(nmax))
    if cached is not None:
        return cached
    lad, shots = flagship_schedule(int(nmax), r=1.45, cap_divisor=1.3)
    step = grid_spacing(lad, shots)
    grid = np.arange(1e-6, 0.5 * math.pi, step, dtype=np.float64)
    depths_all = np.arange(int(nmax) + 1, dtype=np.float64)
    d_all = 2.0 * depths_all + 1.0
    p_all = np.clip(np.cos(np.outer(d_all, grid)) ** 2, 1e-12, 1.0 - 1e-12)
    log_p_all = np.log(p_all)
    log_1mp_all = np.log1p(-p_all)
    out = (grid, step, d_all, log_p_all, log_1mp_all)
    _ADAPT_GRID_CACHE[int(nmax)] = out
    return out


class ReferenceBayesianAdaptiveSubmission:
    """Reference multi-round Bayesian adaptive QAE baseline (cf. Ramôa & Santos 2024 / Grinko et al. 2021).

    Stage 1 executes a coarse geometric ladder to form a Bayesian posterior distribution
    over theta, and Stage 2 selects supplementary depths where the posterior variance is
    highest before computing the global MAP / ML estimate.
    """

    def __init__(self, r: float = 1.45):
        self.r = r

    def estimate(self, oracle: GuardedOracle, nmax: int) -> float:
        nmax_int = int(nmax)
        base_lad = geom_ladder(nmax_int, self.r)
        base_shots = canonical_shots(len(base_lad), base=1, slope=0.92)

        split_idx = max(2, len(base_lad) - 2)
        r1_depths = base_lad[:split_idx]
        r1_shots = base_shots[:split_idx]
        r1_counts = oracle.measure_batch(r1_depths, r1_shots)

        grid, step, _, log_p_all, log_1mp_all = _get_adaptive_full_table(nmax_int)
        k1 = r1_counts.astype(np.float64)
        s1 = np.asarray(r1_shots, dtype=np.float64)
        f1 = np.asarray(r1_depths, dtype=np.int64)

        ll_1 = k1 @ log_p_all[f1] + (s1 - k1) @ log_1mp_all[f1]
        ib1 = int(np.clip(np.argmax(ll_1), 1, len(grid) - 2))
        th1 = float(grid[ib1])

        r2_depths: List[int] = []
        r2_shots: List[int] = []
        for nominal_d, nom_s in zip(base_lad[split_idx:], base_shots[split_idx:]):
            w_lo = max(r1_depths[-1] + 1, int(round(0.82 * nominal_d)))
            w_hi = min(nmax_int, nominal_d)
            best_d = nominal_d
            best_info = -1e9
            for cand_d in range(w_lo, w_hi + 1):
                d_odd = 2.0 * cand_d + 1.0
                info = abs(math.sin(2.0 * d_odd * th1))
                if info > best_info:
                    best_info = info
                    best_d = cand_d
            r2_depths.append(best_d)
            r2_shots.append(nom_s)

        extra_d = int(round(nmax_int / 1.3))
        if extra_d > 0 and extra_d not in r1_depths and extra_d not in r2_depths:
            r2_depths.append(extra_d)
            r2_shots.append(1)

        r2_counts = oracle.measure_batch(r2_depths, r2_shots)

        k2 = r2_counts.astype(np.float64)
        s2 = np.asarray(r2_shots, dtype=np.float64)
        f2 = np.asarray(r2_depths, dtype=np.int64)
        ll_tot = ll_1 + (k2 @ log_p_all[f2] + (s2 - k2) @ log_1mp_all[f2])

        ib_final = int(np.clip(np.argmax(ll_tot), 1, len(grid) - 2))
        all_d_odd = np.concatenate([2.0 * f1 + 1.0, 2.0 * f2 + 1.0])
        all_k = np.concatenate([k1, k2])
        all_s = np.concatenate([s1, s2])
        th_hat = _local_refine_single_trial(float(grid[ib_final]), step, all_d_odd, all_k, all_s)
        return float(math.sin(th_hat))


AdaptiveTwoRoundSubmission = ReferenceBayesianAdaptiveSubmission


def _load_submission_from_dir(sub_dir: str) -> Tuple[Any, str, str]:
    """Load a submission object and its source file path from a submission directory or file."""
    import importlib.util
    import json

    # Always scrub any secret seed env vars prior to importing untrusted submission code
    scrub_secret_seed_env()

    abs_target = os.path.abspath(sub_dir)
    if os.path.isfile(abs_target):
        py_path = abs_target
        folder = os.path.dirname(abs_target)
        algo_name = os.path.splitext(os.path.basename(abs_target))[0]
        entry_cls_name = None
    else:
        folder = abs_target
        meta_path = os.path.join(folder, "metadata.json")
        entry_spec = "submission.py"
        algo_name = os.path.basename(folder)
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            algo_name = meta.get("algorithm_name") or meta.get("name") or meta.get("algo_id") or algo_name
            entry_spec = meta.get("entrypoint", "submission.py")
        if ":" in entry_spec:
            file_part, entry_cls_name = entry_spec.split(":", 1)
        else:
            file_part, entry_cls_name = entry_spec, None
        if not file_part.endswith(".py"):
            file_part += ".py"
        py_path = os.path.join(folder, file_part)

    spec = importlib.util.spec_from_file_location("submitted_module", py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module specification from {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if entry_cls_name and hasattr(mod, entry_cls_name):
        obj = getattr(mod, entry_cls_name)
        return (obj() if _host_inspect.isclass(obj) else obj), algo_name, py_path

    for cand_attr in ("Submission", "submission", "estimate", "run_trial"):
        if hasattr(mod, cand_attr):
            obj = getattr(mod, cand_attr)
            return (obj() if _host_inspect.isclass(obj) else obj), algo_name, py_path

    for _, obj in _host_inspect.getmembers(mod, _host_inspect.isclass):
        if obj.__module__ == mod.__name__ and (hasattr(obj, "run_trial") or hasattr(obj, "estimate") or hasattr(obj, "get_schedule")):
            return obj(), algo_name, py_path

    raise RuntimeError(f"No valid submission entrypoint found in {py_path}")


def _cli_main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="CI Verifier & Statistical Benchmark Runner for AE Leaderboard")
    parser.add_argument("--mode", choices=["demo", "verify", "certify"], default="demo")
    parser.add_argument("--submission", type=str, default=None, help="Path to submission directory or Python file")
    parser.add_argument("--ntrials", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=HOLDOUT_SEED)
    parser.add_argument("--salted-seed", type=int, default=None)
    parser.add_argument(
        "--secret-seed-env",
        type=str,
        default="LEADERBOARD_SECRET_SEED",
        help="Environment variable name holding the blind secret seed (popped & scrubbed before submission import)",
    )
    parser.add_argument("--json-out", type=str, default=None)
    parser.add_argument("--md-out", type=str, default=None)
    args = parser.parse_args()

    # Pop and scrub the secret seed from os.environ BEFORE importing any submission code
    active_secret = scrub_secret_seed_env(args.secret_seed_env)

    if args.mode == "demo" and args.submission is None:
        print("Running CI Verifier & Benchmark Demo on Reference Literature Baselines...\n")
        for label, sub in [
            ("MLQAE-Flagship (1R Non-Adaptive)", MLQAEFlagshipSubmission()),
            ("Reference-Bayesian-Adaptive (2R)", ReferenceBayesianAdaptiveSubmission()),
        ]:
            rep = verify_submission(sub, name=label, secret_seed=active_secret)
            print(rep.summary())
            bench = benchmark_submission(
                sub, name=label, ntrials=args.ntrials, boot_B=400, verify_first=False, secret_seed=active_secret
            )
            print(format_benchmark_report(bench))
            print()
        return

    if args.submission is not None:
        sub_obj, algo_name, py_path = _load_submission_from_dir(args.submission)
    else:
        sub_obj, algo_name, py_path = ReferenceBayesianAdaptiveSubmission(), "Reference-Bayesian-Adaptive", None

    rep = verify_submission(
        sub_obj,
        name=algo_name,
        submission_path=py_path,
        allowed_dir=os.path.dirname(py_path) if py_path else _HERE,
        secret_seed=active_secret,
    )
    print(rep.summary())
    if not rep.passed:
        if args.md_out:
            with open(args.md_out, "w", encoding="utf-8") as fh:
                fh.write(f"```\n{rep.summary()}\n```\n")
        sys.exit(1)

    bench = benchmark_submission(
        sub_obj,
        name=algo_name,
        ntrials=args.ntrials,
        seed_band=args.seed,
        verify_first=False,
        submission_path=py_path,
        secret_seed=active_secret,
    )
    md_table = format_benchmark_report(bench)
    print(md_table)

    if args.salted_seed is not None:
        salted_bench = benchmark_submission(
            sub_obj,
            name=f"{algo_name} (Salted Check)",
            ntrials=max(500, args.ntrials // 4),
            seed_band=args.salted_seed,
            verify_first=False,
            submission_path=py_path,
            secret_seed=active_secret,
        )
        c95_holdout = bench["standard_prior"]["summary"]["C95_ave"]
        c95_salted = salted_bench["standard_prior"]["summary"]["C95_ave"]
        if c95_salted > 1.35 * c95_holdout + 0.5:
            print(
                f"ERROR: Salted seed C95_ave ({c95_salted:.3f}) diverged from HOLDOUT_SEED ({c95_holdout:.3f}); "
                "possible seed-table memorization!",
                file=sys.stderr,
            )
            sys.exit(2)

    if args.md_out:
        with open(args.md_out, "w", encoding="utf-8") as fh:
            fh.write(f"```\n{rep.summary()}\n\n{md_table}\n```\n")

    if args.json_out:
        import json as _json

        std_sum = bench["standard_prior"]["summary"]
        wide_sum = bench["wide_prior"]["summary"]
        legacy = {
            "name": algo_name,
            "verified": rep.passed,
            "seed_band": args.seed,
            "blind_secret_active": bool(active_secret),
            "ntrials": args.ntrials,
            "headline_C68_ave": std_sum["C68_ave"],
            "headline_C95_ave": std_sum["C95_ave"],
            "headline_C95_max": std_sum["C95_max"],
            "headline_C99_ave": std_sum["C99_ave"],
            "headline_C99_max": std_sum["C99_max"],
            "headline_C95_seq": std_sum["C95_seq"],
            "rounds_mean": std_sum["rounds_mean"],
            "query_spread": std_sum["query_spread"],
            "runtime_per_trial_ms": std_sum["runtime_per_trial_ms"],
            "wide_prior_C95_ave": wide_sum["C95_ave"],
            "wide_prior_C99_ave": wide_sum["C99_ave"],
        }
        # A submission directory with metadata.json gets a FULL leaderboard entry (per-scale
        # metrics, track, authors, source="challenge") in the same schema as the literature
        # baselines, so the page can render it straight from leaderboard.json.
        payload: Dict[str, Any] = dict(legacy)
        build_error: Optional[str] = None
        meta_path = os.path.join(os.path.dirname(py_path), "metadata.json") if py_path else None
        if meta_path and os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = _json.load(fh)
            algo_id = os.path.basename(os.path.dirname(py_path))
            try:
                from leaderboard.entry_builder import entry_from_benchmark

                full_entry = entry_from_benchmark(bench, meta=meta, algo_id=algo_id, verified=rep.passed)
                payload = {**full_entry, **legacy, "name": full_entry["name"], "algo_id": algo_id}
            except Exception as exc:  # noqa: BLE001 - surfaced below as a hard CI error
                build_error = f"{type(exc).__name__}: {exc}"
        with open(args.json_out, "w", encoding="utf-8") as fh:
            _json.dump(payload, fh, indent=2)
        if build_error is not None:
            print(f"::error::Could not build a full leaderboard entry from the benchmark: {build_error}", file=sys.stderr)
            sys.exit(3)


if __name__ == "__main__":
    _cli_main()
