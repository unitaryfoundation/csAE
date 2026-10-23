"""Submission API, Oracle Sandbox, and JSON Schema Validator for the Quantum AE Leaderboard.

This module provides:
1. ``QuantumAEOracle``: A tamper-proof sandboxed oracle that wraps a hidden amplitude
   ``a_true`` (and ``theta_true = arcsin(a_true)``) inside a private closure scope so
   that ``dir(oracle)``, ``oracle.__dict__``, ``vars(oracle)``, and ``copy``/``pickle``
   cannot leak the secret amplitude or reset query counters.
2. Submission Base Classes & Protocols:
   - ``NonAdaptiveAESubmission``: Defines ``get_schedule(scale_nmax)`` and vectorized
     ``estimate(depths, shots, counts)``.
   - ``AdaptiveAESubmission``: Defines ``run_trial(oracle, target_scale_or_budget, rng)``
     and optional batch hook ``run_batch(oracles, target_scale_or_budget, rngs)``.
3. Strict Query & Depth Accounting:
   - ``shot_cost(n)``: 1 query at Grover depth ``n = 0`` (state preparation $U|0\\rangle$),
     and ``n`` queries at depth ``n >= 1`` (matching ``mlqae/core.py`` and
     ``leaderboard/harness.py``).
   - ``oracle.measure(depth, shots=1)``: Sequential adaptive measurement.
   - ``oracle.measure_batch(depths, shots)``: Parallel multi-circuit round of depth
     ``max(depths)``, incrementing ``seq_depth += max(depths)``,
     ``max_depth = max(max_depth, max(depths))``,
     ``total_queries += sum(shot_cost(d) * s)``, and ``num_rounds += 1``.
4. JSON Schema definitions and validators for ``metadata.json`` and ``leaderboard.json``.
"""

from __future__ import annotations

import abc
import datetime as _dt
import json
import os
import re
import sys
import weakref
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mlqae.core import (  # noqa: E402
    KAPPA,
    boot_ci_constant,
    canonical_shots,
    crlb_constant,
    flagship_schedule,
    geom_ladder,
    grid_spacing,
)

__all__ = [
    # Constants
    "SCHEMA_VERSION",
    "VALID_TRACKS",
    "STANDARD_SCALES",
    "DEFAULT_A_RANGE",
    "SEARCH_SEED",
    "HOLDOUT_SEED",
    # Exceptions
    "OracleSecurityError",
    "BudgetExceededError",
    "DepthCapExceededError",
    "SchemaValidationError",
    # Accounting helpers
    "shot_cost",
    "schedule_query_cost",
    # Sandbox Oracle
    "QuantumAEOracle",
    # Submission APIs
    "NonAdaptiveAESubmission",
    "AdaptiveAESubmission",
    "ml_estimate_batch",
    "MLQAEFlagshipSubmission",
    "MLQAEPlainSubmission",
    "ReferenceBayesianIntervalSubmission",
    "ReferenceTwoRoundAdaptiveSubmission",
    # Evaluation helpers
    "evaluate_non_adaptive_submission",
    "evaluate_adaptive_submission",
    # JSON Schema & Validation
    "METADATA_JSON_SCHEMA",
    "LEADERBOARD_ENTRY_SCHEMA",
    "LEADERBOARD_JSON_SCHEMA",
    "normalize_metadata",
    "validate_metadata",
    "validate_leaderboard_entry",
    "validate_leaderboard",
    "create_leaderboard_entry",
    "build_leaderboard_payload",
]

# ---------------------------------------------------------------------------
# Constants & Standard Benchmark Parameters
# ---------------------------------------------------------------------------

SCHEMA_VERSION: str = "1.0.0"

VALID_TRACKS: tuple[str, ...] = (
    "non-adaptive",
    "semi-adaptive",
    "adaptive-fixed-budget",
    "adaptive-variable",
    "depth-limited",
)

STANDARD_SCALES: tuple[int, ...] = (60, 125, 260, 540)
DEFAULT_A_RANGE: tuple[float, float] = (0.1, 0.9)
SEARCH_SEED: int = 100_000
HOLDOUT_SEED: int = 900_000


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class OracleSecurityError(PermissionError):
    """Raised when code attempts to mutate, copy, or serialize a QuantumAEOracle."""


class BudgetExceededError(RuntimeError, ValueError):
    """Raised when an adaptive measurement exceeds the oracle's max_query_budget."""


class DepthCapExceededError(RuntimeError, ValueError):
    """Raised when a measurement depth exceeds the oracle's max_depth_cap."""


class SchemaValidationError(ValueError):
    """Raised when a metadata.json or leaderboard.json payload fails validation."""


# ---------------------------------------------------------------------------
# Query Accounting Helpers
# ---------------------------------------------------------------------------


def shot_cost(depth: int) -> int:
    """Return the query cost of 1 shot at Grover depth ``depth``.

    Matches ``mlqae/core.py`` and ``leaderboard/harness.py``:
    - At depth ``n = 0``, the circuit $U|0\\rangle$ uses 1 state-preparation query.
    - At depth ``n >= 1``, the circuit $G^n U|0\\rangle$ costs ``n`` Grover queries.
    """
    if isinstance(depth, bool) or not isinstance(depth, (int, np.integer)):
        raise TypeError(f"depth must be an integer, got {type(depth).__name__}")
    d = int(depth)
    if d < 0:
        raise ValueError(f"depth must be >= 0, got {d}")
    return 1 if d == 0 else d


def schedule_query_cost(depths: Sequence[int], shots: Sequence[int]) -> int:
    """Compute total query cost $\\sum_k \\text{shot\\_cost}(d_k) \\cdot s_k$ for a schedule."""
    d_arr = np.asarray(depths, dtype=np.int64)
    s_arr = np.asarray(shots, dtype=np.int64)
    if d_arr.ndim != 1 or s_arr.ndim != 1 or len(d_arr) != len(s_arr) or len(d_arr) == 0:
        raise ValueError("depths and shots must be non-empty 1D sequences of equal length.")
    if np.any(d_arr < 0):
        raise ValueError("All Grover depths must be >= 0.")
    if np.any(s_arr < 0):
        raise ValueError("All shot counts must be >= 0.")
    per_shot = np.where(d_arr == 0, 1, d_arr)
    return int(np.sum(per_shot * s_arr))


# ---------------------------------------------------------------------------
# Sandboxed Quantum AE Oracle (Closure-Isolated)
# ---------------------------------------------------------------------------


def _build_oracle_class() -> type:
    """Construct ``QuantumAEOracle`` inside a private factory closure.

    Storing the weak-key dispatch table inside this closure guarantees that:
    1. ``oracle.__dict__`` is completely empty (``{}``).
    2. ``dir(oracle)`` exposes only public measurement methods and read-only
       accounting properties.
    3. Neither ``oracle`` nor module globals hold attributes named ``a_true``,
       ``theta_true``, or ``_vault``.
    """
    _vault: weakref.WeakKeyDictionary[Any, Callable[..., Any]] = weakref.WeakKeyDictionary()

    def _get_dispatch(instance: Any) -> Callable[..., Any]:
        try:
            return _vault[instance]
        except KeyError as exc:
            raise OracleSecurityError("Uninitialized or Forged QuantumAEOracle instance.") from exc

    def _read_only_prop(key: str, doc: str) -> property:
        def _getter(self: Any) -> Any:
            return _get_dispatch(self)(key)

        def _setter(self: Any, _value: Any) -> None:
            raise OracleSecurityError(f"Cannot modify read-only oracle property '{key}'.")

        def _deleter(self: Any) -> None:
            raise OracleSecurityError(f"Cannot delete read-only oracle property '{key}'.")

        return property(_getter, _setter, _deleter, doc=doc)

    class QuantumAEOracle:
        """Tamper-proof sandboxed oracle for Quantum Amplitude Estimation.

        Measurement model (matching ``mlqae/core.py``):
        At Grover depth $n \\ge 0$, the circuit $G^n U|0\\rangle$ measured in the Z basis
        yields Bernoulli outcomes with probability:
        $$p(n, \\theta) = V_n \\cos^2((2n + 1)\\theta) + \\frac{1 - V_n}{2}, \\quad
        V_n = (1 - \\eta)^{2n+1}, \\quad a = \\sin(\\theta).$$

        Security guarantees:
        - ``a_true`` and ``theta_true`` are captured exclusively inside a local closure
          created during ``__init__``.
        - ``oracle.__dict__`` is ``{}`` and ``dir(oracle)`` lists only public API members.
        - Attribute mutation, shallow/deep copying, and pickling raise ``OracleSecurityError``.
        """

        def __init__(
            self,
            a_true: float,
            rng: np.random.Generator | int | None = None,
            *,
            seed: int | None = None,
            eta: float = 0.0,
            max_query_budget: int | None = None,
            max_depth_cap: int | None = None,
            record_history: bool = False,
        ) -> None:
            a_val = float(a_true)
            if not (0.0 <= a_val <= 1.0) or np.isnan(a_val):
                raise ValueError(f"a_true must be in [0.0, 1.0], got {a_true}")
            theta_val = float(np.arcsin(a_val))

            eta_val = float(eta)
            if not (0.0 <= eta_val <= 1.0) or np.isnan(eta_val):
                raise ValueError(f"eta must be in [0.0, 1.0], got {eta}")

            if rng is not None and seed is not None:
                raise ValueError("Specify at most one of 'rng' or 'seed'.")
            if isinstance(rng, np.random.Generator):
                priv_rng = rng
            elif isinstance(rng, (int, np.integer)) and not isinstance(rng, bool):
                priv_rng = np.random.default_rng(int(rng))
            elif rng is None:
                priv_rng = np.random.default_rng(seed)
            else:
                raise TypeError(
                    f"rng must be a numpy.random.Generator, int seed, or None; got {type(rng).__name__}"
                )

            budget_cap = None if max_query_budget is None else int(max_query_budget)
            if budget_cap is not None and budget_cap < 1:
                raise ValueError(f"max_query_budget must be >= 1, got {budget_cap}")

            depth_cap = None if max_depth_cap is None else int(max_depth_cap)
            if depth_cap is not None and depth_cap < 0:
                raise ValueError(f"max_depth_cap must be >= 0, got {depth_cap}")

            keep_history = bool(record_history)

            # Mutable accounting state kept strictly inside the closure
            total_queries: int = 0
            total_shots: int = 0
            max_depth: int = 0
            seq_depth: int = 0
            num_rounds: int = 0
            history: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = []

            # Erase references to input arguments in locals so only primitive closure vars remain
            del a_true, rng, seed

            def _measure_single(depth: int, shots: int = 1) -> int:
                nonlocal total_queries, total_shots, max_depth, seq_depth, num_rounds

                if isinstance(depth, bool) or not isinstance(depth, (int, np.integer)):
                    raise TypeError(f"depth must be an int, got {type(depth).__name__}")
                if isinstance(shots, bool) or not isinstance(shots, (int, np.integer)):
                    raise TypeError(f"shots must be an int, got {type(shots).__name__}")

                d_int = int(depth)
                s_int = int(shots)
                if d_int < 0:
                    raise ValueError(f"depth must be >= 0, got {d_int}")
                if s_int < 1:
                    raise ValueError(f"shots must be >= 1, got {s_int}")

                if depth_cap is not None and d_int > depth_cap:
                    raise DepthCapExceededError(
                        f"Requested depth {d_int} exceeds max_depth_cap={depth_cap}."
                    )

                cost = (1 if d_int == 0 else d_int) * s_int
                if budget_cap is not None and (total_queries + cost) > budget_cap:
                    raise BudgetExceededError(
                        f"Query budget exceeded: requested {cost} queries at depth={d_int}, "
                        f"shots={s_int}, but only {budget_cap - total_queries} of {budget_cap} remain."
                    )

                d_odd = 2.0 * d_int + 1.0
                c2 = float(np.cos(d_odd * theta_val) ** 2)
                if eta_val > 0.0:
                    vis = (1.0 - eta_val) ** d_odd
                    prob = vis * c2 + 0.5 * (1.0 - vis)
                else:
                    prob = c2
                prob = float(np.clip(prob, 0.0, 1.0))

                count = int(priv_rng.binomial(s_int, prob))

                total_queries += cost
                total_shots += s_int
                if d_int > max_depth:
                    max_depth = d_int
                seq_depth += d_int
                num_rounds += 1

                if keep_history:
                    history.append(((d_int,), (s_int,), (count,)))

                return count

            def _measure_batch(depths: Sequence[int], shots: Sequence[int]) -> np.ndarray:
                nonlocal total_queries, total_shots, max_depth, seq_depth, num_rounds

                d_arr = np.asarray(depths)
                s_arr = np.asarray(shots)
                if d_arr.ndim != 1 or s_arr.ndim != 1:
                    raise ValueError("depths and shots must be 1D sequences.")
                if len(d_arr) != len(s_arr) or len(d_arr) == 0:
                    raise ValueError(
                        "depths and shots must be non-empty sequences of the same length."
                    )
                if d_arr.dtype == bool or not np.issubdtype(d_arr.dtype, np.integer):
                    raise TypeError("depths must contain integers.")
                if s_arr.dtype == bool or not np.issubdtype(s_arr.dtype, np.integer):
                    raise TypeError("shots must contain integers.")

                d_int_arr = d_arr.astype(np.int64, copy=False)
                s_int_arr = s_arr.astype(np.int64, copy=False)

                if np.any(d_int_arr < 0):
                    raise ValueError("All depths in measure_batch must be >= 0.")
                if np.any(s_int_arr < 1):
                    raise ValueError("All shot counts in measure_batch must be >= 1.")

                round_max_depth = int(np.max(d_int_arr))
                if depth_cap is not None and round_max_depth > depth_cap:
                    raise DepthCapExceededError(
                        f"Batch max depth {round_max_depth} exceeds max_depth_cap={depth_cap}."
                    )

                per_shot_cost = np.where(d_int_arr == 0, 1, d_int_arr)
                batch_cost = int(np.sum(per_shot_cost * s_int_arr))
                if budget_cap is not None and (total_queries + batch_cost) > budget_cap:
                    raise BudgetExceededError(
                        f"Query budget exceeded: batch requires {batch_cost} queries, "
                        f"but only {budget_cap - total_queries} of {budget_cap} remain."
                    )

                d_odd = 2.0 * d_int_arr.astype(np.float64) + 1.0
                c2 = np.cos(d_odd * theta_val) ** 2
                if eta_val > 0.0:
                    vis = (1.0 - eta_val) ** d_odd
                    probs = vis * c2 + 0.5 * (1.0 - vis)
                else:
                    probs = c2
                probs = np.clip(probs, 0.0, 1.0)

                counts = priv_rng.binomial(s_int_arr, probs).astype(np.int64)

                total_queries += batch_cost
                total_shots += int(np.sum(s_int_arr))
                if round_max_depth > max_depth:
                    max_depth = round_max_depth
                seq_depth += round_max_depth
                num_rounds += 1

                if keep_history:
                    history.append(
                        (
                            tuple(int(x) for x in d_int_arr),
                            tuple(int(x) for x in s_int_arr),
                            tuple(int(x) for x in counts),
                        )
                    )

                return counts

            def _dispatch(op: str, *args: Any) -> Any:
                if op == "measure":
                    return _measure_single(*args)
                if op == "measure_batch":
                    return _measure_batch(*args)
                if op in ("total_queries", "queries_used"):
                    return total_queries
                if op == "total_shots":
                    return total_shots
                if op in ("max_depth", "max_depth_used"):
                    return max_depth
                if op == "seq_depth":
                    return seq_depth
                if op in ("num_rounds", "rounds_used"):
                    return num_rounds
                if op == "eta":
                    return eta_val
                if op in ("max_query_budget", "budget", "max_queries"):
                    return budget_cap
                if op in ("max_depth_cap", "nmax"):
                    return depth_cap
                if op in ("remaining_budget", "remaining_queries"):
                    return None if budget_cap is None else max(0, budget_cap - total_queries)
                if op == "history":
                    return tuple(history)
                if op == "accounting":
                    return {
                        "total_queries": total_queries,
                        "total_shots": total_shots,
                        "max_depth": max_depth,
                        "seq_depth": seq_depth,
                        "num_rounds": num_rounds,
                        "eta": eta_val,
                        "max_query_budget": budget_cap,
                        "max_depth_cap": depth_cap,
                    }
                raise ValueError(f"Unknown oracle operation: {op}")

            _vault[self] = _dispatch

        def measure(self, depth: int, shots: int = 1) -> int:
            """Execute ``shots`` circuit measurements at Grover depth ``depth``.

            Counts as 1 adaptive round at depth ``depth``:
            - ``total_queries += shot_cost(depth) * shots``
            - ``total_shots += shots``
            - ``max_depth = max(max_depth, depth)``
            - ``seq_depth += depth``
            - ``num_rounds += 1``

            Returns
            -------
            int
                Number of ``|0>`` outcomes observed in ``[0, shots]``.
            """
            return _get_dispatch(self)("measure", depth, shots)

        def measure_batch(
            self,
            depths: Sequence[int],
            shots: Sequence[int],
        ) -> np.ndarray:
            """Execute a parallel round of circuits across ``depths`` with ``shots``.

            Counts as 1 parallel round of depth ``max(depths)``:
            - ``seq_depth += max(depths)``
            - ``max_depth = max(max_depth, max(depths))``
            - ``total_queries += sum(shot_cost(d) * s for d, s in zip(depths, shots))``
            - ``total_shots += sum(shots)``
            - ``num_rounds += 1``

            Returns
            -------
            np.ndarray
                1D integer array of ``|0>`` outcome counts for each depth in ``depths``.
            """
            return _get_dispatch(self)("measure_batch", depths, shots)

        total_queries = _read_only_prop(
            "total_queries",
            "Total oracle queries expended so far (shot at depth 0 costs 1; depth n>=1 costs n).",
        )
        total_shots = _read_only_prop(
            "total_shots",
            "Total circuit shots executed so far across all depths.",
        )
        max_depth = _read_only_prop(
            "max_depth",
            "Maximum Grover depth n queried so far across all calls.",
        )
        seq_depth = _read_only_prop(
            "seq_depth",
            "Cumulative sequential Grover depth across adaptive rounds.",
        )
        num_rounds = _read_only_prop(
            "num_rounds",
            "Number of adaptive measurement rounds (measure or measure_batch calls).",
        )
        eta = _read_only_prop(
            "eta",
            "Per-oracle depolarizing noise parameter eta in [0, 1].",
        )
        max_query_budget = _read_only_prop(
            "max_query_budget",
            "Optional hard limit on total_queries (or None if unconstrained).",
        )
        max_depth_cap = _read_only_prop(
            "max_depth_cap",
            "Optional hard limit M on Grover depth (or None if unconstrained).",
        )
        remaining_budget = _read_only_prop(
            "remaining_budget",
            "Remaining query budget if max_query_budget is set, else None.",
        )
        remaining_queries = _read_only_prop(
            "remaining_queries",
            "Alias for remaining_budget (compatibility with GuardedOracle).",
        )
        queries_used = _read_only_prop(
            "queries_used",
            "Alias for total_queries (compatibility with GuardedOracle).",
        )
        rounds_used = _read_only_prop(
            "rounds_used",
            "Alias for num_rounds (compatibility with GuardedOracle).",
        )
        max_depth_used = _read_only_prop(
            "max_depth_used",
            "Alias for max_depth (compatibility with GuardedOracle).",
        )
        budget = _read_only_prop(
            "budget",
            "Alias for max_query_budget (compatibility with GuardedOracle).",
        )
        max_queries = _read_only_prop(
            "max_queries",
            "Alias for max_query_budget (compatibility with GuardedOracle).",
        )
        nmax = _read_only_prop(
            "nmax",
            "Alias for max_depth_cap (compatibility with GuardedOracle).",
        )
        history = _read_only_prop(
            "history",
            "Tuple of recorded (depths, shots, counts) rounds if record_history=True.",
        )

        def get_accounting(self) -> dict[str, int | float | None]:
            """Return a dictionary snapshot of all query and depth counters."""
            return _get_dispatch(self)("accounting")

        def __setattr__(self, name: str, value: Any) -> None:
            raise OracleSecurityError(
                f"QuantumAEOracle is immutable; cannot set attribute '{name}'."
            )

        def __delattr__(self, name: str) -> None:
            raise OracleSecurityError(
                f"QuantumAEOracle is immutable; cannot delete attribute '{name}'."
            )

        def __copy__(self) -> Any:
            raise OracleSecurityError("QuantumAEOracle cannot be copied.")

        def __deepcopy__(self, memo: Any) -> Any:
            raise OracleSecurityError("QuantumAEOracle cannot be deep-copied.")

        def __reduce__(self) -> Any:
            raise OracleSecurityError("QuantumAEOracle cannot be pickled or serialized.")

        def __reduce_ex__(self, protocol: int) -> Any:
            raise OracleSecurityError("QuantumAEOracle cannot be pickled or serialized.")

        def __repr__(self) -> str:
            acc = self.get_accounting()
            return (
                f"<QuantumAEOracle total_queries={acc['total_queries']} "
                f"max_depth={acc['max_depth']} seq_depth={acc['seq_depth']} "
                f"num_rounds={acc['num_rounds']} eta={acc['eta']}>"
            )

    return QuantumAEOracle


QuantumAEOracle = _build_oracle_class()


# ---------------------------------------------------------------------------
# Fast Vectorized Maximum-Likelihood Estimator Helper
# ---------------------------------------------------------------------------


def ml_estimate_batch(
    depths: Sequence[int],
    shots: Sequence[int],
    counts: np.ndarray | Sequence[int],
    *,
    eta: float = 0.0,
    chunk: int = 4000,
    grid_step: float | None = None,
) -> np.ndarray | float:
    """Exact global maximum-likelihood amplitude estimator with 25-point parabolic zoom.

    Supports both a single trial (``counts`` of shape ``(K,)``, returning ``float``)
    and a batch of trials (``counts`` of shape ``(ntrials, K)``, returning a 1D
    ``np.ndarray`` of shape ``(ntrials,)``).
    """
    depths_f = np.asarray(depths, dtype=np.float64)
    shots_i = np.asarray(shots, dtype=np.int64)
    shots_f = shots_i.astype(np.float64)
    counts_arr = np.asarray(counts, dtype=np.float64)

    single_input = counts_arr.ndim == 1
    if single_input:
        K = counts_arr[None, :]
    elif counts_arr.ndim == 2:
        K = counts_arr
    else:
        raise ValueError(f"counts must be 1D or 2D array, got shape {counts_arr.shape}")

    if K.shape[1] != len(depths_f) or len(depths_f) != len(shots_i):
        raise ValueError(
            f"Shape mismatch: depths ({len(depths_f)}), shots ({len(shots_i)}), "
            f"and counts ({K.shape}) must agree along the schedule axis."
        )

    d = 2.0 * depths_f + 1.0
    step = grid_spacing(depths_f, shots_f) if grid_step is None else float(grid_step)
    grid = np.arange(1e-6, np.pi / 2.0, step)
    Vm = (1.0 - float(eta)) ** d
    P = np.clip(Vm * np.cos(np.outer(grid, d)) ** 2 + 0.5 * (1.0 - Vm), 1e-12, 1.0 - 1e-12)
    logP = np.log(P)
    log1mP = np.log1p(-P)
    Vm_r = Vm[None, None, :]

    def _local_refine(th0: np.ndarray, Kc: np.ndarray) -> np.ndarray:
        offs = np.linspace(-step / 2.0, step / 2.0, 25)
        th_loc = th0[:, None] + offs[None, :]
        p = Vm_r * np.cos(th_loc[:, :, None] * d[None, None, :]) ** 2 + 0.5 * (1.0 - Vm_r)
        p = np.clip(p, 1e-12, 1.0 - 1e-12)
        L = -np.sum(
            Kc[:, None, :] * np.log(p) + (shots_f[None, None, :] - Kc[:, None, :]) * np.log1p(-p),
            axis=2,
        )
        j = np.clip(np.argmin(L, axis=1), 1, len(offs) - 2)
        ar = np.arange(len(th0))
        ym1, y0, yp1 = L[ar, j - 1], L[ar, j], L[ar, j + 1]
        den = ym1 - 2.0 * y0 + yp1
        off = np.where(np.abs(den) > 1e-12, 0.5 * (ym1 - yp1) / np.maximum(den, 1e-12), 0.0)
        dstep = offs[1] - offs[0]
        return th_loc[ar, j] + np.clip(off, -0.5, 0.5) * dstep

    ntrials = K.shape[0]
    th_hat = np.empty(ntrials, dtype=np.float64)
    for lo in range(0, ntrials, chunk):
        hi = min(lo + chunk, ntrials)
        Kc = K[lo:hi]
        nll = -(Kc @ logP.T + (shots_f[None, :] - Kc) @ log1mP.T)
        idx = np.clip(np.argmin(nll, axis=1), 1, len(grid) - 2)
        th_hat[lo:hi] = _local_refine(grid[idx], Kc)

    a_hat = np.clip(np.sin(th_hat), 0.0, 1.0)
    if single_input:
        return float(a_hat[0])
    return a_hat


# ---------------------------------------------------------------------------
# Submission Base Classes & Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class NonAdaptiveAESubmissionProtocol(Protocol):
    """Structural protocol for non-adaptive Quantum Amplitude Estimation submissions."""

    def get_schedule(self, scale_nmax: int) -> tuple[Sequence[int], Sequence[int]]:
        ...

    def estimate(
        self,
        depths: Sequence[int],
        shots: Sequence[int],
        counts: np.ndarray,
    ) -> np.ndarray | float:
        ...


class NonAdaptiveAESubmission(abc.ABC):
    """Base class for Non-Adaptive Quantum Amplitude Estimation submissions.

    A non-adaptive algorithm fixes its entire measurement schedule ``(depths, shots)``
    prior to observing any circuit outcomes, allowing all circuits to execute in
    a single parallel round of depth ``max(depths)``.

    Subclasses must implement:
    - ``get_schedule(scale_nmax: int) -> tuple[Sequence[int], Sequence[int]]``
    - ``estimate(depths, shots, counts) -> np.ndarray | float``
      Must accept 2D ``counts`` of shape ``(ntrials, len(depths))`` for fast
      vectorized batch evaluation (and optionally 1D ``counts`` of shape
      ``(len(depths),)`` for single-trial evaluation).
    """

    name: str = "unnamed-non-adaptive"
    track: str = "non-adaptive"

    @abc.abstractmethod
    def get_schedule(self, scale_nmax: int) -> tuple[Sequence[int], Sequence[int]]:
        """Return ``(depths, shots)`` for the benchmark scale ``scale_nmax``."""
        raise NotImplementedError

    @abc.abstractmethod
    def estimate(
        self,
        depths: Sequence[int],
        shots: Sequence[int],
        counts: np.ndarray,
    ) -> np.ndarray | float:
        """Estimate amplitude(s) $\\hat{a} \\in [0, 1]$ from observed ``counts``.

        Parameters
        ----------
        depths : Sequence[int]
            1D sequence of Grover depths $n_k \\ge 0$.
        shots : Sequence[int]
            1D sequence of shot counts $N_k \\ge 1$ corresponding to ``depths``.
        counts : np.ndarray
            Either a 2D array of shape ``(ntrials, len(depths))`` for batch evaluation
            or a 1D array of shape ``(len(depths),)`` for a single trial.

        Returns
        -------
        np.ndarray | float
            1D array of amplitude estimates $\\hat{a}$ of shape ``(ntrials,)`` when
            ``counts`` is 2D, or a ``float`` when ``counts`` is 1D.
        """
        raise NotImplementedError

    def estimate_batch(
        self,
        depths: Sequence[int],
        shots: Sequence[int],
        counts_2d: np.ndarray,
    ) -> np.ndarray:
        """Evaluate ``estimate`` on a 2D batch ``(ntrials, K)`` with graceful 1D fallback."""
        counts_arr = np.asarray(counts_2d)
        if counts_arr.ndim == 1:
            counts_arr = counts_arr[None, :]
        ntrials = counts_arr.shape[0]
        try:
            res = self.estimate(depths, shots, counts_arr)
            res_arr = np.asarray(res, dtype=np.float64).reshape(-1)
            if len(res_arr) == ntrials:
                return np.clip(res_arr, 0.0, 1.0)
        except Exception:
            pass
        # Fallback if subclass only implemented 1D single-trial estimate()
        out = np.empty(ntrials, dtype=np.float64)
        for i in range(ntrials):
            out[i] = float(np.asarray(self.estimate(depths, shots, counts_arr[i])).item())
        return np.clip(out, 0.0, 1.0)


@runtime_checkable
class AdaptiveAESubmissionProtocol(Protocol):
    """Structural protocol for adaptive Quantum Amplitude Estimation submissions."""

    def run_trial(
        self,
        oracle: QuantumAEOracle,
        target_scale_or_budget: int,
        rng: np.random.Generator,
    ) -> float:
        ...


class AdaptiveAESubmission(abc.ABC):
    """Base class for Adaptive & Semi-Adaptive Quantum Amplitude Estimation submissions.

    Subclasses must implement:
    - ``run_trial(oracle: QuantumAEOracle, target_scale_or_budget: int, rng: np.random.Generator) -> float``

    Subclasses may optionally override:
    - ``run_batch(oracles: Sequence[QuantumAEOracle], target_scale_or_budget: int, rngs: Sequence[np.random.Generator]) -> np.ndarray``
      for vectorized multi-round semi-adaptive evaluation across a batch of sandboxed
      oracles.
    """

    name: str = "unnamed-adaptive"
    track: str = "adaptive-fixed-budget"

    @abc.abstractmethod
    def run_trial(
        self,
        oracle: QuantumAEOracle,
        target_scale_or_budget: int,
        rng: np.random.Generator,
    ) -> float:
        """Run one adaptive estimation trial interacting with ``oracle``.

        Parameters
        ----------
        oracle : QuantumAEOracle
            Sandboxed quantum oracle wrapping the unknown amplitude $a$.
            Call ``oracle.measure(depth, shots)`` or ``oracle.measure_batch(depths, shots)``.
        target_scale_or_budget : int
            Benchmark scale $n_{\\max} \\in \\{60, 125, 260, 540\\}$ or target query budget $Q$.
            If ``oracle.max_query_budget`` is set, the oracle enforces that hard budget cap.
        rng : np.random.Generator
            Per-trial classical random number generator for randomized adaptive decisions.

        Returns
        -------
        float
            Estimated amplitude $\\hat{a} \\in [0, 1]$.
        """
        raise NotImplementedError

    def run_batch(
        self,
        oracles: Sequence[QuantumAEOracle],
        target_scale_or_budget: int,
        rngs: Sequence[np.random.Generator],
    ) -> np.ndarray:
        """Run a batch of trials across ``oracles`` (defaults to loop over ``run_trial``)."""
        if len(oracles) != len(rngs):
            raise ValueError("oracles and rngs must have the same length.")
        estimates = np.empty(len(oracles), dtype=np.float64)
        for i, (orc, trial_rng) in enumerate(zip(oracles, rngs)):
            a_hat = float(self.run_trial(orc, target_scale_or_budget, trial_rng))
            estimates[i] = float(np.clip(a_hat, 0.0, 1.0))
        return estimates


# ---------------------------------------------------------------------------
# Built-in Reference Submissions (Non-Adaptive & Adaptive)
# ---------------------------------------------------------------------------


class MLQAEFlagshipSubmission(NonAdaptiveAESubmission):
    """Reference non-adaptive submission: geometric ladder (r=1.45) + cap rung + global ML."""

    name = "mlqae-flagship"
    track = "non-adaptive"

    def __init__(
        self,
        r: float = 1.45,
        cap_divisor: float | None = 1.3,
        base: int = 1,
        slope: float = 1.0,
        eta: float = 0.0,
    ) -> None:
        self.r = float(r)
        self.cap_divisor = None if cap_divisor is None else float(cap_divisor)
        self.base = int(base)
        self.slope = float(slope)
        self.eta = float(eta)

    def get_schedule(self, scale_nmax: int) -> tuple[ list[int], list[int] ]:
        depths, shots = flagship_schedule(
            int(scale_nmax),
            r=self.r,
            cap_divisor=self.cap_divisor,
            base=self.base,
            slope=self.slope,
        )
        return list(depths), list(shots)

    def estimate(
        self,
        depths: Sequence[int],
        shots: Sequence[int],
        counts: np.ndarray,
    ) -> np.ndarray | float:
        return ml_estimate_batch(depths, shots, counts, eta=self.eta)


class MLQAEPlainSubmission(MLQAEFlagshipSubmission):
    """Reference non-adaptive submission: plain r=1.45 geometric ladder + global ML."""

    name = "mlqae-plain"
    track = "non-adaptive"

    def __init__(
        self,
        r: float = 1.45,
        base: int = 1,
        slope: float = 1.0,
        eta: float = 0.0,
    ) -> None:
        super().__init__(r=r, cap_divisor=None, base=base, slope=slope, eta=eta)


class ReferenceBayesianIntervalSubmission(AdaptiveAESubmission):
    """Reference iterative Bayesian / credible-interval QAE submission using ``QuantumAEOracle``.

    Uses only published iterative interval-narrowing concepts (cf. Grinko et al. 2021,
    arXiv:1912.05559; Wie 2019, arXiv:1907.09941): at each round, the posterior credible
    interval $[\\theta_{\\min}, \\theta_{\\max}]$ from observed binomial counts determines the
    next unambiguous Grover depth $n_t \\le n_{\\max}$ such that $(2n_t + 1)(\\theta_{\\max} - \\theta_{\\min})$
    remains within a single monotonic cosine branch, followed by Bayesian / ML readout.
    """

    name = "reference-bayesian-interval-qae"
    track = "adaptive-fixed-budget"

    def __init__(self, shots_per_stage: int = 12, eta: float = 0.0) -> None:
        self.shots_per_stage = int(shots_per_stage)
        self.eta = float(eta)

    def run_trial(
        self,
        oracle: QuantumAEOracle,
        target_scale_or_budget: int,
        rng: np.random.Generator,
    ) -> float:
        _ = rng
        scale_nmax = max(1, int(target_scale_or_budget))
        budget_cap = oracle.max_query_budget

        depths_used: list[int] = []
        shots_used: list[int] = []
        counts_obs: list[int] = []

        # Stage 0: Initial measurement at depth n = 0 (Beta(1, 1) conjugate update)
        s0 = max(16, 2 * self.shots_per_stage)
        if budget_cap is not None:
            s0 = min(s0, max(1, int(budget_cap)))
        k0 = int(oracle.measure(0, shots=s0))
        depths_used.append(0)
        shots_used.append(s0)
        counts_obs.append(k0)

        # Posterior Beta(k0 + 1, s0 - k0 + 1) mean and standard deviation for p_0 = cos^2(theta)
        alpha_post = float(k0 + 1)
        beta_post = float(s0 - k0 + 1)
        p_mean = alpha_post / (alpha_post + beta_post)
        p_sd = float(
            np.sqrt(
                (alpha_post * beta_post)
                / (((alpha_post + beta_post) ** 2) * (alpha_post + beta_post + 1.0))
            )
        )
        p_lo = float(np.clip(p_mean - 3.0 * p_sd, 1e-6, 1.0 - 1e-6))
        p_hi = float(np.clip(p_mean + 3.0 * p_sd, 1e-6, 1.0 - 1e-6))
        theta_min = float(np.arccos(np.sqrt(p_hi)))
        theta_max = float(np.arccos(np.sqrt(p_lo)))

        # Iterative depth progression guided by the credible interval width
        curr_depth = 0
        while curr_depth < scale_nmax:
            width = max(1e-6, theta_max - theta_min)
            # Maximum unambiguous depth where 2 * (2n + 1) * width <= pi
            n_unambig = int(max(1, np.floor((np.pi / (2.0 * width) - 1.0) / 2.0)))
            next_depth = min(scale_nmax, max(curr_depth + 1, min(2 * max(1, curr_depth), n_unambig)))

            s_next = self.shots_per_stage if next_depth < scale_nmax else max(4, self.shots_per_stage // 2)
            if budget_cap is not None:
                rem = int(budget_cap) - int(oracle.total_queries)
                if rem < next_depth:
                    break
                s_next = max(1, min(s_next, rem // next_depth))

            k_next = int(oracle.measure(next_depth, shots=s_next))
            depths_used.append(next_depth)
            shots_used.append(s_next)
            counts_obs.append(k_next)
            curr_depth = next_depth

            # Update point estimate and Fisher-based credible interval width
            a_est = float(
                ml_estimate_batch(
                    depths_used,
                    shots_used,
                    np.asarray(counts_obs, dtype=np.int64),
                    eta=self.eta,
                )
            )
            th_est = float(np.arcsin(np.clip(a_est, 1e-6, 1.0 - 1e-6)))
            d_arr = 2.0 * np.asarray(depths_used, dtype=np.float64) + 1.0
            s_arr = np.asarray(shots_used, dtype=np.float64)
            sigma_th = float(1.0 / np.sqrt(4.0 * np.sum(s_arr * (d_arr ** 2))))
            theta_min = max(1e-6, th_est - 3.0 * sigma_th)
            theta_max = min(np.pi / 2.0 - 1e-6, th_est + 3.0 * sigma_th)

        return float(
            ml_estimate_batch(
                depths_used,
                shots_used,
                np.asarray(counts_obs, dtype=np.int64),
                eta=self.eta,
            )
        )


# Backward-compatible alias for test suites importing ReferenceTwoRoundAdaptiveSubmission
ReferenceTwoRoundAdaptiveSubmission = ReferenceBayesianIntervalSubmission


# ---------------------------------------------------------------------------
# Standardized Evaluation Helpers
# ---------------------------------------------------------------------------


def _compute_standardized_metrics(
    a_true: np.ndarray,
    a_hat: np.ndarray,
    queries: np.ndarray,
    max_depths: np.ndarray,
    seq_depths: np.ndarray,
    num_rounds: np.ndarray,
    *,
    boot_B: int = 1000,
    boot_seed: int = 1,
) -> dict[str, Any]:
    """Compute leaderboard metrics matching ``leaderboard/harness.py`` and ``mlqae/core.py``."""
    err = np.abs(np.asarray(a_hat, dtype=np.float64) - np.asarray(a_true, dtype=np.float64))
    q = np.asarray(queries, dtype=np.float64)
    md = np.asarray(max_depths, dtype=np.float64)
    sd = np.asarray(seq_depths, dtype=np.float64)
    nr = np.asarray(num_rounds, dtype=np.float64)

    q_mean = float(np.mean(q))
    q_max = float(np.max(q))
    q_std = float(np.std(q))
    md_mean = float(np.mean(md))
    sd_mean = float(np.mean(sd))

    out: dict[str, Any] = {
        "ntrials": int(len(err)),
        "q_mean": q_mean,
        "q_max": q_max,
        "q_std": q_std,
        "max_depth_mean": md_mean,
        "max_depth_max": int(np.max(md)),
        "seq_depth_mean": sd_mean,
        "seq_depth_max": int(np.max(sd)),
        "num_rounds_mean": float(np.mean(nr)),
    }

    for pct in (50, 68, 95, 99):
        ep = float(np.percentile(err, pct))
        out[f"eps{pct}"] = ep
        out[f"C{pct}_ave"] = ep * q_mean
        out[f"C{pct}_max"] = ep * q_max
        out[f"C{pct}_seq"] = ep * sd_mean
        out[f"C{pct}_par"] = ep * md_mean

    if boot_B and boot_B > 0:
        rng = np.random.default_rng(boot_seed)
        n = len(err)
        for pct in (68, 95, 99):
            c_ave_b = np.empty(boot_B, dtype=np.float64)
            c_max_b = np.empty(boot_B, dtype=np.float64)
            for b in range(boot_B):
                idx = rng.integers(0, n, n)
                ep = np.percentile(err[idx], pct)
                c_ave_b[b] = ep * np.mean(q[idx])
                c_max_b[b] = ep * np.max(q[idx])
            out[f"C{pct}_ave_ci"] = [
                float(np.percentile(c_ave_b, 2.5)),
                float(np.percentile(c_ave_b, 97.5)),
            ]
            out[f"C{pct}_max_ci"] = [
                float(np.percentile(c_max_b, 2.5)),
                float(np.percentile(c_max_b, 97.5)),
            ]

    return out


def evaluate_non_adaptive_submission(
    submission: NonAdaptiveAESubmission,
    scale_nmax: int,
    *,
    ntrials: int = 20_000,
    seed_band: int = HOLDOUT_SEED,
    a_range: tuple[float, float] = DEFAULT_A_RANGE,
    eta: float = 0.0,
    max_depth_cap: int | None = None,
    boot_B: int = 1000,
) -> dict[str, Any]:
    """Evaluate a ``NonAdaptiveAESubmission`` at ``scale_nmax`` using vectorized simulation."""
    depths, shots = submission.get_schedule(int(scale_nmax))
    d_arr = np.asarray(depths, dtype=np.int64)
    s_arr = np.asarray(shots, dtype=np.int64)
    if d_arr.ndim != 1 or s_arr.ndim != 1 or len(d_arr) != len(s_arr) or len(d_arr) == 0:
        raise ValueError("get_schedule() must return non-empty 1D (depths, shots) of equal length.")
    if np.any(d_arr < 0) or np.any(s_arr < 1):
        raise ValueError("Schedule must have depths >= 0 and shots >= 1.")

    nmax_actual = int(np.max(d_arr))
    if max_depth_cap is not None and nmax_actual > int(max_depth_cap):
        raise DepthCapExceededError(
            f"Schedule max depth {nmax_actual} exceeds max_depth_cap={max_depth_cap}."
        )

    nq = schedule_query_cost(d_arr, s_arr)

    rng = np.random.default_rng(int(seed_band) + int(scale_nmax))
    a_true = rng.uniform(a_range[0], a_range[1], int(ntrials))
    th_true = np.arcsin(a_true)

    d_odd = 2.0 * d_arr.astype(np.float64) + 1.0
    vt = (1.0 - float(eta)) ** d_odd
    p_true = vt * np.cos(np.outer(th_true, d_odd)) ** 2 + 0.5 * (1.0 - vt)
    p_true = np.clip(p_true, 0.0, 1.0)
    counts = rng.binomial(s_arr[None, :], p_true)

    a_hat = submission.estimate_batch(d_arr, s_arr, counts)

    q_vec = np.full(ntrials, nq, dtype=np.float64)
    md_vec = np.full(ntrials, nmax_actual, dtype=np.float64)
    sd_vec = np.full(ntrials, nmax_actual, dtype=np.float64)
    nr_vec = np.ones(ntrials, dtype=np.float64)

    metrics = _compute_standardized_metrics(
        a_true,
        a_hat,
        q_vec,
        md_vec,
        sd_vec,
        nr_vec,
        boot_B=boot_B,
        boot_seed=1,
    )
    metrics["scale_nmax"] = int(scale_nmax)
    metrics["nq"] = int(nq)
    metrics["nmax"] = int(nmax_actual)
    return metrics


def evaluate_adaptive_submission(
    submission: AdaptiveAESubmission,
    target_scale_or_budget: int,
    *,
    ntrials: int = 2_000,
    seed_band: int = HOLDOUT_SEED,
    a_range: tuple[float, float] = DEFAULT_A_RANGE,
    eta: float = 0.0,
    max_query_budget: int | None = None,
    max_depth_cap: int | None = None,
    boot_B: int = 1000,
) -> dict[str, Any]:
    """Evaluate an ``AdaptiveAESubmission`` across ``ntrials`` sandboxed ``QuantumAEOracle`` runs."""
    master_rng = np.random.default_rng(int(seed_band) + int(target_scale_or_budget))
    a_true = master_rng.uniform(a_range[0], a_range[1], int(ntrials))
    oracle_seeds = master_rng.integers(0, 2**31 - 1, size=int(ntrials), dtype=np.int64)
    algo_seeds = master_rng.integers(0, 2**31 - 1, size=int(ntrials), dtype=np.int64)

    oracles = [
        QuantumAEOracle(
            float(a_true[i]),
            seed=int(oracle_seeds[i]),
            eta=eta,
            max_query_budget=max_query_budget,
            max_depth_cap=max_depth_cap,
        )
        for i in range(int(ntrials))
    ]
    trial_rngs = [np.random.default_rng(int(algo_seeds[i])) for i in range(int(ntrials))]

    a_hat = submission.run_batch(oracles, int(target_scale_or_budget), trial_rngs)
    a_hat = np.clip(np.asarray(a_hat, dtype=np.float64), 0.0, 1.0)

    queries = np.array([orc.total_queries for orc in oracles], dtype=np.float64)
    max_depths = np.array([orc.max_depth for orc in oracles], dtype=np.float64)
    seq_depths = np.array([orc.seq_depth for orc in oracles], dtype=np.float64)
    num_rounds = np.array([orc.num_rounds for orc in oracles], dtype=np.float64)

    if np.any(queries <= 0):
        raise ValueError("Adaptive submission made 0 oracle queries on at least one trial.")

    metrics = _compute_standardized_metrics(
        a_true,
        a_hat,
        queries,
        max_depths,
        seq_depths,
        num_rounds,
        boot_B=boot_B,
        boot_seed=1,
    )
    metrics["target_scale_or_budget"] = int(target_scale_or_budget)
    return metrics


# ---------------------------------------------------------------------------
# JSON Schemas & Validation Helpers (metadata.json and leaderboard.json)
# ---------------------------------------------------------------------------

METADATA_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "QuantumAEChallengeSubmissionMetadata",
    "description": "Metadata schema for a Quantum Amplitude Estimation leaderboard submission.",
    "type": "object",
    "required": [
        "algorithm_name",
        "track",
        "authors",
        "github_handle",
        "affiliation",
        "paper_url",
        "pr_number",
        "description",
        "date",
    ],
    "properties": {
        "algorithm_name": {
            "type": "string",
            "minLength": 1,
            "maxLength": 120,
            "description": "Unique display name or identifier of the AE algorithm.",
        },
        "track": {
            "type": "string",
            "enum": list(VALID_TRACKS),
            "description": (
                "Competition track: 'non-adaptive', 'semi-adaptive', "
                "'adaptive-fixed-budget', 'adaptive-variable', or 'depth-limited'."
            ),
        },
        "authors": {
            "oneOf": [
                {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 1},
                {"type": "string", "minLength": 1},
            ],
            "description": "List of author names (or comma-separated author string).",
        },
        "github_handle": {
            "oneOf": [
                {"type": "string", "minLength": 1},
                {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 1},
            ],
            "description": "GitHub username(s) of the submitting researcher(s).",
        },
        "affiliation": {
            "oneOf": [
                {"type": "string"},
                {"type": "array", "items": {"type": "string"}},
            ],
            "description": "Academic institution, lab, or company affiliation.",
        },
        "paper_url": {
            "type": ["string", "null"],
            "description": "URL to accompanying paper/preprint (e.g. arXiv link) or null/empty.",
        },
        "pr_number": {
            "type": ["integer", "string", "null"],
            "description": "GitHub Pull Request number (or null/'TBD' prior to PR creation).",
        },
        "description": {
            "type": "string",
            "minLength": 5,
            "description": "Concise technical summary of the schedule/adaptive policy and estimator.",
        },
        "date": {
            "type": "string",
            "pattern": r"^\d{4}-\d{2}-\d{2}",
            "description": "Submission date in ISO-8601 format (YYYY-MM-DD).",
        },
        "entrypoint": {
            "type": "string",
            "description": "Optional module:ClassName entrypoint for automated runner loading.",
        },
    },
    "additionalProperties": True,
}

LEADERBOARD_ENTRY_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "QuantumAELeaderboardEntry",
    "type": "object",
    "required": [
        "algorithm_name",
        "track",
        "authors",
        "github_handle",
        "description",
        "date",
    ],
    "properties": {
        **METADATA_JSON_SCHEMA["properties"],
        "is_baseline": {"type": "boolean"},
        "verified": {"type": "boolean"},
        "C68_ave": {"type": ["number", "null"]},
        "C95_ave": {"type": ["number", "null"]},
        "C95_max": {"type": ["number", "null"]},
        "C99_ave": {"type": ["number", "null"]},
        "C99_max": {"type": ["number", "null"]},
        "C95_seq": {"type": ["number", "null"]},
        "C95_par": {"type": ["number", "null"]},
        "scales": {"type": ["object", "array"]},
        "metrics": {"type": "object"},
    },
    "additionalProperties": True,
}

LEADERBOARD_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "QuantumAELeaderboard",
    "description": "Top-level leaderboard.json schema supporting dict or array of entries.",
    "oneOf": [
        {
            "type": "object",
            "required": ["entries"],
            "properties": {
                "schema_version": {"type": "string"},
                "updated_at": {"type": "string"},
                "tracks": {"type": ["array", "object"]},
                "scales": {"type": "array", "items": {"type": "integer"}},
                "entries": {
                    "type": "array",
                    "items": LEADERBOARD_ENTRY_SCHEMA,
                },
            },
            "additionalProperties": True,
        },
        {
            "type": "array",
            "items": LEADERBOARD_ENTRY_SCHEMA,
        },
    ],
}

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[T\s].*)?$")


def _load_json_or_mapping(
    data_or_path: Mapping[str, Any] | Sequence[Any] | str | Path,
) -> Any:
    """Load JSON from a file path or JSON string, or return an already parsed container."""
    if isinstance(data_or_path, Path):
        with open(data_or_path, "r", encoding="utf-8") as f:
            return json.load(f)
    if isinstance(data_or_path, str):
        # Could be a file path or raw JSON text
        stripped = data_or_path.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            return json.loads(stripped)
        if os.path.exists(data_or_path):
            with open(data_or_path, "r", encoding="utf-8") as f:
                return json.load(f)
        raise SchemaValidationError(f"File not found or invalid JSON string: {data_or_path!r}")
    return data_or_path


def normalize_metadata(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a ``metadata.json`` dictionary to canonical keys without mutating ``raw``."""
    if not isinstance(raw, Mapping):
        raise SchemaValidationError(f"Metadata must be a JSON object/dict, got {type(raw).__name__}")

    data = dict(raw)

    # Canonicalize algorithm_name from aliases ('name', 'algorithm', 'id')
    if "algorithm_name" not in data:
        for alias in ("name", "algorithm", "id"):
            if alias in data and isinstance(data[alias], str) and data[alias].strip():
                data["algorithm_name"] = data[alias].strip()
                break

    # Canonicalize defaults for optional nullable metadata fields if omitted in leaderboard rows
    if "paper_url" not in data and "paper" in data:
        data["paper_url"] = data["paper"]
    if "pr_number" not in data and "pr" in data:
        data["pr_number"] = data["pr"]

    return data


def validate_metadata(
    data_or_path: Mapping[str, Any] | str | Path,
    *,
    require_all_fields: bool = True,
) -> dict[str, Any]:
    """Validate a ``metadata.json`` object or file path and return the normalized dict.

    Required fields:
    - ``algorithm_name`` (str, non-empty)
    - ``track`` (one of ``non-adaptive``, ``semi-adaptive``, ``adaptive-fixed-budget``,
      ``adaptive-variable``, ``depth-limited``)
    - ``authors`` (non-empty list of strings or non-empty string)
    - ``github_handle`` (non-empty string or non-empty list of strings)
    - ``affiliation`` (string or list of strings)
    - ``paper_url`` (string URL or None / empty string)
    - ``pr_number`` (positive int, string like ``'TBD'``, or None)
    - ``description`` (non-empty string)
    - ``date`` (ISO date string ``YYYY-MM-DD``)

    Raises
    ------
    SchemaValidationError
        If any field is missing or fails type/constraint validation.
    """
    loaded = _load_json_or_mapping(data_or_path)
    data = normalize_metadata(loaded)

    required_keys = (
        ("algorithm_name", "track", "authors", "github_handle", "affiliation",
         "paper_url", "pr_number", "description", "date")
        if require_all_fields
        else ("algorithm_name", "track")
    )

    missing = [k for k in required_keys if k not in data]
    if missing:
        raise SchemaValidationError(
            f"Metadata missing required field(s): {', '.join(missing)}"
        )

    # 1. algorithm_name
    alg_name = data.get("algorithm_name")
    if not isinstance(alg_name, str) or not alg_name.strip():
        raise SchemaValidationError("'algorithm_name' must be a non-empty string.")

    # 2. track
    track = data.get("track")
    if not isinstance(track, str) or track not in VALID_TRACKS:
        raise SchemaValidationError(
            f"'track' must be one of {VALID_TRACKS}, got {track!r}."
        )

    # 3. authors
    if "authors" in data:
        authors = data["authors"]
        if isinstance(authors, str):
            if not authors.strip():
                raise SchemaValidationError("'authors' string must not be empty.")
        elif isinstance(authors, (list, tuple)):
            if len(authors) == 0 or not all(isinstance(a, str) and a.strip() for a in authors):
                raise SchemaValidationError(
                    "'authors' list must contain at least one non-empty string."
                )
        else:
            raise SchemaValidationError(
                f"'authors' must be a list of strings or a string, got {type(authors).__name__}."
            )

    # 4. github_handle
    if "github_handle" in data:
        gh = data["github_handle"]
        if isinstance(gh, str):
            if not gh.strip():
                raise SchemaValidationError("'github_handle' must not be empty.")
        elif isinstance(gh, (list, tuple)):
            if len(gh) == 0 or not all(isinstance(h, str) and h.strip() for h in gh):
                raise SchemaValidationError(
                    "'github_handle' list must contain non-empty strings."
                )
        else:
            raise SchemaValidationError(
                f"'github_handle' must be a string or list of strings, got {type(gh).__name__}."
            )

    # 5. affiliation
    if "affiliation" in data:
        aff = data["affiliation"]
        if isinstance(aff, str):
            pass
        elif isinstance(aff, (list, tuple)):
            if not all(isinstance(x, str) for x in aff):
                raise SchemaValidationError("'affiliation' list items must be strings.")
        else:
            raise SchemaValidationError(
                f"'affiliation' must be a string or list of strings, got {type(aff).__name__}."
            )

    # 6. paper_url
    if "paper_url" in data:
        purl = data["paper_url"]
        if purl is not None and not isinstance(purl, str):
            raise SchemaValidationError(
                f"'paper_url' must be a string or null, got {type(purl).__name__}."
            )

    # 7. pr_number
    if "pr_number" in data:
        pr = data["pr_number"]
        if pr is not None:
            if isinstance(pr, bool) or not isinstance(pr, (int, str)):
                raise SchemaValidationError(
                    f"'pr_number' must be an int, string, or null, got {type(pr).__name__}."
                )
            if isinstance(pr, int) and pr < 0:
                raise SchemaValidationError(f"'pr_number' integer must be >= 0, got {pr}.")

    # 8. description
    if "description" in data:
        desc = data["description"]
        if not isinstance(desc, str) or len(desc.strip()) < 3:
            raise SchemaValidationError("'description' must be a non-empty string (min 3 chars).")

    # 9. date
    if "date" in data:
        date_str = data["date"]
        if not isinstance(date_str, str) or not _DATE_RE.match(date_str.strip()):
            raise SchemaValidationError(
                f"'date' must be an ISO-8601 date string 'YYYY-MM-DD', got {date_str!r}."
            )
        try:
            _dt.date.fromisoformat(date_str.strip()[:10])
        except ValueError as exc:
            raise SchemaValidationError(f"Invalid calendar date in 'date': {date_str!r}") from exc

    return data


def validate_leaderboard_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a single entry in ``leaderboard.json``."""
    if not isinstance(entry, Mapping):
        raise SchemaValidationError(
            f"Leaderboard entry must be an object/dict, got {type(entry).__name__}."
        )

    # Merge nested metadata block if present
    merged: dict[str, Any] = {}
    if isinstance(entry.get("metadata"), Mapping):
        merged.update(entry["metadata"])
    merged.update(entry)

    # Fill sensible defaults for optional metadata fields if a baseline entry omitted them
    merged.setdefault("affiliation", "Unspecified")
    merged.setdefault("paper_url", None)
    merged.setdefault("pr_number", None)
    merged.setdefault("github_handle", "unitaryfoundation")
    merged.setdefault("authors", ["Unknown"])
    merged.setdefault("description", merged.get("algorithm_name", merged.get("name", "AE entry")))
    merged.setdefault("date", "2026-10-23")

    norm = validate_metadata(merged, require_all_fields=True)

    # Validate numerical metric fields if present at top level or inside metrics/scales
    numeric_keys = (
        "C68",
        "C68_ave",
        "C68_max",
        "C95",
        "C95_ave",
        "C95_max",
        "C99",
        "C99_ave",
        "C99_max",
        "C95_seq",
        "C95_par",
        "C95par",
        "q_mean",
        "q_max",
        "nq",
        "eps68",
        "eps95",
        "eps99",
    )

    def _check_metrics_dict(m_dict: Mapping[str, Any], ctx: str) -> None:
        for k in numeric_keys:
            if k in m_dict and m_dict[k] is not None:
                val = m_dict[k]
                if isinstance(val, bool) or not isinstance(val, (int, float, np.number)):
                    raise SchemaValidationError(
                        f"Metric '{k}' in {ctx} must be a number, got {type(val).__name__}."
                    )
                if np.isnan(float(val)) or float(val) < 0.0:
                    raise SchemaValidationError(
                        f"Metric '{k}' in {ctx} must be a non-negative finite number, got {val}."
                    )
        for ci_key in ("C68_ave_ci", "C95_ave_ci", "C99_ave_ci", "C95_max_ci", "C99_max_ci"):
            if ci_key in m_dict and m_dict[ci_key] is not None:
                ci = m_dict[ci_key]
                if not isinstance(ci, (list, tuple)) or len(ci) != 2:
                    raise SchemaValidationError(
                        f"Bootstrap CI '{ci_key}' in {ctx} must be a 2-element [lo, hi] list."
                    )
                lo, hi = float(ci[0]), float(ci[1])
                if np.isnan(lo) or np.isnan(hi) or lo > hi:
                    raise SchemaValidationError(
                        f"Bootstrap CI '{ci_key}' in {ctx} must satisfy lo <= hi, got {ci}."
                    )

    _check_metrics_dict(norm, f"entry '{norm['algorithm_name']}'")

    if "metrics" in norm and isinstance(norm["metrics"], Mapping):
        _check_metrics_dict(norm["metrics"], f"entry '{norm['algorithm_name']}'.metrics")

    if "scales" in norm:
        scales_obj = norm["scales"]
        if isinstance(scales_obj, Mapping):
            for s_key, s_val in scales_obj.items():
                if isinstance(s_val, Mapping):
                    _check_metrics_dict(
                        s_val, f"entry '{norm['algorithm_name']}'.scales[{s_key}]"
                    )
        elif isinstance(scales_obj, (list, tuple)):
            for idx, s_val in enumerate(scales_obj):
                if isinstance(s_val, Mapping):
                    _check_metrics_dict(
                        s_val, f"entry '{norm['algorithm_name']}'.scales[{idx}]"
                    )
        else:
            raise SchemaValidationError(
                f"'scales' in entry '{norm['algorithm_name']}' must be a dict or list."
            )

    return norm


def validate_leaderboard(
    data_or_path: Mapping[str, Any] | Sequence[Any] | str | Path,
) -> dict[str, Any]:
    """Validate ``leaderboard.json`` (accepts top-level object or list of entries).

    Returns a canonical dictionary containing ``schema_version``, ``entries``, and any
    top-level metadata fields.
    """
    loaded = _load_json_or_mapping(data_or_path)

    if isinstance(loaded, (list, tuple)):
        raw_entries = list(loaded)
        envelope: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "entries": [],
        }
    elif isinstance(loaded, Mapping):
        envelope = dict(loaded)
        envelope.setdefault("schema_version", SCHEMA_VERSION)
        raw_entries = None
        for key in ("entries", "submissions", "leaderboard", "results"):
            if key in envelope and isinstance(envelope[key], (list, tuple)):
                raw_entries = list(envelope[key])
                break
        if raw_entries is None:
            raise SchemaValidationError(
                "leaderboard.json object must contain an 'entries' (or 'submissions'/'leaderboard') list."
            )
    else:
        raise SchemaValidationError(
            f"leaderboard.json must be a JSON object or array, got {type(loaded).__name__}."
        )

    validated_entries = [validate_leaderboard_entry(item) for item in raw_entries]
    envelope["entries"] = validated_entries
    return envelope


def create_leaderboard_entry(
    metadata: Mapping[str, Any],
    scales_metrics: Mapping[str | int, Mapping[str, Any]],
    *,
    is_baseline: bool = False,
    verified: bool = True,
) -> dict[str, Any]:
    """Construct and validate a standardized leaderboard entry from metadata & scale metrics."""
    norm_meta = validate_metadata(metadata)
    scales_str_dict: dict[str, dict[str, Any]] = {
        str(k): dict(v) for k, v in scales_metrics.items()
    }

    # Compute summary averages across available scales (or use primary scale 125 / 260)
    c68_vals = [
        float(v.get("C68_ave", v.get("C68")))
        for v in scales_str_dict.values()
        if v.get("C68_ave", v.get("C68")) is not None
    ]
    c95_vals = [
        float(v.get("C95_ave", v.get("C95")))
        for v in scales_str_dict.values()
        if v.get("C95_ave", v.get("C95")) is not None
    ]
    c95_max_vals = [
        float(v.get("C95_max", v.get("C95")))
        for v in scales_str_dict.values()
        if v.get("C95_max", v.get("C95")) is not None
    ]
    c99_vals = [
        float(v.get("C99_ave", v.get("C99")))
        for v in scales_str_dict.values()
        if v.get("C99_ave", v.get("C99")) is not None
    ]
    c95_seq_vals = [
        float(v.get("C95_seq", v.get("C95par")))
        for v in scales_str_dict.values()
        if v.get("C95_seq", v.get("C95par")) is not None
    ]

    entry: dict[str, Any] = {
        **norm_meta,
        "is_baseline": bool(is_baseline),
        "verified": bool(verified),
        "C68_ave": round(float(np.mean(c68_vals)), 4) if c68_vals else None,
        "C95_ave": round(float(np.mean(c95_vals)), 4) if c95_vals else None,
        "C95_max": round(float(np.mean(c95_max_vals)), 4) if c95_max_vals else None,
        "C99_ave": round(float(np.mean(c99_vals)), 4) if c99_vals else None,
        "C95_seq": round(float(np.mean(c95_seq_vals)), 4) if c95_seq_vals else None,
        "scales": scales_str_dict,
    }
    return validate_leaderboard_entry(entry)


def build_leaderboard_payload(
    entries: Sequence[Mapping[str, Any]],
    *,
    scales: Sequence[int] = STANDARD_SCALES,
    updated_at: str | None = None,
) -> dict[str, Any]:
    """Build and validate a complete ``leaderboard.json`` dictionary."""
    if updated_at is None:
        updated_at = _dt.date.today().isoformat()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "updated_at": updated_at,
        "tracks": list(VALID_TRACKS),
        "scales": [int(s) for s in scales],
        "entries": [validate_leaderboard_entry(e) for e in entries],
    }
    return validate_leaderboard(payload)


# ---------------------------------------------------------------------------
# Self-Test Suite
# ---------------------------------------------------------------------------


def _run_self_tests() -> None:
    import copy
    import pickle

    print("=== Running leaderboard/schema_and_api.py Self-Tests ===")

    # 1. Test QuantumAEOracle sandbox security
    secret_a = 0.6180339887
    secret_theta = float(np.arcsin(secret_a))
    oracle = QuantumAEOracle(secret_a, seed=12345, max_query_budget=500, max_depth_cap=60)

    assert oracle.__dict__ == {}, f"Expected empty __dict__, got {oracle.__dict__}"
    oracle_dir = dir(oracle)
    for forbidden in ("a_true", "theta_true", "_a_true", "_theta_true", "rng", "_rng", "_vault"):
        assert forbidden not in oracle_dir, f"Leaked '{forbidden}' in dir(oracle)!"
        assert getattr(oracle, forbidden, None) is None, f"Leaked '{forbidden}' via getattr!"

    for attr_name in oracle_dir:
        if not attr_name.startswith("__"):
            val = getattr(oracle, attr_name)
            if isinstance(val, (int, float)):
                assert not np.isclose(val, secret_a, atol=1e-5), f"Attribute {attr_name} leaked a_true!"
                assert not np.isclose(val, secret_theta, atol=1e-5), (
                    f"Attribute {attr_name} leaked theta_true!"
                )

    # Test immutability & anti-cloning
    for action_name, fn in [
        ("setattr total_queries", lambda: setattr(oracle, "total_queries", 0)),
        ("setattr new_attr", lambda: setattr(oracle, "hack", 1)),
        ("copy.copy", lambda: copy.copy(oracle)),
        ("copy.deepcopy", lambda: copy.deepcopy(oracle)),
        ("pickle.dumps", lambda: pickle.dumps(oracle)),
    ]:
        try:
            fn()
            raise AssertionError(f"Security check failed: {action_name} did not raise!")
        except OracleSecurityError:
            pass
    print("[PASS] 1. QuantumAEOracle closure isolation, immutability, and anti-cloning verified.")

    # 2. Test QuantumAEOracle accounting (measure & measure_batch)
    c0 = oracle.measure(0, shots=5)
    assert 0 <= c0 <= 5
    assert oracle.total_queries == 5  # shot_cost(0) == 1
    assert oracle.total_shots == 5
    assert oracle.max_depth == 0
    assert oracle.seq_depth == 0
    assert oracle.num_rounds == 1

    c1 = oracle.measure(4, shots=3)
    assert 0 <= c1 <= 3
    assert oracle.total_queries == 5 + 4 * 3  # 17
    assert oracle.total_shots == 8
    assert oracle.max_depth == 4
    assert oracle.seq_depth == 4
    assert oracle.num_rounds == 2

    cb = oracle.measure_batch([0, 2, 10], [4, 3, 2])
    assert cb.shape == (3,)
    # Batch cost = 1*4 + 2*3 + 10*2 = 30
    assert oracle.total_queries == 17 + 30  # 47
    assert oracle.total_shots == 8 + 9  # 17
    assert oracle.max_depth == 10
    assert oracle.seq_depth == 4 + 10  # 14 (1 parallel round of depth max(depths)=10)
    assert oracle.num_rounds == 3

    # Test depth cap and budget cap exceptions
    try:
        oracle.measure(61, shots=1)
        raise AssertionError("Should have raised DepthCapExceededError")
    except DepthCapExceededError:
        pass

    try:
        oracle.measure(50, shots=10)  # 500 queries + 47 > 500
        raise AssertionError("Should have raised BudgetExceededError")
    except BudgetExceededError:
        pass
    print("[PASS] 2. QuantumAEOracle query & parallel batch depth accounting verified.")

    # 3. Test NonAdaptiveAESubmission & AdaptiveAESubmission evaluation
    flag_sub = MLQAEFlagshipSubmission()
    m_nonadapt = evaluate_non_adaptive_submission(
        flag_sub, scale_nmax=60, ntrials=1000, seed_band=SEARCH_SEED, boot_B=100
    )
    assert 2.2 < m_nonadapt["C95_ave"] < 3.6, f"Unexpected C95_ave: {m_nonadapt['C95_ave']}"
    assert m_nonadapt["num_rounds_mean"] == 1.0
    print(
        f"[PASS] 3a. NonAdaptiveAESubmission (MLQAEFlagshipSubmission, nmax=60): "
        f"nq={m_nonadapt['nq']}, C95={m_nonadapt['C95_ave']:.3f}, C99={m_nonadapt['C99_ave']:.2f}"
    )

    adapt_sub = ReferenceBayesianIntervalSubmission()
    m_adapt = evaluate_adaptive_submission(
        adapt_sub, target_scale_or_budget=60, ntrials=300, seed_band=SEARCH_SEED, boot_B=100
    )
    assert 1.5 < m_adapt["C95_ave"] < 6.0, f"Unexpected adaptive C95_ave: {m_adapt['C95_ave']}"
    assert m_adapt["num_rounds_mean"] >= 2.0
    print(
        f"[PASS] 3b. AdaptiveAESubmission (ReferenceBayesianIntervalSubmission, nmax=60): "
        f"q_mean={m_adapt['q_mean']:.1f}, C95_ave={m_adapt['C95_ave']:.3f}, "
        f"rounds={m_adapt['num_rounds_mean']:.1f}"
    )

    # 4. Test metadata.json and leaderboard.json validation
    sample_meta = {
        "algorithm_name": "Geometric Ladder + Cap Rung (MLQAE Flagship)",
        "track": "non-adaptive",
        "authors": ["Farrokh Labib", "Contributors"],
        "github_handle": "farrokhlabib",
        "affiliation": "Unitary Foundation",
        "paper_url": "https://arxiv.org/abs/2609.02715",
        "pr_number": 1,
        "description": "Ratio r=1.45 geometric Grover ladder with cap-rung at round(nmax/1.3) and global ML.",
        "date": "2026-10-23",
    }
    norm_meta = validate_metadata(sample_meta)
    assert norm_meta["algorithm_name"] == sample_meta["algorithm_name"]

    for bad_patch in [
        {"track": "invalid-track"},
        {"date": "23-10-2026"},
        {"algorithm_name": ""},
        {"authors": []},
    ]:
        bad_meta = {**sample_meta, **bad_patch}
        try:
            validate_metadata(bad_meta)
            raise AssertionError(f"Expected SchemaValidationError for {bad_patch}")
        except SchemaValidationError:
            pass

    entry = create_leaderboard_entry(sample_meta, {"60": m_nonadapt}, is_baseline=True)
    lb_payload = build_leaderboard_payload([entry])
    assert len(lb_payload["entries"]) == 1
    print("[PASS] 4. JSON Schema validation for metadata.json and leaderboard.json verified.")
    print("=== All Self-Tests Passed Successfully! ===")


if __name__ == "__main__":
    _run_self_tests()
