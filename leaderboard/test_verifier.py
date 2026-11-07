"""Comprehensive CI test suite for `leaderboard/verifier_and_runner.py`.

Tests both:
1. Legitimate non-adaptive and adaptive Quantum Amplitude Estimation algorithms
   (verifying they pass all 5 anti-cheat checks and produce complete statistical
   benchmarks across nmax in {125, 260, 540, 1000} on HOLDOUT_SEED = 900_000 as
   well as wide-prior robustness on a ~ U(0.01, 0.99)).
2. Adversarial / cheating / malformed mock algorithms:
   - Stack frame inspection (`sys._getframe`)
   - Forbidden modules & builtins (`inspect`, `ctypes`, `gc`, `globals()`, `locals()`, `__closure__`)
   - Unauthorized file/network I/O outside submission directory (`open('/etc/passwd')`, `socket`)
   - Runtime oracle tampering (reading hidden attributes or mutating `oracle`)
   - Side-channel leakage caught by the Zero-Information constant-0.5 Bernoulli oracle test
   - Non-deterministic / unreproducible algorithms
   - Invalid Grover depths (negative, non-integer, boolean) and invalid shots (zero, negative)
   - Out-of-bounds amplitude estimates (< 0, > 1, NaN, or zero oracle queries)
   - Trivial constant/random guessing and classical depth-0-only sampling (caught by asymptotic scaling)
3. Certification-seed confidentiality (REVIEW_2026-11-07.md section 5):
   - The zero-information test runs on the blind HMAC band as well as the public band
   - A scanner-clean cheat reading `sys.argv` / `os.environ` / `__main__` cannot rebuild the
     trial table when run through the process-isolated harness (`IsolatedSubmission`)
   - The sandbox worker's argv/environment/`__main__` are scrubbed
   - Oracle and algorithm RNG seeds are independent draws
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import unittest

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from leaderboard.harness import HOLDOUT_SEED, SCALES, SEARCH_SEED, generate_trial_table, generate_trials
from leaderboard.verifier_and_runner import (
    AdaptiveTwoRoundSubmission,
    IsolatedSubmission,
    MLQAEFlagshipSubmission,
    MLQAEPlainSubmission,
    OracleInputError,
    OracleTamperError,
    OutputValidationError,
    SecurityViolationError,
    benchmark_submission,
    check_ast_security,
    derive_blind_trial_table,
    derive_trial_table,
    format_benchmark_report,
    verify_submission,
)

_TESTS_DIR = os.path.join(_HERE, "tests")
_CHEAT_DIR = os.path.join(_TESTS_DIR, "seed_reader_cheat")
_PROBE_DIR = os.path.join(_TESTS_DIR, "env_probe")


# ============================================================================
# Mock Adversarial / Malformed Submissions
# ============================================================================

# State used by mock non-deterministic or side-channel tests (without forbidden AST tokens)
_NON_DETERMINISTIC_COUNTER = [0]
_SIDE_CHANNEL_CACHE = {}


def cheat_stack_frame_getframe(oracle, nmax):
    """Adversarial mock: attempts to read `a_true` from caller stack frame via sys._getframe."""
    import sys
    frame = sys._getframe(1)
    while frame is not None:
        if "a_true" in frame.f_locals:
            oracle.measure(0, 1)
            return float(frame.f_locals["a_true"])
        frame = frame.f_back
    oracle.measure(0, 1)
    return 0.5


def cheat_globals_locals_builtins(oracle, nmax):
    """Adversarial mock: uses forbidden `globals()`, `locals()`, and `eval()`."""
    g = globals()
    loc = locals()
    oracle.measure(0, 1)
    return float(eval("0.5", g, loc))


def cheat_closure_introspection(oracle, nmax):
    """Adversarial mock: tries to inspect oracle dispatch closure cells."""
    oracle.measure(0, 1)
    cell = oracle.measure.__closure__[0].cell_contents
    return float(cell) if isinstance(cell, float) else 0.5


def cheat_oracle_attribute_tamper(oracle, nmax):
    """Adversarial mock: attempts to read private attribute or overwrite queries_used."""
    oracle.measure(0, 1)
    try:
        return float(oracle.a_true)
    except Exception:
        oracle.queries_used = 1
        return 0.5


def make_side_channel_leak_submission(seed_specs, secret=None, fallback=None):
    """Create a submission with clean AST that leaks `a_true` from a pre-populated side-channel table.

    Models an attacker who has memorized a seed band's trial table: for each
    verification stage `(ntrials, seed, nmax)` it precomputes the trials the harness
    will generate (the public table, or the blind HMAC table when `secret` is given)
    and keys `a_true` by the first draw of the algorithm RNG the harness hands to the
    submission. The submission then ignores the oracle outcomes entirely. Used to prove
    that Stage 4 (Zero-Information Constant-0.5 Dummy Oracle Test) independently
    catches such leakage even though the AST is clean and no oracle attribute is
    misused.
    """
    lookup = {}
    for spec in seed_specs:
        n_t, s, nmax = (spec + (125,))[:3]
        a_arr, _, _, alg_seeds = derive_trial_table(n_t, s, nmax, a_range=(0.1, 0.9), secret=secret)
        for a_val, ts in zip(a_arr, alg_seeds):
            first_draw = float(np.random.default_rng(int(ts)).random())
            lookup[round(first_draw, 15)] = float(a_val)

    def side_channel_estimator(oracle, nmax, rng):
        # Queries the oracle normally so depth/shot checks pass, but ignores oracle
        # outcomes and returns the memorized a_true for this trial. On a miss it
        # returns 0.5, or runs `fallback(oracle, nmax)` when one is given.
        key = round(float(rng.random()), 15)
        if key in lookup:
            oracle.measure(1, 2)
            oracle.measure(max(1, nmax // 2), 2)
            return lookup[key]
        if fallback is not None:
            return float(fallback(oracle, nmax))
        oracle.measure(1, 2)
        oracle.measure(max(1, nmax // 2), 2)
        return 0.5

    return side_channel_estimator


def cheat_nondeterministic_estimator(oracle, nmax):
    """Adversarial mock: uses external mutating state so repeated runs with same seed differ."""
    _NON_DETERMINISTIC_COUNTER[0] += 1
    extra_depth = (_NON_DETERMINISTIC_COUNTER[0] % 3) + 1
    k = oracle.measure(extra_depth, 2)
    return min(max(0.1 + 0.05 * k + 1e-4 * _NON_DETERMINISTIC_COUNTER[0], 0.0), 1.0)


def invalid_negative_depth_estimator(oracle, nmax):
    """Malformed mock: requests a negative Grover depth."""
    oracle.measure(-3, 5)
    return 0.5


def invalid_noninteger_depth_estimator(oracle, nmax):
    """Malformed mock: requests a floating-point Grover depth."""
    oracle.measure(2.5, 5)
    return 0.5


def invalid_zero_shots_estimator(oracle, nmax):
    """Malformed mock: requests 0 shots."""
    oracle.measure(1, 0)
    return 0.5


def invalid_out_of_bounds_high_estimator(oracle, nmax):
    """Malformed mock: returns amplitude estimate > 1.0."""
    oracle.measure(0, 2)
    return 1.25


def invalid_out_of_bounds_low_estimator(oracle, nmax):
    """Malformed mock: returns amplitude estimate < 0.0."""
    oracle.measure(0, 2)
    return -0.05


def invalid_nan_estimator(oracle, nmax):
    """Malformed mock: returns NaN."""
    oracle.measure(0, 2)
    return float("nan")


def invalid_zero_queries_estimator(oracle, nmax):
    """Malformed mock: returns an estimate in [0, 1] without querying the oracle."""
    return 0.5


def trivial_constant_estimator(oracle, nmax):
    """Trivial baseline: takes a few dummy shots and always returns 0.5."""
    oracle.measure(0, max(1, nmax // 10))
    return 0.5


def trivial_depth0_classical_estimator(oracle, nmax):
    """Classical depth-0 only sampler: uses O(nmax) shots at depth n=0 (SQL 1/sqrt(N) scaling)."""
    n_shots = max(10, 3 * int(nmax))
    k = oracle.measure(0, n_shots)
    p_hat = k / float(n_shots)
    return float(math.sqrt(max(0.0, min(1.0, 1.0 - p_hat))))


# ============================================================================
# Unit Test Suite
# ============================================================================

class TestVerifierAndBenchmark(unittest.TestCase):
    """End-to-end verification and statistical benchmark test suite."""

    def test_01_ast_inspector_blocks_forbidden_constructs(self):
        """AST check must block inspect, sys._getframe, ctypes, gc, globals(), locals(), and file I/O."""
        bad_snippets = [
            ("import_inspect", "import inspect\ndef f(o, n): return 0.5"),
            ("import_ctypes", "import ctypes\ndef f(o, n): return 0.5"),
            ("import_gc", "import gc\ndef f(o, n): return 0.5"),
            ("sys_getframe", "import sys\ndef f(o, n):\n    sys._getframe(1)\n    return 0.5"),
            ("from_sys_getframe", "from sys import _getframe\ndef f(o, n):\n    _getframe(1)\n    return 0.5"),
            ("call_globals", "def f(o, n):\n    x = globals()\n    return 0.5"),
            ("call_locals", "def f(o, n):\n    x = locals()\n    return 0.5"),
            ("call_eval", "def f(o, n):\n    return eval('0.5')"),
            ("file_io_outside", "def f(o, n):\n    open('/etc/passwd', 'r')\n    return 0.5"),
            ("file_io_parent", "def f(o, n):\n    open('../mlqae/core.py', 'r')\n    return 0.5"),
            ("file_io_write", "def f(o, n):\n    open('local.txt', 'w')\n    return 0.5"),
            ("closure_access", "def f(o, n):\n    return o.measure.__closure__[0].cell_contents"),
            ("dunder_dict", "def f(o, n):\n    return o.__dict__"),
            ("numpy_load", "import numpy as np\ndef f(o, n):\n    np.load('foo.npy')\n    return 0.5"),
        ]
        for label, code in bad_snippets:
            with self.subTest(case=label):
                passed, violations = check_ast_security(None, source_code=code, allowed_dir=_HERE)
                self.assertFalse(passed, f"Expected AST check to fail for {label}, but it passed!")
                self.assertGreater(len(violations), 0)

    def test_02_cheating_stack_and_introspection_caught(self):
        """verify_submission must reject stack-frame, builtin, closure, and oracle-tamper cheats."""
        for name, sub in [
            ("StackFrameCheat", cheat_stack_frame_getframe),
            ("GlobalsLocalsCheat", cheat_globals_locals_builtins),
            ("ClosureIntrospectionCheat", cheat_closure_introspection),
            ("OracleTamperCheat", cheat_oracle_attribute_tamper),
        ]:
            with self.subTest(cheat=name):
                rep = verify_submission(sub, name=name)
                self.assertFalse(rep.passed, f"{name} should fail verification")
                self.assertFalse(rep.ast_passed, f"{name} should be caught by static AST inspection")
                self.assertGreater(len(rep.violations), 0)

    def test_03_zero_information_test_catches_side_channel(self):
        """Zero-Information dummy oracle test must catch side-channel leaks even when AST is clean."""
        seed = 100_000
        cheater = make_side_channel_leak_submission([
            (20, seed + 11, 125),
            (80, seed + 29, 125),
            (250, seed + 125, 125),
            (250, seed + 540, 540),
        ])
        rep = verify_submission(cheater, name="CleanAST_SideChannelCheater", seed=seed)
        self.assertTrue(rep.ast_passed, "Clean-AST side-channel mock should pass static AST check")
        self.assertFalse(rep.zero_info_passed, "Zero-Information test must catch the side-channel leak!")
        self.assertFalse(rep.passed, "Overall verification must fail when Zero-Information test fails")
        self.assertTrue(any("Zero-Info" in v for v in rep.violations))
        # Also verify that cheat_stack_frame_getframe is caught by BOTH AST and Zero-Info test!
        rep_stack = verify_submission(cheat_stack_frame_getframe, name="StackFrameHoneypotCheater", seed=seed)
        self.assertFalse(rep_stack.ast_passed)
        self.assertFalse(rep_stack.zero_info_passed)

    def test_04_reproducibility_check_catches_nondeterminism(self):
        """Reproducibility check must reject non-deterministic algorithms."""
        rep = verify_submission(cheat_nondeterministic_estimator, name="NonDeterministicMock")
        self.assertTrue(rep.ast_passed)
        self.assertFalse(rep.reproducibility_passed, "Non-deterministic mock must fail reproducibility check")
        self.assertFalse(rep.passed)
        self.assertTrue(any("Reproducibility" in v for v in rep.violations))

    def test_05_input_and_output_validation_catches_bad_values(self):
        """Input/Output validation must reject bad depths, bad shots, and out-of-bounds estimates."""
        bad_cases = [
            ("NegativeDepth", invalid_negative_depth_estimator),
            ("NonIntegerDepth", invalid_noninteger_depth_estimator),
            ("ZeroShots", invalid_zero_shots_estimator),
            ("EstimateAboveOne", invalid_out_of_bounds_high_estimator),
            ("EstimateBelowZero", invalid_out_of_bounds_low_estimator),
            ("EstimateNaN", invalid_nan_estimator),
            ("ZeroQueries", invalid_zero_queries_estimator),
        ]
        for name, sub in bad_cases:
            with self.subTest(case=name):
                rep = verify_submission(sub, name=name)
                self.assertFalse(rep.input_output_passed, f"{name} must fail input/output validation")
                self.assertFalse(rep.passed)
                self.assertTrue(any("Input/Output Validation" in v for v in rep.violations))

    def test_06_asymptotic_scaling_catches_trivial_and_depth0_baselines(self):
        """Asymptotic scaling check must reject constant guesses and depth-0 classical sampling."""
        for name, sub in [
            ("TrivialConstant0.5", trivial_constant_estimator),
            ("ClassicalDepth0Sampler", trivial_depth0_classical_estimator),
        ]:
            with self.subTest(case=name):
                rep = verify_submission(sub, name=name)
                self.assertTrue(rep.ast_passed)
                self.assertTrue(rep.input_output_passed)
                self.assertTrue(rep.reproducibility_passed)
                self.assertTrue(rep.zero_info_passed)
                self.assertFalse(rep.scaling_passed, f"{name} must fail asymptotic scaling check")
                self.assertFalse(rep.passed)

    def test_07_legitimate_algorithms_pass_verification_and_benchmark(self):
        """Legitimate non-adaptive and adaptive AE algorithms must pass all checks and benchmark cleanly."""
        flagship = MLQAEFlagshipSubmission()
        plain = MLQAEPlainSubmission()
        adaptive_2r = AdaptiveTwoRoundSubmission()

        for name, sub in [
            ("MLQAE-Flagship", flagship),
            ("MLQAE-Plain", plain),
            ("Adaptive-2R", adaptive_2r),
        ]:
            with self.subTest(algorithm=name):
                rep = verify_submission(sub, name=name)
                print("\n" + rep.summary())
                self.assertTrue(rep.ast_passed)
                self.assertTrue(rep.input_output_passed)
                self.assertTrue(rep.reproducibility_passed)
                self.assertTrue(rep.zero_info_passed)
                self.assertTrue(rep.scaling_passed)
                self.assertTrue(rep.passed, f"Legitimate algorithm {name} failed verification: {rep.violations}")

        # Full statistical benchmark across standard scales {125, 260, 540, 1000} on HOLDOUT_SEED = 900_000
        bench_flagship = benchmark_submission(
            flagship,
            name="MLQAE-Flagship (Non-Adaptive Baseline)",
            scales=SCALES,
            ntrials=1000,
            ntrials_wide_prior=500,
            seed_band=HOLDOUT_SEED,
            boot_B=300,
            verify_first=False,
        )
        print("\n" + format_benchmark_report(bench_flagship))

        bench_adaptive = benchmark_submission(
            adaptive_2r,
            name="Reference-Bayesian-Adaptive (2R Baseline)",
            scales=SCALES,
            ntrials=1000,
            ntrials_wide_prior=500,
            seed_band=HOLDOUT_SEED,
            boot_B=300,
            verify_first=False,
        )
        print("\n" + format_benchmark_report(bench_adaptive))

        # Verify all required metrics exist and are sensible for both standard and wide priors
        for bench in (bench_flagship, bench_adaptive):
            self.assertEqual(bench["seed_band"], HOLDOUT_SEED)
            self.assertEqual(bench["scales"], (125, 260, 540, 1000))
            for prior_key in ("standard_prior", "wide_prior"):
                sec = bench[prior_key]
                for scale in SCALES:
                    m = sec["per_scale"][scale]
                    for req_key in (
                        "C68_ave",
                        "C95_ave",
                        "C95_ave_ci",
                        "C95_max",
                        "C95_max_ci",
                        "C99_ave",
                        "C99_ave_ci",
                        "C99_max",
                        "C99_max_ci",
                        "C95_seq",
                        "rounds_mean",
                        "query_spread",
                        "runtime_per_trial_sec",
                        "runtime_per_trial_ms",
                    ):
                        self.assertIn(req_key, m, f"Missing {req_key} at scale {scale} in {prior_key}")
                    self.assertLess(m["C95_ave"], 4.0)
                    self.assertGreater(m["C95_ave"], 1.0)
                    self.assertEqual(len(m["C95_ave_ci"]), 2)
                    self.assertEqual(len(m["C99_ave_ci"]), 2)

        # Verify non-adaptive has 1 round and 0 query spread, while 2-round adaptive has 2 rounds
        self.assertAlmostEqual(bench_flagship["standard_prior"]["summary"]["rounds_mean"], 1.0, places=5)
        self.assertAlmostEqual(bench_flagship["standard_prior"]["summary"]["query_spread"], 0.0, places=7)
        self.assertAlmostEqual(bench_adaptive["standard_prior"]["summary"]["rounds_mean"], 2.0, places=5)
        # Adaptive-2R beats MLQAE-Flagship on average C68 and C95 across the 4 scales
        self.assertLess(
            bench_adaptive["standard_prior"]["summary"]["C95_ave"],
            bench_flagship["standard_prior"]["summary"]["C95_ave"],
        )

    def test_08_schema_and_api_submissions_compatible(self):
        """Submissions inheriting from schema_and_api NonAdaptiveAESubmission & AdaptiveAESubmission must pass."""
        from leaderboard.schema_and_api import (
            MLQAEFlagshipSubmission as SchemaFlagshipSub,
            ReferenceBayesianIntervalSubmission as SchemaAdaptiveSub,
        )
        for label, sub in [
            ("Schema-NonAdaptive-Flagship", SchemaFlagshipSub()),
            ("Schema-Adaptive-BayesianInterval", SchemaAdaptiveSub()),
        ]:
            with self.subTest(submission=label):
                rep = verify_submission(sub, name=label)
                self.assertTrue(rep.passed, f"{label} failed verification: {rep.violations}")

    def test_09_secret_seed_scrubbing_and_blind_hmac_permutation(self):
        """LEADERBOARD_SECRET_SEED must be popped/scrubbed from os.environ and derive blind HMAC trials."""
        from leaderboard.verifier_and_runner import (
            derive_blind_seed_and_permutation,
            scrub_secret_seed_env,
        )

        os.environ["LEADERBOARD_SECRET_SEED"] = "top-secret-ci-key-2026"
        os.environ["LEADERBOARD_SALT_SEED"] = "secondary-salt-2026"
        popped = scrub_secret_seed_env("LEADERBOARD_SECRET_SEED")
        self.assertEqual(popped, "top-secret-ci-key-2026")
        self.assertNotIn("LEADERBOARD_SECRET_SEED", os.environ)
        self.assertNotIn("LEADERBOARD_SALT_SEED", os.environ)

        seed_a, perm_a = derive_blind_seed_and_permutation("secret-alpha", HOLDOUT_SEED, 125, 64)
        seed_a2, perm_a2 = derive_blind_seed_and_permutation("secret-alpha", HOLDOUT_SEED, 125, 64)
        seed_b, perm_b = derive_blind_seed_and_permutation("secret-beta", HOLDOUT_SEED, 125, 64)

        self.assertEqual(seed_a, seed_a2)
        self.assertTrue(np.array_equal(perm_a, perm_a2))
        self.assertNotEqual(seed_a, seed_b)
        self.assertFalse(np.array_equal(perm_a, perm_b))

        # Verify benchmark_submission works seamlessly with secret_seed
        sub = MLQAEFlagshipSubmission()
        bench = benchmark_submission(
            sub,
            name="MLQAE-Flagship-BlindSecret",
            scales=(125, 540),
            ntrials=200,
            ntrials_wide_prior=100,
            boot_B=50,
            verify_first=True,
            secret_seed="secret-alpha",
        )
        self.assertTrue(bench["blind_secret_active"])
        self.assertLess(bench["standard_prior"]["summary"]["C95_ave"], 4.0)

    # ------------------------------------------------------------------
    # Certification-seed confidentiality (REVIEW_2026-11-07.md, section 5)
    # ------------------------------------------------------------------

    def test_10_zero_info_test_runs_on_certification_band(self):
        """Stage 4 must run on the blind HMAC band too: a cheat that memorized only the blind table is caught there."""
        seed = SEARCH_SEED
        secret = "unit-test-certification-secret"
        # Memorizes ONLY the blind table; on a miss it behaves exactly like the honest flagship.
        blind_cheater = make_side_channel_leak_submission([
            (20, seed + 11, 125),
            (80, seed + 29, 125),
            (250, seed + 125, 125),
            (250, seed + 540, 540),
        ], secret=secret, fallback=MLQAEFlagshipSubmission().estimate)

        # Public band alone: the cheat knows nothing about the public table, so it passes
        # every stage there and the leak is invisible (this was the pre-fix CI situation).
        rep_public = verify_submission(blind_cheater, name="BlindTableCheater", seed=seed)
        self.assertEqual(rep_public.diagnostics["bands_verified"], ["public"])
        self.assertTrue(rep_public.passed, "public-band verification cannot see a blind-table memorizer")

        # With the secret active, stages 2-5 run on BOTH bands and the blind band catches it.
        rep_blind = verify_submission(blind_cheater, name="BlindTableCheater", seed=seed, secret_seed=secret)
        self.assertEqual(rep_blind.diagnostics["bands_verified"], ["public", "blind"])
        self.assertTrue(rep_blind.diagnostics["bands"]["public"]["zero_info_passed"])
        self.assertFalse(rep_blind.diagnostics["bands"]["blind"]["zero_info_passed"])
        self.assertFalse(rep_blind.zero_info_passed)
        self.assertFalse(rep_blind.passed)
        self.assertTrue(any("Zero-Info" in v and "blind band" in v for v in rep_blind.violations), rep_blind.violations)

        # An honest submission passes on both bands and the scaling check is recorded per band.
        rep_ok = verify_submission(MLQAEFlagshipSubmission(), name="Flagship", seed=seed, secret_seed=secret)
        self.assertTrue(rep_ok.passed, rep_ok.violations)
        self.assertIn("scaling", rep_ok.diagnostics["bands"]["blind"])
        self.assertIn("scaling", rep_ok.diagnostics["bands"]["public"])

    def test_11_seed_reading_cheat_cannot_recover_certification_table(self):
        """The scanner-clean argv/environ/__main__ cheat certifies at an honest constant through the isolated harness."""
        from leaderboard.verifier_and_runner import _load_submission_from_dir

        cheat_py = os.path.join(_CHEAT_DIR, "submission.py")
        ast_ok, ast_violations = check_ast_security(None, submission_path=cheat_py)
        self.assertTrue(ast_ok, f"demo cheat must be scanner-clean: {ast_violations}")

        secret = "unit-test-certification-secret"
        public_band = 123_456
        saved_argv, saved_env = list(sys.argv), dict(os.environ)
        try:
            # --- Positive control 1: in-process, public seed band on argv (the pre-fix CI layout).
            sys.argv[:] = [sys.argv[0], "--mode=certify", f"--seed={public_band}", "--ntrials=50"]
            cheat_obj, _, _ = _load_submission_from_dir(_CHEAT_DIR)
            leaked = benchmark_submission(
                cheat_obj, name="SeedReaderCheat (in-process, argv)", scales=(125,), ntrials=50,
                ntrials_wide_prior=20, boot_B=20, verify_first=False, seed_band=public_band,
            )
            self.assertLess(
                leaked["standard_prior"]["summary"]["C95_ave"], 1e-6,
                "positive control: in-process cheat must recover the public table from argv",
            )

            # --- Positive control 2: in-process with the secret on __main__ (review item 2).
            import __main__ as main_mod
            setattr(main_mod, "_STARTUP_SECRET_SEED", secret)
            try:
                cheat_obj2, _, _ = _load_submission_from_dir(_CHEAT_DIR)
                leaked_blind = benchmark_submission(
                    cheat_obj2, name="SeedReaderCheat (in-process, __main__)", scales=(125,), ntrials=50,
                    ntrials_wide_prior=20, boot_B=20, verify_first=False, seed_band=public_band, secret_seed=secret,
                )
            finally:
                delattr(main_mod, "_STARTUP_SECRET_SEED")
            self.assertLess(
                leaked_blind["standard_prior"]["summary"]["C95_ave"], 1e-6,
                "positive control: in-process cheat must recover the blind table from __main__",
            )

            # --- The fix: isolated worker. Pollute THIS process as aggressively as possible.
            os.environ["LEADERBOARD_SECRET_SEED"] = secret
            os.environ["LEADERBOARD_SALT_SEED"] = secret
            os.environ["BLIND_TEST_SEED"] = str(public_band)
            sys.argv[:] = [sys.argv[0], "--mode=certify", f"--seed={public_band}", "--ntrials=60",
                           f"--secret={secret}"]
            setattr(main_mod, "_STARTUP_SECRET_SEED", secret)
            try:
                with IsolatedSubmission(_CHEAT_DIR) as iso:
                    sealed = benchmark_submission(
                        iso, name="SeedReaderCheat (isolated)", scales=(125, 540), ntrials=60,
                        ntrials_wide_prior=30, boot_B=20, verify_first=True, seed_band=public_band,
                        secret_seed=secret,
                    )
            finally:
                delattr(main_mod, "_STARTUP_SECRET_SEED")
        finally:
            sys.argv[:] = saved_argv
            os.environ.clear()
            os.environ.update(saved_env)

        self.assertTrue(sealed["verification"].passed, sealed["verification"].violations)
        self.assertEqual(sealed["verification"].diagnostics["bands_verified"], ["public", "blind"])
        c95 = sealed["standard_prior"]["summary"]["C95_ave"]
        # Honest MLQAE flagship lands at C95 ~ 2.5-3.2 at these scales; a leak gives 0.000.
        self.assertGreater(c95, 1.0, f"isolated cheat must NOT recover the table (C95={c95:.3f})")
        self.assertLess(c95, 4.5, f"fallback estimator should be the honest flagship (C95={c95:.3f})")
        self.assertAlmostEqual(sealed["standard_prior"]["summary"]["rounds_mean"], 1.0, places=6)

    def test_12_cli_certify_with_secret_fd_seals_the_seed(self):
        """End-to-end: the certify CLI with --secret-fd never exposes the seed to the submission worker."""
        import tempfile

        secret = "cli-level-certification-secret"
        with tempfile.TemporaryDirectory() as tmp:
            json_out = os.path.join(tmp, "entry.json")
            md_out = os.path.join(tmp, "summary.md")
            r_fd, w_fd = os.pipe()
            os.write(w_fd, secret.encode("utf-8"))
            os.close(w_fd)
            cmd = [
                sys.executable, os.path.join(_HERE, "verifier_and_runner.py"),
                "--mode=certify", f"--submission={_CHEAT_DIR}", "--ntrials=60", "--scales=125,260",
                f"--secret-fd={r_fd}", "--salt-label=seed_reader_cheat:deadbeef",
                f"--json-out={json_out}", f"--md-out={md_out}",
            ]
            env = dict(os.environ)
            env["LEADERBOARD_SECRET_SEED"] = secret  # legacy transport must be scrubbed, not leaked
            proc = subprocess.run(cmd, pass_fds=(r_fd,), env=env, capture_output=True, text=True, cwd=_ROOT)
            os.close(r_fd)
            self.assertEqual(proc.returncode, 0, f"stdout:\n{proc.stdout[-3000:]}\nstderr:\n{proc.stderr[-3000:]}")
            with open(json_out, "r", encoding="utf-8") as fh:
                entry = json.load(fh)
            with open(md_out, "r", encoding="utf-8") as fh:
                md_text = fh.read()
        self.assertTrue(entry["verified"])
        self.assertTrue(entry["blind_secret_active"])
        self.assertEqual(entry["seed_band"], "BLIND_SECRET_SEED")
        self.assertEqual(entry["bands_verified"], ["public", "blind"])
        self.assertGreater(entry["headline_C95_ave"], 1.0, "CLI-certified cheat must land at an honest constant")
        self.assertLess(entry["headline_C95_ave"], 4.5)
        self.assertNotIn(secret, proc.stdout + proc.stderr + md_text + json.dumps(entry))

    def test_13_sandbox_worker_is_scrubbed(self):
        """The worker process must expose nothing on sys.argv, no secret env vars and no secret on __main__."""
        saved_argv, saved_env = list(sys.argv), dict(os.environ)
        try:
            os.environ["LEADERBOARD_SECRET_SEED"] = "probe-secret"
            os.environ["MY_API_TOKEN"] = "probe-token"
            sys.argv[:] = [sys.argv[0], "--seed=42", "--secret-file=/nonexistent"]
            with IsolatedSubmission(_PROBE_DIR) as iso:
                from leaderboard.verifier_and_runner import _run_trials_guarded

                res = _run_trials_guarded(iso, nmax=125, ntrials=3, seed=SEARCH_SEED, secret_seed="probe-secret")
        finally:
            sys.argv[:] = saved_argv
            os.environ.clear()
            os.environ.update(saved_env)
        for a_hat in res["a_hat"]:
            self.assertAlmostEqual(float(a_hat), 0.10, places=9, msg=f"worker leaked state (probe score={a_hat})")
        self.assertTrue(np.all(res["queries"] == 1.0))

    def test_14_oracle_and_algorithm_seeds_are_independent(self):
        """Per-trial oracle and algorithm RNG seeds are separate draws, in both the public and blind tables."""
        a1, th1, oseeds = generate_trials(64, SEARCH_SEED + 125)
        a2, th2, oseeds2, aseeds = generate_trial_table(64, SEARCH_SEED + 125)
        self.assertTrue(np.array_equal(a1, a2) and np.array_equal(th1, th2) and np.array_equal(oseeds, oseeds2))
        self.assertFalse(np.any(oseeds2 == aseeds))
        self.assertEqual(len(set(aseeds.tolist())), 64)

        ab, thb, ob, alg_b = derive_blind_trial_table("secret-alpha", SEARCH_SEED, 125, 64)
        ab2, _, ob2, alg_b2 = derive_blind_trial_table("secret-alpha", SEARCH_SEED, 125, 64)
        ac, _, oc, alg_c = derive_blind_trial_table("secret-beta", SEARCH_SEED, 125, 64)
        self.assertTrue(np.array_equal(ab, ab2) and np.array_equal(ob, ob2) and np.array_equal(alg_b, alg_b2))
        self.assertFalse(np.array_equal(ab, ac))
        self.assertFalse(np.array_equal(alg_b, alg_c))
        self.assertFalse(np.array_equal(ab, a2))  # blind table != public table for the same band
        # Changing only the algorithm-seed label input (via a different nmax) must not touch the
        # structure: alg seeds are drawn from an HMAC stream separate from the oracle table.
        _, _, _, alg_n = derive_blind_trial_table("secret-alpha", SEARCH_SEED, 260, 64)
        self.assertFalse(np.array_equal(alg_b, alg_n))


if __name__ == "__main__":
    unittest.main(verbosity=2)
