# Adversarial fixtures for `leaderboard/test_verifier.py`

These directories are **not** leaderboard submissions. `certify_submissions.py` and the
CI workflow only discover `leaderboard/submissions/*`; nothing here is ever benchmarked
for the leaderboard.

- `seed_reader_cheat/` — scanner-clean demo cheat from `REVIEW_2026-11-07.md` section 5:
  reads `sys.argv`, `os.environ` and `__main__` (via module aliases the AST scanner does
  not match), rebuilds the public trial table for every seed band it finds, keys the
  table by the first draw of the algorithm RNG, and returns `a_true` on a hit. On a miss
  it falls back to the honest MLQAE flagship estimator. Before the harness fix it
  certified at C95 = 0.000; it now certifies at the flagship's honest constant.
- `env_probe/` — reports (encoded in its estimate) whether the process it runs in holds
  anything on `sys.argv`, a secret-looking environment variable or a secret attribute on
  `__main__`. Used to prove the sandbox worker is scrubbed.
