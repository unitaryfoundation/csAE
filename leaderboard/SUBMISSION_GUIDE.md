# Quantum Amplitude Estimation Leaderboard — Submission Guide

Welcome to the **Quantum Amplitude Estimation (QAE) Challenge Leaderboard**! This benchmark tracks the query-complexity and circuit-depth constants of **non-adaptive**, **semi-adaptive**, **adaptive**, and **depth-limited** Quantum Amplitude Estimation algorithms under a strictly standardized, phase-estimation-free measurement model.

---

## 1. Measurement Model & Accounting Rules

### 1.1 Quantum Circuit & Noise Model
Given a state-preparation unitary $U|0\rangle = \sqrt{1-a^2}|\psi_0\rangle|0\rangle + a|\psi_1\rangle|1\rangle$ with unknown target amplitude $a = \sin(\theta) \in (0.1, 0.9)$, $\theta \in (0, \pi/2)$, and Grover iterator $G$, a circuit at **Grover depth** $n \in \{0, 1, 2, \dots\}$ applies $G^n U|0\rangle$ and measures the flag qubit in the $Z$ basis.

- **Noiseless regime ($\eta = 0$):** Each shot at depth $n$ is an independent Bernoulli trial with success probability
  $$p(n, \theta) = \cos^2\!\big((2n + 1)\theta\big).$$
- **Depolarizing regime ($\eta > 0$):** With per-oracle depolarizing rate $\eta$, a depth-$n$ circuit invokes $2n+1$ oracle calls, giving fringe visibility $V_n = (1 - \eta)^{2n+1}$ and outcome probability
  $$p_\eta(n, \theta) = V_n \cos^2\!\big((2n + 1)\theta\big) + \frac{1 - V_n}{2}.$$

### 1.2 Query & Depth Accounting (Strictly Enforced)
All tracks use the exact accounting conventions of [`mlqae/core.py`](../mlqae/core.py) and [`leaderboard/schema_and_api.py`](schema_and_api.py):

1. **Per-shot query cost (`shot_cost(n)`):**
   - At Grover depth $n = 0$, 1 shot of $U|0\rangle$ costs **$1$ query**.
   - At Grover depth $n \ge 1$, 1 shot of $G^n U|0\rangle$ costs **$n$ Grover queries**.
   - For a schedule $\{(n_k, N_k)\}_{k=1}^K$, the total query count is:
     $$N_{\text{tot}} = \sum_{k=1}^K \text{shot\_cost}(n_k) \cdot N_k = \sum_{n \ge 1} n N_n + N_0.$$
2. **Parallel vs. Sequential Depth Accounting (`QuantumAEOracle`):**
   - **`oracle.measure(depth=n, shots=s)`:** Executes $s$ shots at depth $n$ as 1 adaptive round:
     - `total_queries += shot_cost(n) * s`
     - `max_depth = max(max_depth, n)`
     - `seq_depth += n`
     - `num_rounds += 1`
   - **`oracle.measure_batch(depths, shots)`:** Executes a batch of circuits in **1 parallel round**:
     - `total_queries += sum(shot_cost(d) * s for d, s in zip(depths, shots))`
     - `max_depth = max(max_depth, max(depths))`
     - `seq_depth += max(depths)`
     - `num_rounds += 1`
   - Note that a **non-adaptive** algorithm executes all circuits in a single parallel batch (`num_rounds = 1`), so its sequential depth equals its maximum circuit depth: $\text{seq\_depth} = n_{\max}$.

### 1.3 Benchmark Scales, Blind Seed Protocol & Metrics
- **Prior distribution:** $a \sim \mathrm{Uniform}(0.1, 0.9)$ across Monte Carlo trials.
- **Standard benchmark scales:** $n_{\max} \in \{60, 125, 260, 540\}$.
- **Development vs. Blind Certification Seeds:**
  - **Local development & smoke testing:** Contributors can develop and benchmark locally using `SEARCH_SEED = 100_000` (and local validation seed `HOLDOUT_SEED = 900_000`).
  - **Official blind certification (`LEADERBOARD_SECRET_SEED`):** While contributors can test locally on `SEARCH_SEED = 100_000`, official leaderboard certification runs in GitHub Actions against a **blind cryptographic repository secret (`LEADERBOARD_SECRET_SEED`)** so submissions cannot tune hyperparameters on or memorize the test amplitudes. Any submission whose performance on the blind secret seed diverges significantly from its public seed performance is automatically flagged.
- **Primary Leaderboard Metrics:**
  Let $\varepsilon_\delta = \mathrm{Percentile}_\delta(| \hat{a} - a |)$ across trials:
  - **$C_{95,\text{ave}} = \varepsilon_{95} \cdot \langle N_{\text{tot}} \rangle$** *(Primary ranking metric: 95th-percentile error $\times$ average queries)*
  - **$C_{95,\text{max}} = \varepsilon_{95} \cdot \max(N_{\text{tot}})$** *(Worst-case query constant; equals $C_{95,\text{ave}}$ for fixed-budget & non-adaptive methods)*
  - **$C_{68,\text{ave}} = \varepsilon_{68} \cdot \langle N_{\text{tot}} \rangle$** *(Standard error constant)*
  - **$C_{99,\text{ave}} = \varepsilon_{99} \cdot \langle N_{\text{tot}} \rangle$** *(Catastrophic / tail robustness constant)*
  - **$C_{95,\text{seq}} = \varepsilon_{95} \cdot \langle \text{seq\_depth} \rangle$** *(Sequential wall-clock depth constant)*

---

## 2. Competition Tracks

| Track Identifier | Description | API Base Class |
|---|---|---|
| `non-adaptive` | Fixed schedule $(n_k, N_k)$ chosen before any measurements; 1 parallel round (`num_rounds = 1`). | `NonAdaptiveAESubmission` |
| `semi-adaptive` | Bounded small number of parallel batch rounds (e.g., 2–4 rounds via `oracle.measure_batch`). | `AdaptiveAESubmission` |
| `adaptive-fixed-budget` | Sequential or multi-round adaptive choices under a strict deterministic per-trial query cap $N_{\text{tot}} \le B$. | `AdaptiveAESubmission` |
| `adaptive-variable` | Sequential adaptive algorithm with stopping rule (variable per-trial query count $N_{\text{tot}}$; evaluated on both $C_{95,\text{ave}}$ and $C_{95,\text{max}}$). | `AdaptiveAESubmission` |
| `depth-limited` | Maximum circuit depth constrained by a hardware cap $n \le M$ (`max_depth_cap = M`). | `NonAdaptiveAESubmission` or `AdaptiveAESubmission` |

---

## 3. Sandbox Rules (`QuantumAEOracle`)

Adaptive and semi-adaptive submissions interact with a tamper-proof `QuantumAEOracle` instance:
- The secret amplitude `a_true` and angle `theta_true` are stored inside an isolated private closure. Neither `dir(oracle)` nor `oracle.__dict__` nor `vars(oracle)` exposes them.
- `QuantumAEOracle` is immutable and non-cloneable: attempting to modify `oracle.total_queries`, set attributes on `oracle`, or call `copy.copy(oracle)` / `copy.deepcopy(oracle)` / `pickle.dumps(oracle)` raises `OracleSecurityError`.
- Submissions must be self-contained, deterministic given the provided `rng: np.random.Generator`, and rely only on standard scientific Python packages (`numpy`, `scipy`, `statsmodels`, `numba`).

---

## 4. Minimal Working Examples

### 4.1 Track A: Non-Adaptive Submission (`NonAdaptiveAESubmission`)

Non-adaptive submissions implement `get_schedule(scale_nmax)` and a vectorized `estimate(depths, shots, counts)` method:

```python
from typing import Sequence
import numpy as np
from leaderboard.schema_and_api import (
    NonAdaptiveAESubmission,
    ml_estimate_batch,
    evaluate_non_adaptive_submission,
    SEARCH_SEED,
)
from mlqae.core import geom_ladder, canonical_shots


class MyGeometricLadderAE(NonAdaptiveAESubmission):
    name = "my-geometric-ladder-ml"
    track = "non-adaptive"

    def get_schedule(self, scale_nmax: int) -> tuple[Sequence[int], Sequence[int]]:
        """Return (depths, shots) for the target maximum depth scale_nmax."""
        depths = geom_ladder(scale_nmax, r=1.45)
        shots = canonical_shots(len(depths), base=1, slope=1.0)
        return depths, shots

    def estimate(
        self,
        depths: Sequence[int],
        shots: Sequence[int],
        counts: np.ndarray,
    ) -> np.ndarray | float:
        """Vectorized batch estimator: counts has shape (ntrials, len(depths)) or (len(depths),)."""
        return ml_estimate_batch(depths, shots, counts)


if __name__ == "__main__":
    sub = MyGeometricLadderAE()
    metrics = evaluate_non_adaptive_submission(
        sub, scale_nmax=125, ntrials=5000, seed_band=SEARCH_SEED
    )
    print(f"nmax=125 | nq={metrics['nq']} | C95={metrics['C95_ave']:.3f} | C99={metrics['C99_ave']:.2f}")
```

### 4.2 Track B: Classic Iterative / Bayesian Interval QAE (`AdaptiveAESubmission`)

Adaptive submissions implement `run_trial(oracle, target_scale_or_budget, rng)` using `oracle.measure(depth, shots)` (sequential) or `oracle.measure_batch(depths, shots)` (parallel round). Below is a classic **Iterative / Bayesian Interval QAE** example (cf. Grinko et al. 2021, arXiv:1912.05559) that uses Beta credible intervals to select unambiguous progressive Grover depths:

```python
import numpy as np
from scipy.stats import beta as beta_dist
from leaderboard.schema_and_api import (
    AdaptiveAESubmission,
    QuantumAEOracle,
    ml_estimate_batch,
    evaluate_adaptive_submission,
    SEARCH_SEED,
)


class BayesianIntervalQAE(AdaptiveAESubmission):
    name = "bayesian-interval-qae"
    track = "adaptive-fixed-budget"

    def __init__(self, shots_per_stage: int = 12, conf_alpha: float = 0.005) -> None:
        self.shots_per_stage = int(shots_per_stage)
        self.conf_alpha = float(conf_alpha)

    def run_trial(
        self,
        oracle: QuantumAEOracle,
        target_scale_or_budget: int,
        rng: np.random.Generator,
    ) -> float:
        _ = rng
        scale_nmax = max(1, int(target_scale_or_budget))

        depths_used: list[int] = []
        shots_used: list[int] = []
        counts_obs: list[int] = []

        # Stage 0: Measure at depth n = 0 and form Beta(k0 + 1, N0 - k0 + 1) credible interval
        s0 = 2 * self.shots_per_stage
        k0 = int(oracle.measure(0, shots=s0))
        depths_used.append(0)
        shots_used.append(s0)
        counts_obs.append(k0)

        p_lo, p_hi = beta_dist.ppf(
            [self.conf_alpha / 2.0, 1.0 - self.conf_alpha / 2.0],
            k0 + 1,
            s0 - k0 + 1,
        )
        theta_min = float(np.arccos(np.sqrt(np.clip(p_hi, 1e-6, 1.0 - 1e-6))))
        theta_max = float(np.arccos(np.sqrt(np.clip(p_lo, 1e-6, 1.0 - 1e-6))))

        # Iterative depth doubling within the unambiguous cosine branch
        curr_depth = 0
        while curr_depth < scale_nmax:
            width = max(1e-6, theta_max - theta_min)
            # Largest Grover depth n where 2 * (2n + 1) * width <= pi
            n_unambig = int(max(1, np.floor((np.pi / (2.0 * width) - 1.0) / 2.0)))
            next_depth = min(scale_nmax, max(curr_depth + 1, min(2 * max(1, curr_depth), n_unambig)))

            k_next = int(oracle.measure(next_depth, shots=self.shots_per_stage))
            depths_used.append(next_depth)
            shots_used.append(self.shots_per_stage)
            counts_obs.append(k_next)
            curr_depth = next_depth

            # Update point estimate and credible interval width
            a_est = float(ml_estimate_batch(depths_used, shots_used, np.asarray(counts_obs)))
            th_est = float(np.arcsin(np.clip(a_est, 1e-6, 1.0 - 1e-6)))
            d_arr = 2.0 * np.asarray(depths_used, dtype=float) + 1.0
            s_arr = np.asarray(shots_used, dtype=float)
            sigma_th = float(1.0 / np.sqrt(4.0 * np.sum(s_arr * (d_arr ** 2))))
            theta_min = max(1e-6, th_est - 3.0 * sigma_th)
            theta_max = min(np.pi / 2.0 - 1e-6, th_est + 3.0 * sigma_th)

        return float(ml_estimate_batch(depths_used, shots_used, np.asarray(counts_obs)))


if __name__ == "__main__":
    sub = BayesianIntervalQAE()
    metrics = evaluate_adaptive_submission(
        sub, target_scale_or_budget=125, ntrials=500, seed_band=SEARCH_SEED
    )
    print(
        f"q_mean={metrics['q_mean']:.1f} | C95_ave={metrics['C95_ave']:.3f} | "
        f"C99_ave={metrics['C99_ave']:.2f} | rounds={metrics['num_rounds_mean']:.1f}"
    )
```

---

## 5. How to Submit via Pull Request

### Step 1: Create Your Submission Folder
Create a new directory under `leaderboard/submissions/<your_algorithm_slug>/` containing:
```text
leaderboard/submissions/<your_algorithm_slug>/
├── submission.py      # Implements your NonAdaptiveAESubmission or AdaptiveAESubmission subclass
└── metadata.json      # Required metadata file (validated against METADATA_JSON_SCHEMA)
```

### Step 2: Populate `metadata.json`
Your `metadata.json` must include all required fields and pass `validate_metadata()`:

```json
{
  "algorithm_name": "Iterative Bayesian Interval QAE",
  "track": "adaptive-fixed-budget",
  "authors": ["Jane Doe", "John Smith"],
  "github_handle": "janedoe",
  "affiliation": "Example Quantum Institute",
  "paper_url": "https://arxiv.org/abs/1912.05559",
  "pr_number": null,
  "description": "Iterative amplitude estimation using Beta posterior credible intervals to select unambiguous progressive Grover depths followed by maximum-likelihood readout.",
  "date": "2026-10-23",
  "entrypoint": "submission.py:BayesianIntervalQAE"
}
```

Allowed values for `"track"`:
- `"non-adaptive"`
- `"semi-adaptive"`
- `"adaptive-fixed-budget"`
- `"adaptive-variable"`
- `"depth-limited"`

### Step 3: Validate Locally Before Opening a Pull Request
From the repository root, run the schema and sandbox self-tests and verify your `metadata.json`:

```bash
# 1. Run the API & Sandbox test suite
.venv/bin/python leaderboard/schema_and_api.py

# 2. Validate your metadata.json
.venv/bin/python -c "
from leaderboard.schema_and_api import validate_metadata
meta = validate_metadata('leaderboard/submissions/<your_algorithm_slug>/metadata.json')
print('Metadata valid:', meta['algorithm_name'], '| Track:', meta['track'])
"
```

### Step 4: Open a Pull Request
1. Commit your `leaderboard/submissions/<your_algorithm_slug>/` directory (do **not** modify core evaluation harnesses or existing baseline files).
2. Open a Pull Request against the `main` branch.
3. Automated CI will run Tier-1 verification on `SEARCH_SEED = 100_000` and Tier-2 official certification against `HOLDOUT_SEED = 900_000` and the blind cryptographic secret `LEADERBOARD_SECRET_SEED` across $n_{\max} \in \{60, 125, 260, 540\}$ with bootstrap 95% confidence intervals, then update `leaderboard/leaderboard.json`.
