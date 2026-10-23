# Quantum Amplitude Estimation Challenge Leaderboard (`csAE` & `mlqae`)

Welcome to the **Quantum Amplitude Estimation (QAE) Challenge Leaderboard** and reference repository! This repository hosts:
1. **The QAE Challenge Leaderboard & Automated CI Benchmark Suite** ([`leaderboard/`](leaderboard/), live dashboard at **https://unitaryfoundation.github.io/csAE/leaderboard.html**) — an open, standardized competition comparing **non-adaptive**, **semi-adaptive**, **adaptive**, **depth-limited**, and **noisy** Quantum Amplitude Estimation algorithms under identical physical measurement and query-accounting rules.
2. **Geometric-Ladder Maximum-Likelihood QAE (`mlqae`)** ([`mlqae/`](mlqae/), [arXiv:2609.02715](https://arxiv.org/abs/2609.02715), interactive demo at **https://unitaryfoundation.github.io/csAE/**).
3. **Compressed-Sensing / Virtual-Array ESPRIT QAE (`csAE`)** ([`csae/`](csae/), [arXiv:2405.14697](https://arxiv.org/abs/2405.14697)).

---

## 🏆 The Quantum Amplitude Estimation Challenge

Can you design a **non-adaptive** or **adaptive** amplitude estimation schedule/estimator that beats the current state of the art?

- **Live Interactive Leaderboard**: **https://unitaryfoundation.github.io/csAE/leaderboard.html** (source: [`docs/leaderboard.html`](docs/leaderboard.html))
- **Live Geometric-Ladder Interactive Demo**: **https://unitaryfoundation.github.io/csAE/** (source: [`docs/index.html`](docs/index.html))
- **How to Submit (Pull Request Guide)**: [`leaderboard/SUBMISSION_GUIDE.md`](leaderboard/SUBMISSION_GUIDE.md)
- **CI Security & Architecture Plan**: [`leaderboard/ARCHITECTURE_PLAN.md`](leaderboard/ARCHITECTURE_PLAN.md)

### Standardized Rules & Blind Evaluation
- **Physical Model**: Estimating unknown amplitude $a = \sin\theta \sim \mathrm{Uniform}(0.1, 0.9)$ from measurements of $G^n U|0\rangle$ with success probability $p(n,\theta) = \cos^2((2n+1)\theta)$.
- **Query Cost Accounting**: A shot at Grover depth $n = 0$ costs $1$ query; a shot at Grover depth $n \ge 1$ costs $n$ queries ($N_{\mathrm{tot}} = \sum_{n \ge 1} n N_n + N_0$).
- **Primary Metric**: $C_{95}^{\mathrm{ave}} = \varepsilon_{95} \cdot \langle N_{\mathrm{tot}} \rangle$ across standard scales $n_{\max} \in \{60, 125, 260, 540\}$ ($20{,}000$ Monte Carlo trials per scale with $1{,}000$-resample bootstrap 95% confidence intervals), tracked alongside worst-case query cost $C_{95}^{\max}$, tail constant $C_{99}^{\mathrm{ave}}$, and sequential depth constant $C_{95}^{\mathrm{seq}}$.
- **5-Stage Anti-Cheat & Blind Secret Test Seed (`LEADERBOARD_SECRET_SEED`)**:
  Every Pull Request touching `leaderboard/submissions/<your_algo>/` triggers an isolated, network-disabled GitHub Actions workflow ([`.github/workflows/leaderboard_ci.yml`](.github/workflows/leaderboard_ci.yml)) running [`leaderboard/verifier_and_runner.py`](leaderboard/verifier_and_runner.py):
  1. **Static AST Security Scanner** (blocks forbidden imports, builtins, and underscore/dunder introspection).
  2. **Input/Output & Budget Contract Validation** (enforces valid depths, shot counts, and estimates in $[0,1]$).
  3. **Bitwise Seed Reproducibility Check**.
  4. **Zero-Information Coin-Flip Oracle Test** (proves the algorithm cannot side-channel `a_true` from the closure-isolated [`QuantumAEOracle`](leaderboard/schema_and_api.py)).
  5. **Asymptotic Heisenberg Scaling Check** & **Blind Secret-Seed Certification** (official rankings are evaluated against an unguessable `HMAC-SHA256` seed derived from `secrets.LEADERBOARD_SECRET_SEED` so submissions cannot tune hyperparameters on or memorize the test set).

### Seeded Literature Baselines (`11/11` Certified in [`leaderboard/submissions/`](leaderboard/submissions/))

| Rank | Algorithm | Track | Rounds | $C_{68}^{\mathrm{ave}}$ | $C_{95}^{\mathrm{ave}}$ `[95% CI]` | $C_{95}^{\max}$ | $C_{99}^{\mathrm{ave}}$ | $C_{95}^{\mathrm{seq}}$ | Reference |
| ---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **`mlqae-plain`** | `non-adaptive` | $1.0$ | $1.199$ | **2.818** `[2.76, 2.89]` | **2.818** | $6.45$ | $0.284$ | [arXiv:2609.02715](https://arxiv.org/abs/2609.02715) |
| **2** | **`mlqae-flagship`** | `non-adaptive` | $1.0$ | $1.275$ | **2.883** `[2.82, 2.94]` | $2.883$ | **5.50** | **0.212** | [arXiv:2609.02715](https://arxiv.org/abs/2609.02715) |
| **3** | **`BAE-Adaptive`** | `adaptive-fixed-budget` | $21.1$ | **1.068** | **3.034** `[2.91, 3.16]` | $3.034$ | $10.97$ | $3.013$ | [arXiv:2401.15072](https://arxiv.org/abs/2401.15072) |
| **4** | **`chebAE`** | `adaptive-variable` | $15.0$ | $1.391$ | **3.141** `[2.93, 3.42]` | $4.833$ | $9.97$ | $2.905$ | [arXiv:2207.08628](https://arxiv.org/abs/2207.08628) |
| **5** | **`BIQAE (Beta-Posterior)`** | `adaptive-variable` | $10.0$ | $1.990$ | **4.691** `[4.61, 4.79]` | $9.513$ | $8.65$ | $4.691$ | Bayesian Beta-IQAE |
| **6** | **`csAE (ESPRIT)`** | `non-adaptive` | $1.0$ | $1.760$ | **4.815** `[4.16, 6.11]` | $4.815$ | $15.64$ | $0.296$ | [arXiv:2405.14697](https://arxiv.org/abs/2405.14697) |
| **7** | **`MLAE-Linear`** | `non-adaptive` | $1.0$ | $2.521$ | **5.320** `[5.24, 5.40]` | $5.320$ | $7.45$ | $0.051$ | [arXiv:1904.10246](https://arxiv.org/abs/1904.10246) |
| **8** | **`IQAE-CP`** | `adaptive-variable` | $5.2$ | $2.735$ | **6.838** `[6.68, 6.98]` | $14.626$ | $19.40$ | $6.838$ | [arXiv:1912.05559](https://arxiv.org/abs/1912.05559) |
| **9** | **`MLAE-Pow2`** | `non-adaptive` | $1.0$ | $3.592$ | **7.880** `[7.72, 8.02]` | $7.880$ | $13.42$ | **0.039** | [arXiv:1904.10246](https://arxiv.org/abs/1904.10246) |
| **10** | **`FAE`** | `adaptive-variable` | $8.1$ | $3.932$ | **8.735** `[8.58, 8.90]` | $11.011$ | $14.10$ | $8.735$ | [arXiv:2003.02417](https://arxiv.org/abs/2003.02417) |
| **11** | **`IQAE-Chernoff`** | `adaptive-variable` | $19.1$ | $4.035$ | **9.371** `[9.20, 9.54]` | $38.952$ | $13.54$ | $9.371$ | [arXiv:1912.05559](https://arxiv.org/abs/1912.05559) |

### Quickstart: Verify & Benchmark Locally
```bash
# Run the 9-stage anti-cheat & sandbox unit test suite
python leaderboard/test_verifier.py

# Certify all 11 seeded literature baseline submissions
python leaderboard/certify_submissions.py

# Verify your own submission before opening a Pull Request
python leaderboard/verifier_and_runner.py --mode=verify --submission=leaderboard/submissions/<your_algo>
```

---

## Core Packages in This Repository: `csAE` and `mlqae`

Two non-adaptive amplitude-estimation algorithms originally developed in this repository measure the Grover iterator at a fixed set of depths and post-process the outcomes classically, so every circuit can run in parallel; they differ in the depth schedule and in the estimator.

| | csAE | Geometric ladder + exact ML (`mlqae`) |
|---|---|---|
| Paper | [arXiv:2405.14697](https://arxiv.org/abs/2405.14697) | [arXiv:2609.02715](https://arxiv.org/abs/2609.02715) |
| Depth schedule | powers of two | ratio 1.45 ladder, one extra rung near the cap |
| Estimator | ESPRIT on a sparse virtual array | exact global maximum likelihood (one matrix product) |
| Constants at 95% | 4.3/ε total, 0.26/ε sequential depth | 2.8–3.1/ε total, 0.21/ε sequential depth |
| Where | [`csae/`](csae/): `estimator.py`, `csAE_example.ipynb`, `plots.ipynb` | [`mlqae/`](mlqae/) with its own [README](mlqae/README.md); core module `mlqae/core.py` |
| Live demo | — | https://unitaryfoundation.github.io/csAE/ (source: [`docs/index.html`](docs/index.html)) |

For the geometric-ladder maximum-likelihood method, start at [`mlqae/README.md`](mlqae/README.md). The section below documents `csAE`, which lives in the `csae/` package (import it from the repository root: `from csae import *`).

## csAE

This repository contains all of the code required to generate the tables and plots provided in arXiv:2405.14697. To recreate the images simply run the provided notebook `csae/plots.ipynb`. This will use the precomputed results stored in pickle files to generate the plots.

Should you want to run your own simulations and learn how to use the compressed sensing amplitude estimation approach, we recommend looking at the notebook `csae/csAE_example.ipynb`. This contains a minimal working example demonstrating the functionality.

All of the data used to generate the results in the paper can be recreated by running the `csae/run_ae_sims.py` python script from the repository root with the following commands. This code block generates the data needed for Fig. 3 and Tab. 1. It runs 500 Monte Carlo trials of the csAE approach for random amplitudes in the range a=(0.1, ..., 0.9). With 12 threads on a workstation this takes about XXX hours to complete.

```
python csae/run_ae_sims.py --save --dir=csae/sims_final/ --nthreads=24 --num_lengths=6 --C=4 --adjacency=5 --num_mc=500 > csae/sims_final/outfileC4.0.dat
python csae/run_ae_sims.py --save --dir=csae/sims_final/ --nthreads=24 --num_lengths=6 --C=8 --adjacency=5 --num_mc=500 > csae/sims_final/outfileC8.0.dat

```

Once the data is generated, simply run the notebook `csae/plots.ipynb` to generate the figures and tables. For your convenience, the precomputed data is already stored so there is no need to rerun these long simulations, except for reproducibility.
