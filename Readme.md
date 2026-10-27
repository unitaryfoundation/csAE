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
- **Depth Cap**: the benchmark scale $n_{\max}$ is a hard limit on the Grover depth of every circuit, for every track; the evaluators and the CI oracle reject deeper circuits. Sequential rounds are not capped and are charged in the sequential-depth constant.
- **Primary Metric**: $C_{95}^{\mathrm{ave}} = \varepsilon_{95} \cdot \langle N_{\mathrm{tot}} \rangle$ across standard scales $n_{\max} \in \{125, 260, 540, 1000\}$ ($20{,}000$ Monte Carlo trials per scale with $1{,}000$-resample bootstrap 95% confidence intervals), tracked alongside worst-case query cost $C_{95}^{\max}$, tail constant $C_{99}^{\mathrm{ave}}$, and sequential depth constant $C_{95}^{\mathrm{seq}}$.
- **5-Stage Anti-Cheat & Blind Secret Test Seed (`LEADERBOARD_SECRET_SEED`)**:
  Every Pull Request touching `leaderboard/submissions/<your_algo>/` triggers an isolated, network-disabled GitHub Actions workflow ([`.github/workflows/leaderboard_ci.yml`](.github/workflows/leaderboard_ci.yml)) running [`leaderboard/verifier_and_runner.py`](leaderboard/verifier_and_runner.py):
  1. **Static AST Security Scanner** (blocks forbidden imports, builtins, and underscore/dunder introspection).
  2. **Input/Output & Budget Contract Validation** (enforces valid depths, shot counts, and estimates in $[0,1]$).
  3. **Bitwise Seed Reproducibility Check**.
  4. **Zero-Information Coin-Flip Oracle Test** (proves the algorithm cannot side-channel `a_true` from the closure-isolated [`QuantumAEOracle`](leaderboard/schema_and_api.py)).
  5. **Asymptotic Heisenberg Scaling Check** & **Blind Secret-Seed Certification** (official rankings are evaluated against an unguessable `HMAC-SHA256` seed derived from `secrets.LEADERBOARD_SECRET_SEED` so submissions cannot tune hyperparameters on or memorize the test set).

### Seeded Literature Baselines (`11/11` Certified in [`leaderboard/submissions/`](leaderboard/submissions/))

Literature baselines are run exactly as specified in their papers; the benchmark scale fixes only the sequence length. The two Suzuki et al. MLAE entries use $N_{\mathrm{shot}} = 100$ per circuit with the paper's LIS ($m_k = k$) and EIS ($m_k = 2^{k-1}$) sequences, so the LIS constant grows with scale as its $O(N_q^{-3/4})$ error law implies. The live page shows the per-scale values and the $C_{68}/C_{95}/C_{99}$ frontiers.

| Rank | Algorithm | Track | Rounds | $C_{68}^{\mathrm{ave}}$ | $C_{95}^{\mathrm{ave}}$ `[95% CI]` | $C_{95}^{\max}$ | $C_{99}^{\mathrm{ave}}$ | $C_{95}^{\mathrm{seq}}$ | Reference |
| ---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **`mlqae-plain`** | `non-adaptive` | $1.0$ | $1.222$ | **2.881** `[2.82, 2.96]` | $2.881$ | $6.67$ | $0.285$ | [arXiv:2609.02715](https://arxiv.org/abs/2609.02715) |
| **2** | **`mlqae-flagship`** | `non-adaptive` | $1.0$ | $1.300$ | **2.936** `[2.88, 3.00]` | $2.936$ | $5.71$ | $0.212$ | [arXiv:2609.02715](https://arxiv.org/abs/2609.02715) |
| **3** | **`BAE-Adaptive`** | `adaptive-fixed-budget` | $26.6$ | $1.098$ | **3.034** `[2.92, 3.15]` | $3.034$ | $11.25$ | $3.023$ | [arXiv:2401.15072](https://arxiv.org/abs/2401.15072) |
| **4** | **`chebAE`** | `adaptive-variable` | $15.0$ | $1.411$ | **3.188** `[2.95, 3.44]` | $4.871$ | $15.24$ | $2.920$ | [arXiv:2207.08628](https://arxiv.org/abs/2207.08628) |
| **5** | **`csAE (ESPRIT)`** | `non-adaptive` | $1.0$ | $1.401$ | **3.741** `[3.60, 3.88]` | $3.741$ | $12.55$ | $0.231$ | [arXiv:2405.14697](https://arxiv.org/abs/2405.14697) |
| **6** | **`BIQAE (Beta-Posterior)`** | `adaptive-variable` | $10.8$ | $1.957$ | **4.652** `[4.57, 4.75]` | $9.396$ | $9.42$ | $4.652$ | Bayesian Beta-IQAE |
| **7** | **`IQAE-CP`** | `adaptive-variable` | $5.7$ | $2.703$ | **6.805** `[6.66, 6.95]` | $14.605$ | $31.03$ | $6.805$ | [arXiv:1912.05559](https://arxiv.org/abs/1912.05559) |
| **8** | **`MLAE-EIS (Suzuki)`** | `non-adaptive` | $1.0$ | $3.628$ | **7.942** `[7.80, 8.09]` | $7.942$ | $14.05$ | $0.040$ | [arXiv:1904.10246](https://arxiv.org/abs/1904.10246) |
| **9** | **`FAE`** | `adaptive-variable` | $9.1$ | $3.947$ | **8.785** `[8.63, 8.95]` | $11.278$ | $14.32$ | $8.785$ | [arXiv:2003.02417](https://arxiv.org/abs/2003.02417) |
| **10** | **`IQAE-Chernoff`** | `adaptive-variable` | $19.9$ | $3.881$ | **9.066** `[8.90, 9.23]` | $30.157$ | $13.16$ | $9.066$ | [arXiv:1912.05559](https://arxiv.org/abs/1912.05559) |
| **11** | **`MLAE-LIS (Suzuki)`** | `non-adaptive` | $1.0$ | $35.354$ | **74.588** `[73.38, 75.62]` | $74.588$ | $100.86$ | $0.004$ | [arXiv:1904.10246](https://arxiv.org/abs/1904.10246) |

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
