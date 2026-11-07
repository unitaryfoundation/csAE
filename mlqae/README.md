# Quantum amplitude estimation beyond power-of-two schedules (`mlqae`)

**Paper:** [arXiv:2609.02715](https://arxiv.org/abs/2609.02715) · **Live demo:** https://unitaryfoundation.github.io/csAE/

Code and paper source for [arXiv:2609.02715](https://arxiv.org/abs/2609.02715), the follow-up to csAE (arXiv:2405.14697):
same measurement framework (Grover-depth schedules, Z-basis measurements only,
non-adaptive / fully parallel), two changes:

1. **Estimator**: exact global maximum likelihood over θ (one matrix
   multiplication per batch of trials, plus a local zoom refinement),
   replacing sign recovery + virtual array + ESPRIT.
2. **Schedule**: geometric depth ladder with ratio **r ≈ 1.45** instead of
   powers of two, with an untuned linear-decay shot profile (deepest level
   gets 1 shot, one more per shallower level), plus **one extra rung at
   round(nmax/1.3)**. That rung breaks the deepest level's mirror degeneracy,
   which is what the 99% tail is made of: C99 falls 16–20% and the parallel
   constant ~27% at no cost in C95, for ~7% of C68. The plain ladder is the
   object the theory analyses; the flagship is the plain ladder plus the rung. In the depth-limited regime
   (cap M), the same ladder capped at M with all shots scaled uniformly.

Amplitude convention, measurement model, query accounting, and the amplitude
range a ∈ (0.1, 0.9) match the csAE paper's simulations.

## Headline results (certified, bootstrap 95% CIs)

| method | C95 = ε95·Ntot | C99 | seq. depth ε95·nmax |
|---|---|---|---|
| csAE (published) | 4.3 | 8.9–12.3 | 0.26 |
| plain r=1.45 ladder + global ML | 2.72–3.10 | 6.0–7.5 | 0.29 |
| **flagship: + one rung at nmax/1.3** | **2.78–3.07** (ε95 from 3.5e-3 to 9.9e-7) | **5.1–5.8** | **0.21** |
| chebAE average (matched targets) | 3.00–3.23 | 5.47 | 2.8–3.0 |
| chebAE max observed | 4.56–5.00 | 8.14 | — |

Depth-limited regime (cap M, uniform scaling): trade-off constant
C̃95 = ε95²·M·Ntot = 0.39–0.58 for scale s ≥ 4, within 1.02–1.11× of the
schedule CRLB, i.e. Ntot ≈ (0.4–0.6)/(M ε95²) — explicit constants for the
M·N ≈ 1/ε² frontier. See `paper/main.pdf` for the full write-up, mechanism
(aliasing / Chernoff analysis), and the optimality accounting.

## Environment

From the repository root:

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # numpy scipy matplotlib numba statsmodels seaborn
```

`mlqae/` itself needs only numpy, scipy, matplotlib, and (for the chebAE
comparisons) statsmodels; numba and seaborn are used by the sibling `csae/` package.

**Run every script from the repository root**, e.g.
`python mlqae/head_to_head.py`.

## Dependence on csAE tools

The package is self-contained except for two deliberate uses of the rest of
the repository:

- `head_to_head.py` and `extend_scaling.py` import **`chebAE.py`** (the
  repository's copy of the Rall–Fuller reference implementation) for the
  matched comparisons.
- `compare_to_csae.py` imports the sibling **`csae/`** package (`estimator.py`,
  `util.py`, `signals.py`, `frequencyestimator.py`) to run the published csAE
  pipeline on paired trials. It requires the sign-learning version of these
  modules (present on this branch; not in pre-2024 history).

Everything else (`core.py` and the schedule/ladder studies) uses only
numpy/scipy.

## Reproducing the paper's tables and figures

All scripts use fixed seeds; reruns reproduce the committed `.pkl` files and
the numbers printed in the paper up to the stated bootstrap CIs (the `.pkl`
files in this folder are the snapshots the paper was built from). Runtimes are
measured wall-clock on an 11-core laptop (Python 3.12, numpy 1.26.4, 4 BLAS
threads per script), one script at a time. The whole set is ~7 minutes plus two slower items
(`depth_tails.py` ~4 min, `zzb_table.py` ~15 min); run `test_mlqae.py` first.

| paper artifact | script(s) | output | runtime |
|---|---|---|---|
| Table I (flagship scales), cap-rung variant, Table IV (noise) | `final_numbers.py` | `paper_numbers.pkl` + printed tables | 55 s |
| Table III + Sec. VI.B anchored-design constants | `depth_tradeoff.py` (stages 1–4; `python mlqae/depth_tradeoff.py 4` for Table III only) | `depth_tradeoff.pkl` | 59 s |
| Sec. V / VI.B tail numbers (stage-1 stall at ε95·M ≈ 0.14–0.28, pair-anchor thin tails, uniform-scaling clean tails) | `depth_tails.py` | `depth_tails.pkl` + printed table | ~4 min |
| Table II + Fig. 3 data (head-to-head vs chebAE) | `head_to_head.py`, then `extend_scaling.py` | `head_to_head.pkl`, `head_to_head_ext.pkl` | 70 s + 156 s |
| Table V (Ziv–Zakai bound columns; achieved column = Table I) | `zzb_table.py` | printed table | ~15 min |
| Sec. VIII window-rule numbers (Bayes rule for the tolerance loss vs plain ML, both scales) | `window_map.py` | printed table | ~3 min |
| Table VI + Appendix A prose numbers (mode width, grid resolution, bracketing) | `grid_resolution.py` | `grid_resolution.pkl` + printed tables | 48 s |
| Sec. VI.A deep-shot sweep (base = 1 optimal at every scale; the C95/C99 trade) | `scale_rule.py` | `scale_rule.pkl` | 28 s |
| Fig. 1 data (ratio sweep, flip rates) | `ratio_sweep.py` | `ratio_sweep.pkl` | 18 s |
| Fig. 2 data (aliasing exponents + measured flip rates) | `alias_analysis.py` | `alias_analysis.pkl` | < 1 s |
| Fig. 4 data (depth-limited fan) | `depth_fan.py` | `depth_fan.pkl` | 9 s |
| Sec. III paired numbers (csAE 4.37 vs polish 4.06 vs global ML 3.92) | `compare_to_csae.py` | printed | 158 s |
| Figures (PDFs in `paper/figures/`) | `make_figures.py` (after the data scripts above) | `ratio_sweep.pdf`, `alias_exponent.pdf`, `scaling.pdf`, `depth_fan.pdf` | seconds |
| The paper itself | `cd mlqae/paper && pdflatex main && bibtex main && pdflatex main && pdflatex main` | `main.pdf` | seconds |

Order matters only in two places: `extend_scaling.py` reads
`head_to_head.pkl`, and `make_figures.py` reads the data pickles (it prefers
`head_to_head_ext.pkl` for Fig. 3 when present).

Peak memory is ~6 GB, in `extend_scaling.py`'s deepest point (nmax ~ 2e5, a
3.4e6-point likelihood grid); everything else stays well under 1 GB.

Sanity anchors while reproducing: `grid_resolution.py` prints measured mode
widths within ~1% of `pi sqrt(Lambda)` for s ≥ 16 and basin-level disagreement
rates of order 1e-3 with sub-0.1-nat likelihood gaps; `head_to_head.py` prints
`nmax=125 … C95=2.82…2.87` for the flagship and `C_ave≈3.1` for chebAE at the
matched target; `compare_to_csae.py`'s csAE column reproduces the r=3 row of
`csae/sims/csae_C4.000_mc0500.pkl` trial-for-trial.

## Files

- `core.py` (imported as `mlqae`) — the core evaluator: ladders, canonical shots, the coarse-grid
  spacing rule of Appendix A, vectorized global-ML simulation (with noise model
  and local zoom refinement), bootstrap CIs.
- `test_mlqae.py` — validation suite (~10 s); every test checks the code
  against an analytic limit, an independent bound, a published external number,
  or a pinned historical value. Run this first when reproducing.
- `head_to_head.py`, `extend_scaling.py` — chebAE comparison at matched
  achieved error, down to ε95 ≈ 1.4e-6.
- `ratio_sweep.py` — ladder-ratio sweep with basin-flip rates.
- `alias_analysis.py` — Chernoff/Bhattacharyya aliasing bound vs measured
  catastrophic-flip rates.
- `depth_tradeoff.py` — depth-limited schedule designs (single anchor /
  pair / band / uniform scaling) documenting the degeneracy cascade.
- `depth_tails.py` — tail quantification for those designs: the single-anchor
  stall and its diverging tails, the pair anchor's polynomially suppressed
  catastrophic tail (rate 1e-5..1e-4, Ctilde99.9 up to ~90), and uniform
  scaling's exponentially clean tails.
- `depth_fan.py` — fixed-M fan data (Fig. 4).
- `scale_rule.py` — shot-base-vs-scale sweep; confirms base = 1 is optimal
  for C95 at every scale, and quantifies the C95/C99 trade from extra deep
  shots (Sec. VI.A).
- `final_numbers.py` — certified holdout numbers for Tables I and IV and the
  cap-rung tail variant, with its three controls (divisor sweep, uniform
  scaling at matched budget, transfer to a second scale).
- `compare_to_csae.py` — paired estimator comparison on the published csAE
  pipeline and schedule.
- `grid_resolution.py` — Appendix A: measured likelihood mode width vs the
  Fisher prediction, the two grid-spacing rules against a reference search, and
  the bracketing check against a 32× finer coarse grid.
- `zzb_table.py` — the Ziv–Zakai lower-bound columns of Table V (exact
  Monte-Carlo evaluation of the binary-test error mass, Appendix B protocol).
- `window_map.py` — the window rule of Sec. VIII (center of the highest-mass
  posterior window of half-width h) against plain ML on identical records.
- `make_figures.py` — renders all figure PDFs from the pickles.
- `paper/` — REVTeX source, figures, bibliography.

## Interactive demo

`docs/index.html` is a self-contained browser demo of the ladder mechanism:
the likelihood over θ is drawn shot by shot as a geometric ladder is measured,
with a zoom on the rival basins, a scoreboard of wrong-basin runs, and
switches for r ∈ {1.25, 1.45, 2}, n_max, and shot order. Open the file
directly in a browser, or use the GitHub Pages copy at
https://unitaryfoundation.github.io/csAE/.

## Caveats / open items

- Shot-noise-only model (plus the depolarizing study of Table IV); no
  hardware-noise or endpoint (a → 0, 1) study.
- The class-level lower bound on ε95·Ntot for non-adaptive schedules is open
  (see the optimality paragraph in the paper's Discussion).
- chebAE comparisons use this repository's implementation and the csAE
  paper's accounting conventions; same-seed pairing across algorithms is not
  possible (chebAE draws adaptively).
