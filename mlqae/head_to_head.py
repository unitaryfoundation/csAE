"""Rigorous head-to-head: dense-ladder ML-QAE vs chebAE (Rall & Fuller,
Quantum 7, 937 (2023)), using the paper repository's own chebae()
implementation, in one harness.

Protocol
--------
Our method: geometric ratio-1.45 ladders at four scales, UNTUNED canonical
shot allocation (linear decay to 1 shot at the deepest level; no per-scale
tuning, to avoid any winner's-curse advantage). 50,000 trials per scale,
amplitudes uniform in (0.1, 0.9).

chebAE: for each scale, the target error is set to OUR achieved eps_95 at that
scale with delta = 0.05 (the same convention the paper uses to compare csAE to
chebAE), 2,000 trials, amplitudes uniform in (0.1, 0.9). Constants use
chebAE's own achieved 95th-percentile error times its average (resp. maximum
observed) query count. Separate 68% and 99% comparisons run chebAE at
delta = 0.32 and delta = 0.01 against our eps_68 and eps_99, so that Table III
of the paper can report all three confidence levels under one convention.

All constants come with nonparametric bootstrap 95% CIs.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pickle
import multiprocessing
from mlqae import (geom_ladder, canonical_shots, flagship_schedule,
                   evaluate_schedule, boot_ci_constant)

RATIO = 1.45
SCALES = [60, 125, 260, 540]
NT_OURS = 50000
NT_CHEB = 2000
OUTFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'head_to_head.pkl')


def cheb_worker(args):
    a, eps, delta, seed = args
    import numpy as np
    from chebAE import chebae
    np.random.seed(seed)
    r = chebae(a, eps, delta)
    # time_complexity = sum of circuit depths over the adaptive rounds; this is
    # chebAE's sequential-depth (parallel) cost in the paper's convention,
    # since adaptive rounds cannot be parallelized. max_depth is the deepest
    # single circuit (Grover depth floor(deg/2)) the run executed, which is
    # what a hardware depth cap constrains.
    return abs(r['a_hat'] - a), r['queries'], r['time_complexity'], r['max_depth']


def run_chebae(eps, delta, ntrials, seed0, return_max_depth=False):
    rng = np.random.default_rng(seed0)
    avals = rng.uniform(0.1, 0.9, ntrials)
    args = [(avals[i], eps, delta, seed0 + i) for i in range(ntrials)]
    with multiprocessing.Pool(7) as pool:
        out = pool.map(cheb_worker, args, chunksize=25)
    err = np.array([o[0] for o in out])
    q = np.array([o[1] for o in out])
    md = np.array([o[2] for o in out])
    if return_max_depth:
        mx = np.array([o[3] for o in out])
        return err, q, md, mx
    return err, q, md


def rerun_chebae_max_depth():
    """Re-run only the 95% chebAE comparison at the four scales, with the same
    seeds and targets as the stored results, and store each trial's maximum
    circuit depth under results['cheb95_maxdepth'] = {nmax: array}.

    The stored (err, q, md) tuples are left untouched; the rerun is checked
    against them so the new depths are known to belong to the same trials.
    """
    with open(OUTFILE, 'rb') as h:
        results = pickle.load(h)
    results['cheb95_maxdepth'] = {}
    for (nmax, eps, err0, q0, md0) in results['cheb95']:
        if nmax not in SCALES:
            continue
        err, q, md, mx = run_chebae(eps, 0.05, NT_CHEB, seed0=555000 + nmax,
                                    return_max_depth=True)
        same = (np.allclose(err, err0) and np.array_equal(q, q0)
                and np.array_equal(md, md0))
        results['cheb95_maxdepth'][int(nmax)] = mx
        print(f"nmax={nmax:4d}: reproduced stored trials={same}  "
              f"<max depth>={mx.mean():.1f} (min {mx.min()}, max {mx.max()})  "
              f"<seq depth>={md.mean():.1f}  <queries>={q.mean():.1f}", flush=True)
        if not same:
            raise RuntimeError(f"chebAE rerun at nmax={nmax} does not reproduce "
                               "the stored trials; refusing to mix depths from "
                               "different runs.")
    with open(OUTFILE, 'wb') as h:
        pickle.dump(results, h)
    print(f'saved {OUTFILE} with cheb95_maxdepth')


def cheb_constants(err, q, md, pct, B=2000, seed=1):
    rng = np.random.default_rng(seed)
    n = len(err)
    cav, cmx, par = np.empty(B), np.empty(B), np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n)
        e = np.percentile(err[i], pct)
        cav[b] = e * q[i].mean()
        cmx[b] = e * q[i].max()
        par[b] = e * md[i].mean()
    point = (np.percentile(err, pct) * q.mean(),
             np.percentile(err, pct) * q.max(),
             np.percentile(err, pct) * md.mean())
    cis = (np.percentile(cav, [2.5, 97.5]), np.percentile(cmx, [2.5, 97.5]),
           np.percentile(par, [2.5, 97.5]))
    return point, cis


if __name__ == '__main__':
    if '--chebae-only' in sys.argv:
        rerun_chebae_max_depth()
        sys.exit(0)
    results = {'ours': [], 'cheb95': [], 'cheb68': None, 'cheb99': None,
               'ours99': None, 'cheb95_maxdepth': {}}
    print(f'=== ours: ratio-{RATIO} ladder + cap rung, canonical shots ===')
    for nmax in SCALES:
        ladder, shots = flagship_schedule(nmax, RATIO)
        r = evaluate_schedule(ladder, shots, NT_OURS, seed=987000 + nmax,
                              return_errors=True)
        lo68, hi68 = boot_ci_constant(r['errors'], r['nq'], 68)
        lo95, hi95 = boot_ci_constant(r['errors'], r['nq'], 95)
        lo99, hi99 = boot_ci_constant(r['errors'], r['nq'], 99)
        results['ours'].append((nmax, ladder, shots, r, (lo95, hi95), (lo99, hi99),
                                (lo68, hi68)))
        print(f"nmax={r['nmax']:4d} q={r['nq']:5d}  eps95={r['p95']:.2e}  "
              f"C68={r['C68']:.2f} [{lo68:.2f},{hi68:.2f}]  "
              f"C95={r['C95']:.2f} [{lo95:.2f},{hi95:.2f}]  "
              f"C99={r['C99']:.2f} [{lo99:.2f},{hi99:.2f}]  C95par={r['C95par']:.3f}",
              flush=True)

    print(f'\n=== chebAE at matched targets (delta=0.05, {NT_CHEB} trials/scale) ===')
    for (nmax, ladder, shots, r, _, _, _) in results['ours']:
        eps = r['p95']
        err, q, md, mx = run_chebae(eps, 0.05, NT_CHEB, seed0=555000 + nmax,
                                    return_max_depth=True)
        (cav, cmx, cpar), (ciav, cimx, cipar) = cheb_constants(err, q, md, 95)
        results['cheb95'].append((nmax, eps, err, q, md))
        results['cheb95_maxdepth'][int(nmax)] = mx
        cov = np.mean(err <= eps)
        print(f"target eps={eps:.2e}: achieved eps95={np.percentile(err,95):.2e} "
              f"(P[err<=target]={cov:.2f})  C_ave={cav:.2f} [{ciav[0]:.2f},{ciav[1]:.2f}]  "
              f"C_max={cmx:.2f} [{cimx[0]:.2f},{cimx[1]:.2f}]  "
              f"C_par={cpar:.1f} [{cipar[0]:.1f},{cipar[1]:.1f}]", flush=True)

    print('\n=== 68% comparison (chebAE at delta=0.32, target = our eps68) ===')
    results['cheb68'] = []
    for (nmax, ladder, shots, r, _, _, ci68) in results['ours'][1:3]:
        err, q, md = run_chebae(r['p68'], 0.32, NT_CHEB, seed0=555000 + nmax)
        (cav, cmx, cpar), (ciav, cimx, cipar) = cheb_constants(err, q, md, 68)
        results['cheb68'].append((nmax, r['p68'], err, q, md))
        print(f"nmax={r['nmax']:4d}: ours C68={r['C68']:.2f} [{ci68[0]:.2f},{ci68[1]:.2f}]"
              f"  eps68*nmax={r['p68']*r['nmax']:.3f}  ||  chebAE achieved "
              f"eps68={np.percentile(err,68):.2e}  C_ave={cav:.2f} "
              f"[{ciav[0]:.2f},{ciav[1]:.2f}]  C_max={cmx:.2f} "
              f"[{cimx[0]:.2f},{cimx[1]:.2f}]  C_par={cpar:.2f}", flush=True)

    print('\n=== 99% comparison at the nmax=260 scale ===')
    nmax, ladder, shots, r, _, ci99, _ = results['ours'][2]
    print(f"ours: eps99={r['p99']:.2e}, C99={r['C99']:.2f} [{ci99[0]:.2f},{ci99[1]:.2f}]")
    err, q, md = run_chebae(r['p99'], 0.01, NT_CHEB, seed0=777001)
    (cav, cmx, cpar), (ciav, cimx, cipar) = cheb_constants(err, q, md, 99)
    results['cheb99'] = (r['p99'], err, q, md)
    print(f"chebAE(delta=0.01, target=our eps99): achieved eps99={np.percentile(err,99):.2e}  "
          f"C99_ave={cav:.2f} [{ciav[0]:.2f},{ciav[1]:.2f}]  "
          f"C99_max={cmx:.2f} [{cimx[0]:.2f},{cimx[1]:.2f}]", flush=True)

    # ---- amplitude-prior robustness (quoted in the Discussion) ----------
    # The a ~ U(0.1, 0.9) convention is inherited from Refs. [6,8,11,14]. Check
    # it is not doing work: widen to U(0.01, 0.99) and re-run BOTH methods. Note
    # the wider prior is slightly EASIER for everyone -- cos(theta) crushes the
    # additive error as a -> 1 -- so the honest test is whether our margin over
    # chebAE survives, not whether our own constant improves.
    print('\n=== amplitude-prior robustness: U(0.1,0.9) vs U(0.01,0.99) ===')
    results['prior_robustness'] = {}
    for nmax in (125, 382):
        ladder, shots = flagship_schedule(nmax)
        for ar in ((0.1, 0.9), (0.01, 0.99)):
            r = evaluate_schedule(ladder, shots, 200000, seed=88000 + nmax,
                                  a_range=ar, chunk=2000)
            rng = np.random.default_rng(666000 + nmax)
            av = rng.uniform(ar[0], ar[1], NT_CHEB)
            with multiprocessing.Pool(7) as pool:
                out = pool.map(cheb_worker,
                               [(av[i], r['p95'], 0.05, 666000 + nmax + i)
                                for i in range(NT_CHEB)], chunksize=25)
            err = np.array([o[0] for o in out]); q = np.array([o[1] for o in out])
            md = np.array([o[2] for o in out])
            (cav, _, _), _ = cheb_constants(err, q, md, 95)
            results['prior_robustness'][(nmax, ar)] = (r['C95'], cav)
            print(f"  nmax={r['nmax']:4d} a~U{ar}: ours C95={r['C95']:.3f}  "
                  f"chebAE ave={cav:.2f}  margin={100*(1-r['C95']/cav):.1f}%",
                  flush=True)

    with open(OUTFILE, 'wb') as h:
        pickle.dump(results, h)
    print(f'\nsaved {OUTFILE}')
