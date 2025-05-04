# Used to generate data for figures. Run the following commands. Each run takes about 7 hours for a single command to run with 12 threads on a server.
# python run_ae_sims.py --save --dir=sims/ --nthreads=12 --num_lengths=6 --C=4 --adjacency=5 --num_mc=500
# python run_ae_sims.py --save --dir=sims/ --nthreads=12 --num_lengths=6 --C=8 --adjacency=5 --num_mc=500


import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS']='1'

import numpy as np
from signals import *
from frequencyestimator import *
from csae import *
from util import *
import time
import multiprocessing
import pickle
import argparse
import pathlib

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


def run(theta, n_samples, ula_signal, espirit, heavy_signs, seed, eta=0.0, adjacency=2):

    np.random.seed(seed)

    csignal, measurements = simulate_signal(ula_signal.depths, n_samples, theta)
    ula_signal.set_measurements(measurements)
    res = csae_with_local_minimization(ula_signal, espirit, heavy_signs, sample=True, correction=True, optimize=True, disp=False, adjacency=adjacency)

    theta_est = res['theta_est']
    error = np.abs(np.sin(theta) - np.sin(theta_est))

    cR = ula_signal.get_cov_matrix_toeplitz(csignal)
    theta_est_exact_signs, _ = espirit.estimate_theta_toeplitz(cR)
    error_exact_signs = np.abs(np.sin(theta) - np.sin(theta_est_exact_signs))
    
    return error, theta_est, error_exact_signs, theta_est_exact_signs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Run compressed sensing amplitude estimation simulation',
                                    description="This program creates the simulation files. Running this program will generate all data files needed by plots.ipynb which will generate the figures in the paper. \n\n Use the following commands from the command line to run the correct simulations and store the output:\n python run_ae_sims.py --save --dir sims --nthreads=4 --num_mc=500 --num_lengths=8 --eta=0.0 \n python run_ae_sims.py --save --dir sims_eta0.01 --nthreads=4 --num_mc=500 --num_lengths=8 --eta=0.01 --C=1.5\n python run_ae_sims.py --save --dir sims_eta0.05 --nthreads=4 --num_mc=500 --num_lengths=8 --eta=0.05")
    parser.add_argument('--save', action='store_true', help="Set to true if you want to save output files (default: False).")
    parser.add_argument('--dir', type=str, help="Directory to save output files (default: sims/).", default="sims/")
    parser.add_argument('--nthreads', type=int, help="Number of threads to use for simulation (default: 1).", default=1)
    parser.add_argument('--num_mc', type=int, help="Number of Monte Carlo trials (default: 500)", default=500)
    parser.add_argument('--num_lengths', type=int, help="Maximum length array to use (default: 8)", default=5)
    parser.add_argument('--eta', type=float, help="Add a bias term to the estimated output probabilities. This biases the output towards a 50/50 mixture assuming noise in the circuit causes depolarization (default=0.0)", default=0.0)
    parser.add_argument('--fixed_sample', type=int, help="If set, this sets the sampling strategy to do a fixed number of samples at each depth, rather than one that samples more at lower depth and less and longer depth (default=None)", default=None)
    parser.add_argument('--C', type=float, help="This is a free parameter that determines how many shots to take at each step. (default=1.5)", default=1.5)
    parser.add_argument('--adjacency', type=int, help="The maximum Hamming distance to consider for the sign variations. (default=2)", default=2)
    args = parser.parse_args()

    print('Command Line Arguments')
    print(args)
    
    np.random.seed(7)
    
    pathlib.Path(args.dir).mkdir(parents=True, exist_ok=True) 

    # In paper, we use 8, but it takes about four hours to run this in total on a 4 core laptop using 4 threads. If you want to just test this out, set num_lenghts to 6 and it should finish within minutes.
    num_lengths = args.num_lengths
    num_mc = args.num_mc

    num_queries = np.zeros(num_lengths, dtype=int)
    max_single_query = np.zeros(num_lengths, dtype=int)
    errors = np.zeros((num_lengths, num_mc), dtype = float)
    thetas = np.zeros((num_lengths, num_mc), dtype = float)
    avals  = np.zeros((num_lengths, num_mc), dtype = float)

    errors_exact_signs = np.zeros((num_lengths, num_mc), dtype=float)
    thetas_exact_signs = np.zeros((num_lengths, num_mc), dtype=float)
    
    num_threads = args.nthreads

    arrays = []

    avals = [np.random.uniform(0.1, 0.9) for _ in range(num_mc)]
    thetas_mc = np.arcsin(np.array(avals))

    filename = args.dir+f'/csae_C{args.C:0.3f}_mc{num_mc:04d}.pkl'

    for r in range(num_lengths):

        print(f'Trial {r+1} of {num_lengths}')

        espirit = ESPIRIT()
        narray = [2]*(2*r+2)
        arrays.append(narray)

        ula_signal = TwoqULASignal(M=narray, C=args.C)
        heavy_signs = get_heavy_signs(ula_signal.depths, ula_signal.n_samples, len(narray) ** 2)

        if args.fixed_sample:
            n_samples = [args.fixed_sample]*len(ula_signal.n_samples)
        else:
            n_samples = ula_signal.n_samples

        # Compute the total number of queries. The additional count of ula_signal.n_samples[0] is to
        # account for the fact that the Grover oracle has two invocations of the unitary U, but is
        # preceded by a single invocation of U (see Eq. 2 in paper). This accounts for the shots required
        # for that single U operator. Most papers negect this cost
        num_queries[r] = np.sum(np.array(ula_signal.depths)*np.array(n_samples)) + n_samples[0]
        max_single_query[r] = np.max(ula_signal.depths)

        pool = multiprocessing.Pool(num_threads)
        start = time.time()
        processes = [pool.apply_async(run, args=(theta, n_samples, ula_signal, espirit, heavy_signs, seed, args.eta, args.adjacency)) for seed, theta in enumerate(thetas_mc)]
        sims = [p.get() for p in processes]
        for k in range(num_mc):
            errors[r,k], thetas[r,k], errors_exact_signs[r,k], thetas_exact_signs[r,k] = sims[k]
        end = time.time()
        print(f'Time for trial {r+1}: {end-start} (s)')

        print(f'Array parameters: {narray}')
        print(f'Query Depths: {ula_signal.depths}')
        print(f'Number of Samples per query: {n_samples}')
        print(f'Number of queries: {num_queries[r]}')
        print(f'Max Single Query: {max_single_query[r]}')
        print(f'99% percentile: {np.percentile(errors[r], 99):e}')
        print(f'95% percentile: {np.percentile(errors[r], 95):e}')
        print(f'68% percentile: {np.percentile(errors[r], 68):e}')
        print(f'99% Constant: {num_queries[r] * np.percentile(errors[r], 99):e}')
        print(f'95% Constant: {num_queries[r] * np.percentile(errors[r], 95):e}')
        print(f'68% Constant: {num_queries[r] * np.percentile(errors[r], 68):e}')
        print(f'99% Max Constant: {max_single_query[r] * np.percentile(errors[r], 99):e}')
        print(f'95% Max Constant: {max_single_query[r] * np.percentile(errors[r], 95):e}')
        print(f'68% Max Constant: {max_single_query[r] * np.percentile(errors[r], 68):e}')
        print(f'Error per oracle: {args.eta:e}')
        print(f'Maximum Error: {1.0 - (1.0-args.eta)**(max_single_query[r]+1):e}')
        print(f'C parameter: {args.C:0.3f}')
        print()

    if args.save:
        with open(filename, 'wb') as handle:
            pickle.dump((errors, thetas, errors_exact_signs, thetas_exact_signs, num_queries, max_single_query, arrays, num_lengths, num_mc, avals),
                        handle, protocol=pickle.HIGHEST_PROTOCOL)
