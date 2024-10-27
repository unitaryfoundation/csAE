# Used to generate data for figures. Run the following commands. Warning this takes about 4 hours for a single command to run on a 4 core laptop.
# python run_ae_sims.py --save --dir sims_C1.5_final --nthreads=4 --num_mc=500 --num_lengths=8 --C=1.5
# python run_ae_sims.py --save --dir sims_C1.8_final --nthreads=4 --num_mc=500 --num_lengths=8 --C=1.8
# python run_ae_sims.py --save --dir sims_C1.5_eta1e-4_final --nthreads=4 --num_mc=500 --num_lengths=8 --C=1.5 --eta=1e-4
# python run_ae_sims.py --save --dir sims_C1.5_eta1e-5_final --nthreads=4 --num_mc=500 --num_lengths=8 --C=1.5 --eta=1e-5
# python run_ae_sims.py --save --dir sims_C1.5_eta1e-5_final --nthreads=4 --num_mc=500 --num_lengths=8 --C=1.5 --eta=1e-6

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS']='1'

import numpy as np
from signals import *
from frequencyestimator import *
import time
import multiprocessing
import pickle
import argparse
import pathlib
from scipy.stats import binom

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

def adjust_angle(depths, n_samples, measurements, theta_found):

    p_o2 = np.cos((2 * depths + 1) * (theta_found / 2.0)) ** 2
    p_o4 = np.cos((2 * depths + 1) * (theta_found / 4.0)) ** 2
    p_same = np.cos((2 * depths + 1) * (theta_found)) ** 2
    p_s2 = np.cos((2 * depths + 1) * (np.pi / 2 - theta_found)) ** 2
    p_s4 = np.cos((2 * depths + 1) * (np.pi / 4 - theta_found)) ** 2
    p_s2_o2 = np.cos((2 * depths + 1) * (np.pi / 2 - theta_found / 2)) ** 2
    p_s4_o2 = np.cos((2 * depths + 1) * (np.pi / 4 - theta_found/ 2)) ** 2

    l_o2 = np.sum(
        np.log(
            [1e-75+binom.pmf(n_samples[kk] * measurements[kk], n_samples[kk], p_o2[kk]) for
             kk in
             range(len(n_samples))]))
    l_o4 = np.sum(
        np.log(
            [1e-75+binom.pmf(n_samples[kk] * measurements[kk], n_samples[kk], p_o4[kk]) for
             kk in
             range(len(n_samples))]))
    l_same = np.sum(
        np.log(
            [1e-75+binom.pmf(n_samples[kk] * measurements[kk], n_samples[kk], p_same[kk]) for
             kk in
             range(len(n_samples))]))
    l_s2 = np.sum(
        np.log(
            [1e-75+binom.pmf(n_samples[kk] * measurements[kk], n_samples[kk], p_s2[kk]) for
             kk in
             range(len(n_samples))]))
    l_s4 = np.sum(
        np.log(
            [1e-75+binom.pmf(n_samples[kk] * measurements[kk], n_samples[kk], p_s4[kk]) for
             kk in
             range(len(n_samples))]))
    l_s2_o2 = np.sum(
        np.log([1e-75+binom.pmf(n_samples[kk] * measurements[kk], n_samples[kk], p_s2_o2[kk])
                for kk in
                range(len(n_samples))]))
    l_s4_o2 = np.sum(
        np.log([1e-75+binom.pmf(n_samples[kk] * measurements[kk], n_samples[kk], p_s4_o2[kk])
                for kk in
                range(len(n_samples))]))


    # which_correction = np.argmin([obj_same, obj_s2, obj_s4, obj_o2, obj_s2_o2, obj_s4_o2])
    which_correction = np.argmax([l_same, l_s2, l_s4, l_o2, l_o4, l_s2_o2, l_s4_o2])
    if which_correction == 1:
        theta_found = np.pi / 2.0 - theta_found
    elif which_correction == 2:
        theta_found = np.pi / 4.0 - theta_found
    elif which_correction == 3:
        theta_found =  0.5 * theta_found
    elif which_correction == 4:
        theta_found =  0.25 * theta_found
    elif which_correction == 5:
        theta_found = np.pi / 2.0 - 0.5 * theta_found
    elif which_correction == 6:
        theta_found = np.pi / 4.0 - 0.5 * theta_found

    return np.abs(theta_found)


def run(theta, n_samples, ula_signal, espirit, seed, eta):
    n=2
    signal = ula_signal.estimate_signal(n_samples, theta, seed, eta)
    R = ula_signal.get_cov_matrix_toeplitz(signal)
    theta_est, _ = espirit.estimate_theta_toeplitz(R, n=n)
    theta_est = adjust_angle(ula_signal.depths, n_samples, ula_signal.measurements, theta_est)
    error = np.abs(np.sin(theta)-np.sin(theta_est)) 
    theta = theta_est
    
    return error, theta

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Run ULA Simulation',
                                    description="This program creates the simulation files. Running this program will generate all data files needed by plots.ipynb which will generate the figures in the paper. \n\n Use the following commands from the command line to run the correct simulations and store the output:\n python run_ae_sims.py --save --dir sims --nthreads=4 --num_mc=500 --num_lengths=8 --eta=0.0 \n python run_ae_sims.py --save --dir sims_eta0.01 --nthreads=4 --num_mc=500 --num_lengths=8 --eta=0.01 --C=1.5\n python run_ae_sims.py --save --dir sims_eta0.05 --nthreads=4 --num_mc=500 --num_lengths=8 --eta=0.05")
    parser.add_argument('--save', action='store_true', help="Set to true if you want to save output files (default: False).")
    parser.add_argument('--dir', type=str, help="Directory to save output files (default: sims/).", default="sims/")
    parser.add_argument('--nthreads', type=int, help="Number of threads to use for simulation (default: 1).", default=1)
    parser.add_argument('--num_mc', type=int, help="Number of Monte Carlo trials (default: 500)", default=500)
    parser.add_argument('--num_lengths', type=int, help="Maximum length array to use (default: 8)", default=5)
    parser.add_argument('--eta', type=float, help="Add a bias term to the estimated output probabilities. This biases the output towards a 50/50 mixture assuming noise in the circuit causes depolarization (default=0.0)", default=0.0)
    parser.add_argument('--aval', type=float, help="If set, this defines the amplitude to be estimated. If not set, the range [0.1, 0.2, ..., 0.9] is used instead (default=None)", default=None)
    parser.add_argument('--fixed_sample', type=int, help="If set, this sets the sampling strategy to do a fixed number of samples at each depth, rather than one that samples more at lower depth and less and longer depth (default=None)", default=None)
    parser.add_argument('--C', type=float, help="This is a free parameter that determines how many shots to take at each step. (default=1.5)", default=1.5)
    args = parser.parse_args()

    print('Command Line Arguments')
    print(args)
    
    pathlib.Path(args.dir).mkdir(parents=True, exist_ok=True) 

    seed = 7
    np.random.seed(seed)


    # In paper, we use 8, but it takes about four hours to run this in total on a 4 core laptop using 4 threads. If you want to just test this out, set num_lenghts to 6 and it should finish within minutes.
    num_lengths = args.num_lengths
    num_mc = args.num_mc

    num_queries = np.zeros(num_lengths, dtype=int)
    max_single_query = np.zeros(num_lengths, dtype=int)
    errors = np.zeros((num_lengths, num_mc), dtype = float)
    thetas = np.zeros((num_lengths, num_mc), dtype = float)
    
    
    num_threads = args.nthreads

    arrays = []

    if args.aval:
        avals = [args.aval]
    else:
        avals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for a in avals:
        theta = np.arcsin(a)
        print(f'a = {a}')
        print(f'theta = {theta}')
        print(f'theta*4/pi = {theta*4/np.pi}')

        filename = args.dir+f'/a{a:0.3f}_mc{num_mc:04d}.pkl'

        for r in range(num_lengths):

            print(f'Trial {r+1} of {num_lengths}')

            espirit = ESPIRIT()
            narray = [2]*(2*r+2)
            arrays.append(narray)
            print(f'Array parameters: {narray}')
            ula_signal = TwoqULASignal(M=narray, C=args.C)

            if args.fixed_sample:
                n_samples = [args.fixed_sample]*len(ula_signal.n_samples)
            else:
                n_samples = ula_signal.n_samples
            print(f'Shots per depth: {n_samples}')
            print(f'C parameter: {args.C:0.3f}')

            # Compute the total number of queries. The additional count of ula_signal.n_samples[0] is to 
            # account for the fact that the Grover oracle has two invocations of the unitary U, but is 
            # preceded by a single invocation of U (see Eq. 2 in paper). This accounts for the shots required
            # for that single U operator, which costs half as much as the Grover oracle.
            # num_queries[r] = 2*np.sum(np.array(ula_signal.depths)*np.array(n_samples)) + n_samples[0]
            num_queries[r] = np.sum(np.array(ula_signal.depths)*np.array(n_samples)) + n_samples[0]/2
            max_single_query[r] = np.max(ula_signal.depths)

            pool = multiprocessing.Pool(num_threads)
            start = time.time()
            processes = [pool.apply_async(run, args=(theta, n_samples, ula_signal, espirit, seed+mci, args.eta)) for mci in range(num_mc)]
            sims = [p.get() for p in processes]
            for k in range(num_mc):
                errors[r,k], thetas[r,k] = sims[k]
            end = time.time()
            print(f'Time for trial {r+1}: {end-start} (s)')

            
            print(f'Number of queries: {num_queries[r]}')
            print(f'Depths: {ula_signal.depths}')
            print(f'Samples: {ula_signal.n_samples}')
            print(f'Max Single Query: {max_single_query[r]}')
            print(f'99% percentile: {np.percentile(errors[r], 99):e}')
            print(f'95% percentile: {np.percentile(errors[r], 95):e}')
            print(f'99% Constant: {num_queries[r] * np.percentile(errors[r], 99):e}')
            print(f'95% Constant: {num_queries[r] * np.percentile(errors[r], 95):e}')
            print(f'99% Max Constant: {max_single_query[r] * np.percentile(errors[r], 99):e}')
            print(f'95% Max Constant: {max_single_query[r] * np.percentile(errors[r], 95):e}')
            print(f'Error per oracle: {args.eta:e}')
            print(f'Maximum Error: {1.0 - (1.0-args.eta)**(max_single_query[r]+1):e}')
            print()
        
        if args.save:
            with open(filename, 'wb') as handle:
                pickle.dump((errors, thetas, num_queries, max_single_query, arrays, num_lengths, num_mc), 
                            handle, protocol=pickle.HIGHEST_PROTOCOL)    