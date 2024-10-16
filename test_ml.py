import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from signals import *
from frequencyestimator import *
import itertools
import math
from scipy.stats import binom
import torch
from ml_optsigns import SignModel

sns.set_style("whitegrid")
sns.despine(left=True, bottom=True)
sns.set_context("poster", font_scale = .45, rc={"grid.linewidth": 0.8})

# For reproducibility
#22, 26
np.random.seed(14)
# Set the per oracle noise parameter (See Eq. 18)
eta = 0
# Set the array parameters (See Thm. II.2 and Eq. 12)
narray = [2, 2, 2, 2, 2, 2]
# Set the actual amplitude
a = 0.2
theta = np.arcsin(a)
print(theta)
print(theta/np.pi)
C=5.0
ula_signal = TwoqULASignal(M=narray, C=C)
print(ula_signal.depths)
NUM_FEATURES = len(ula_signal.depths)
NUM_CLASSES = len(ula_signal.depths) - 1

file_subscript = ''
for x in narray:
    file_subscript += f'{x}'
filename = f'ml_models/sign_model_{file_subscript}_C{C:0.2f}.pt'

sign_model = SignModel(input_features=NUM_FEATURES,
                        output_features=NUM_CLASSES,
                        hidden_units=16*NUM_FEATURES).to('cpu')

file_subscript = ''
for x in narray:
    file_subscript += f'{x}'
sign_model.load_state_dict(torch.load(filename, weights_only=True))
sign_model.eval()

# This sets up the simulation that simulates the measured amplitudes at the various physical locations.
# It uses a C=1.5 value, which corresponds to the sampling schedule given in Eq. 16. The variable C here
# is the parameter K in the paper.


# Number of Monte Carlo trials used to estimate statistics. We tend to use 500 in the paper. Choose 100 here for speed.
num_mc = 1
num_median = 3
thetas = np.zeros(num_mc, dtype=float)
errors = np.zeros(num_mc, dtype=float)

# Sets up the ESPIRIT object to estimate the amplitude
espirit = ESPIRIT()

for k in range(num_mc):
    print(f'Trial {k+1} of {num_mc}')
    # This estimates the covariance matrix of Eq. 8 using the approch given in DOI:10.1109/LSP.2015.2409153
    median_theta = np.zeros(num_median, dtype=float)
    ula_signal.estimate_signal(n_samples=ula_signal.n_samples, theta=theta, eta=eta, offset=0.0)
    X = torch.from_numpy(ula_signal.measurements).type(torch.float)
    # signs = [-1+2*(sign_model(X) > 0.5).float().numpy()]
    signs = -1 + 2 * (sign_model(X) > 0.5).float().numpy()
    signs = [1.0] + list(signs)
    print(signs)

    for j in range(num_median):
        # offset = 0.1*np.pi # figure this out
        offset = np.random.uniform(0.0, 0.25*np.pi)
        offset = 0.0
        # offset = 0.07957110621802466*np.pi
        # print(signs)
        # signal = ula_signal.estimate_signal(n_samples=ula_signal.n_samples, theta=theta, eta=eta, signs=signs)
        # offset_signal = ula_signal.add_signal_offset(ula_signal.signal, offset)
        # signal = ula_signal.update_signal_signs(offset_signal, signs)
        noise_signal = ula_signal.add_signal_noise(ula_signal.signal, 0.0001)
        signal = ula_signal.update_signal_signs(noise_signal, signs)
        # signal = ula_signal.add_signal_offset(signal, offset)
        R = ula_signal.get_cov_matrix_toeplitz(signal)
        # This estimates the angle using the ESPIRIT algorithm
        theta_est, eigs = espirit.estimate_theta_toeplitz(R)
        print(f'theta_est: {theta_est/np.pi}')
        print(f'offset:    {offset/np.pi}')

        # if offset < theta_est:
        if -np.angle(eigs) < 0:
            theta_est = theta_est - offset
        else:
            theta_est = np.pi / 2.0 - theta_est - offset
        print(f'theta_est: {theta_est / np.pi}')

        if math.isclose(np.linalg.norm(np.array(ula_signal.signs_exact) - np.array(signs)), 0):
            print(f'     CORRECT ANSWER FOUND')
            print(f'     angle {-np.angle(eigs) / np.pi / 4}')
            print(f'     angle {theta_est/np.pi}')

        p_exact = np.cos((2*ula_signal.depths+1)*(theta))**2
        p_o2 = np.cos((2 * ula_signal.depths + 1) * (theta_est/2.0)) ** 2
        p_o4 = np.cos((2 * ula_signal.depths + 1) * (theta_est / 4.0)) ** 2
        p_same = np.cos((2*ula_signal.depths+1)*(theta_est))**2
        p_s2 = np.cos((2 * ula_signal.depths + 1) * (np.pi/2-theta_est)) ** 2
        p_s4 = np.cos((2 * ula_signal.depths + 1) * (np.pi / 4 - theta_est)) ** 2
        p_s2_o2 = np.cos((2 * ula_signal.depths + 1) * (np.pi / 2 - theta_est/2)) ** 2
        p_s4_o2 = np.cos((2 * ula_signal.depths + 1) * (np.pi / 4 - theta_est/2)) ** 2

        l_exact = np.sum(
            np.log([1e-75 + binom.pmf(ula_signal.n_samples[kk] * ula_signal.measurements[kk], ula_signal.n_samples[kk],
                                      p_exact[kk]) for kk in
                    range(len(ula_signal.n_samples))]))

        l_o2 = np.sum(
            np.log([1e-75+binom.pmf(ula_signal.n_samples[kk]*ula_signal.measurements[kk], ula_signal.n_samples[kk], p_o2[kk]) for kk in
             range(len(ula_signal.n_samples))]))
        l_o4 = np.sum(
            np.log([1e-75+binom.pmf(ula_signal.n_samples[kk] * ula_signal.measurements[kk], ula_signal.n_samples[kk], p_o4[kk]) for kk in
             range(len(ula_signal.n_samples))]))
        l_same = np.sum(
            np.log([1e-75+binom.pmf(ula_signal.n_samples[kk] * ula_signal.measurements[kk], ula_signal.n_samples[kk], p_same[kk]) for kk in
             range(len(ula_signal.n_samples))]))
        l_s2 = np.sum(
            np.log([1e-75+binom.pmf(ula_signal.n_samples[kk] * ula_signal.measurements[kk], ula_signal.n_samples[kk], p_s2[kk]) for kk in
             range(len(ula_signal.n_samples))]))
        l_s4 = np.sum(
            np.log([1e-75+binom.pmf(ula_signal.n_samples[kk] * ula_signal.measurements[kk], ula_signal.n_samples[kk], p_s4[kk]) for kk in
             range(len(ula_signal.n_samples))]))
        l_s2_o2 = np.sum(
            np.log([1e-75+binom.pmf(ula_signal.n_samples[kk] * ula_signal.measurements[kk], ula_signal.n_samples[kk], p_s2_o2[kk]) for kk in
             range(len(ula_signal.n_samples))]))
        l_s4_o2 = np.sum(
            np.log([1e-75+binom.pmf(ula_signal.n_samples[kk] * ula_signal.measurements[kk], ula_signal.n_samples[kk], p_s4_o2[kk]) for kk in
             range(len(ula_signal.n_samples))]))
        print(f'theta_exact obj:           {l_exact}\n')

        print(f'2*theta_found obj:         {l_same}')
        print(f'2*theta found:             {2*theta_est / np.pi}\n')

        print(f'pi-2*theta_found obj:      {l_s2}')
        print(f'pi-2*theta found:          {1.0-2 * theta_est / np.pi}\n')

        print(f'pi/2-2*theta_found obj:    {l_s4}')
        print(f'pi/2-2*theta found:        {0.5 - 2 * theta_est / np.pi}\n')

        print(f'theta_found obj:           {l_o2}')
        print(f'theta found:               {theta_est / np.pi}\n')

        print(f'theta_found/2 obj:         {l_o4}')
        print(f'theta found:               {theta_est / 2.0 / np.pi}\n')

        print(f'pi - theta_found obj:    {l_s2_o2}')
        print(f'pi - theta found:        {1.0 - theta_est / np.pi}\n')

        print(f'pi/2 - theta_found obj:    {l_s4_o2}')
        print(f'pi/2 - theta found:        {0.5 - theta_est / np.pi}\n')

        print('FINAL ANGLE FOUND')
        which_correction = np.argmax([l_same, l_s2, l_s4, l_o2, l_o4, l_s2_o2, l_s4_o2])
        if which_correction == 1:
            theta_est = np.pi/2.0 - theta_est
        elif which_correction == 2:
            theta_est = np.pi/4.0 - theta_est
        elif which_correction == 3:
            theta_est = 0.5*theta_est
        elif which_correction == 4:
            theta_est = 0.25*theta_est
        elif which_correction == 5:
            theta_est = np.pi / 2.0 - 0.5 * theta_est
        elif which_correction == 6:
            theta_est = np.pi / 4.0 - 0.5 * theta_est

        print(f'2*theta corrected:         {2*theta_est / np.pi}')
        print(f'2*theta exact:             {2*theta / np.pi}')
        # print(f'Correct Objective: {correct_objective}')
        print(f'Signs found: {signs}')
        print(f'Signs exact: {ula_signal.signs_exact}\n')

        median_theta[j] = theta_est

    print(median_theta)
    thetas[k] = np.median(median_theta)
    errors[k] = np.abs(np.abs(np.sin(theta)) - np.abs(np.sin(thetas[k])))

# Compute the total number of queries. The additional count of ula_signal.n_samples[0]/2 is to
# account for the fact that the Grover oracle has two invocations of the unitary U, but is
# preceded by a single invocation of U (see Eq. 2 in paper). This accounts for the shots required
# for that single U operator, which costs half as much as the Grover oracle.
num_queries = np.sum(np.array(ula_signal.depths) * np.array(ula_signal.n_samples)) + ula_signal.n_samples[0] / 2
# Compute the maximum single query
max_single_query = np.max(ula_signal.depths)

print(f'Array parameters: {narray}')
print(f'Number of queries: {num_queries}')
print(f'theta: {theta}')
print(f'Ave theta estimated: {np.mean(thetas)}')
print(f'a = {a}; a_est = {np.sin(np.mean(thetas))}')
print(f'Max Single Query: {max_single_query}')
print(f'99% percentile: {np.percentile(errors, 99):e}')
print(f'95% percentile: {np.percentile(errors, 95):e}')
print(f'68% percentile: {np.percentile(errors, 68):e}')
print(f'99% percentile constant: {np.percentile(errors, 99) * num_queries:f}')
print(f'95% percentile constant: {np.percentile(errors, 95) * num_queries:f}')
print(f'68% percentile constant: {np.percentile(errors, 68) * num_queries:f}')
print()

signal = ula_signal.update_signal_signs(ula_signal.signal, ula_signal.signs_exact)
ula_signal_exact = ula_signal.get_ula_signal(signal)
# signal = ula_signal.update_signal_signs([1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0])
# ula_signal_bad = ula_signal.get_ula_signal(signal)
# plt.plot(np.real(ula_signal_exact))
# plt.plot(np.real(ula_signal_found))
# plt.plot(np.real(ula_signal_bad))
# plt.plot(np.real(ula_signal.measurements))
# plt.plot(np.cos((2*ula_signal.depths+1)*(thetas[k]))**2)
# plt.plot(np.cos((2*ula_signal.depths+1)*(np.pi/2-thetas[k]))**2)
# plt.plot(np.cos((2*ula_signal.depths+1)*(np.pi/4-thetas[k]))**2)
# plt.show()

print(np.cos((2*np.array(ula_signal.depths)+1)*0.2013579207903308))
print(np.cos((2*np.array(ula_signal.depths)+1)*0.05923209*np.pi))
print()
print(np.sin((2*np.array(ula_signal.depths)+1)*0.2013579207903308))
print(np.sin((2*np.array(ula_signal.depths)+1)*0.05923209*np.pi))