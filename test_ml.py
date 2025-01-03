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


def objective_function_ll(lp, cos_signal, abs_sin, ula_signal, esprit):
    signal = cos_signal + 1.0j * lp * abs_sin
    R = ula_signal.get_cov_matrix_toeplitz(signal)
    theta_est, _ = esprit.estimate_theta_toeplitz(R)

    theta_est = apply_correction(ula_signal, ula_signal.measurements, theta_est, theta_est)

    # print(f'2*theta_est: {2*theta_est}')
    p_same = np.cos((2 * ula_signal.depths + 1) * (theta_est)) ** 2

    obj = -np.sum(
        np.log(
            [1e-75 + binom.pmf(ula_signal.n_samples[kk] * ula_signal.measurements[kk], ula_signal.n_samples[kk], p_same[kk]) for kk
             in
             range(len(ula_signal.n_samples))]))


    # eigs = np.abs(esprit.eigs)[:2]
    # obj = eigs[1] - eigs[0]

    return obj
def objective_function(lp, cos_signal, abs_sin, ula_signal, esprit):
    signal = cos_signal + 1.0j * lp * abs_sin
    R = ula_signal.get_cov_matrix_toeplitz(signal)
    _, _ = esprit.estimate_theta_toeplitz(R)
    eigs = np.abs(esprit.eigs)[:2]
    obj = eigs[1] - eigs[0]

    return obj

def minimize_obj(all_signs, cos_signal, abs_sin, ula_signal, esprit, disp):
    """
    Find the sign variation that minimizes the objective function.

    Args:
        all_signs (list): A list of all possible sign variations to consider.
        cos_signal (numpy.ndarray): The real part of the estimated signal.
        abs_sin (numpy.ndarray): The absolute value of the imaginary part of the estimated signal.
        ula_signal (ULASignal): An instance of the ULASignal class, containing information about the signal.
        esprit (ESPRIT): An instance of the ESPRIT class, used for estimating the signal parameters.
        disp (bool): If True, prints the current objective and the current best sign variation.

    Returns:
        numpy.ndarray: The sign variation that minimizes the objective function.
    """
    x_star = all_signs[0]
    obj_ll = objective_function_ll(np.array(all_signs[0]), cos_signal, abs_sin, ula_signal, esprit)
    obj_eig = objective_function(np.array(all_signs[0]), cos_signal, abs_sin, ula_signal, esprit)
    if disp:
        print('original objective ll : ', obj_ll)
        print('original objective eig: ', obj_eig)

    for x in all_signs:
        curr_obj_ll = objective_function_ll(np.array(x), cos_signal, abs_sin, ula_signal, esprit)
        curr_obj_eig = objective_function(np.array(x), cos_signal, abs_sin, ula_signal, esprit)
        # print(f'current objective: {curr_obj}')
        # print(f'current best signs: {x}')
        # if (curr_obj_ll*1 < obj_ll) and (curr_obj_eig*1 < obj_eig):
        if (curr_obj_ll * 1 < obj_ll):
        # if (curr_obj_eig * 1 < obj_eig):
            if disp:
                print(f'new objective ll:  {curr_obj_ll}')
                print(f'new objective eig: {curr_obj_eig}')
                print(f'new best signs: {x}')
            obj_ll = curr_obj_ll
            obj_eig = curr_obj_eig
            x_star = np.array(x)

    return x_star




def apply_correction(ula_signal, measurements, theta_est, theta):
    theta_est = np.abs(theta_est)
    p_exact = np.cos((2 * ula_signal.depths + 1) * (theta)) ** 2
    p_neg = np.cos((2 * ula_signal.depths + 1) * (-theta)) ** 2
    p_o2 = np.cos((2 * ula_signal.depths + 1) * (theta_est / 2.0)) ** 2
    p_o4 = np.cos((2 * ula_signal.depths + 1) * (theta_est / 4.0)) ** 2
    p_same = np.cos((2 * ula_signal.depths + 1) * (theta_est)) ** 2
    p_s2 = np.cos((2 * ula_signal.depths + 1) * (np.pi / 2 - theta_est)) ** 2
    p_s4 = np.cos((2 * ula_signal.depths + 1) * (np.pi / 4 - theta_est)) ** 2
    p_s2_o2 = np.cos((2 * ula_signal.depths + 1) * (np.pi / 2 - theta_est / 2)) ** 2
    p_s4_o2 = np.cos((2 * ula_signal.depths + 1) * (np.pi / 4 - theta_est / 2)) ** 2

    l_exact = np.sum(
        np.log([1e-75 + binom.pmf(ula_signal.n_samples[kk] * measurements[kk], ula_signal.n_samples[kk],
                                  p_exact[kk]) for kk in
                range(len(ula_signal.n_samples))]))
    l_neg = np.sum(
        np.log([1e-75 + binom.pmf(ula_signal.n_samples[kk] * measurements[kk], ula_signal.n_samples[kk],
                                  p_neg[kk]) for kk in
                range(len(ula_signal.n_samples))]))
    l_o2 = np.sum(
        np.log(
            [1e-75 + binom.pmf(ula_signal.n_samples[kk] * measurements[kk], ula_signal.n_samples[kk], p_o2[kk]) for kk
             in
             range(len(ula_signal.n_samples))]))
    l_o4 = np.sum(
        np.log(
            [1e-75 + binom.pmf(ula_signal.n_samples[kk] * measurements[kk], ula_signal.n_samples[kk], p_o4[kk]) for kk
             in
             range(len(ula_signal.n_samples))]))
    l_same = np.sum(
        np.log(
            [1e-75 + binom.pmf(ula_signal.n_samples[kk] * measurements[kk], ula_signal.n_samples[kk], p_same[kk]) for kk
             in
             range(len(ula_signal.n_samples))]))
    l_s2 = np.sum(
        np.log(
            [1e-75 + binom.pmf(ula_signal.n_samples[kk] * measurements[kk], ula_signal.n_samples[kk], p_s2[kk]) for kk
             in
             range(len(ula_signal.n_samples))]))
    l_s4 = np.sum(
        np.log(
            [1e-75 + binom.pmf(ula_signal.n_samples[kk] * measurements[kk], ula_signal.n_samples[kk], p_s4[kk]) for kk
             in
             range(len(ula_signal.n_samples))]))
    l_s2_o2 = np.sum(
        np.log(
            [1e-75 + binom.pmf(ula_signal.n_samples[kk] * measurements[kk], ula_signal.n_samples[kk], p_s2_o2[kk]) for
             kk in
             range(len(ula_signal.n_samples))]))
    l_s4_o2 = np.sum(
        np.log(
            [1e-75 + binom.pmf(ula_signal.n_samples[kk] * measurements[kk], ula_signal.n_samples[kk], p_s4_o2[kk]) for
             kk in
             range(len(ula_signal.n_samples))]))

    which_correction = np.argmax([l_same, l_s2, l_s4, l_o2, l_o4, l_s2_o2, l_s4_o2])
    if which_correction == 1:
        theta_est = np.pi / 2.0 - theta_est
    elif which_correction == 2:
        theta_est = np.pi / 4.0 - theta_est
    elif which_correction == 3:
        theta_est = theta_est / 2
    elif which_correction == 4:
        theta_est = theta_est / 4
    elif which_correction == 5:
        theta_est = np.pi / 2.0 - 0.5 * theta_est
    elif which_correction == 6:
        theta_est = np.pi / 4.0 - 0.5 * theta_est
    # elif which_correction == 7:
    #     theta_est = -theta_est

    # print(f'FINAL ANGLE FOUND: {theta_est, theta}')

    return np.abs(theta_est)

# For reproducibility
#22, 26
np.random.seed(14)
# Set the per oracle noise parameter (See Eq. 18)
eta = 0
# Set the array parameters (See Thm. II.2 and Eq. 12)
narray = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
# Set the actual amplitude
a = 0.2
theta = np.arcsin(a)
print(theta)
print(theta/np.pi)
C=5.0
thresh = 1.5
# Number of Monte Carlo trials used to estimate statistics. We tend to use 500 in the paper. Choose 100 here for speed.
num_mc = 100


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

thetas = np.zeros(num_mc, dtype=float)
errors = np.zeros(num_mc, dtype=float)

# Sets up the ESPIRIT object to estimate the amplitude
espirit = ESPIRIT()

percent_correct = 0

for k in range(num_mc):
    print(f'Trial {k+1} of {num_mc}')
    # This estimates the covariance matrix of Eq. 8 using the approch given in DOI:10.1109/LSP.2015.2409153
    ula_signal.estimate_signal(n_samples=ula_signal.n_samples, theta=theta, eta=eta, offset=0.0)
    X = torch.from_numpy(ula_signal.measurements).type(torch.float)
    # signs = [-1+2*(sign_model(X) > 0.5).float().numpy()]
    sign_confidence = sign_model(X).detach().numpy()
    print('sign_confidence: ', sign_confidence)
    sign_confidence_sort = np.argsort(np.abs(sign_model(X).detach().numpy()))
    # print(sign_confidence)
    num_signs_to_switch = np.sum(np.abs(sign_confidence) < thresh)
    signs_to_switch = sign_confidence_sort[:num_signs_to_switch]
    print(f'signs to switch: {signs_to_switch+2}')
    sign_combinations = [s for s in itertools.product([1.0, -1.0], repeat=num_signs_to_switch)]
    # print(sign_combinations)

    signs = -1 + 2 * (sign_model(X) > 0.5).float().numpy()
    signs = [1.0] + list(signs)
    # print(signs)

    all_signs = [[x for x in signs] for _ in range(len(sign_combinations))]
    # print("before")
    # print(all_signs)
    for i in range(len(sign_combinations)):
        for j in range(num_signs_to_switch):
            all_signs[i][signs_to_switch[j]+1] = sign_combinations[i][j]
    # print("after")
    # print(all_signs)
    # for combo in sign_combinations:
    #     row_iteration = signs_to_sw

    cos_signal = np.real(ula_signal.signal)
    abs_sin    = np.abs(np.imag(ula_signal.signal))
    # x_star = all_signs[0]
    x_star = minimize_obj(all_signs, cos_signal, abs_sin, ula_signal, espirit, True)
    print("x_star: ", x_star)
    signal = cos_signal + 1.0j * np.array(x_star) * abs_sin
    # signal = ula_signal.update_signal_signs(ula_signal.signs_exact, x_star)

    # signal = ula_signal.update_signal_signs(ula_signal.signal, signs)
    R = ula_signal.get_cov_matrix_toeplitz(signal)
    # This estimates the angle using the ESPIRIT algorithm
    theta_est, eigs = espirit.estimate_theta_toeplitz(R)
    print(f'theta_est: {theta_est/np.pi}')

    if math.isclose(np.linalg.norm(np.array(ula_signal.signs_exact) - np.array(x_star)), 0):
        percent_correct += 1/num_mc
        print(f'     CORRECT ANSWER FOUND')
        print(f'     angle {-np.angle(eigs) / np.pi / 4}')
        print(f'     angle {theta_est/np.pi}')

    theta_est = apply_correction(ula_signal, ula_signal.measurements, theta_est, theta)

    print(f'2*theta corrected:         {2*theta_est / np.pi}')
    print(f'2*theta exact:             {2*theta / np.pi}')
    # print(f'Correct Objective: {correct_objective}')
    print(f'Signs found: {x_star}')
    print(f'Signs exact: {ula_signal.signs_exact}\n')



    thetas[k] = theta_est
    errors[k] = np.abs(np.abs(np.sin(theta)) - np.abs(np.sin(thetas[k])))

# Compute the total number of queries. The additional count of ula_signal.n_samples[0]/2 is to
# account for the fact that the Grover oracle has two invocations of the unitary U, but is
# preceded by a single invocation of U (see Eq. 2 in paper). This accounts for the shots required
# for that single U operator, which costs half as much as the Grover oracle.
num_queries = np.sum(np.array(ula_signal.depths) * np.array(ula_signal.n_samples)) + ula_signal.n_samples[0] / 2
# Compute the maximum single query
max_single_query = np.max(ula_signal.depths)

print(f'Percent Signs Correct: {percent_correct*100:0.2f}%')
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