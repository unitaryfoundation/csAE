import numpy as np

from frequencyestimator import *
from util import *
import ast

def estimate_amplitude(ula_signal, heavy_signs):
    """
    Estimates the amplitude, a, of a state |psi> = a|0,x> + sqrt{1-a^2}|1,x'>
    given a set of measurements, depths (number of queries), and number of samples taken.

    Usage: amplitude = csae.estimate_amplitude(measurements, depths, n_samples)
    Inputs:
        ula_signal: object of type signal that contains the measurements, depths, and number of samples along with other array parameters
        heavy_signs: Simulated sign distribution obtained by calling the function get_heavy_signs
    Returns:
        amplitude: float containing the estimate of the amplitude of state |0,x>
    """

    espirit = ESPIRIT()
    results = csae_with_local_minimization(ula_signal, espirit, heavy_signs, sample=True, correction=True, optimize=True,
                                           disp=False)

    a = np.sin(results['theta_est'])
    return a

def objective_function(lp, cos_signal, abs_sin, ula_signal, esprit):
    signal = cos_signal + 1.0j * lp * abs_sin
    R = ula_signal.get_cov_matrix_toeplitz(signal)
    theta_est, _ = esprit.estimate_theta_toeplitz(R)

    theta_est = apply_correction(ula_signal, theta_est)

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

def get_heavy_signs(depths, n_samples, steps):
    thetas = np.linspace(np.arcsin(0.09), np.arcsin(0.91), steps)
    avals = np.sin(thetas)

    num_mc = 1000

    sign_distributions = {}

    for a, theta in zip(avals, thetas):
        # theta = np.arcsin(a)
        distr = {}
        for i in range(num_mc):
            signal, _ = simulate_signal(depths, n_samples, theta)
            # print(measurements)
            signs = tuple(np.sign(np.imag(signal)))
            # print(signs)

            distr[signs] = distr.get(signs, 0.0) + 1 / num_mc
        sign_distributions[str(a)] = distr

    ret_dict = {}
    for a in avals:
        distr = get_signs(sign_distributions[str(a)])
        ret_dict[str(a)] = distr

    # Normalize the values
    normalized_data = {}
    for key, sub_dict in ret_dict.items():
        total_sum = sum(sub_dict.values())
        normalized_data[key] = {k: v / total_sum for k, v in sub_dict.items()}

    return normalized_data

def get_signs(sign_distribution):
    # Sort the dictionary by values in descending order
    sorted_data = dict(sorted(sign_distribution.items(), key=lambda item: item[1], reverse=True))

    # Create a new dictionary with cumulative sum close to 0.68
    cumulative_dict = {}
    cumulative_sum = 0.0

    for key, value in sorted_data.items():
        cumulative_dict[key] = value
        if cumulative_sum + value > 0.9:
            break
        cumulative_sum += value

    # Output the result
    return cumulative_dict

# Taking a random sample based on the normalized probabilities
def sample_from_normalized(data, sample_size=1):
    keys = list(map(str, list(data.keys())))
    probabilities = list(data.values())
    sampled_keys = np.random.choice(a=keys, size=sample_size, p=probabilities)
    return [ast.literal_eval(t) for t in sampled_keys]


def sample_signs(heavy_signs, sample_size=3):
    """
    Sample a set of sign variations from a learned sign distribution.

    Args:
        heavy_signs (dict): A dictionary where the keys are strings representing float values,
                           and the values are dictionaries with keys as sign vectors and values
                           as their corresponding probabilities.
        sample_size (int, optional): The number of sign variations to sample. Defaults to 3.

    Returns:
        dict: A dictionary where the keys are the same as the keys in `heavy_signs`,
              and the values are lists of sampled sign variations.
    """
    avals = list(heavy_signs.keys())
    signs_to_try = {}
    for a in avals:
        signs_to_try[a] = list(set(sample_from_normalized(heavy_signs[a], sample_size=sample_size)))

    return signs_to_try


def avals_to_usef(a0, avals, L=3):
    """
    Find the `L` values in `avals` that are closest to the given `a0` value.

    Args:
        a0 (float): The reference value to find the closest `L` values for.
        avals (list): A list of float values representing the keys in `heavy_signs`.
        L (int, optional): The number of closest values to return. Defaults to 3.

    Returns:
        list: A list of string representations of the `L` closest values to `a0` in `avals`.
    """
    avals = list(map(float, avals))

    left = 0
    right = len(avals)

    while left < right:
        mid = (left + right) // 2

        if avals[mid] <= a0:
            left = mid + 1
        else:
            right = mid

    if left >= 4:
        avals_to_use = list(map(str, avals[left - L: left + L]))
    if left >= 3:
        avals_to_use = list(map(str, avals[left - 3: left + L]))
    elif left >= 2:
        avals_to_use = list(map(str, avals[left - 2: left + L]))
    elif left >= 1:
        avals_to_use = list(map(str, avals[left - 1: left + L]))
    else:
        avals_to_use = list(map(str, avals[left: left + L]))

    return avals_to_use


def all_signs_to_try(avals_to_use, signs_to_try, adjacency=2):
    """
    Generate all possible sign variations within a Hamming distance of 2 from the given `avals_to_use`.

    Args:
        avals_to_use (list): A list of string representations of float values.
        signs_to_try (dict): A dictionary where the keys are string representations of float values,
                            and the values are lists of sign variations.
        adjacency (int, optional): The maximum Hamming distance to consider for the sign variations.
                                  Defaults to 2.

    Returns:
        list: A list of all possible sign variations within the specified Hamming distance.
    """
    all_signs = []
    for a in avals_to_use:
        for x in signs_to_try[a]:
            hamming_distance_two_signs = generate_adjacent_sign_variations(np.array(x), adjacency)
            all_signs.extend(hamming_distance_two_signs)
    all_signs = list(set(all_signs))

    return all_signs


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
    # obj = 1.0
    x_star = np.array(all_signs[0])
    obj = objective_function(x_star, cos_signal, abs_sin, ula_signal, esprit)
    for x in all_signs:
        curr_obj = objective_function(np.array(x), cos_signal, abs_sin, ula_signal, esprit)
        if curr_obj < obj:
            if disp:
                print(f'current objective: {curr_obj}')
                print(f'current best signs: {x}')
            obj = curr_obj
            x_star = np.array(x)

    return x_star


def csae_with_local_minimization(ula_signal, esprit, heavy_signs, sample=False, correction=False, optimize=False,
                                 disp=False):
    """
    Perform CSAE (Compressive Sensing Angle Estimation) with local minimization.

    Args:
        theta (float): The true angle of the signal.
        ula_signal (ULASignal): An instance of the ULASignal class, containing information about the signal.
        esprit (ESPRIT): An instance of the ESPRIT class, used for estimating the signal parameters.
        heavy_signs (dict): A dictionary where the keys are strings representing float values,
                           and the values are dictionaries with keys as sign vectors and values
                           as their corresponding probabilities.
        sample (bool, optional): If True, samples sign variations from the learned sign distribution.
                                If False, uses the learned sign distribution directly. Defaults to False.
        correction (bool, optional): If True, applies a correction to the estimated angle. Defaults to False.
        optimize (bool, optional): If True, uses a sliding window approach to find the best sign variation.
                                  If False, uses the learned sign distribution directly. Defaults to False.
        disp (bool, optional): If True, prints additional information during the process. Defaults to False.

    Returns:
        dict: A dictionary containing the estimated angles, errors, number of queries, maximum depth,
              and other relevant information.
    """


    depths = ula_signal.depths
    n_samples = ula_signal.n_samples
    # csignal, measurements = estimate_signal(depths, n_samples, theta)
    # ula_signal.measurements = measurements
    # cos_signal = np.real(csignal)
    cos_signal = np.cos(2*np.arccos(np.sqrt(ula_signal.measurements)))

    # Get a complex signal with just signs of 1
    csignal = ula_signal.get_complex_signal([1]*len(ula_signal.measurements))

    # correct_signs = np.sign(np.imag(csignal))
    # if disp:
    #     print(f'correct signs: {correct_signs}')
    abs_sin = np.abs(np.imag(csignal))

    if sample:
        # step 1: sample signs from learned sign distribution
        signs_to_try = sample_signs(heavy_signs=heavy_signs, sample_size=3)
        avals = list(heavy_signs.keys())

        if optimize:
            # step 2: using rough estimate where the amplitude is, use the a-values around that estimate
            a0 = np.sqrt(0.5 - 0.5 * cos_signal[0])
            avals_to_use = avals_to_usef(a0, avals, L=3)

            if disp:
                print(f'rough estimate a: {a0}')
                print(f'avals to use: {avals_to_use}')

            # step 3: now vary the signs in a sliding window of size "adjacency"
            all_signs = all_signs_to_try(avals_to_use, signs_to_try, adjacency=2)

            if disp:
                print(f'number of signs Hamming distance two: {len(all_signs)}')

            # step 4: try all the signs pick the ones that minimize the objective function
            if disp:
                print('debug')
                print(all_signs)
            x_star = minimize_obj(all_signs, cos_signal, abs_sin, ula_signal, esprit, disp)

            # step 5 (optional): do one more sweep
            hamming_distance_one_signs = list(set(generate_adjacent_sign_variations(x_star, 1)))
            x_star = minimize_obj(hamming_distance_one_signs, cos_signal, abs_sin, ula_signal, esprit, disp)

        else:
            # here we don't vary the signs using a sliding window, but directly use the learned sign distribution (poor performance)
            all_signs = []
            for a in avals:
                all_signs.extend(heavy_signs[a])
            all_signs = list(set(all_signs))
            x_star = minimize_obj(all_signs, cos_signal, abs_sin, ula_signal, esprit, disp)

    else:
        avals = list(heavy_signs.keys())
        signs_to_try = {}
        for a in avals:
            signs_to_try[a] = heavy_signs[a]

        if optimize:
            # step 2: using rough estimate where the amplitude is, use the a-values around that estimate
            a0 = np.sqrt(0.5 - 0.5 * cos_signal[0])
            avals_to_use = avals_to_usef(a0, avals, L=3)

            if disp:
                print(f'rough estimate a: {a0}')
                print(f'avals to use: {avals_to_use}')

            # step 3: now vary the signs in a sliding window of size "adjacency"
            all_signs = all_signs_to_try(avals_to_use, signs_to_try, adjacency=2)

            if disp:
                print(f'number of signs Hamming distance two: {len(all_signs)}')

            # step 4: try all the signs pick the ones that minimize the objective function
            x_star = minimize_obj(all_signs, cos_signal, abs_sin, ula_signal, esprit, disp, measu)

            # step 5 (optional): do one more sweep
            hamming_distance_one_signs = list(set(generate_adjacent_sign_variations(x_star, 1)))
            x_star = minimize_obj(hamming_distance_one_signs, cos_signal, abs_sin, ula_signal, esprit, disp)

        else:
            # here we don't vary the signs using a sliding window, but directly use the learned sign distribution (poor performance)
            all_signs = []
            for a in avals:
                all_signs.extend(heavy_signs[a])
            all_signs = list(set(all_signs))
            x_star = minimize_obj(all_signs, cos_signal, abs_sin, ula_signal, esprit, disp)

    # Optimization is done.

    if disp:
        print(x_star)

    signal = cos_signal + 1.0j * x_star * abs_sin
    R = ula_signal.get_cov_matrix_toeplitz(signal)
    theta_est, _ = esprit.estimate_theta_toeplitz(R)
    theta_est = np.abs(theta_est)
    if correction:
        theta_est = apply_correction(ula_signal, theta_est)  # apply correction


    # cR = ula_signal.get_cov_matrix_toeplitz(csignal)
    # theta_est1, _ = esprit.estimate_theta_toeplitz(cR)

    # compute queries required
    num_queries = np.sum(np.array(ula_signal.depths) * np.array(ula_signal.n_samples)) + ula_signal.n_samples[0]
    max_single_query = np.max(ula_signal.depths)

    ret_dict = {'theta_est': theta_est,
                'queries': num_queries, 'depth': max_single_query,
                'x_star': x_star}

    return ret_dict