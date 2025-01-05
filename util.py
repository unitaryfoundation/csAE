import numpy as np
from scipy.stats import binom
import itertools
from typing import List

def simulate_signal(depths, n_samples, theta, eta=0.0):
    P0 = lambda n, theta: np.cos((2 * n + 1) * theta) ** 2
    P0x = lambda n, theta: (1.0 + np.sin(2 * (2 * n + 1) * theta)) / 2.0

    signals = np.zeros(len(depths), dtype=np.complex128)
    measurements = np.zeros(len(depths))
    signs_exact = [1] * len(depths)
    for i, n in enumerate(depths):
        # Get the exact measurement probabilities (assuming access to both z and x basis measurements, which we don't have)
        # The x basis measurements are only used for simulation purposes and not for reconstruction
        p0 = P0(n, theta)
        p1 = 1.0 - p0
        p0x = P0x(n, theta)
        p1x = 1.0 - p0x

        # Get the "noisy" probabilities by sampling and adding a bias term that pushes towards 50/50 mixture
        eta_n = (1.0 - eta) ** (n + 1)  # The error at depth n increases as more queries are implemented
        p0_estimate = np.random.binomial(n_samples[i], eta_n * p0 + (1.0 - eta_n) * 0.5) / n_samples[i]
        p1_estimate = 1.0 - p0_estimate
        # p0x_estimate = np.random.binomial(n_samples[i], eta_n * p0x + (1.0 - eta_n) * 0.5) / n_samples[i]
        # p1x_estimate = 1.0 - p0x_estimate

        # Estimate theta
        # theta_estimated = np.arctan2(p0x_estimate - p1x_estimate, p0_estimate - p1_estimate)

        theta_cos = 2 * np.arccos(np.sqrt(p0_estimate))
        # theta_cos = np.arccos(np.sqrt(p0_estimate))
        theta_estimated = theta_cos

        # For simulation purposes
        theta_exact = np.arctan2(p0x - p1x, p0 - p1)
        signs_exact[i] = np.sign(np.imag(np.exp(1j * theta_exact)))  # Sign of the sine term

        # Compute f(n) - Eq. 3
        # fi_estimate = np.exp(1.0j * theta_estimated)
        fi_estimate = np.cos(theta_cos) + 1.0j * signs_exact[i] * np.sin(theta_cos)
        signals[i] = fi_estimate
        measurements[i] = p0_estimate

    return signals, measurements

def apply_correction(ula_signal, theta_est):
    theta_est = np.abs(theta_est)
    p_o2 = np.cos((2 * ula_signal.depths + 1) * (theta_est / 2.0)) ** 2
    p_o4 = np.cos((2 * ula_signal.depths + 1) * (theta_est / 4.0)) ** 2
    p_same = np.cos((2 * ula_signal.depths + 1) * (theta_est)) ** 2
    p_s2 = np.cos((2 * ula_signal.depths + 1) * (np.pi / 2 - theta_est)) ** 2
    p_s4 = np.cos((2 * ula_signal.depths + 1) * (np.pi / 4 - theta_est)) ** 2
    p_s2_o2 = np.cos((2 * ula_signal.depths + 1) * (np.pi / 2 - theta_est / 2)) ** 2
    p_s4_o2 = np.cos((2 * ula_signal.depths + 1) * (np.pi / 4 - theta_est / 2)) ** 2


    l_o2 = np.sum(
        np.log(
            [1e-75 + binom.pmf(ula_signal.n_samples[kk] * ula_signal.measurements[kk], ula_signal.n_samples[kk], p_o2[kk]) for kk
             in
             range(len(ula_signal.n_samples))]))
    l_o4 = np.sum(
        np.log(
            [1e-75 + binom.pmf(ula_signal.n_samples[kk] * ula_signal.measurements[kk], ula_signal.n_samples[kk], p_o4[kk]) for kk
             in
             range(len(ula_signal.n_samples))]))
    l_same = np.sum(
        np.log(
            [1e-75 + binom.pmf(ula_signal.n_samples[kk] * ula_signal.measurements[kk], ula_signal.n_samples[kk], p_same[kk]) for kk
             in
             range(len(ula_signal.n_samples))]))
    l_s2 = np.sum(
        np.log(
            [1e-75 + binom.pmf(ula_signal.n_samples[kk] * ula_signal.measurements[kk], ula_signal.n_samples[kk], p_s2[kk]) for kk
             in
             range(len(ula_signal.n_samples))]))
    l_s4 = np.sum(
        np.log(
            [1e-75 + binom.pmf(ula_signal.n_samples[kk] * ula_signal.measurements[kk], ula_signal.n_samples[kk], p_s4[kk]) for kk
             in
             range(len(ula_signal.n_samples))]))
    l_s2_o2 = np.sum(
        np.log(
            [1e-75 + binom.pmf(ula_signal.n_samples[kk] * ula_signal.measurements[kk], ula_signal.n_samples[kk], p_s2_o2[kk]) for
             kk in
             range(len(ula_signal.n_samples))]))
    l_s4_o2 = np.sum(
        np.log(
            [1e-75 + binom.pmf(ula_signal.n_samples[kk] * ula_signal.measurements[kk], ula_signal.n_samples[kk], p_s4_o2[kk]) for
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

    return np.abs(theta_est)

def generate_all_sign_variations(signs_array: np.ndarray) -> np.ndarray:
    """
    Generate all possible sign variations by changing signs at all possible pairs of positions.

    Parameters:
    -----------
    signs_array : numpy.ndarray
        Original array of signs (typically containing 1 or -1)

    Returns:
    --------
    numpy.ndarray
        Array of all possible sign variation arrays
    """
    # Get all possible unique pairs of positions
    n = len(signs_array)
    position_pairs = list(itertools.combinations(range(n), 2))

    # Will store all variations
    all_variations = []

    # Iterate through all possible position pairs
    for pos1, pos2 in position_pairs:
        # Generate variations for this pair of positions
        pair_variations = generate_pair_variations(signs_array, pos1, pos2)
        all_variations.extend(pair_variations)

    return all_variations


def generate_adjacent_sign_variations(signs_array: np.ndarray, size: int) -> np.ndarray:
    """
    Generate sign variations by sliding a two-position window across the array.

    Parameters:
    -----------
    signs_array : numpy.ndarray
        Original array of signs (typically containing 1 or -1)

    Returns:
    --------
    numpy.ndarray
        Array of all possible sign variation arrays
    """
    # Total length of the array
    n = len(signs_array)

    # Will store all variations
    all_variations = []

    # Iterate through adjacent position pairs
    # (0,1), (1,2), (2,3), ... until the second-to-last pair
    for pos1 in range(1, n - size + 1):
        pos = [pos1 + i for i in range(size)]

        # Generate variations for this pair of adjacent positions
        pair_variations = generate_pair_variations(signs_array, pos)

        all_variations.extend(pair_variations)

    return all_variations


def generate_pair_variations(signs_array: np.ndarray, pos: List[int]) -> List[np.ndarray]:
    """
    Generate sign variations for a specific pair of positions.

    Parameters:
    -----------
    signs_array : numpy.ndarray
        Original array of signs
    pos1 : int
        First position to modify
    pos2 : int
        Second position to modify

    Returns:
    --------
    List[numpy.ndarray]
        List of sign variation arrays
    """
    # Validate input positions
    # if pos1 < 0 or pos2 < 0 or pos1 >= len(signs_array) or pos2 >= len(signs_array):
    #     raise ValueError("Positions must be within the array bounds")

    # Generate all possible sign combinations for the two positions
    sign_combinations = list(itertools.product([-1, 1], repeat=len(pos)))

    # Create variations
    variations = []
    for combo in sign_combinations:
        # Create a copy of the original array
        variation = signs_array.copy()

        # Modify the two specified positions
        for i in range(len(combo)):
            variation[pos[i]] *= combo[i]

        variations.append(tuple(variation))

    return variations