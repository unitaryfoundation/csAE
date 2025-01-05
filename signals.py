import numpy as np
from abc import ABCMeta, abstractmethod
import pickle
from scipy.linalg import toeplitz
from numba import njit

@njit
def get_ula_signal(q, idx, signal):
    p = np.outer(signal, np.conj(signal)).T.ravel()  # Compute outer product
    p = p[idx[0]]  # Restrict to indices
    cp = np.conj(p)
    for i in range(1, q):
        p = np.outer(p, cp).T.ravel() # Compute outer product iteratively
        p = p[idx[i]]  # Restrict to indices
    return p

class ULASignal(metaclass = ABCMeta):
    
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def get_cov_matrix(self):
        pass
    
class TwoqULASignal(ULASignal):

    def __init__(self, M, ula=None, seed=None, C=1.2):
        '''
        Constructor for wrapper class around signal.
            ULA_signal_dict: either a dictionary containing the signal or path to a pickle file to load with the signal
        '''
        if seed: np.random.seed(seed)
        
        if isinstance(M, (list, np.ndarray)):
            self.M = M
            depths, n_samples = self._get_depths(self.M, C=C)
            self.depths = depths
            self.n_samples = n_samples
            self.q = len(self.M)//2 if len(self.M) % 2 == 0 else len(self.M)//2 + 1
            self.idx = self.get_idx()
            self.C = C
            self.measurements = None
        elif isinstance(ula, str):
            with open(ula, 'rb') as handle:
                self.idx, self.depths, self.n_samples, self.M, self.C = pickle.load(handle)
            self.q = len(self.M)//2 if len(self.M) % 2 == 0 else len(self.M)//2 + 1
        else:
            raise TypeError("Input ULA must by array of indices or path to pickle file")

    def save_ula(self, filename='ula.pkl'):
        with open(filename, 'wb') as handle:
            pickle.dump((self.idx, self.depths, self.n_samples, self.M, self.C), handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    
    def get_cov_matrix(self, signal):
        '''
        This generates Eq. 13 in the paper DOI: 10.1109/DSP-SPE.2011.5739227 using the 
        technique from DOI:10.1109/LSP.2015.2409153
        '''
        self.ULA_signal = self.get_ula_signal(self.q, self.idx, signal)
        total_size = len(self.ULA_signal)
        ULA_signal = self.ULA_signal

        '''
        This uses the techinque from DOI:10.1109/LSP.2015.2409153
        '''
        subarray_col = ULA_signal[total_size//2:]
        subarray_row = np.conj(subarray_col)
        covariance_matrix = toeplitz(subarray_col, subarray_row)
        
        
        self.cov_matrix = covariance_matrix
        self.m = np.shape(self.cov_matrix)[0]
        self.R = covariance_matrix
        return covariance_matrix
    

    def get_cov_matrix_toeplitz(self, signal):
        '''
        This generates R tilde of DOI: 10.1109/LSP.2015.2409153 and only stores a column and row, which entirely 
        defines a Toeplitz matrix
        '''
        self.ULA_signal = get_ula_signal(self.q, self.idx, signal)
        total_size = len(self.ULA_signal)
        ULA_signal = self.ULA_signal
        
        subarray_col = ULA_signal[total_size//2:]
        subarray_row = np.conj(subarray_col)
        
        return subarray_col
    
    def get_idx(self):
        virtual_locations = []
        depths = self.depths
        q = self.q
        list_of_idx = []
        difference_matrix = np.zeros((len(depths), len(depths)), dtype=int)
        for r, rval in enumerate(depths):
            for c, cval in enumerate(depths):
                difference_matrix[r][c] = rval-cval
        depths0 = difference_matrix.flatten(order='F')
        depths0, idx = np.unique(depths0, return_index = True)
        new_depths = depths0
        list_of_idx.append(idx)

        virtual_locations.append(depths0)
        for i in range(q-1):
            difference_matrix = np.zeros((len(new_depths), len(depths0)), dtype=int)
            for r, rval in enumerate(new_depths):
                for c, cval in enumerate(depths0):
                    difference_matrix[r][c] = rval-cval
            new_depths = difference_matrix.flatten(order='F')
            new_depths, idx = np.unique(new_depths, return_index = True)
            virtual_locations.append(new_depths)

            if i<q-2:
                list_of_idx.append(idx)

        self.virtual_locations = virtual_locations

        difference_set = new_depths
        a = difference_set
        b = np.diff(a)
        b = b[:len(b)//2]
        try:
            start_idx = np.max(np.argwhere(b>1)) + 1
            list_of_idx.append(idx[start_idx:-start_idx])
        except:
            list_of_idx.append(idx)

        return list_of_idx

    def _get_depths(self, narray, C=1.2):
        physLoc = []
        n_samples = []

        r = (len(narray)-2)//2

        for i,m in enumerate(narray):
            c = int(np.prod(narray[:i]))
            for j in range(m):
                physLoc.append(j*c)

        physLoc = np.sort(list(set(physLoc)))

        for i in range(len(physLoc)):
            x = int((np.ceil(C*(len(physLoc)-i)))) # sims_99
            n_samples.append(x if x!=0 else 1)
        n_samples[0] = n_samples[0] * 2
        return physLoc, n_samples

    @classmethod
    def get_depths_and_samples(cls, narray, C=1.2):
        return cls._get_depths(cls, narray, C=C)

    def set_measurements(self, measurements):
        assert(len(measurements) == len(self.depths)), "Length of measurements does not match length of depths"
        self.measurements = np.array(measurements)

    def get_complex_signal(self, signs):
        if self.measurements is not None:
            theta_cos = 2 * np.arccos(np.sqrt(np.array(self.measurements)))
            csignal = np.cos(theta_cos) + 1.0j * np.array(signs) * np.sin(theta_cos)
            return csignal
        else:
            print("No measurements. Must set measurement values using set_measurements function before using this method.")
            return None
    
    # def estimate_signal(self, n_samples, theta, eta=0.0):
    #     depths = self.depths
    #     signals = np.zeros(len(depths), dtype = np.complex128)
    #     measurements = np.zeros(len(depths))
    #     for i,n in enumerate(depths):
    #         # Get the exact measurement probabilities (assuming access to both z and x basis measurements, which we don't have)
    #         # The x basis measurements are only used for simulation purposes and not for reconstruction
    #         p0 = P0(n, theta)
    #         p0x = P0x(n,theta)
    #
    #         # Get the "noisy" probabilities by sampling and adding a bias term that pushes towards 50/50 mixture
    #         eta_n = (1.0-eta)**(n+1) # The error at depth n increases as more queries are implemented
    #         p0_estimate = np.random.binomial(n_samples[i], eta_n*p0 + (1.0-eta_n)*0.5)/n_samples[i]
    #         p1_estimate = 1.0 - p0_estimate
    #         p0x_estimate = np.random.binomial(n_samples[i], eta_n*p0x + (1.0-eta_n)*0.5)/n_samples[i]
    #         p1x_estimate = 1.0 - p0x_estimate
    #
    #         # Estimate theta
    #         theta_estimated = np.arctan2(p0x_estimate - p1x_estimate, p0_estimate - p1_estimate)
    #
    #         # Compute f(n) - Eq. 3
    #         fi_estimate = np.exp(1.0j*theta_estimated)
    #         signals[i] = fi_estimate
    #         measurements[i] = p0_estimate
    #
    #     return signals, measurements

