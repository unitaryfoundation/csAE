import numpy as np
from abc import ABCMeta, abstractmethod
import pickle
from scipy.linalg import toeplitz
from numba import njit

P0 = lambda n, theta: np.cos((2*n+1)*theta)**2
P1 = lambda n, theta: np.sin((2*n+1)*theta)**2
P0x = lambda n, theta: (1.0 + np.sin(2*(2*n+1)*theta))/2.0
P1x = lambda n, theta: (1.0 - np.sin(2*(2*n+1)*theta))/2.0

P00 = lambda n, theta: 0.25*(np.cos(theta) + np.cos((2*n+1)*theta))**2
P10 = lambda n, theta: 0.25*(np.cos(theta) - np.cos((2*n+1)*theta))**2
P11 = lambda n, theta: 0.25*(np.sin(theta) - np.sin((2*n+1)*theta))**2
P01 = lambda n, theta: 0.25*(np.sin(theta) + np.sin((2*n+1)*theta))**2

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

    def __init__(self, M=None, ula=None, seed=None, C=1.2):
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
        elif isinstance(ula, str):
            with open(ula, 'rb') as handle:
                self.idx, self.depths, self.n_samples, self.M = pickle.load(handle)
            self.q = len(self.M)//2 if len(self.M) % 2 == 0 else len(self.M)//2 + 1
        else:
            raise TypeError("Input ULA must by array of indices or path to pickle file")

    def save_ula(self, filename='ula.pkl'):
        with open(filename, 'wb') as handle:
            pickle.dump((self.idx, self.depths, self.n_samples, self.M), handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    
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
            # x = int((np.ceil(C*(len(physLoc)-i/2)))) # sims_99
            n_samples.append(x if x!=0 else 1)

        return physLoc, n_samples
    
    def estimate_signal(self, n_samples, theta, eta=0.0):
        depths = self.depths
        signals = np.zeros(len(depths), dtype = np.complex128)
        self.measurements = np.zeros(len(depths), dtype = np.float64)
        self.signs = {'sin_est':[], 'cos_est':[], 'sin_exact':[], 'cos_exact':[]}
        for i,n in enumerate(depths):
            # Get the exact measuremnt probabilities
            p0 = P0(n, theta)
            p1 = P1(n, theta)

            p00 = P00(n, theta)
            p01 = P01(n, theta)
            p10 = P10(n, theta)
            p11 = P11(n, theta)

            p0x = P0x(n,theta)
            p1x = P1x(n,theta)

            # Get the "noisy" probabilities by sampling and adding a bias term that pushes towards 50/50 mixture
            eta_n = (1.0-eta)**(n+1) # The error at depth n increases as more queries are implemented
            p0_estimate = np.random.binomial(n_samples[i], eta_n*p0 + (1.0-eta_n)*0.5)/n_samples[i]
            p1_estimate = 1.0 - p0_estimate
            p0x_estimate = np.random.binomial(n_samples[i], eta_n*p0x + (1.0-eta_n)*0.5)/n_samples[i]
            p1x_estimate = 1.0 - p0x_estimate

            # Save the first measurement
            # if i==0:
            #     p0_estimate_n0 = p0_estimate

            rng = np.random.default_rng(seed=9)
            p00_n = eta_n*p00 + (1.0-eta_n)*0.5
            p01_n = eta_n*p01 + (1.0-eta_n)*0.5
            p10_n = eta_n*p10 + (1.0-eta_n)*0.5
            p11_n = eta_n*p11 + (1.0-eta_n)*0.5
            p00_estimate, p01_estimate, p10_estimate, p11_estimate = \
                rng.multinomial(n_samples[i], [p00_n, p01_n, p10_n, p11_n])
            
            # p00_estimate = np.random.binomial(n_samples[i], eta_n*p00 + (1.0-eta_n)*0.5)/n_samples[i]
            # p01_estimate = np.random.binomial(n_samples[i], eta_n*p01 + (1.0-eta_n)*0.5)/n_samples[i]
            # p10_estimate = np.random.binomial(n_samples[i], eta_n*p10 + (1.0-eta_n)*0.5)/n_samples[i]
            # p11_estimate = 1.0-p10_estimate-p01_estimate-p00_estimate
            # p11_estimate = np.random.binomial(n_samples[i], eta_n*p11 + (1.0-eta_n)*0.5)/n_samples[i]

            # Save the first measurement
            if i==0:
                p00_estimate_n0 = p00_estimate # cos^2 at n=0
                p01_estimate_n0 = p01_estimate # sin^2 at n=0
                cos_estimated = np.sqrt(p00_estimate)
                sin_estimated = np.sqrt(p01_estimate)
                self.measurements[i] = p00_estimate
            else:
                # cos_estimated = 2*np.sqrt(p00_estimate) - np.sqrt(p00_estimate_n0)
                # sin_estimated = 2*np.sqrt(p00_estimate) - np.sqrt(p00_estimate_n0)

                # or
                # print(p00_estimate)
                # print(p10_estimate)
                # print(p01_estimate)
                # print(p11_estimate)
                # print(2*(p00_estimate + p10_estimate) - p00_estimate_n0)
                # print(2*(p01_estimate + p11_estimate) - p01_estimate_n0)
                # print()
                csq = 2*(p00_estimate + p10_estimate) - p00_estimate_n0
                ssq = 2*(p01_estimate + p11_estimate) - p01_estimate_n0

                if csq < 0:
                    csq = 0
                    ssq = 1
                elif ssq < 0:
                    ssq = 0
                    csq = 1
                # cos_estimated = np.sqrt(2*(p00_estimate + p10_estimate) - p00_estimate_n0)
                # sin_estimated = np.sqrt(2*(p01_estimate + p11_estimate) - p01_estimate_n0)
                cos_estimated = np.sqrt(csq)
                sin_estimated = np.sqrt(ssq)

                self.measurements[i] = csq

                # cos_estimated = (p00_estimate - p10_estimate)

            # print(cos_estimated)
            # print(np.sqrt(p0_estimate))
            # print()
            cos_estimated = np.sqrt(p0_estimate)
            sin_estimated = np.sqrt(p1_estimate)
            self.measurements[i] = p0_estimate

            cos_sign = 1
            sin_sign = 1

            if p11_estimate > p01_estimate:
                sin_sign = -1
            if p10_estimate > p00_estimate:
                cos_sign = -1
            self.signs['sin_est'].append(sin_sign)
            self.signs['cos_est'].append(cos_sign)
            
            theta_cos = np.arccos(np.sqrt(p0_estimate))
            theta_estimated = theta_cos
            # Determine which quadrant to place theta estimated in
            theta_old = np.arctan2(p0x_estimate - p1x_estimate, p0_estimate - p1_estimate)
            # signs_exact = np.sign(np.imag(np.exp(1j * theta_old)))
            self.signs['sin_exact'].append(int(np.sign(np.imag(np.exp(1j * (2*n+1)*theta)))))
            self.signs['cos_exact'].append(int(np.sign(np.real(np.exp(1j * (2*n+1)*theta)))))

            # sin_sign = np.sign(np.imag(np.exp(1j * (2*n+1)*theta)))
            
            
            # Store this to determine angle at theta = 0 or pi/2
            if i==0:
                self.p0mp1 = p0_estimate - p1_estimate

            # Compute f(n) - Eq. 3
            # fi_estimate = np.exp(1.0j*theta_estimated)
            # signals[i] = fi_estimate
            signals[i] = cos_estimated*cos_sign + 1.0j*sin_estimated*sin_sign
            # print(n)
            # print(np.sign(np.sin((2*n+1)*theta)))
            # print(np.sign(np.cos((2*n+1)*theta)))
            # print(np.angle(signals[i])/np.pi)
            # # signals[i] = np.exp(1.0j*(2*n+1)*theta) + np.random.normal(0.00001)
            # print(np.angle(np.exp(1.0j*(2*n+1)*theta))/np.pi)
            # print(theta_cos)
            # print()
        return signals    