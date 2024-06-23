import numpy as np
from abc import ABCMeta, abstractmethod
import pickle
from scipy.linalg import toeplitz
from numba import njit

P0 = lambda n, theta: np.cos((2*n+1)*theta)**2
P1 = lambda n, theta: np.sin((2*n+1)*theta)**2
P0x = lambda n, theta: (1.0 + np.sin(2*(2*n+1)*theta))/2.0
P1x = lambda n, theta: (1.0 - np.sin(2*(2*n+1)*theta))/2.0

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
            self.n_samples = np.array(n_samples)
            # self.depths = self._get_depths(self.M)
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
        self.ULA_signal = get_ula_signal(self.q, self.idx, signal)
        total_size = len(self.ULA_signal)
        ULA_signal = self.ULA_signal

        '''
        This uses the techinque from DOI:10.1109/LSP.2015.2409153
        '''
        subarray_col = ULA_signal[total_size//2:]
        subarray_row = ULA_signal[:total_size//2+1] #np.conj(subarray_col)
        print(len(subarray_col), len(subarray_row))
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
        # signal = self.estimate_signal(n_samples, theta, eta=eta)
        self.ULA_signal = get_ula_signal(self.q, self.idx, signal)
        total_size = len(self.ULA_signal)
        ULA_signal = self.ULA_signal
        
        subarray_col = ULA_signal[total_size//2:]
        subarray_row = np.conj(subarray_col)

        #TODO: R: row i = ULA_signal[i : total_size//2 + i]
        # return R
        
        return subarray_col
    
    #TODO: this can be more efficiently implemented
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

        return physLoc, n_samples
    
    def estimate_signal(self, n_samples, theta, eta=0.0):
        depths = self.depths
        signals = np.zeros(len(depths), dtype = np.complex128)
        cos_signal = np.zeros(len(depths), dtype = np.complex128)
        for i,n in enumerate(depths):
            # Get the exact measuremnt probabilities
            p0 = P0(n, theta) #cos((2n+1)theta)^2 = 0.5 + 0.5*cos(n*w'+w) 
            p1 = P1(n, theta)

            p0x = P0x(n,theta)
            p1x = P1x(n,theta)

            # Get the "noisy" probabilities by sampling and adding a bias term that pushes towards 50/50 mixture
            eta_n = (1.0-eta)**(n+1) # The error at depth n increases as more queries are implemented
            p0_estimate = np.random.binomial(n_samples[i], eta_n*p0 + (1.0-eta_n)*0.5)/n_samples[i]
            p1_estimate = np.random.binomial(n_samples[i], eta_n*p1 + (1.0-eta_n)*0.5)/n_samples[i]
            p0x_estimate = np.random.binomial(n_samples[i], eta_n*p0x + (1.0-eta_n)*0.5)/n_samples[i]
            p1x_estimate = 1.0 - p0x_estimate
            
            # Estimate theta
            theta_estimated = np.arctan2(p0x_estimate - p1x_estimate, p0_estimate - p1_estimate)

            # estimate cos(2n+1)2theta
            cos_signal[i] = 2*p0_estimate - 1
            
            # Store this to determine angle at theta = 0 or pi/2
            if i==0:
                self.p0mp1 = p0_estimate - p1_estimate

            # Compute f(n) - Eq. 3
            # theta1 = 0.3
            # theta2 = 0.6
            # fi_estimate = 0.5*np.exp(-1.0j*(n+1)*theta1) + 0.5*np.exp(-1.0j*(n+1)*theta2)# + np.random.normal(0.0, 0.0001)
            fi_estimate = np.exp(1.0j*theta_estimated)
            # fi_estimate = (2*p0_estimate - 1)*np.exp(1.0j*np.random.normal(0.0, 0.0001))
            # fi_estimate = (2*p0_estimate - 1)*np.exp(1.0j*np.pi/4)
            # fi_estimate = (2*p0_estimate - 1) + np.random.normal(0.0, 0.0001)
            # fi_estimate = (2*p0_estimate - 1)
            # fi_estimate = np.cos((2*n+1)*2*theta)

            # fi_estimate = 2*p0_estimate - 1 + 1.0j*np.sqrt(1-(2*p0_estimate - 1)**2)
            signals[i] = fi_estimate
        self.cos_signal = cos_signal
        
        return signals    