import numpy as np
from numba import njit, objmode
from abc import ABCMeta, abstractmethod 
from scipy.linalg import matmul_toeplitz


@njit
def Lanczos( A, v, m=100):
    n = len(v)
    if m>n: m = n;
    # from here https://en.wikipedia.org/wiki/Lanczos_algorithm
    V = np.zeros( (m,n) , dtype = np.complex128)
    T = np.zeros( (m,m) , dtype = np.complex128)
    V[0, :] = v

    # step 2.1 - 2.3 in https://en.wikipedia.org/wiki/Lanczos_algorithm
    w = np.dot(A, v)
    alfa = np.dot(np.conj(w),v)
    w = w - alfa*v
    T[0,0] = alfa

    # needs to start the iterations from indices 1
    for j in range(1, m-1 ):
        # beta = np.sqrt( np.abs( np.dot( np.conj(w), w ) ) )
        beta = np.linalg.norm(w)

        V[j,:] = w/beta

        # This performs some rediagonalization to make sure all the vectors are orthogonal to eachother
        for i in range(j-1):
            V[j, :] = V[j,:] - np.dot(V[j,:], np.conj(V[i, :]))*V[i,:]
        V[j, :] = V[j, :]/np.linalg.norm(V[j, :])


        w = np.dot(A, V[j, :])
        alfa = np.dot(np.conj(w), V[j, :])
        w = w - alfa * V[j, :] - beta*V[j-1, :]

        T[j,j  ] = alfa
        T[j-1 ,j] = beta
        T[j,j-1] = beta


    return T, V

@njit
def Lanczost( A, v, m=100):
    n = len(v)
    if m>n: m = n;
    # from here https://en.wikipedia.org/wiki/Lanczos_algorithm
    V = np.zeros( (m,n) , dtype = np.complex128)
    T = np.zeros( (m,m) , dtype = np.complex128)
    V[0, :] = v

    # step 2.1 - 2.3 in https://en.wikipedia.org/wiki/Lanczos_algorithm
    w = np.zeros(n, dtype=np.complex128)
    with objmode(w='complex128[:]'):
        w = matmul_toeplitz(A, v)
    alfa = np.dot(np.conj(w),v)
    w = w - alfa*v
    T[0,0] = alfa
    
    # needs to start the iterations from indices 1
    for j in range(1, m-1 ):
        # beta = np.sqrt( np.abs( np.dot( np.conj(w), w ) ) )
        beta = np.linalg.norm(w)

        V[j,:] = w/beta

        # This performs some rediagonalization to make sure all the vectors are orthogonal to eachother
        for i in range(j-1):
            V[j, :] = V[j,:] - np.dot(V[j,:], np.conj(V[i, :]))*V[i,:]
        V[j, :] = V[j, :]/np.linalg.norm(V[j, :])

        with objmode(w='complex128[:]'):
            w = matmul_toeplitz(A, V[j, :])

        alfa = np.dot(np.conj(w), V[j, :])
        w = w - alfa * V[j, :] - beta*V[j-1, :]

        T[j,j  ] = alfa
        T[j-1 ,j] = beta
        T[j,j-1] = beta

    return T, V


class EstimateFrequency(metaclass = ABCMeta):
    
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def estimate_theta(self):
        pass
    
    def eig(self, R, n=1, lanczos=False):
        self.R = R
        if lanczos:   
            self.R = np.ascontiguousarray(self.R)
            self._eig_decomp_lanczos(n)
        else:
            self._eig_decomp(n)
            
    def _eig_decomp(self, n=1):
        eig_values, eig_vectors = np.linalg.eigh(self.R)
        idx = np.argsort(eig_values)[::-1]
        eig_values = eig_values[idx]
        eig_vectors = eig_vectors[:, idx]
        
        # Form S and G
        self.S = np.matrix(eig_vectors[:, : n])
        self.G = np.matrix(eig_vectors[:, n :])
        
    def _eig_decomp_lanczos(self, n=1, m=100):        
        v0   = np.array(np.random.rand( np.shape(self.R)[0]) , dtype=np.complex128); v0 /= np.sqrt( np.abs(np.dot( v0, np.conjugate(v0) ) ) )

        T, V = Lanczos( self.R, v0, m=m )
        esT, vsT = np.linalg.eigh( T )
        esT_sort_idx = np.argsort(esT)[::-1]
        lm_eig = np.matrix(V.T @ (vsT[:, esT_sort_idx[:n].squeeze() ]))
        
        # Form S and G
        self.S = np.matrix(lm_eig)
        self.S.reshape((np.shape(self.S)[1], np.shape(self.S)[0]))
        
        
    def _eig_decomp_lanczost(self, n=1, m=100):
        v0   = np.array(np.random.rand( np.shape(self.R)[0]) + 1.0j*np.random.rand( np.shape(self.R)[0]), dtype=np.complex128); v0 /= np.sqrt( np.abs(np.dot( v0, np.conjugate(v0) ) ) )
        T, V = Lanczost( self.R, v0, m=m )

        esT, vsT = np.linalg.eigh( T )
        esT_sort_idx = np.argsort(esT)[::-1]
        lm_eig = np.matrix(V.T @ (vsT[:, esT_sort_idx[:n].squeeze() ]))
        
        # Form S and G
        self.S = np.matrix(lm_eig)
        self.eigs = esT[esT_sort_idx]


class ESPIRIT(EstimateFrequency):
    
    def __init__(self):
        pass
    
    def estimate_theta_toeplitz(self, R, n=2, p0mp1=1.0, s0=1.0):
        return self.estimate_theta(R, True, True, n=2, p0mp1=p0mp1, s0=s0)
    
    def estimate_theta(self, R, lanczos=False, lanczos_toeplitz=False, n=2, p0mp1=1.0, s0=1.0):

        self.R = R
        if lanczos:   
            self.R = np.ascontiguousarray(self.R, dtype=np.complex128)
            if lanczos_toeplitz:
                self._eig_decomp_lanczost(n, m=50)
            else:
                self._eig_decomp_lanczos(n, m=100)
        else:
            self._eig_decomp(n)
                

        if self.R.ndim == 1:
            m = len(self.R)
        else:
            m = np.shape(self.R)[0]
        
        S1 = np.matrix(self.S[0:-1, :], dtype=np.complex128)
        S2 = np.matrix(self.S[1:, :], dtype=np.complex128)

        # S1 = np.matrix(self.S[0:-1, 0], dtype=np.complex128)
        # S2 = np.matrix(self.S[1:, 0], dtype=np.complex128)

        # print(S1)
        # print(S2)

        Phi = np.linalg.pinv(S1) @ S2

        eigs, _ = np.linalg.eig(Phi)

        angle = -np.angle(eigs)
        self.w = np.array([angle[0]/4])
        # self.w = np.max(np.abs(angle))/4
        # max_angle_idx = np.argmax(np.abs(angle))
        # self.w = angle[max_angle_idx]/4
        # self.w = np.sum(angle)/4

        # w = self.w


        # Fix the quadrant
        # if (self.w < 0.0):
        #     w = np.abs(self.w)
        # else:
        #     w = np.abs(np.pi/2.0 - np.abs(self.w))

        # if (s0 > 0.5):
        #     w = np.abs(np.pi / 2.0 - np.abs(self.w[0]))
        # else:
        #     w = np.abs(self.w[0])

        w = np.abs(self.w[0])

        # Fix pi/2 shifts that happen due to degeneracies in the signal space
        
        # Fix right around theta = 0 or pi/2
        # if (np.abs(w) < 1/m) and (p0mp1 < -0.999999):
        #     w = np.pi/2.0 + w
        #
        # elif (np.abs(w-np.pi/2) < 1/m) and (p0mp1 > 0.999999):
        #     w = np.abs(w-np.pi/2.0)

        return w, eigs