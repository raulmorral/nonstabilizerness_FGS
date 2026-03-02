import numpy as np
import sys
import os
import pickle

from scipy.optimize import minimize
from scipy.stats import ortho_group
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph

from pfapack import pfaffian as pf



import itertools
from itertools import chain

from numba import jit


def create_path(path):
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed. Might already exist!" % path)
    else:
        print ("Successfully created the directory %s " % path)

# Functions to save and load python type objects to file using pickle.
def save_obj(data, filename ):
    with open(filename + '.pkl', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def load_obj(filename ):
    with open(filename + '.pkl', 'rb') as f:
        return pickle.load(f)
    

def flatten(matrix):
    return list(chain.from_iterable(matrix))

_cov_0 = np.array([[0.,1.],[-1.,0.]])
_c = np.cos(np.pi/8)
_s = np.sin(np.pi/8)
_Orth_T = np.array([[_c, _s],[-_s, _c]])

def compute_all_williamson_eigenvalues(Cov_reduced):

    l = np.linalg.eigvalsh(1j*Cov_reduced)
    #are from lowest to largest, which means that first half is negative
    l=np.flip(l)
    l = l[:len(l)//2] # only positive values

    return l  

def compute_williamson_eigenvalues(Cov_reduced, tol = 1e-10):

    l = np.linalg.eigvalsh(1j*Cov_reduced)
    #are from lowest to largest, which means that first half is negative
    l=np.flip(l)
    l = l[:len(l)//2] 
    #now have all the lambda_i that are positive and can use to calculate S
    #but first remove those that are too close to one
    l = l[(1-l)>tol]

    return l    

@jit(nopython=True)
def ent_entropy_from_williamson(l, alpha=1):
    
    if alpha == 1:
        return -np.sum((1+l)/2*np.log2((1+l)/2)+(1-l)/2*np.log2((1-l)/2))
    elif alpha == np.inf:
        return -np.sum(np.log2((1+l)/2))
    elif alpha == 0:
        # numerically unstable
        return(len(l))
    else:
        return np.sum(np.log2(((1+l)/2)**alpha+((1-l)/2)**alpha))/(1-alpha)
    
@jit(nopython=True)
def entangling_orthogonal_matrix(K):
    O = np.array([
        [np.cos(K[1]) * np.cos(K[2]), np.cos(K[0]) * np.sin(K[2]), np.sin(K[0]) * np.sin(K[2]), np.cos(K[2]) * np.sin(K[1])],
        [-np.cos(K[1]) * np.sin(K[2]), np.cos(K[0]) * np.cos(K[2]), np.cos(K[2]) * np.sin(K[0]), -np.sin(K[1]) * np.sin(K[2])],
        [-np.sin(K[1]) * np.sin(K[3]), -np.cos(K[3]) * np.sin(K[0]), np.cos(K[0]) * np.cos(K[3]), np.cos(K[1]) * np.sin(K[3])],
        [-np.cos(K[3]) * np.sin(K[1]), np.sin(K[0]) * np.sin(K[3]), -np.cos(K[0]) * np.sin(K[3]), np.cos(K[1]) * np.cos(K[3])]
        ])
    return O

def Haar_random_O():
    """
    Returns a Haar random orthogonal O acting on Majorana operators.
    """
    O = ortho_group.rvs(4)
    O[1,] *= np.linalg.det(O)
    return O

def random_braiding():
    """
    Returns a random braiding gate acting on Majorana operators.
    """
    majos = np.arange(4)
    np.random.shuffle(majos)
    signs = (2*np.random.randint(0,2,size=4)-1)
    
    O = np.zeros((4,4))
    for ii in range(4):
        O[ii, majos[ii]] = signs[ii]

    O[1,] *= np.linalg.det(O)
    return O

def singular_values_from_Williamson_eigenvalues(l):
    
    if len(l) > 16:
        raise ValueError('Too many singular values to calculate!')
    s = np.array([[(1-l[0])/2, 0], [0, (1+l[0])/2]])
    for w in l[1:]:
        s = np.kron(s, np.array([[(1-w)/2, 0], [0, (1+w)/2]]))
        
    s = np.diag(s)
    return np.flip(s)

def consecutive_spacing(s):
    # s is in decreasing order
    spacings = s[:-1] - s[1:]
    # only consider non-zero spacings
    spacings = spacings[spacings>1e-8]
    return spacings

def ratio_spacing(s):
    spacings = consecutive_spacing(s)
    return spacings[1:] / spacings[:-1]





class FGS(object):
    """
    Class of Fermionic Gaussian States.
    
    We consider a 1D system with sites indexed with i from 0 to L-1.
    We assume a generic free fermion state, with no particle number conservation.
    
    Parameters
    ----------
    Cov:
        Same as attributes
    
    Attributes
    ----------
    R: np.Array[ndim=2]
        Evolution operator with shape 2Lx2L. 
        This operator is used to reduce the simulation time of unitary gates:
        intead of evolving the covariance matrix for each unitary, keeps track of all unitary evolution until unitarity is broken (e.g. by a measurement).
        R is an orthogonal matrix.
    Cov: np.Array[ndim=2]
        Correlation matrix. Only updated if updated = True, otherwise X needs to be applied to update.
    udpdated: boolean
        True if there is no unitary evolution still to be applied to C (when X is the identity).
        False when the C needs to be updated.
    L: int
        Number of sites
        
        
    bonds = 0 (0, 1) ,1 (1, 2),...,L-2 (L-2, L-1)
    qubits = 0,1,2,...,L-1
    """
    
    def __init__(self, Cov):
        self.Cov = Cov
        self.L = len(Cov) // 2
        self.R = np.eye(2*self.L)
        self.updated = True
       
        
    def copy(self):
        return FGS(self.Cov.copy())
    
    def update_covariance_matrix(self):
        """
        Updates the covariance matrix of the state by applying the accumulated evolution in R.
        The correlation matrix is defined as C = <alpha alpha+> with alpha = (a1+, ..., aL+, a1, ..., aL)^T
        (+ = dagger).
        """
        
        if not self.updated:
            
            self.Cov = self.R.T@self.Cov@self.R
            self.R = np.eye(2*self.L)
            self.updated = True
    
    def apply_U_bond(self, O, i):
        """
        Applies unitary U in the state at bond i.
        """
                
        self.R[:, 2*i : 2*i + 4] = self.R[:, 2*i : 2*i+4] @ O
        
        self.updated = False

    def apply_U_global(self, O):
        """
        Applies unitary U in the whole state.
        """
                
        self.R = self.R @ O
        
        self.updated = False
        self.update_covariance_matrix()
        
    def apply_T_gate(self, k):
        """
        Applies a T gate in site k.
        """
        self.R[:, 2*k : 2*k + 2] = self.R[:, 2*k : 2*k+2] @ _Orth_T # maybe this is not T acting with the phase at 1 but at 0. Should check, just for consistency.
        self.updated = False
    
    def compute_entanglement_entropy(self, bond, alpha=1, tol=1e-10):
        
        if bond >= self.L-1:
            print('Algo ha ido mal')
        
        self.update_covariance_matrix()
        
        if bond <= self.L//2:
            Cov_reduced = self.Cov[:2*(bond+1), :2*(bond+1)]
        else:
            Cov_reduced = self.Cov[2*(bond+1):, 2*(bond+1):]        
        
        l = compute_williamson_eigenvalues(Cov_reduced, tol)
        
        return ent_entropy_from_williamson(l, alpha)
    
    def compute_entanglement_entropies(self, bond, alpha_list, tol=1e-10):
        
        if bond >= self.L-1:
            print('Algo ha ido mal')
        
        self.update_covariance_matrix()
        
        how_many = len(alpha_list)
        entropies = np.zeros(how_many)
        
        if bond <= self.L//2:
            Cov_reduced = self.Cov[:2*(bond+1), :2*(bond+1)]
        else:
            Cov_reduced = self.Cov[2*(bond+1):, 2*(bond+1):]  

        l = compute_williamson_eigenvalues(Cov_reduced, tol)
        for jj, alpha in enumerate(alpha_list):
            entropies[jj] = ent_entropy_from_williamson(l, alpha)
           
        return entropies
    
    def compute_profile(self, alpha, tol=1e-10):
        
        profile = np.zeros(self.L + 1)
        for ii in range(self.L-1):
            profile[ii+1] = self.compute_entanglement_entropy(ii, alpha, tol=tol)
        return profile
    
    def compute_profiles(self, alpha_list, tol=1e-10):
        
        self.update_covariance_matrix()
        
        how_many = len(alpha_list)
        profiles = np.zeros((how_many, self.L + 1))
        
        for ii in range(self.L-1):
            if ii+1 < self.L//2:
                Cov_reduced = self.Cov[0:2*(ii+1),0:2*(ii+1)]
            else:
                Cov_reduced = self.Cov[2*(ii+1):,2*(ii+1):]
            
            l = compute_williamson_eigenvalues(Cov_reduced, tol)
            for jj, alpha in enumerate(alpha_list):
                profiles[jj, ii+1] = ent_entropy_from_williamson(l, alpha)
           
        return profiles
    
    def compute_entanglement_entropy_sites_contiguous(self, lower, higher_inclusive, alpha = 1, tol = 1e-10):

        # qubits from 0 to L-1
        self.update_covariance_matrix()

        lower_maj_index = 2*lower
        higher_maj_index = 2*higher_inclusive+1

        Cov_reduced = self.Cov[lower_maj_index:higher_maj_index+1,lower_maj_index:higher_maj_index+1] #+1 to make inclusive
        
        l = compute_williamson_eigenvalues(Cov_reduced, tol)

        return ent_entropy_from_williamson(l, alpha)

    def compute_entanglement_entropy_sites(self, sites, alpha = 1, tol=1e-10):
        
        self.update_covariance_matrix()
    
        maj_array=[]
        for j in sites:
            maj_array.append(2*j)
            maj_array.append(2*j+1)

        maj_array.sort() #smallest first
        r = len(sites)    

        rows = np.array([[maj for ii in range(2*r)] for maj in maj_array])
        columns = np.array([[maj for maj in maj_array] for ii in range(2*r)])
        
        Cov_reduced = self.Cov[rows,columns]
        
        l = compute_williamson_eigenvalues(Cov_reduced, tol)

        return ent_entropy_from_williamson(l, alpha)

    def mutual_information(self, A, B, alpha = 1, tol = 1e-10):
        """
        A and B are arrays of sites. They cannot be overlapping, otherwise error
        """
        
        union = np.concatenate((A, B))
        
        return self.compute_entanglement_entropy_sites(A, alpha, tol) + self.compute_entanglement_entropy_sites(B, alpha, tol) - self.compute_entanglement_entropy_sites(union, alpha, tol)
    
    def compute_exp_val_Majorana_string(self, majoranas):
        """
        majoranas is a list of 0 and 1 of length 2*L, where L indicates that that majorana is included.
        """
        self.update_covariance_matrix()
        
        majoranas_bool = majoranas.astype(bool)
        
        Cov_reduced = self.Cov[majoranas_bool][:, majoranas_bool]
        
        return pf.pfaffian(Cov_reduced) # is the pfaffian directly the expectation value of the string?
        
    
    
    def fermionic_negativity(self, bond):
        """
        Return the fermionic negativity for a bipartition at a given bond. 
        
        NEED TO STILL CHECK THIS
        """    
        
        gamma = self.Cov()
        
        # now we need to compute gamma_plus and minus according to their definition

        gamma_plus = np.zeros((2*L, 2*L), complex)
        gamma_minus = np.zeros((2*L, 2*L), complex)

        gamma_plus[:2*r, :2*r] = - gamma[:2*r, :2*r]
        gamma_plus[2*r:, 2*r:] = gamma[2*r:, 2*r:]
        gamma_plus[:2*r, 2*r:] = + 1j * gamma[:2*r, 2*r:]    
        gamma_plus[2*r:, :2*r] = + 1j * gamma[2*r:, :2*r]

        gamma_minus[:2*r, :2*r] = - gamma[:2*r, :2*r]
        gamma_minus[2*r:, 2*r:] = gamma[2*r:, 2*r:]
        gamma_minus[:2*r, 2*r:] = - 1j * gamma[:2*r, 2*r:]    
        gamma_minus[2*r:, :2*r] = - 1j * gamma[2*r:, :2*r]

        # calculate the covariance matrix gamma_x with the formula for G_x=(1+i gamma_x)/2
        G_cross = ((np.eye(2*L)+1j*gamma_minus)/2)@np.linalg.inv((np.eye(2*L)-gamma_plus@gamma_minus)/2)@((np.eye(2*L)+1j*gamma_plus)/2)
    
        eigval_gamma_X = np.real_if_close(np.linalg.eigvals(G_cross))
        eigval_gamma_X[eigval_gamma_X<0] = 0
        eigval_gamma_X[eigval_gamma_X>1] = 1
        
        negativity = np.sum(np.log2((1-eigval_gamma_X)**0.5+eigval_gamma_X**0.5))/2
    
        return np.real_if_close(negativity)     

    def apply_disentangling_unitary(self, bond, alpha = 1, method = 'L-BFGS-B', tol = 1e-10):
        """
        Finds the unitary that minimizes the Rényi n entanglement entropy (alpha>0) in the given bond. 
        """
        
        ent = self.compute_entanglement_entropy(bond, alpha, tol=tol)
        
        if ent > 1e-10:

            cost_func = lambda K: entanglement_applying_gate(K, self.Cov, bond, alpha, self.L, tol=tol)
            res = minimize(cost_func, 0.5-np.random.randn(4), method=method, tol=1e-14)

            self.apply_U_bond(entangling_orthogonal_matrix(res.x), bond)
            
    def apply_measurement_particle_number(self, k):
        """
        Applies a particle number measurement at site k.
        """
        
        L = self.L
        
        self.update_covariance_matrix()
        
        Id = np.eye(L)

        prob_1 = np.real_if_close(1/2 * (1 - self.Cov[2*k, 2*k+1]))

        gamma = self.Cov
        
        gamma_post = np.zeros((2*L, 2*L), float)

        if np.random.rand() < prob_1:
            for i in range(2*L):
                for j in range(i):
                    if (i,j) == (2*k+1, 2*k):
                        gamma_post[i,j] = 1
                    elif i == 2*k or i == 2*k+1 or j == 2*k or j == 2*k+1:
                        gamma_post[i,j] = 0
                    else:
                        gamma_post[i,j] = gamma[i,j] - (gamma[2*k, j] * gamma[2*k+1, i] - gamma[2*k, i] * gamma[2*k+1, j]) / (2 * prob_1)

        else:        
            for i in range(2*L):
                for j in range(i):
                    if (i,j) == (2*k+1, 2*k):
                        gamma_post[i,j] = -1
                    elif i == 2*k or i == 2*k+1 or j == 2*k or j == 2*k+1:
                        gamma_post[i,j] = 0
                    else:
                        gamma_post[i,j] = gamma[i,j] + (gamma[2*k, j] * gamma[2*k+1, i] - gamma[2*k, i] * gamma[2*k+1, j]) / (2 * (1-prob_1))

        self.Cov = gamma_post - gamma_post.T
        
        
    def correlation_matrix(self):
        """
        Returns the correlation matrix of the state in the form of Surace paper.
        """
        L = self.L
        order = [2*ii for ii in range(self.L)] + [2*ii+1 for ii in range(self.L)]
        covariance_matrix = covariance_matrix[order, :]
        covariance_matrix = covariance_matrix[:, order]

        Omega = np.zeros((2*L, 2*L), complex)
        Omega[:L, :L] = np.eye(L)
        Omega[:L, L:] = np.eye(L)
        Omega[L:, :L] = 1j*np.eye(L)
        Omega[L:, L:] = -1j*np.eye(L)
        Omega /= np.sqrt(2)

        return (1j * np.conj(Omega.T) @ self.Cov @ Omega + np.eye(2*L)) / 2
        

def product_0_FGS(L):
    """
    Returns the fermionic gaussian state |0000> in L sites.
    
    Parameters:
    ----------
    L: int
        Number of sites of the state.
        
    Returns:
    --------
    state: FGS_class
        State in the FGS class.
    """
    
    Gamma0 = np.zeros((2*L, 2*L))
    for k in range(L):
        Gamma0[2*k:2*k+2, 2*k: 2*k+2] = _cov_0
    state = FGS(Gamma0)
    return state
        
    
def random_FGS(L):
    """
    Returns a random fermionic gaussian state in L sites.
    
    Parameters:
    ----------
    L: int
        Number of sites of the state.
        
    Returns:
    --------
    state: FGS_class
        A random Gaussian Fermionic State in L sites.
    """
    
    Gamma0 = np.zeros((2*L, 2*L))
    for k in range(L):
        Gamma0[2*k:2*k+2, 2*k: 2*k+2] = _cov_0
    
    O = ortho_group.rvs(2*L)
    O[1,] *= np.linalg.det(O)
    
    state = FGS(O@Gamma0@O.T)
    
    return state


def random_braiding_state(L):
    """
    Returns a random braiding state in L sites.
    
    Parameters:
    ----------
    L: int
        Number of sites of the state.
        
    Returns:
    --------
    state: FGS_class
        A random braiding state in L sites.
    """
    
    majoranas = [ii for ii in range(2*L)]
    np.random.shuffle(majoranas)
    
    Gamma = np.zeros((2*L, 2*L))
        
    for ii in range(L):
        Gamma[majoranas[2*ii],majoranas[2*ii+1]] = (-1)**(np.random.randint(2))
        
    Gamma = Gamma - Gamma.T
        
    return FGS(Gamma)

def entanglement_applying_gate(K, Cov, bond, alpha, L, tol=1e-10):
    """
    Return the n-th Rényi entropy for a bipartition at any of the bonds for a state given by covariance matrix Cov
    after applying an entangling gate given by 4 real parameters K.
    (bond = 0 means bipartition between 1st qubit and rest).
    """    
    C = Cov.copy()
        
    O = entangling_orthogonal_matrix(K)
    
    i = bond
    C[:, 2*i : 2*i + 4] = C[:, 2*i : 2*i+4] @ O
    C[2*i : 2*i+4, :] = O.T @ C[2*i : 2*i+4, :]
    
    if bond < L//2:
        Cov_reduced = C[:2*(bond+1), :2*(bond+1)]
    else:
        Cov_reduced = C[2*(bond+1):, 2*(bond+1):]
    
    l = compute_williamson_eigenvalues(Cov_reduced, tol)

    return ent_entropy_from_williamson(l, alpha) 


def stabilizer_Renyi_entropy_exact(state, alpha = 2):
    """
    Given a covariance matrix, it computes exactly the stabilizer Rényi entropy.
    """
    assert state.L <= 10
    L = state.L
    state.update_covariance_matrix()
    
    # Generate all possible binary arrays of length 2*L
    all_binary_arrays = itertools.product([0, 1], repeat=2*L)
    
    # Filter out arrays where the number of 1s is not even
    even_ones_arrays = (arr for arr in all_binary_arrays if arr.count(1) % 2 == 0 and arr.count(1) > 0)

    results = [1]
    for majoranas in even_ones_arrays:
        results.append(state.compute_exp_val_Majorana_string(np.array(majoranas)))
    
    results = np.array(results)**2
    results = results[results>1e-12]
    
    if alpha == 1:
        return -np.sum(results*np.log2(results)) / 2**L
    elif alpha == 0:
        raise ValueError('Not implemented for alpha 0')
    elif alpha == np.inf:
        raise ValueError('Not implemented for alpha infinity')
    else:
        return np.log2(np.sum(results**(alpha))/2**L)/(1-alpha)
    
def quadratic_Tsallis_entropy(state, alpha = 2):
    """
    Given a covariance matrix, it computes exactly the stabilizer Rényi entropy.
    """
    L = state.L
    state.update_covariance_matrix()
    
    pij = state.Cov**2
    
    if alpha == 1:
        pass
    elif alpha == 0:
        pass
    elif alpha == np.inf:
        pass
    else:
        return 2*L-np.sum((pij)**alpha)

        
def stabilizer_nullity(state):
    
    L = state.L
    
    state.update_covariance_matrix()
    A = np.abs(state.Cov)    
        
    A[A<1e-15] = 0
    
    n_components = csgraph.connected_components(sp.csr_matrix(A), directed = False, return_labels = False)    
    
    return L - n_components
