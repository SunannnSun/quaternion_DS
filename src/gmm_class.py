import numpy as np

from scipy.spatial.transform import Rotation as R
from scipy.stats import multivariate_normal
from sklearn.mixture import BayesianGaussianMixture

from .util.quat_tools import *
from .util.plot_tools import *



class gmm_class:
    def __init__(self, q_in:list, q_att:R, K_init:int):
        """
        Initialize a GMM class

        Parameters:
        ----------
            q_in (list):            M-length List of Rotation objects for ORIENTATION INPUT

            q_att (Rotation):       Single Rotation object for ORIENTATION ATTRACTOR
        """

        # store parameters
        self.q_in     = q_in
        self.q_att    = q_att
        self.K_init   = K_init

        self.M = len(q_in)
        self.N = 4

        # form projected states
        self.q_in_att    = riem_log(q_att, q_in)



    def fit(self):
        """ 
        Fit model to data; 
        predict and store assignment label;
        extract and store Gaussian component 
        """

        gmm = BayesianGaussianMixture(n_components=self.K_init, n_init=3, random_state=2).fit(self.q_in_att)
        assignment_arr = gmm.predict(self.q_in_att)

        self._rearrange_array(assignment_arr)
        self._extract_gaussian()

        return self.logProb(self.q_in)



    def _rearrange_array(self, assignment_arr):
        """ Remove empty components and arrange components in order """
        rearrange_list = []
        for idx, entry in enumerate(assignment_arr):
            if not rearrange_list:
                rearrange_list.append(entry)
            if entry not in rearrange_list:
                rearrange_list.append(entry)
                assignment_arr[idx] = len(rearrange_list) - 1
            else:
                assignment_arr[idx] = rearrange_list.index(entry)   
        
        self.K = int(assignment_arr.max()+1)
        self.assignment_arr = assignment_arr



    def _extract_gaussian(self):
        """
        Extract Gaussian components from assignment labels and data

        Parameters:
        ----------
            Priors(list): K-length list of priors

            Mu(list):     K-length list of tuple: ([3, ] NumPy array, Rotation)

            Sigma(list):  K-length list of [N, N] NumPy array 
        """

        assignment_arr = self.assignment_arr

        Prior   = [0] * self.K
        Mu      = [R.identity()] * self.K 
        Sigma   = [np.zeros((self.N, self.N), dtype=np.float32)] * self.K

        gaussian_list = [] 

        for k in range(self.K):
            q_k      = [q for index, q in enumerate(self.q_in) if assignment_arr[index]==k] 
            q_k_mean = quat_mean(q_k)

            q_diff = riem_log(q_k_mean, q_k) 

            Prior[k]  = len(q_k)/self.M
            Mu[k]     = q_k_mean
            Sigma[k]  = q_diff.T @ q_diff / (len(q_k)-1)  + 10E-6 * np.eye(self.N)

            gaussian_list.append(
                {   
                    "prior" : Prior[k],
                    "mu"    : Mu[k],
                    "sigma" : Sigma[k],
                    "rv"    : multivariate_normal(np.zeros(4), Sigma[k], allow_singular=True)
                }
            )

        self.gaussian_list = gaussian_list

        self.Prior  = Prior
        self.Mu     = Mu
        self.Sigma  = Sigma



    def logProb(self, q_in):
        """ Compute log probability"""
        if isinstance(q_in, list):
            logProb = np.zeros((self.K, len(q_in)))
        else:
            logProb = np.zeros((self.K, 1))


        for k in range(self.K):
            prior_k, mu_k, _, normal_k = tuple(self.gaussian_list[k].values())

            q_k  = riem_log(mu_k, q_in)

            logProb[k, :] = np.log(prior_k) + normal_k.logpdf(q_k)

        maxPostLogProb = np.max(logProb, axis=0, keepdims=True)
        expProb = np.exp(logProb - np.tile(maxPostLogProb, (self.K, 1)))
        postProb = expProb / np.sum(expProb, axis = 0, keepdims=True)

        return postProb
    



    def _dual_gmm(self):
        """ dual GMM to cover the entire quaternion space """

        K_dual = self.K * 2
        # Prior_dual   = [0] * K_dual
        dual_gaussian_list = [] 

        for k in range(int(K_dual/2)):      
            # Prior_dual[k]  = self.Prior[k] / 2
            Prior, Mu, Sigma, _ = tuple(self.gaussian_list[k].values())

            dual_gaussian_list.append({
                    "prior" : Prior/2,
                    "mu"    : Mu,
                    "sigma" : Sigma,
                    "rv"    : multivariate_normal(np.zeros(4), Sigma, allow_singular=True)
                }
            )

        for k in np.arange(int(K_dual/2), K_dual):          
            k_prev = k - int(K_dual/2)
            # Prior_dual[k]  = Prior_dual[k_prev]
            Prior, Mu, Sigma, _ = tuple(self.gaussian_list[k_prev].values())
            Mu     = R.from_quat(-Mu.as_quat())

            dual_gaussian_list.append({
                    "prior" : Prior/2,
                    "mu"    : Mu,
                    "sigma" : Sigma,
                    "rv"    : multivariate_normal(np.zeros(4), Sigma, allow_singular=True)
                }
            )
        
        dual_gmm = gmm_class(self.q_in, self.q_att, 1)
        dual_gmm.K = K_dual
        dual_gmm.gaussian_list = dual_gaussian_list

        return dual_gmm