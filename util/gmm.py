import numpy as np

from scipy.spatial.transform import Rotation as R
from scipy.stats import multivariate_normal
from sklearn.mixture import BayesianGaussianMixture

from .quat_tools import *
from .plot_tools import *


class gmm:
    def __init__(self, q_in, q_att, K_init):
        self.q_in     = q_in
        self.q_att    = q_att
        self.K_init = K_init

        self.N = len(q_in)
        self.M = 4
    


    def _rearrange_array(self, assignment_arr):
        
        rearrange_list = []
        for idx, entry in enumerate(assignment_arr):
            if not rearrange_list:
                rearrange_list.append(entry)
            if entry not in rearrange_list:
                rearrange_list.append(entry)
                assignment_arr[idx] = len(rearrange_list) - 1
            else:
                assignment_arr[idx] = rearrange_list.index(entry)   
        
        return assignment_arr



    def fit(self):
        q_in_att = riem_log(self.q_att, self.q_in)

        gmm = BayesianGaussianMixture(n_components=self.K_init, n_init=3, random_state=2).fit(q_in_att)

        assignment_arr = gmm.predict(q_in_att)
        assignment_arr = self._rearrange_array(assignment_arr)


        self.gmm = gmm
        self.K = int(assignment_arr.max()+1)
        self._return_normal_class(assignment_arr)




    def predict(self, q_in, index_list):
        q_in_att = riem_log(self.q_att, q_in)

        assignment_arr = self.gmm.predict(q_in_att)
        assignment_arr = self._rearrange_array(assignment_arr)
  
        postProb = self.postLogProb(q_in)
        
        plot_gmm(q_in, index_list, assignment_arr, interp=False)

        return assignment_arr, postProb



    def _return_normal_class(self, assignment_arr):
        """
        Return normal_class: 
            Assuming one distribution if only q_list is given;
            Output a list of normal_class if assignment_arr is provided too;
            Each normal class is defined by Mu, Sigma, (Prior is optional);
            Store the result normal class and Prior in gmm class

        @note verify later if the R.mean() yields the same results as manually computing the Krechet mean of quaternion
        """


        Prior   = [0] * self.K
        Mu      = [R.identity()] * self.K
        Sigma   = [np.zeros((self.M, self.M), dtype=np.float32)] * self.K

        q_normal_list = [] 

        for k in range(self.K):
            q_k      = [q for index, q in enumerate(self.q_in) if assignment_arr[index]==k] 
            q_k_mean = quat_mean(q_k)

    
            Prior[k]  = len(q_k)/self.N
            Mu[k]     = q_k_mean
            Sigma[k]  = riem_cov(q_k_mean, q_k) + 10E-6 * np.eye(4)


            q_normal_list.append(
                {
                    # "prior" : Prior[k],
                    "mu"    : Mu[k],
                    "sigma" : Sigma[k],
                    "rv"    : multivariate_normal(np.zeros((self.M, )), Sigma[k], allow_singular=True)
                }
            )

        self.Prior = Prior
        self.q_normal_list = q_normal_list




    def logProb(self, q_list):

        """
        vectorized operation
        """


        if isinstance(q_list, R):
            logProb = np.zeros((self.K, 1))
        elif isinstance(q_list, list):
            logProb = np.zeros((self.K, len(q_list)))
        else:
            print("Invalid q_list type")
            sys.exit()


        for k in range(self.K):
            mu_k, _, normal_k = tuple(self.q_normal_list[k].values())

            q_list_k = riem_log(mu_k, q_list)

            logProb[k, :] = normal_k.logpdf(q_list_k)
        

        return logProb
    


    def postLogProb(self, q_list):

        """
        vectorized operation
        """

        logPrior = np.log(self.Prior)
        logProb  = self.logProb(q_list)

        if isinstance(q_list, R):
            postLogProb  = logPrior[:, np.newaxis] + logProb
        elif isinstance(q_list, list):
            postLogProb  = np.tile(logPrior[:, np.newaxis], (1, len(q_list))) + logProb

        maxPostLogProb = np.max(postLogProb, axis=0, keepdims=True)
        expProb = np.exp(postLogProb - np.tile(maxPostLogProb, (self.K, 1)))
        postProb = expProb / np.sum(expProb, axis = 0, keepdims=True)


        return postProb
    
    

    def _dual_gmm(self):

        """
        Double the normal list and the priors
        """

        K_dual = self.K * 2
        Prior_dual   = [0] * K_dual
        q_normal_list_dual = [] 

        for k in range(int(K_dual/2)):      
            Prior_dual[k]  = self.Prior[k] / 2
            Mu, Sigma, _ = tuple(self.q_normal_list[k].values())
    
            q_normal_list_dual.append(
                {
                    "mu"    : Mu,
                    "sigma" : Sigma,
                    "rv"    : multivariate_normal(np.zeros((self.M, )), Sigma, allow_singular=True)
                }
            )

        for k in np.arange(int(K_dual/2), K_dual):          
            k_prev = k - int(K_dual/2)
            Prior_dual[k]  = Prior_dual[k_prev]
            Mu, Sigma, _ = tuple(self.q_normal_list[k_prev].values())
            Mu     =  R.from_quat(-Mu.as_quat())

            q_normal_list_dual.append(
                {
                    "mu"    : Mu,
                    "sigma" : Sigma,
                    "rv"    : multivariate_normal(np.zeros((self.M, )), Sigma, allow_singular=True)
                }
            )
        
        dual_gmm = gmm(self.q_in, self.q_att, self.K_init)
        dual_gmm.K = K_dual
        dual_gmm.Prior = Prior_dual
        dual_gmm.q_normal_list = q_normal_list_dual

        return dual_gmm




