import numpy as np

from scipy.spatial.transform import Rotation as R
from scipy.stats import multivariate_normal
from sklearn.mixture import BayesianGaussianMixture

from .quat_tools import *

class gmm:
    def __init__(self, q_att, q_train):
        
        self.q_att   = q_att
        self.q_train = q_train

        self.q_train_att = riem_log(q_att, q_train)
        self.N = len(q_train)
        self.M = 4

        # self.q_att   = q_att
        # self.q_att_q = canonical_quat(q_att.as_quat())
        # self.q_list  = q_list
        # self.q_list_q   = list_to_arr(self.q_list)
        # self.q_list_att = riem_log(self.q_att_q, self.q_list_q)



    def begin(self, *args, **argv): #
        """
        Fill in the actual clustering algorithm and return the result assignment_arr
        """

        if len(args) == 1:
            assignment_arr = args[0] # pseudo clustering
        elif len(args) == 0:
            assignment_arr = BayesianGaussianMixture(n_components=1, random_state=2).fit_predict(self.q_train_att)

        rearrange_list = []
        for idx, entry in enumerate(assignment_arr):
            if not rearrange_list:
                rearrange_list.append(entry)
            if entry not in rearrange_list:
                rearrange_list.append(entry)
                assignment_arr[idx] = len(rearrange_list) - 1
            else:
                assignment_arr[idx] = rearrange_list.index(entry)   

        self.assignment_arr = assignment_arr 
        self.K = int(assignment_arr.max()+1)
        self._return_norma_class()

        return self.assignment_arr


    def _return_norma_class(self):
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
            q_k      = [q for index, q in enumerate(self.q_train) if self.assignment_arr[index]==k] 
            r_k      = R.from_quat([canonical_quat(q.as_quat()) for q in q_k])
            q_k_mean = r_k.mean()
        

            Prior[k]  = len(q_k)/self.N
            Mu[k]     = q_k_mean
            Sigma[k]  = riem_cov(q_k_mean, q_k) + 10**(-6) * np.eye(4)


            q_normal_list.append(
                {
                    "prior" : Prior[k],
                    "mu"    : Mu[k],
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
            _, mu_k, normal_k = tuple(self.q_normal_list[k].values())

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


