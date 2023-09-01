import numpy as np
from scipy.spatial.transform import Rotation as R
from .quat_tools import *
from .normal import normal as normal_class
    


class gmm:
    def __init__(self, *args_):

        self.q_list = args_[0]
        


    def cluster(self, dummy_arr): #dummy
        """
        Fill in the actual clustering algorithm and return the result assignment_arr
        """

        self.assignment_arr = dummy_arr



    def return_norma_class(self, q_list, *args_):
        """
        Return normal_class: 
            Assuming one distribution if only q_list is given;
            Output a list of normal_class if assignment_arr is provided too;
            Each normal class is defined by Mu, Sigma, (Prior is optional);
            Store the result normal class and Prior in gmm class

        @note verify later if the R.mean() yields the same results as manually computing the Krechet mean of quaternion
        """


        N = len(q_list)
        M = 4

        if len(args_) == 1:
            assignment_arr = args_[0]
        else:
            assignment_arr = np.zeros((N, ), dtype=int)

        K = assignment_arr.max()+1

        Prior  = np.zeros((K, ))
        Mu      = np.zeros((K, M))
        Sigma   = np.zeros((K, M, M))

        q_normal_list = []  # easier for list comprehension later

        for k in range(K):
            q_k = [q for index, q in enumerate(q_list) if assignment_arr[index]==k] 
            r_k = R.from_quat([canonical_quat(q.as_quat()) for q in q_k])
            q_k_mean = r_k.mean()
        

            Prior[k]       = len(q_k)/N

            # Mu[k, :]        = canonical_quat(q_k_mean.as_quat())
            Mu[k, :]        = canonical_quat(R.identity().as_quat())

            # Sigma[k, :, :]  = riem_cov(q_k, q_k_mean)
            Sigma[k, :, :]  = riem_cov(q_k, R.from_quat(Mu[k, :]))

            q_normal_list.append(normal_class(Mu[k, :], Sigma[k, :, :], Prior[k]))
        

        self.Prior = Prior
        self.q_normal_list = q_normal_list


        return q_normal_list
    

    def prob(self, q):

        """
        Not yet vectorized
        """

        



    


    




        
    




