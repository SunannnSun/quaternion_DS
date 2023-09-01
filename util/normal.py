import numpy as np
from scipy.spatial.transform import Rotation as R
from .quat_tools import *


class normal:
    def __init__(self, mu, sigma, *args_):
        self.mu = mu
        self.sigma = sigma

        if len(args_) == 1:
            self.prior = args_[0]




    
    def logProb(self, q_list):

        """
        not yet vectorize (high priority given the expense of computation)
        """

        N = len(q_list)
        M = 4

        q_arr = list_to_arr(q_list)
        q_diff = riem_log(self.mu, q_arr)

        logProb = np.zeros((N, ))

        logProb += -1/2 * M * np.log(2 * np.pi)
        logProb += -1/2 * np.linalg.det(self.sigma)

        a = np.linalg.det(self.sigma)

        logProb += -1/2 * np.sum( q_diff.T * (self.sigma @ q_diff.T), axis= 0) #vectorize quadratic function


        return logProb


    def prob(self, q_list):

        prob = np.exp(self.logProb(q_list))

        return  prob