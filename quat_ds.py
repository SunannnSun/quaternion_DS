import numpy as np
from scipy.spatial.transform import Rotation as R

from util import optimize_tools, plot_tools
from util.quat_tools import *
from util.gmm import gmm as gmm_class   




class quat_ds:
    def __init__(self, q_in, q_out, q_att, index_list, K_init) -> None:
        self.q_in  = q_in
        self.q_out = q_out
        self.q_att = q_att
        self.index_list = index_list
        self.K_init = K_init

        self.N = len(q_in)
        
    
    def _cluster(self):
        gmm = gmm_class(self.q_in, self.q_att, self.index_list, self.K_init)
        gmm.begin()
        
        self.postProb = gmm.postLogProb(self.q_in)
        self.gmm      = gmm
        self.K        = gmm.K
        


    def _optimize(self):
        self.A = optimize_tools.optimize_quat_system(self.q_in, self.q_out, self.q_att, self.postProb)


    def begin(self):
        self._cluster()
        self._optimize()


    def sim(self, q_init, dt=0.1):
        """
        Forward simulation given an initial point

        Args:
            q_init: A Rotation object for starting point

        Returns:
            q_test: List of Rotation objects
            w_test: Array of weights (N by K)
        
        To-do:
            Control the scale of displacement by setting a celling and floor 
        """    
        
        N = len(self.index_list) + 500
        K = self.K
        A = self.A
        q_att = self.q_att
        gmm   = self.gmm
        
        q_test = [q_init]
        w_test = np.zeros((N, K))      
        for i in range(N):
            q_in      = q_test[i]
            q_in_att  = riem_log(q_att, q_in)
            q_out_att = np.zeros((4, 1))

            w_test[i, :] = gmm.postLogProb(q_in).T
            for k in range(K):
                q_out_att += w_test[i, k] * A[k] @ q_in_att.reshape(-1, 1)

            q_out_body = parallel_transport(q_att, q_in, q_out_att.T)
            q_out_q    = riem_exp(q_in, q_out_body) 
            q_out      = R.from_quat(q_out_q.reshape(4,))
            
            q_test.append(q_out)
        
        
        return q_test, w_test