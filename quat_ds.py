import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from util import plot_tools, optimize_tools, quat_tools
from util.quat_tools import *
from util.gmm import gmm as gmm_class



class quat_ds:
    def __init__(self, q_train, w_train, t_train, q_att) -> None:
        self.q_att   = q_att
        self.q_train = q_train
        self.w_train = w_train
        self.t_train = t_train

        self.N = len(q_train)
        
    
    def _cluster(self):
        gmm = gmm_class(self.q_att, self.q_train)
        label = gmm.begin()
        self.postProb = gmm.postLogProb(self.q_train)
        self.gmm = gmm

    def _optimize(self):
        self.A = optimize_tools.optimize_quat_system(self.q_train, self.w_train, self.t_train, self.q_att, self.postProb)
        self.K = self.A.shape[0]


    def begin(self):
        self._cluster()
        self._optimize()


    def sim(self, q_init, dt=0.1):
        
        N_tot = self.N + 20
        q_test = [q_init]

        # w_test = np.zeros((N_tot, 4))

        # d_test = [R.identity()] * N_tot  

        for i in range(N_tot):
            q_curr = q_test[i]
            q_curr_att = riem_log(self.q_att, q_curr)

            h_k = self.gmm.postLogProb(q_curr)  # (K by N)

                      
            d_pred_att = np.zeros((4, 1))
            for k in range(self.K):
                h_k_i =  h_k[k, 0]
                w_k_i =  self.A[k] @ q_curr_att.reshape(-1, 1)
                d_pred_att += h_k_i * w_k_i
            # d_test[i,:] = d_pred_att.T

            d_pred_body = parallel_transport(self.q_att, q_curr, d_pred_att.reshape(4,))
            d_pred_q    = riem_exp(q_curr, d_pred_body.T) 
            d_pred      = R.from_quat(d_pred_q.reshape(4,))

            # q_next = w_pred * q_test[i] # rotate about body frame
            q_test.append(d_pred)
        
        # plot_tools.plot_4d_coord(w_test)
        
    
        return q_test