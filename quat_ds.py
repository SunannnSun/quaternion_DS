import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from util import plot_tools, optimize_tools
from util.quat_tools import *
from util.gmm import gmm as gmm_class



class quat_ds:
    def __init__(self, q_train, w_train, q_att) -> None:
        self.q_att   = q_att
        self.q_train = q_train
        self.w_train = w_train

        self.N = len(q_train)
        
    
    def _cluster(self):
        gmm = gmm_class(self.q_att, self.q_train)
        label = gmm.begin()
        self.postProb = gmm.postLogProb(self.q_train)
        self.gmm = gmm

    def _optimize(self):
        self.A = optimize_tools.optimize_quat_system(self.q_train, self.w_train, self.q_att, self.postProb)
        self.K = self.A.shape[0]


    def begin(self):
        self._cluster()
        self._optimize()


    def sim(self, q_init, dt=0.1):

        q_test = [q_init]

        for i in range(self.N+100):
            q_curr = canonical_quat(q_test[i].as_quat())
            q_curr_att = riem_log(self.q_att, q_curr)

            h_k = self.gmm.postLogProb(q_curr)

                      
            w_pred_att = np.zeros((4, 1))
            for k in range(self.K):
                h_k_i =  h_k[k, 0]
                w_k_i =  self.A[k] @ q_curr_att[:, np.newaxis]
                w_pred_att += h_k_i * w_k_i


            w_pred_id = parallel_transport(self.q_att, self.q_id, w_pred_att)
            w_pred_q  = riem_exp(self.q_id.as_quat(), w_pred_id * dt) # multiplied by dt before projecting back to the quaternion space
            w_pred    = R.from_quat(w_pred_q)

            q_next = w_pred * q_test[i]
            q_test.append(q_next)
    
        return q_test