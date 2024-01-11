import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from util import plot_tools, optimize_tools, quat_tools
from util.quat_tools import *
from util.gmm import gmm as gmm_class



class quat_ds:
    def __init__(self, q_train, w_train, q_att, **argv) -> None:
        self.q_att   = q_att
        self.q_train = q_train
        self.w_train = w_train
        # self.t_train = t_train

        self.N = len(q_train)

        if "index_list" in argv:
            self.index_list = argv["index_list"]
        
    
    def _cluster(self):
        gmm = gmm_class(self.q_att, self.q_train, index_list = self.index_list)
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
        N = self.N + 100
        K = self.K
        h = np.zeros((N, K))
        q_test = [q_init]


        for i in range(N):
            q_curr = q_test[i]

            h_k = self.gmm.postLogProb(q_curr)  
            h[i, :] = h_k.T
            
            q_curr_att = riem_log(self.q_att, q_curr)

            d_pred_att = np.zeros((4, 1))
            for k in range(self.K):
                h_k_i =  h_k[k, 0]
                w_k_i =  self.A[k] @ q_curr_att.reshape(-1, 1)
                d_pred_att += h_k_i * w_k_i

            d_pred_body = parallel_transport(self.q_att, q_curr, d_pred_att.T)
            d_pred_q    = riem_exp(q_curr, d_pred_body) 


            # d_pred_q    = riem_exp(self.q_att, d_pred_att.T) 

            d_pred      = R.from_quat(d_pred_q.reshape(4,))

            # q_next = w_pred * q_test[i] # rotate about body frame
            q_test.append(d_pred)
        
        # plot_tools.plot_4d_coord(w_test)
        
        fig, axs = plt.subplots(K, 1, figsize=(12, 8))
        
        import random 
        colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
        "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]
        
        for k in range(K):
            axs[k].scatter(np.arange(N), h[:, k], s=5, color=colors[k])
            axs[k].set_ylim([0, 1])
        axs[0].set_title("GMM Posterior Probability of Reproduced Data")

            # axs[k].legend(loc="upper left")

        return q_test