import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from util import plot_tools, optimize_tools
from util.quat_tools import *

from util.gmm import gmm as gmm_class



if __name__ == "__main__":
    """
    Demonstrate learning a double linear dynamical system in quaternion space
    """


    ##### Create and plot the synthetic demonstration data ####
    rand_seed =  np.random.RandomState(seed=1)
    # rand_seed =  np.random.RandomState()


    K = 2
    N = 50
    dt = 0.1
    q_init = R.identity()
    q_train = [q_init]

    w_init = np.pi/6 * np.array([1, 0, 0]) 
    w_train = [w_init]


    q_id_q = canonical_quat(R.identity().as_quat())



    assignment_arr = np.zeros((K*N+1, ), dtype=int)

    for k in range(K):
        if len(w_train) != 0:
            w_train.pop()
            w_new = np.pi/6 * np.array([1, 0, 1]) 
            w_train.append(w_new)

        for i in np.arange(N*k, N*(k+1)):
            q_next =  R.from_rotvec(w_train[i] * dt) * q_train[i]
            q_train.append(q_next)
            if k == K-1:
                w_train.append(w_new * (N*K-i)/(N*K))
            else:
                w_train.append(w_init)
            assignment_arr[i+1] = k


    fig = plt.figure()
    ax = fig.add_subplot(projection="3d", proj_type="ortho")
    ax.figure.set_size_inches(10, 8)
    ax.set(xlim=(-2, 2), ylim=(-2, 2), zlim=(-2, 2))
    ax.set_aspect("equal", adjustable="box")
    plot_tools.animate_rotated_axes(ax, q_train)



    #### Perform pseudo clustering, return assignment_arr and sufficient statistics ####
    gmm = gmm_class(q_train)
    gmm.cluster(assignment_arr)
    gmm.return_norma_class(q_train, assignment_arr)
    postProb = gmm.postLogProb(q_train)




    A = optimize_tools.optimize_double_quat_system(q_train, w_train, q_train[-1], postProb)
# """

    #### Reproduce the demonstration ####
    # q_init = R.random()
    dt = 0.001
    q_init = R.identity()
    q_test = [q_init]

    q_att_q = canonical_quat(q_train[-1].as_quat())

    for i in range(N+500):
        q_curr_q = canonical_quat(q_test[i].as_quat())
        q_curr_t = riem_log(q_att_q, q_curr_q)
        h_k = gmm.postLogProb(q_curr_q)
        
        w_pred_att  = h_k[0, 0] * A[0] @ q_curr_t[:, np.newaxis] * dt + h_k[1, 0] * A[1] @ q_curr_t[:, np.newaxis] * dt

        # w_pred_att = np.zeros((4, 1))
        # for k in range(2):
        #     h_k_i =  h_k[k, 0]
        #     w_k_i =  A[k] @ q_curr_t[:, np.newaxis]
        #     w_pred_att += h_k_i * w_k_i * dt
        
        # w_pred_att = A[0] @ q_curr_t[:, np.newaxis] * dt

        w_pred_id = parallel_transport(q_att_q, q_id_q, w_pred_att)

        q_next = R.from_quat(riem_exp(q_id_q, w_pred_id * dt)) * R.from_quat(q_curr_q)

        q_test.append(q_next)



    #### Plot the results ####
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d", proj_type="ortho")
    ax.figure.set_size_inches(10, 8)
    ax.set(xlim=(-2, 2), ylim=(-2, 2), zlim=(-2, 2))
    ax.set_aspect("equal", adjustable="box")

    plot_tools.animate_rotated_axes(ax, q_test)
# """