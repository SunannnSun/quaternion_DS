import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from util import plot_tools, optimize_tools
from util.quat_tools import *




if __name__ == "__main__":
    """
    Multiple behaviour but single quat system
    """

    ##### Create and plot the synthetic demonstration data ####
    N = 20
    dt = 0.1

    q_id_q = canonical_quat(R.identity().as_quat())
    q_init = R.identity()
    w_init = np.pi/6 * np.array([1, 0, 0]) 

    q_train = [q_init]
    w_train = [w_init]
    

    # for k in range(2):
    #     if k == 1:
    #         w_new = np.pi/6 * np.array([0, 0, 1]) 
    #     for i in np.arange(N*k, N*(k+1)):
    #         q_next =  R.from_rotvec(w_train[i] * dt) * q_train[i]
    #         q_train.append(q_next)
    #         if k == 1:
    #             w_train.append(w_new * (N*2-i)/(N*2))
    #         else:
    #             w_train.append(w_init)


    N1 = 20
    N2 = 50
    for i in range(N1):
        q_next =  R.from_rotvec(w_train[i] * dt) * q_train[i]
        q_train.append(q_next)
        w_train.append(w_init)  

    w_new = np.pi/3 * np.array([0, 0, 1]) 
    for i in np.arange(N1, N2):
        q_next =  R.from_rotvec(w_train[i] * dt) * q_train[i]
        q_train.append(q_next)
        w_train.append(w_new * (N2-i)/N2) 



    fig = plt.figure()
    ax = fig.add_subplot(projection="3d", proj_type="ortho")
    ax.figure.set_size_inches(10, 8)
    ax.set(xlim=(-2, 2), ylim=(-2, 2), zlim=(-2, 2))
    ax.set_aspect("equal", adjustable="box")
    plot_tools.animate_rotated_axes(ax, q_train)










    #### Learn the matrix A of single linear DS ####
    A = optimize_tools.optimize_single_quat_system(q_train, w_train,  q_train[-1])


    #### Reproduce the demonstration ####
    # q_init = R.random()
    q_init = R.identity()
    dt = 0.1
    q_test = [q_init]

    q_att_q = canonical_quat(q_train[-1].as_quat())

    for i in range(N+200):
        q_curr_q = canonical_quat(q_test[i].as_quat())
        q_curr_t = riem_log(q_att_q, q_curr_q)
        w_pred_att = A @ q_curr_t[:, np.newaxis]

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
