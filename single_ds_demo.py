import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from util import plot_tools, optimize_tools
from util.quat_tools import *




if __name__ == "__main__":
    """
    Demonstrate learning a single linear dynamical system in quaternion space
    """

    ##### Create and plot the synthetic demonstration data ####
    N = 40
    dt = 0.1
    q_init = R.identity()
    w_init = np.pi/6 * np.array([1, 0, 1]) 

    q_train = [q_init]
    w_train = [w_init]
    

    q_id_q = canonical_quat(R.identity().as_quat())


    for i in range(N):
        q_next =  R.from_rotvec(w_train[i] * dt) * q_train[i]
        q_train.append(q_next)
        w_train.append(w_init*(N-i)/N)
        # w_train.append(w_init)


    fig = plt.figure()
    ax = fig.add_subplot(projection="3d", proj_type="ortho")
    ax.figure.set_size_inches(10, 8)
    ax.set(xlim=(-2, 2), ylim=(-2, 2), zlim=(-2, 2))
    ax.set_aspect("equal", adjustable="box")
    plot_tools.animate_rotated_axes(ax, q_train)


    # q_att = R.from_euler('xyz', np.pi/2 * np.array([0, 1, 0]) )


    #### Learn the matrix A of single linear DS ####
    A = optimize_tools.optimize_single_quat_system(q_train, w_train,  q_train[-1])

    # A = optimize_tools.optimize_single_quat_system(q_train, w_train,  q_att)


    #### Reproduce the demonstration ####
    # q_init = R.random()
    q_init = R.identity()

    q_test = [q_init]

    q_att_q = canonical_quat(q_train[-1].as_quat())

    # q_att_q = canonical_quat(q_att.as_quat())

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
