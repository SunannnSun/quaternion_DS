import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from util import plot_tools, optimize_tools
from util.quat_tools import *


def canonical_quat(q):
    """
    Force all quaternions to have positive scalar part; necessary to ensure proper propagation in DS
    """
    if (q[-1] < 0):
        return -q
    else:
        return q



if __name__ == "__main__":

    N = 50
    dt = 0.2  # unit time
    ang_vel = np.pi/6

    w_axis = np.array([1, 0, 0]) 
    q_train = [R.identity()]
    w_train = [w_axis * ang_vel]

    for i in range(N):
        q_next =  R.from_rotvec(w_train[i] * dt) * q_train[i]
        q_train.append(q_next)
        w_train.append(w_axis * ang_vel * (N-i)/N)  #decaying velocity approaching zero near attractor

    q_att = q_train[-1]
    
    A = optimize_tools.optimize_single_quat_system(q_train[0:-1], w_train[0:-1], q_att)

    print(A)


    q_init = R.from_euler('xyz', [12, 150, 130], degrees=True)
    # q_init = R.identity()


    q_test = [q_init]
    w_test = []

    q_att = canonical_quat(q_att.as_quat())

    for i in range(N+200):
        q_curr_q =  canonical_quat(q_test[i].as_quat())
        q_curr_t = riem_log(q_att, q_curr_q)[:, np.newaxis]
        w_pred = A @ q_curr_t  

        w_new = parallel_transport(q_att, q_curr_q, w_pred)

        q_next = riem_exp(q_curr_q, w_new * dt)

        q_test.append(R.from_quat(q_next))

    



    fig = plt.figure()
    ax = fig.add_subplot(projection="3d", proj_type="ortho")
    ax.figure.set_size_inches(10, 8)
    ax.set(xlim=(-2, 2), ylim=(-2, 2), zlim=(-2, 2))
    ax.set_aspect("equal", adjustable="box")

    plot_tools.animate_rotated_axes(ax, q_test)
    # plot_tools.animate_rotated_axes(ax, q_train)


    plt.show()



        # # r = R.from_quat(q_test)
    # rotations = [canonical_quat(q.as_quat()) for q in q_test]
    # r = R.from_quat(rotations)
    # q_mean = r.mean()
    # # R.mean(q_test)