import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from util import plot_tools, optimize_tools


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
    dt = 0.1  # unit time
    ang_vel = np.pi/6

    w_axis = np.array([1, 0, 0]) 
    q_train = [R.identity()]
    w_train = [w_axis * ang_vel]

    for i in range(N):
        q_next =  R.from_rotvec(w_train[i] * dt) * q_train[i]
        q_train.append(q_next)
        w_train.append(w_axis * ang_vel * (N-i)/N)


    
    q_att = q_train[-1]
    q_now = q_train[0]


    # print(q_att.as_quat())
    # print(q_att.inv().as_quat())
    # print( (q_now * q_att.inv()).as_quat())



    A = optimize_tools.optimize_single_system(q_train, w_train, q_att)

    q_test = [R.identity()]

    w_test = []

    for i in range(N):
        q_diff = (q_test[i] * q_att.inv()).as_quat()
        
        w_pred = A @ canonical_quat(q_diff)[0:3]
        w_test.append(w_pred)

        q_next =  R.from_rotvec(w_test[i] * dt) * q_test[i]
        q_test.append(q_next)

    


    fig = plt.figure()
    ax = fig.add_subplot(projection="3d", proj_type="ortho")
    ax.figure.set_size_inches(10, 8)
    ax.set(xlim=(-2, 2), ylim=(-2, 2), zlim=(-2, 2))
    ax.set_aspect("equal", adjustable="box")


    plot_tools.animate_rotated_axes(ax, q_test)

    # plot_tools.plot_rotated_axes(ax, rotations[0])
    # plot_tools.plot_rotated_axes(ax, rotations[1])

    plt.show()