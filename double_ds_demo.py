import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from util import plot_tools, optimize_tools
from util.quat_tools import *

from util.gmm import gmm as gmm_class
from util.normal import normal as normal_class



if __name__ == "__main__":
    """
    Demonstrate learning a double linear dynamical system in quaternion space
    """


    ##### Create and plot the synthetic demonstration data ####
    N = 60
    dt = 0.1  
    q_init = R.identity()
    w_init = np.pi/6 * np.array([1, 0, 0]) 

    q_train = [q_init]
    w_train = [w_init]


    assignment_arr = np.zeros((N+1, ), dtype=int)

    for i in range(int(N/2)):
        q_next =  R.from_rotvec(w_train[i] * dt) * q_train[i]
        q_train.append(q_next)
        w_train.append(w_init * (N-i)/N)  

    w_new = np.pi/6 * np.array([0, 1, 0]) 
    for i in np.arange(int(N/2), N):
        assignment_arr[i] = 1
        q_next =  R.from_rotvec(w_train[i] * dt) * q_train[i]
        q_train.append(q_next)
        w_train.append(w_new * (N-i)/N) 


    # fig = plt.figure()
    # ax = fig.add_subplot(projection="3d", proj_type="ortho")
    # ax.figure.set_size_inches(10, 8)
    # ax.set(xlim=(-2, 2), ylim=(-2, 2), zlim=(-2, 2))
    # ax.set_aspect("equal", adjustable="box")
    # plot_tools.animate_rotated_axes(ax, q_train)



    #### Perform pseudo clustering, return assignment_arr and sufficient statistics ####
    gmm = gmm_class(q_train)
    gmm.cluster(assignment_arr)
    # gmm.return_norma_class(q_train, assignment_arr)
    normal_list = gmm.return_norma_class(q_train)

    normal_list[0].logProb(q_train)



