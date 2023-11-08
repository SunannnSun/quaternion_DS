import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from util.quat_tools import *



def generate_traj(K=2, N=40, dt=0.1, **kwargs):
    """
    :param K: the number of trajectory
    :param N: the number of point per trajectory


    """
    # rng_seed =  np.random.RandomState(seed=1)
    rot_vel  = np.pi/6
    rng_seed =  np.random.RandomState(seed=1)
    w_list   = [R.random(random_state=rng_seed).as_rotvec() for k in range(K)]
    w_list   = [R.from_rotvec(rot_vel * rot_vec / np.linalg.norm(rot_vec)) for rot_vec in w_list]


    q_train = [R.identity()] * (N * K)
    w_train = [R.identity()] * (N * K -1)


    if "q_init" in kwargs:
        q_init = kwargs["q_init"]
    else:
        q_init = R.identity()


    q_train[0] = q_init
    for i in range (N * K -1):
        k = i // N
        if (k != K-1):
            w_train[i] = w_list[k]
        else:
            w_train[i] = R.from_rotvec(w_list[k].as_rotvec() * (1 - i/(N*K)))
        q_train[i+1] =  R.from_rotvec(w_train[i].as_rotvec() * dt) * q_train[i]         # Rotate wrt the world frame
    q_att = q_train[-1]


    return q_init, q_att, q_train, w_train



    # w_k_dt = R.from_rotvec(w_train[i].as_rotvec() * dt)                           # Rotate wrt the body frame
    # q_train[i+1] = R.from_matrix(w_k_dt.apply(q_train[i].as_matrix()))