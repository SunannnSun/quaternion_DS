import os
import numpy as np
from util import plot_tools
from scipy.spatial.transform import Rotation as R
from util import plot_tools, optimize_tools
import matplotlib.pyplot as plt
from util.gmm import gmm as gmm_class
from util.quat_tools import *


def _get_sequence(seq_file):
    """
    Returns a list of containing each line of `seq_file`
    as an element

    Args:
        seq_file (str): File with name of demonstration files
                        in each line

    Returns:
        [str]: List of demonstration files
    """
    seq = None
    with open(seq_file) as x:
        seq = [line.strip() for line in x]
    return seq

def _angular_velocities(q1, q2, dt):
    """
    https://mariogc.com/post/angular-velocity-quaternions/
    """
    return (2 / dt) * np.array([
        q1[0]*q2[1] - q1[1]*q2[0] - q1[2]*q2[3] + q1[3]*q2[2],
        q1[0]*q2[2] + q1[1]*q2[3] - q1[2]*q2[0] - q1[3]*q2[1],
        q1[0]*q2[3] - q1[1]*q2[2] + q1[2]*q2[1] - q1[3]*q2[0]])
    


def load_clfd_dataset(task_id=1, num_traj=1, sub_sample=3):
    """
    [num_demos=9, trajectory_length=1000, data_dimension=7] 
    A data point consists of 7 elements: px,py,pz,qw,qx,qy,qz (3D position followed by quaternions in the scalar first format).
    """


    dir_path    = os.path.dirname(os.path.realpath(__file__))
    seq_file    = os.path.join(dir_path, "dataset", "pos_ori", "robottasks_pos_ori_sequence_4.txt")
    filenames   = _get_sequence(seq_file)
    datafile    = os.path.join(dir_path, "dataset", "pos_ori", filenames[task_id])
    
    data        = np.load(datafile)[:, ::sub_sample, :]
    L, N, M     = data.shape

    data_flat        = np.zeros((num_traj*N, M))
    for l in range(num_traj):
        data_flat[l*N:(l+1)*N, :] = data[l, :, :]

    data_pos = data_flat[:, :3]
    data_ori = data_flat[:, 3:]

    w     = data_ori[:, 0].copy()
    xyz   = data_ori[:, 1:].copy()

    data_ori[:, -1]  = w
    data_ori[:, 0:3] = xyz


    q_train = [R.from_quat(q) for q in data_ori.tolist()]
    q_init = q_train[0]
    q_att  = q_train[-1]



    w_train = [R.identity()] * (len(q_train) -1)
    for i in range(len(w_train)):
        q_k   = data_flat[i  , 3:]
        q_kp1 = data_flat[i+1, 3:]
        w_train[i] = _angular_velocities(q_k, q_kp1, dt=0.1)
    # w_train.append(dq_dt*0)


    w_train = [R.from_euler('xyz', w_k) for w_k in w_train]
    


    return q_init, q_att, q_train, w_train