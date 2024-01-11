import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter

from util import plot_tools, optimize_tools, quat_tools
from util.gmm import gmm as gmm_class
from util.quat_tools import *
from util.plot_tools import *



def _average_traj(q_list, num_traj, num_per_traj, index_list):
    """
    Given multiple sequence of trajecotry, take the last point of each trajectory and average out as the attractor,
    then shift each trajectory so that they all end up at a common attractor

    Note: averaging and shifting will somehow flip the signs of all quaternions. Nevertheless, the resulting sequence
    is still smooth and continuous; proceeding operations wrt the attractor 
    """

    N = num_per_traj

    q_att_list = [R.identity().as_quat()] * num_traj
    for l in range(num_traj):
        q_att_list[l] = q_list[(l+1) * N - 1].as_quat()

    q_att_list = R.from_quat(q_att_list)
    q_att_avg = q_att_list.mean()

    q_shifted = [R.identity()] * len(q_list)
    for l in range(num_traj):
        q_diff =  q_att_avg * q_att_list[l].inv()
        q_shifted[l*N: (l+1)*N] = [q_diff * q for q in q_list[l*N: (l+1)*N]]

    # ax = plot_tools.plot_demo(q_list, index_list=index_list, title="unshifted demonstration")
    # ax = plot_tools.plot_demo(q_shifted, index_list=index_list, title="shifted demonstration")

    return q_shifted




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

    file_path           = os.path.dirname(os.path.realpath(__file__))
    dir_path            = os.path.dirname(file_path)

    seq_file    = os.path.join(dir_path, "dataset", "pos_ori", "robottasks_pos_ori_sequence_4.txt")
    filenames   = _get_sequence(seq_file)
    datafile    = os.path.join(dir_path, "dataset", "pos_ori", filenames[task_id])
    

    data        = np.load(datafile)[:, ::sub_sample, :]
    L, N, M     = data.shape
    
    N_tot = num_traj * N

    q_in  = [R.identity()] * N_tot
    q_out = [R.identity()] * N_tot 
    index_list = [0] * N_tot
    
    for l in range(num_traj):
        data_ori = np.zeros((N, 4))

        w        = data[l, :, 3 ].copy()
        xyz      = data[l, :, 4:].copy()
        data_ori[:, -1]  = w
        data_ori[:, 0:3] = xyz

        q_in[l*N: (l+1)*N] = [R.from_quat(q) for q in data_ori.tolist()]
        index_list[l*N: (l+1)*N] = [i for i in range(N)]        
        
    q_in = _average_traj(q_in, num_traj, N, index_list)
    

    q_init = q_in[0]
    q_att  = q_in[-1]

    plot_quat(q_in, title='q_train_raw')


    return q_in, q_init, q_att, index_list

