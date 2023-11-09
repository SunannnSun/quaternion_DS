import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter

from util import plot_tools, optimize_tools, quat_tools



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

    Remark: dataset does not provide time stamp. Hence, in order to compute velocity, we woudl like to determine an appropriate dt.
    Given each task contains 1000 observations, we can safely assume the time it takes to complete one trajectory is 10 seconds.

    Manually dealing with multiple trajectories by assigning correct angular velocity near start and end of each trajecotry; i.e. apply 
    Savgol separately for each individual trajectory
    """
    file_path           = os.path.dirname(os.path.realpath(__file__))
    dir_path            = os.path.dirname(file_path)

    seq_file    = os.path.join(dir_path, "dataset", "pos_ori", "robottasks_pos_ori_sequence_4.txt")
    filenames   = _get_sequence(seq_file)
    datafile    = os.path.join(dir_path, "dataset", "pos_ori", filenames[task_id])
    

    data        = np.load(datafile)[:, ::sub_sample, :]
    L, N, M     = data.shape
    T = 10
    dt = T / 1000 * sub_sample
    N_tot = num_traj * N


    q_train = [R.identity()] *  N_tot
    w_train = [R.identity()] * (N_tot-1)


    for l in range(num_traj):
        data_ori = np.zeros((N, 4))

        w        = data[l, :, 3 ].copy()
        xyz      = data[l, :, 4:].copy()
        data_ori[:, -1]  = w
        data_ori[:, 0:3] = xyz
        q_train[l*N: (l+1)*N] = [R.from_quat(q) for q in data_ori.tolist()]

        
        rotvec_l = np.zeros((N-1, 3))
        for i in range(rotvec_l.shape[0]):
            q_k   = data[l, i,   3:]
            q_kp1 = data[l, i+1, 3:]
            rotvec_l[i, :] = _angular_velocities(q_k, q_kp1, dt)
        
        rotvec_l = savgol_filter(rotvec_l, window_length=20, polyorder=2, axis=0, mode="nearest")

        if l != num_traj-1:
            rotvec_l = np.vstack((rotvec_l, rotvec_l[-1, :][np.newaxis, :]))
            w_train[l*N: (l+1)*N]   = [R.from_rotvec(rotvec_l[j, :]) for j in range(rotvec_l.shape[0])]
        else:
            w_train[l*N: (l+1)*N]   = [R.from_rotvec(rotvec_l[j, :]) for j in range(rotvec_l.shape[0])]



    q_init = q_train[0]
    q_att  = q_train[-1]

    

    return q_init, q_att, q_train, w_train, dt