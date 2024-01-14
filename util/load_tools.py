import os
import numpy as np
from scipy.spatial.transform import Rotation as R



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




def load_clfd_dataset(task_id=1, num_traj=1, sub_sample=3):
    """
    Solely used to load the raw dataset

    Note:
        [num_demos=9, trajectory_length=1000, data_dimension=7] 
        A data point consists of 7 elements: px,py,pz,qw,qx,qy,qz (3D position followed by quaternions in the scalar first format).
    """

    file_path           = os.path.dirname(os.path.realpath(__file__))
    dir_path            = os.path.dirname(file_path)

    seq_file    = os.path.join(dir_path, "dataset", "pos_ori", "robottasks_pos_ori_sequence_4.txt")
    filenames   = _get_sequence(seq_file)
    datafile    = os.path.join(dir_path, "dataset", "pos_ori", filenames[task_id])
    

    data        = np.load(datafile)[:, ::sub_sample, :]
    _, N, _     = data.shape
    N_tot = num_traj * N

    q_in  = [R.identity()] * N_tot
    
    for l in range(num_traj):
        data_ori = np.zeros((N, 4))

        w        = data[l, :, 3 ].copy()
        xyz      = data[l, :, 4:].copy()
        data_ori[:, -1]  = w
        data_ori[:, 0:3] = xyz

        q_in[l*N: (l+1)*N] = [R.from_quat(q) for q in data_ori.tolist()]

    index_list = [np.arange(l*N, (l+1)*N) for l in range(num_traj)]
    
    return q_in, index_list

