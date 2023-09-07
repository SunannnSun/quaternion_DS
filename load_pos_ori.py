import os
import numpy as np
from util import plot_tools
from scipy.spatial.transform import Rotation as R
from util import plot_tools, optimize_tools
import matplotlib.pyplot as plt
from util.gmm import gmm as gmm_class


def get_sequence(seq_file):
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
    
dir_path     = os.path.dirname(os.path.realpath(__file__))
seq_file = os.path.join(dir_path, "pos_ori", "robottasks_pos_ori_sequence_4.txt")
filenames = get_sequence(seq_file)

task_id = 0
datafile = os.path.join(dir_path, "pos_ori", filenames[task_id])
data = np.load(datafile)  

"""
[num_demos=9, trajectory_length=1000, data_dimension=7] 
A data point consists of 7 elements: px,py,pz,qw,qx,qy,qz (3D position followed by quaternions in the scalar first format).
"""


sub_sample = 3
data = data[:, ::sub_sample, :]

L, N, M = data.shape

traj_num = 1
Data = np.zeros((traj_num*N, M))

for l in range(traj_num):
    data_l = data[l, :, :]
    Data[l*N:(l+1)*N, :] = data[l, :, :]

Data = Data[:, 3:]

Data_w = Data[:, 0].copy()
Data_xyz = Data[:, 1:].copy()

Data[:, -1] = Data_w
Data[:, 0:3] = Data_xyz

print(Data[0, :])


q_train = [R.from_quat(q) for q in Data.tolist()]

print(len(q_train))

fig = plt.figure()
ax = fig.add_subplot(projection="3d", proj_type="ortho")
ax.figure.set_size_inches(10, 8)
ax.set(xlim=(-2, 2), ylim=(-2, 2), zlim=(-2, 2))
ax.set_aspect("equal", adjustable="box")
plot_tools.animate_rotated_axes(ax, q_train)

gmm = gmm_class(q_train[-1], q_train)
labels = gmm.begin()
    

print(labels)
