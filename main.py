import numpy as np
import matplotlib.pyplot as plt
from quat_ds import quat_ds as quat_ds_class
from util import plot_tools, traj_generator, quat_tools

from scipy.spatial.transform import Rotation as R
# from util.quat_tools import *




"""
####### GENERATE RANDOM TRAJECTORY ########
"""
from load_pos_ori import load_clfd_dataset, _angular_velocities
q_init, q_att, q_train, w_train = load_clfd_dataset(task_id=2, num_traj=1, sub_sample=10)



# q_init, q_att, q_train, w_train = traj_generator.generate_traj(K=1, N=80)

# q_arr = quat_tools.list_to_arr(q_train)
# q_flipped = np.zeros((q_arr.shape))

# w     = q_arr[:, -1].copy()
# xyz   = q_arr[:, :3].copy()

# q_flipped[:, 0]  = w
# q_flipped[:, 1:] = xyz

# w_train = [np.zeros((3, ))] * (len(q_train) -1)
# for i in range(len(w_train)):
#     q_k   = q_flipped[i  , :]
#     q_kp1 = q_flipped[i+1, :]
#     w_train[i] = _angular_velocities(q_k, q_kp1, dt=0.1)

# w_train = [R.from_rotvec(w_k) for w_k in w_train]


# q_list_q   = quat_tools.list_to_arr(q_train)
# q_list_att = quat_tools.riem_log(q_att.as_quat(), q_list_q)
# plot_tools.plot_4d_coord(q_list_att)


"""
############ PERFORM QUAT-DS ############
"""
# quat_ds = quat_ds_class(q_train, w_train, q_att)
# quat_ds.begin()
# q_test = quat_ds.sim(q_init)


"""
############ PLOT RESULTS #############
"""

# plot_tools.animate_rotated_axes(q_train)
# plot_tools.animate_rotated_axes(q_test)
# plot_tools.plot_quat(q_test)
# plot_tools.plot_rot_vec(w_train)

# plot_tools.plot_quat(q_train)

# plot_tools.plot_rotated_axes_sequence(q_train)
# plot_tools.plot_rotated_axes_sequence(q_test)

plt.show()