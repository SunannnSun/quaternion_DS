import numpy as np
import matplotlib.pyplot as plt
from quat_ds import quat_ds as quat_ds_class
from util import plot_tools, traj_generator, quat_tools, load_tools

from scipy.spatial.transform import Rotation as R
# from util.quat_tools import *





"""
####### GENERATE RANDOM TRAJECTORY ########
"""

# q_init, q_att, q_train, w_train, dt = load_tools.load_clfd_dataset(task_id=2, num_traj=1, sub_sample=10)
q_init, q_att, q_train, w_train, dt = traj_generator.generate_traj(K=1, N=40, dt=0.1)

"""
############ PERFORM QUAT-DS ############
"""
quat_ds = quat_ds_class(q_train, w_train, q_att)
quat_ds.begin()
q_test = quat_ds.sim(q_init, dt =0.1)


"""
############ PLOT RESULTS #############
"""

# plot_tools.animate_rotated_axes(q_train)
plot_tools.animate_rotated_axes(q_test)
plot_tools.plot_quat(q_test)
# plot_tools.plot_rot_vec(w_train)

# plot_tools.plot_quat(q_train)

# plot_tools.plot_rotated_axes_sequence(q_train)
# plot_tools.plot_rotated_axes_sequence(q_test)

plt.show()