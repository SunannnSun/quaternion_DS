import numpy as np
import matplotlib.pyplot as plt
from quat_ds import quat_ds as quat_ds_class
from util import plot_tools, traj_generator

# from scipy.spatial.transform import Rotation as R
# from util.quat_tools import *






####### GENERATE RANDOM TRAJECTORY ########

q_init, q_att, q_train, w_train = traj_generator.generate_traj(K=2, N=40)
plot_tools.animate_rotated_axes(q_train)



############ PERFORM QUAT-DS ############

quat_ds = quat_ds_class(q_train, w_train, q_att)
quat_ds.begin()
q_test = quat_ds.sim(q_init)



############ PLOT RESULTS #############

plot_tools.animate_rotated_axes(q_test)
plot_tools.plot_quat(q_test)
plot_tools.plot_rotated_axes_sequence(q_test)
