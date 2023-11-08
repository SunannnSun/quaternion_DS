import numpy as np
import matplotlib.pyplot as plt
from quat_ds import quat_ds as quat_ds_class
from util import plot_tools, traj_generator, quat_tools

from scipy.spatial.transform import Rotation as R
# from util.quat_tools import *


q_0 = R.from_euler("xyz", np.array([np.pi/3, 0, 0]))

w   = R.from_rotvec(np.array([0, np.pi/3, 0]))


q_1 = w * q_0


q = [q_0, q_1]

plot_tools.animate_rotated_axes(q)
