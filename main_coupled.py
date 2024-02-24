import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


from quat_ds_coupled import quat_ds as quat_ds_class
from util import plot_tools, traj_generator, quat_tools, load_tools, process_tools
from util.gmm_coupled import gmm as gmm_class   


# sys.path.append('./damm_lpvds')

# from damm_lpvds import damm_lpvds as damm_lpvds_class



"""####### LOAD AND PROCESS DATA ########"""
p_in, q_in, index_list                  = load_tools.load_clfd_dataset(task_id=2, num_traj=1, sub_sample=1)
q_in, q_out, q_init, q_att, index_list  = process_tools.pre_process(q_in, index_list, opt= "slerp")















plt.show()