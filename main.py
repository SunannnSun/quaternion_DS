import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


from quat_ds import quat_ds as quat_ds_class
from util import plot_tools, traj_generator, quat_tools, load_tools, process_tools




"""####### LOAD AND PROCESS DATA ########"""
# q_init, q_att, q_train, w_train, t_train, dt = traj_generator.generate_traj(K=1, N=20, dt=0.1)

q_in, q_init, q_att, index_list         = load_tools.load_clfd_dataset(task_id=0, num_traj=1, sub_sample=1)
q_in, q_out, q_init, q_att, index_list  = process_tools.pre_process(q_in, q_att, index_list)


"""############ PERFORM QUAT-DS ############"""

quat_ds = quat_ds_class(q_in, q_out, q_att, index_list = index_list)
quat_ds.begin()

q_test, w_test = quat_ds.sim(q_init)


"""############ PLOT RESULTS #############"""

plot_tools.plot_quat(q_test, title='q_test')
plot_tools.plot_gmm_prob(w_test, title="GMM Posterior Probability of Reproduced Data")

plt.show()