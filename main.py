import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


from quat_ds import quat_ds as quat_ds_class
from util import plot_tools, traj_generator, quat_tools, load_tools, process_tools



"""####### LOAD AND PROCESS DATA ########"""
p_in, q_in, index_list                  = load_tools.load_clfd_dataset(task_id=0, num_traj=2, sub_sample=1)
q_in, q_out, q_init, q_att, index_list  = process_tools.pre_process(q_in, index_list, opt= "slerp")

"""############ PERFORM QUAT-DS ############"""

quat_ds = quat_ds_class(q_in, q_out, q_att, index_list, K_init=4)
quat_ds.begin()

# q_init = R.from_quat(-q_init.as_quat())

q_test, w_test = quat_ds.sim(q_init, dt=0.1)


"""############ PLOT RESULTS #############"""

plot_tools.plot_quat(q_test, title='q_test')
plot_tools.plot_gmm_prob(w_test, title="GMM Posterior Probability of Reproduced Data")

plot_tools.plot_reference_trajectories_DS(p_in)

plt.show()

"""############ OUTPUT RESULTS #############"""

# np.save("p_in", p_in)
# q_in_arr = np.array([q.as_quat() for q in q_in])
# np.save("q_in", q_in_arr)

quat_ds.logOut()
print(q_init.as_quat())