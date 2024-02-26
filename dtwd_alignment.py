from dtw import dtw
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


from quat_ds import quat_ds as quat_ds_class
from util import plot_tools, traj_generator, quat_tools, load_tools, process_tools



# We define two sequences x, y as numpy array
# where y is actually a sub-sequence from x

# shape N by M 


# x = np.array([2, 0, 1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)
# y = np.array([1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)

p_in, q_in, index_list_raw                  = load_tools.load_clfd_dataset(task_id=2, num_traj=9, sub_sample=1)
q_in, q_out, q_init, q_att, index_list  = process_tools.pre_process(q_in, index_list_raw, opt= "slerp")


quat_ds = quat_ds_class(q_in, q_out, q_att, index_list, K_init=4)
quat_ds.begin()

q_test, w_test = quat_ds.sim(q_init, dt=0.1, if_perturb=False)

# plot_tools.plot_train_test(q_in, index_list, q_test)

plot_tools.plot_train_test_4d(q_in, index_list, q_test)


# dtw_obj = dtw(x, y, keep_internals=True)

# cost_matrix = dtw_obj.costMatrix

# idx = []

# for i in range(cost_matrix.shape[0]):

#     idx.append(np.argmin(cost_matrix[i, :]))



# dtw_obj.plot()

# print(dtw_obj.index1)

# print(dtw_obj.index2)



plt.show()
a = 1