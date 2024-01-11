import numpy as np
import matplotlib.pyplot as plt
from quat_ds import quat_ds as quat_ds_class
from util import plot_tools, traj_generator, quat_tools, load_tools
from util.gmm import gmm as gmm_class

from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter



"""
Load data
"""
q_init, q_att, q_train, w_train, dt, index_list = load_tools.load_clfd_dataset(task_id=3, num_traj=1, sub_sample=1)



"""
Smoothen data
"""
q_train_att = quat_tools.riem_log(q_att, q_train)

q_train_savgol = savgol_filter(q_train_att, window_length=80, polyorder=2, axis=0, mode="nearest")

q_train_arr = quat_tools.riem_exp(q_att, q_train_savgol)
# plot_tools.plot_4d_coord(q_train_arr, title='q_train_smooth')

q_train = [R.from_quat(q_train_arr[i, :]) for i in range(q_train_arr.shape[0])]

plot_tools.plot_quat(q_train, title='q_train_smooth')


"""
Pre-process data
"""

gmm = gmm_class(q_att, q_train, index_list = index_list)
label = gmm.begin()

N = len(q_train)

max_threshold = 0.03

threshold = max_threshold * np.ones((N, ))
threshold[label==gmm.K-1] = np.linspace(max_threshold, 0, num=np.sum(label==gmm.K-1), endpoint=True)

# threshold = np.linspace(max_threshold, 0, num=N, endpoint=True)

q_new  = [q_train[0]]
w_new  = []
for i in np.arange(1, N):
    q_curr = q_new[-1]
    q_next = q_train[i]
    dis    = q_next * q_curr.inv()
    if np.linalg.norm(dis.as_rotvec()) < threshold[i]:
        pass
    else:
        q_new.append(q_next)
        w_new.append(q_next)
w_new.append(w_new[-1])
N = len(q_new)
index_list_new = [i for i in range(len(q_new))]


plot_tools.plot_quat(q_new, title='q_new')

q_att = q_new[-1]

    

d_train_body = quat_tools.riem_log(q_new, w_new)            # project each displacement wrt their corresponding orientation
d_train_att = quat_tools.parallel_transport(q_new, q_att, d_train_body)


plot_tools.plot_4d_coord(d_train_body, title='d_train_body')
plot_tools.plot_4d_coord(d_train_att, title='w_train_att')



"""
GMM validation
"""
gmm = gmm_class(q_new[-1], q_new, index_list = index_list_new)
label = gmm.begin()

w_train = np.zeros((N, gmm.K))
for i in range(N):
    w_train[i, :] = gmm.postLogProb(q_new[i]).T
plot_tools.plot_gmm_prob(w_train, title="GMM Posterior Probability of Original Data")



"""
Learning
"""
quat_ds = quat_ds_class(q_new, w_new, q_att, index_list = index_list_new)
quat_ds.begin()


"""
Forward Simulation
"""
q_test, w_test = quat_ds.sim(q_init, dt=0.1)
plot_tools.plot_quat(q_test, title='q_test')
plot_tools.plot_gmm_prob(w_test, title="GMM Posterior Probability of Reproduced Data")


plt.show()

