import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


from quat_ds_coupled import quat_ds as quat_ds_class
from util import plot_tools, traj_generator, quat_tools, load_tools, process_tools
from util.gmm_coupled import gmm as gmm_class   



"""####### LOAD AND PROCESS DATA ########"""
p_in, q_in, index_list                  = load_tools.load_clfd_dataset(task_id=2, num_traj=1, sub_sample=1)
q_in, q_out, q_init, q_att, index_list  = process_tools.pre_process(q_in, index_list, opt= "slerp")


# quat_ds = quat_ds_class(p_in, q_in, q_out, q_att, index_list, K_init=5)
# quat_ds._cluster()
# quat_ds.begin()
# p_init = p_in[-1, :].reshape(1, -1)

# q_test, w_test = quat_ds.sim(p_init, q_init, dt=0.1)



# plot_tools.plot_quat(q_test, title='q_test')
# plot_tools.plot_gmm_prob(w_test, title="GMM Posterior Probability of Reproduced Data")

# plot_tools.plot_reference_trajectories_DS(p_in)


# plt.show()
gmm = gmm_class(p_in, q_in, q_att)
gmm.fit(K_init=5)

q_new = []
p_new = []
for l in range(len(index_list)):
    for idx in index_list[l]:
        p_new.append(p_in[idx, :])
        q_new.append(q_in[idx])

# gmm.predict(np.array(p_new), q_new, index_list)



# w_train = np.zeros((len(q_new), gmm.K))
# for i in range(len(q_new)):
#     p_i = p_new[i].reshape(1, -1)
#     w_train[i, :] = gmm.postLogProb(p_i, q_new[i]).T


p_i = p_new[100].reshape(1, -1)

w_train = gmm.postLogProb(p_i, q_new[100]).T

print(w_train)
plot_tools.plot_gmm_prob(w_train, title="GMM Posterior Probability of Original Data")


plt.show()