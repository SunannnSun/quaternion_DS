import numpy as np
import cvxpy as cp

from util.quat_tools import *
from util.plot_tools import *



def optimize_quat_system(q_train, w_train, t_train, q_att, postProb):
    """
    :param q_train:  list of Rotation objects representing orientation, should be length N
    :param w_train:  list of Rotation objects representing angular velocity, should be length N-1
    :param q_att:    single Rotation object represent the target attractor
    :param postProb: posterior probability of each observation, shape (K, N), where K is number of components and N is the number of observations

    """
    
    q_train_att = riem_log(q_att, q_train)

    # w_train_global  = [q_train[i]*w_train[i]*q_train[i].inv() for i in range(len(w_train))]

    # w_train_global_wrt_id  = riem_log(R.identity().inv(), w_train_global)
    # plot_4d_coord(w_train_global_wrt_id, title='w_train_global_wrt_id')

    # w_train_global_wrt_att = parallel_transport(R.identity().inv(), q_att, w_train_global_wrt_id)
    # plot_4d_coord(w_train_global_wrt_att, title='w_train_global_wrt_att')


    d_train = [q_train[0]] * len(q_train)

    for i in range(len(w_train)):

        dq = w_train[i].as_rotvec() * (t_train[i+1] - t_train[i])

        d_train[i+1]  = q_train[i] * R.from_rotvec(dq)

    # plot_quat(d_train, title='displacement in quaternion')
    # plot_quat(q_train, title='displacement in quaternion')

    # plot_rot_vec(d_train, title='displacement in axis angle')

    # d_train_att = riem_log(q_att, d_train)
    # plot_4d_coord(d_train_att, title='d_train_att')

    d_train_body = riem_log(q_train[:-1], d_train[1:])          # project each displacement wrt their corresponding orientation
    # plot_4d_coord(d_train_body, title='d_train_body')


    d_train_att = parallel_transport(q_train[:-1], q_att, d_train_body)
    # plot_4d_coord(d_train_att, title='d_train_att')



    # plot_4d_coord(q_train_att)
    # plot_rot_vec(w_train)
    # w_train     = list_to_arr(w_train)
    # w_train_body  = riem_log(q_train[:-1], w_train)
    # w_train_att = parallel_transport(q_train[:-1], q_att, w_train_body)
    # plot_4d_coord(w_train_att)

    
    K, _ = postProb.shape
    N = len(q_train)
    M = 4


    max_norm = 0.5
    A_vars = []
    constraints = []
    for k in range(K):
        A_vars.append(cp.Variable((M, M), symmetric=True))
        # constraints += [A_vars[k] << 0]

        constraints += [A_vars[k].T + A_vars[k] << -0.01 * np.eye(4)]
        constraints += [cp.norm(A_vars[k], 'fro') <= max_norm]



    for k in range(K):
        w_pred_att_k = A_vars[k] @ q_train_att[:-1].T
        if k == 0:
            w_pred_att  = cp.multiply(np.tile(postProb[k, :-1], (M, 1)), w_pred_att_k)
        else:
            w_pred_att += cp.multiply(np.tile(postProb[k, :-1], (M, 1)), w_pred_att_k)
    w_pred_att = w_pred_att.T

    

    # objective = cp.sum(cp.norm2(w_pred_att - w_train_global_wrt_att, axis=0))
    objective = cp.norm(w_pred_att-d_train_att, 'fro')


    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=cp.MOSEK, verbose=True)


    A_res = np.zeros((K, M, M))
    for k in range(K):
        A_res[k, :, :] = A_vars[k].value
        print(A_vars[k].value)
        print(np.linalg.norm(A_vars[k].value, 'fro'))

    return A_res