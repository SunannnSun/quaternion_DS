import numpy as np
import cvxpy as cp

from util.quat_tools import *
from util.plot_tools import *



def optimize_quat_system(q_train, w_train, q_att, postProb):
    """
    :param q_train:  list of Rotation objects representing orientation, should be length N
    :param w_train:  list of Rotation objects representing angular velocity, should be length N-1
    :param q_att:    single Rotation object represent the target attractor
    :param postProb: posterior probability of each observation, shape (K, N), where K is number of components and N is the number of observations

    """

    
    q_train_att = riem_log(q_att, q_train)


    d_train_body = riem_log(q_train, w_train)            # project each displacement wrt their corresponding orientation
    d_train_att = parallel_transport(q_train, q_att, d_train_body)


    # plot_4d_coord(q_train_att, title='q_train_att')
    # plot_4d_coord(d_train_body, title='d_train_body')
    # plot_4d_coord(d_train_att, title='w_train_att')


    
    K, _ = postProb.shape
    N = len(q_train)
    M = 4


    max_norm = 0.5
    A_vars = []
    constraints = []
    for k in range(K):
        A_vars.append(cp.Variable((M, M), symmetric=False))

        # constraints += [A_vars[k].T + A_vars[k] << -10E-3 * np.eye(4)]
        constraints += [A_vars[k].T + A_vars[k] << np.zeros((4, 4))]

        # constraints += [cp.norm(A_vars[k], 'fro') <= max_norm]



    for k in range(K):
        w_pred_att_k = A_vars[k] @ q_train_att.T
        if k == 0:
            w_pred_att  = cp.multiply(np.tile(postProb[k, :], (M, 1)), w_pred_att_k)
        else:
            w_pred_att += cp.multiply(np.tile(postProb[k, :], (M, 1)), w_pred_att_k)
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

    # print(w_pred_att.value)
    return A_res