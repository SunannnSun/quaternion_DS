import numpy as np
import cvxpy as cp

from util.quat_tools import *
from util.plot_tools import *



def optimize_quat_system(q_in, q_out, q_att, postProb):
    """
    :param q_in:  list of Rotation objects representing orientation, should be length N
    :param q_out:  list of Rotation objects representing angular velocity, should be length N-1
    :param q_att:    single Rotation object represent the target attractor
    :param postProb: posterior probability of each observation, shape (K, N), where K is number of components and N is the number of observations

    """

    
    q_in_att   = riem_log(q_att, q_in)

    q_out_body = riem_log(q_in, q_out)            
    q_out_att  = parallel_transport(q_in, q_att, q_out_body)


    # plot_4d_coord(q_in_att, title='q_in_att')
    # plot_4d_coord(q_out_body, title='q_out_body')
    # plot_4d_coord(q_out_att, title='w_train_att')


    
    K, N = postProb.shape
    M = 4


    max_norm = 0.5
    A_vars = []
    constraints = []
    for k in range(K):
        A_vars.append(cp.Variable((M, M), symmetric=False))

        constraints += [A_vars[k].T + A_vars[k] << np.zeros((4, 4))]

        # constraints += [cp.norm(A_vars[k], 'fro') <= max_norm]



    for k in range(K):
        q_pred_att_k = A_vars[k] @ q_in_att.T
        if k == 0:
            q_pred_att  = cp.multiply(np.tile(postProb[k, :], (M, 1)), q_pred_att_k)
        else:
            q_pred_att += cp.multiply(np.tile(postProb[k, :], (M, 1)), q_pred_att_k)
    q_pred_att = q_pred_att.T

    
    objective = cp.norm(q_pred_att-q_out_att, 'fro')


    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=cp.MOSEK, verbose=True)


    A_res = np.zeros((K, M, M))
    for k in range(K):
        A_res[k, :, :] = A_vars[k].value
        print(A_vars[k].value)
        print(np.linalg.norm(A_vars[k].value, 'fro'))

    return A_res