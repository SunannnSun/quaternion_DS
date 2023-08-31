import numpy as np
import cvxpy as cp

from util.quat_tools import *

def canonical_quat(q):
    """
    Force all quaternions to have positive scalar part; necessary to ensure proper propagation in DS
    """
    if (q[-1] < 0):
        return -q
    else:
        return q


def optimize_single_system(q_train, w_train, q_att):
    N = len(w_train)
    M = 3

    A = cp.Variable((M, M), symmetric=True)

    constraints = []
    constraints += [A << 0]

    objective = 0

    for i in range(N):
        q_diff = (q_train[i] * q_att.inv()).as_quat()

        w_pred = A @ canonical_quat(q_diff)[0:3]

        objective +=cp.norm(w_pred- w_train[i], 2)**2

    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=cp.MOSEK, verbose=True)

    return A.value




def optimize_single_quat_system(q_train, w_train, q_att):

    q_att = canonical_quat(q_att.as_quat())

    N = len(w_train)
    M = 4

    A = cp.Variable((M, M), symmetric=True)

    constraints = []
    constraints += [A << 0]

    objective = 0

    for i in range(N):
        
        w_i = canonical_quat(R.from_rotvec(w_train[i]).as_quat()).reshape(-1, 1)

        q_i = canonical_quat(q_train[i].as_quat())


        q_diff = riem_log(q_att, q_i).reshape(-1, 1)

        w_pred = A @ q_diff


        objective += cp.norm(w_pred - riem_log(q_att, w_i), 2)**2

        # q_diff = (q_train[i] * q_att.inv()).as_quat()

        # w_pred = A @ canonical_quat(q_diff)[0:3]

        # objective +=cp.norm(w_pred- w_train[i], 2)**2

    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=cp.MOSEK, verbose=True)

    return A.value







'''

def optimize_lpv_ds_from_data(Data, attractor, ctr_type, gmm, *args):
    ctr_type = 2
    M = len(Data)
    N = len(Data[0])
    M = int(M / 2)

    # Positions and Velocity Trajectories
    Xi_ref = Data[0:M, :]
    Xi_ref_dot = Data[M:, :]

    # Define Optimization Variables
    K = len(gmm.Priors)
    A_c = np.zeros((K, M, M))
    b_c = np.zeros((M, K))

    # should have switch ctr_type here
    if ctr_type == 0:
        helper = 1  # blank for later use
        symm_constr = 0

    P = args[0]
    symm_constr = args[1]


    # Define Constraints and Assign Initial Values
    # 创建一个object 叫decision variable，which makes it
    A_vars = []
    b_vars = []
    Q_vars = []
    constrains = []

    for k in np.arange(K):
        if symm_constr:
            A_vars.append(cp.Variable((M, M), symmetric=True))
        else:
            A_vars.append(cp.Variable((M, M)))

        if k == 0:
            A_vars[k] = cp.Variable((M, M), symmetric=True)

        if ctr_type != 1:
            if M == 2:
                b_vars.append(cp.Variable((2, 1)))
            else:
                b_vars.append(cp.Variable((3, 1)))
            Q_vars.append(cp.Variable((M, M), symmetric=True))


        epi = 0.001
        epi = epi * -np.eye(M)
        # Define Constraints
        if ctr_type == 0:
            constrains += [A_vars[k].T + A_vars[k] << epi]
            # constrains += [b_vars[k].T == -A_vars[k] @ attractor]
            constrains += [b_vars[k] == -A_vars[k] @ attractor]

        elif ctr_type == 1:
            constrains += [A_vars[k].T @ P + P @ A_vars[k] << epi]
        else:
            constrains += [A_vars[k].T @ P + P @ A_vars[k] == Q_vars[k]]
            constrains += [Q_vars[k] << epi]
            constrains += [b_vars[k] == -A_vars[k] @ attractor]

    # Calculate our estimated velocities caused by each local behavior
    Xi_d_dot_c_raw = []

    for k in np.arange(K):
        h_K = np.repeat(h_k[k, :].reshape(1, len(h_k[0])), M, axis=0)
        if ctr_type == 1:
            f_k = A_vars[k] @ Xi_ref
        else:
            f_k = A_vars[k] @ Xi_ref
            f_k = f_k + b_vars[k]
        Xi_d_dot_c_raw.append(cp.multiply(h_K, f_k))

    # Sum each of the local behaviors to generate the overall behavior at
    # each point
    Xi_dot_error = np.zeros((M, N))
    for k in np.arange(K):
        Xi_dot_error = Xi_dot_error + Xi_d_dot_c_raw[k]
    Xi_dot_error = Xi_dot_error - Xi_ref_dot

    # Defining Objective Function depending on constraints
    if ctr_type == 0:
        Xi_dot_total_error = 0
        for n in np.arange(N):
            Xi_dot_total_error = Xi_dot_total_error + cp.norm(Xi_dot_error[:, n], 2)
        Objective = Xi_dot_total_error
    else:
        Objective = cp.norm(Xi_dot_error, 'fro')

    prob = cp.Problem(cp.Minimize(Objective), constrains)

    prob.solve(solver=cp.MOSEK, verbose=True)

    for k in np.arange(K):
        A_c[k] = A_vars[k].value
        if ctr_type != 1:
            b_c[:, k] = b_vars[k].value.reshape(-1)

    return A_c, b_c


'''