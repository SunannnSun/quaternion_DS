import numpy as np
import cvxpy as cp

from util.quat_tools import *




def optimize_single_rotvec_system(q_train, w_train, q_att):
    """
    Perform optimization in rotation vector space (non-Euclidean, non unit-sphere)
    """

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
    """

    Perform optimization in the tangent space of the quaternion manifold

    @algorithm:
        # 1. Given the q_att as the point of tangency, project the current q_i onto the tangent space via Log map
        # 2. Compute the predicted angular velocity in the tangent space
        # 3. Parallel transport the trained angular velocity from attractor to the current q
        # 4. Compute the objective by measuring the Euclidean distance between w_pred and w_diff from training
    
        
    @note: verify later if it makes difference between using the next point or the angular velocity
    @note: attempt to vectorize the operation in the very late stage...
    """
    q_train_q = list_to_arr(q_train)


    q_att_q = canonical_quat(q_att.as_quat())
    q_id_q = canonical_quat(R.identity().as_quat())
    q_train_att = riem_log(q_att_q, q_train_q)

    N = len(w_train)
    M = 4

    A = cp.Variable((M, M), symmetric=True)

    constraints = [A << 0]
    """
    objective = 0
    for i in range(N):
        q_curr_q = canonical_quat(q_train[i].as_quat())
        q_curr_t = riem_log(q_att_q, q_curr_q)
        w_pred_att = A @ q_curr_t[:, np.newaxis]

        w_curr_q = canonical_quat(R.from_rotvec(w_train[i]).as_quat())
        w_curr_t = riem_log(q_id_q, w_curr_q)
        w_curr_att = parallel_transport(q_id_q, q_att_q, w_curr_t)

        objective += cp.norm(w_pred_att - w_curr_att[:, np.newaxis], 2)**2
    """

    w_pred_att =  A @ q_train_att.T
    w_curr_att = np.zeros((M, N))
    for i in range(N):
        w_curr_q = canonical_quat(R.from_rotvec(w_train[i]).as_quat())
        w_curr_t = riem_log(q_id_q, w_curr_q)
        w_curr_att[:, i] = parallel_transport(q_id_q, q_att_q, w_curr_t)        
    objective = cp.norm(w_pred_att-w_curr_att, 'fro')
    # objective = cp.sum(cp.norm2(w_pred_att - w_curr_att, axis=0))

    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=cp.MOSEK, verbose=False)

    print(A.value)

    return A.value




def optimize_double_quat_system(q_train, w_train, q_att, postProb):
    """
    need to vectorize later

    later change from q_next to the actual w angular velocity

    Require additional information on clustering result
    """

    q_att_q = canonical_quat(q_att.as_quat())
    q_id_q = canonical_quat(R.identity().as_quat())
    q_train_q = list_to_arr(q_train)
    q_train_att = riem_log(q_att_q, q_train_q)


    K = postProb.shape[0]
    N = len(w_train)
    M = 4

    A_res = np.zeros((K, M, M))

    A0 = cp.Variable((M, M), symmetric=True)
    A1 = cp.Variable((M, M), symmetric=True)
    A_vars = [A0, A1]
    
    # objective = 0
    constraints = [A0 << 0, A1<<0]
 
    fK = []
    for k in np.arange(K):
        fk = A_vars[k] @ q_train_att.T
        hk = np.tile(postProb[k, :].T, (M, 1))
        fK.append(cp.multiply(hk ,fk))
    w_pred_att = fK[0] + fK[1]

    # hk0 = np.tile(postProb[0, :].T, (M, 1))
    # hk1 = np.tile(postProb[1, :].T, (M, 1))

    # w_pred_att = cp.multiply(hk0, A_vars[0] @ q_train_q.T) + cp.multiply(hk1,  A_vars[1] @ q_train_q.T)

    w_pred_att = cp.multiply(hk ,fk)


    w_curr_att = np.zeros((M, N))
    for i in range(N):
        w_curr_q = canonical_quat(R.from_rotvec(w_train[i]).as_quat())
        w_curr_t = riem_log(q_id_q, w_curr_q)
        w_curr_att[:, i] = parallel_transport(q_id_q, q_att_q, w_curr_t)
    

    objective = cp.sum(cp.norm2(w_pred_att - w_curr_att, axis=0))
    # objective = cp.norm(w_pred_att-w_curr_att, 'fro')


    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=cp.MOSEK, verbose=False)

    for k in range(K):
        A_res[k, :, :] = A_vars[k].value
        print(A_vars[k].value)

    return A_res




def optimize_quat_system(q_train, w_train, q_att, postProb):
    """
    """

    q_att_q = canonical_quat(q_att.as_quat())
    q_train_q = list_to_arr(q_train)


    K = postProb.shape[0]
    N = len(w_train)
    M = 4


    A_vars = []
    constraints = []
    for k in range(K):
        A_vars.append(cp.Variable((M, M), symmetric=True))
        constraints += [A_vars[k] << 0]


    FK = []
    for k in np.arange(K):
        fk = A_vars[k] @ q_train_q.T
        hk = np.tile(postProb[k, :].T, (M, 1))
        FK.append(cp.multiply(hk ,fk))
    
    w_pred = 0
    for k in np.arange(K):
        w_pred += FK[k]




    w_act = np.zeros((M, N))
    for i in range(N):
        q_curr   = q_train[i]
        q_curr_q = q_train_q[i]

        w_curr = R.from_rotvec(w_train[i])
        q_next = w_curr * q_curr
        q_next_q = canonical_quat(q_next.as_quat())

        q_next_t = riem_log(q_curr_q, q_next_q)
        w_act[:, i] = parallel_transport(q_curr_q, q_att_q, q_next_t)
    
        

    objective = cp.sum(cp.norm2(w_pred - w_act, axis=0))

    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=cp.MOSEK, verbose=True)


    A_res = np.zeros((K, M, M))
    for k in range(K):
        A_res[k, :, :] = A_vars[k].value
        print(A_vars[k].value)

    return A_res



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
    # 创建一个object 叫decision variable, which makes it
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