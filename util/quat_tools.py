import sys
import numpy as np
from scipy.spatial.transform import Rotation as R


"""
@note all operations below, of which the return is a vector, return 1-D array, 
      unless multiple inputs are given in vectorized operations
"""


def _process_x(x):
    """
    x can be either
        - a single R object
        - a list of R objects
    """

    if isinstance(x, list):
        x = list_to_arr(x)
    elif isinstance(x, R):
        x = x.as_quat()[np.newaxis, :]

    return x



def _process_xy(x, y):
    """
    Transform both x and y into (N by M) np.ndarray and normalize to ensure unit quaternions

    x and y can be either
        - 2 single R objects
        - 1 single R object + 1 list of R objects
        - 2 lists of R objects
    
    Except when both x and y are single R objects, always expand and cast the single R object to meet the same shape
    """
    
    M = 4
    if isinstance(x, R) and isinstance(y, list):
        N = len(y)
        x = np.tile(x.as_quat()[np.newaxis, :], (N,1))
        y = list_to_arr(y)

    elif isinstance(y, R) and isinstance(x, list):
        N = len(x)
        y = np.tile(y.as_quat()[np.newaxis, :], (N,1))
        x = list_to_arr(x)

    elif isinstance(x, list) and isinstance(y, list):
        x = list_to_arr(x)
        y = list_to_arr(y)
    
    elif isinstance(x, R) and isinstance(y, R):
        if x.as_quat().ndim == 1:
            x = x.as_quat()[np.newaxis, :]
        else:
            x = x.as_quat()
        if y.as_quat().ndim == 1:
            y = y.as_quat()[np.newaxis, :]
        else:
            y = y.as_quat()

    elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        if x.ndim == 1 and y.ndim == 1:
            x = x[np.newaxis, :]
            y = y[np.newaxis, :]
        M = x.shape[1]

    else:
        print("Invalid inputs in quaternion operation")
        sys.exit()

    x = x / np.tile(np.linalg.norm(x, axis=1, keepdims=True), (1,M))
    y = y / np.tile(np.linalg.norm(x, axis=1, keepdims=True), (1,M))
    

    return x,y




def unsigned_angle(x, y):
    """
    Vectorized operation

    @param x is always a 1D array
    @param y is either a 1D array or 2D array of N by M

    note: "If a is an N-D array and b is a 1-D array, it is a sum product over the last axis of a and b; i.e. sum(a[i,:] * b) "
    note: "/" divide operator equivalent to np.divide, performing element-wise division
    note:  np.dot, np.linalg.norm(keepdims=False) and the return angle are 1-D array
    """
    x, y = _process_xy(x, y)

    dotProduct = np.sum(x * y, axis=1)

    angle = np.arccos(np.clip(dotProduct, -1, 1))

    return angle





def riem_log(x, y):
    """
    Vectorized operation

    @param x is the point of tangency and is always a 1D array
    @param y is either a 1D array or 2D array of N by M


    @note special cases to take care of when x=y and angle(x, y) = pi
    @note IF further normalization needed after adding perturbation?

    - Scenario 1:
        When projecting q_train wrt q_att:
            x is a single R object
            y is a list of R objects
    
    - Scenario 2:
        When projecting each w_train wrt each q_train:
            x is a list of R objects
            y is a list of R objects
    
    - Scenario 3:
        When parallel_transport each projected w_train from respective q_train to q_att:
            x is a list of R objects
            y is a single R object

    - Scenario 4:
        When simulating forward, projecting q_curr wrt q_att:
            x is a single R object
            y is a single R object
    """


    x, y = _process_xy(x, y)
    y_copy = y.copy()

    N, M = x.shape

    angle = unsigned_angle(x, y) 

    y[angle == np.pi] += 0.001

    x_T_y = np.tile(np.sum(x * y, axis=1,keepdims=True), (1,M))
    
    x_T_y_x = x_T_y * x

    u = np.tile(angle[:, np.newaxis]/np.linalg.norm(y_copy-x_T_y_x, axis=1, keepdims=True), (1, M)) * (y_copy-x_T_y_x)

    u[angle == 0] = np.zeros([1, M]) 

    
    return u


def parallel_transport(x, y, v):
    """
    Vectorized operation
    
    parallel transport a vector u from space defined by x to a new space defined by y

    @param: x original tangent point, np.array()
    @param: y new tangent point, np.array
    @param v vector in tangent space (compatible with both 1-D and 2-D)

    """
    v = _process_x(v)
    log_xy = riem_log(x, y)
    log_yx = riem_log(y, x)
    d_xy = unsigned_angle(x, y)


    # a = np.sum(log_xy * v, axis=1) 
    u = v - (log_xy + log_yx) * np.tile(np.sum(log_xy * v, axis=1, keepdims=True) / np.power(d_xy,2)[:, np.newaxis], (1, 4))


    return u


def riem_exp(x, v):
    """
    The only useage of riem_exp so far is during simulation where x is a rotation object, v is a numpy array
    """

    x = _process_x(x)

    if v.ndim == 2:
        v = v[:, 0]

    v_norm = np.linalg.norm(v)

    y = x * np.cos(v_norm) + v / v_norm * np.sin(v_norm)

    return y


def riem_cov(q_mean, q_list):


    q_list_mean = riem_log(q_mean, q_list)
    scatter = q_list_mean.T @ q_list_mean

    cov = scatter/len(q_list)


    return cov




def canonical_quat(q):
    """
    Force all quaternions to have positive scalar part; necessary to ensure proper propagation in DS
    """
    if (q[-1] < 0):
        return -q
    else:
        return q
    


def list_to_arr(q_list):

    N = len(q_list)
    M = 4

    q_arr = np.zeros((N, M))

    for i in range(N):
        q_arr[i, :] = q_list[i].as_quat()

        # q_arr[i, :] = canonical_quat(q_list[i].as_quat())

    return q_arr