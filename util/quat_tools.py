import numpy as np
from scipy.spatial.transform import Rotation as R


"""
@note all operations below, of which the return is a vector, return 1-D array, 
      unless multiple inputs are given in vectorized operations
"""


def parallel_transport(x, y, v):
    """
    Vectorized operation
    
    parallel transport a vector u from space defined by x to a new space defined by y

    @param: x original tangent point, np.array()
    @param: y new tangent point, np.array
    @param v vector in tangent space (compatible with both 1-D and 2-D)

    """

    log_xy = riem_log(x, y)
    log_yx = riem_log(y, x)
    d_xy = unsigned_angle(x, y)

    if d_xy == 0:
        return v

    if v.ndim == 2 and v.shape[1]==1: # 2D vector
        u = v[:, 0] - 1/d_xy**2 * np.dot(log_xy, v) * (log_xy + log_yx)

    elif v.ndim == 2 and v.shape[0]>1: # 2D matrix
        u = v - 1/d_xy**2 * v @ log_xy[:, np.newaxis] * np.tile(log_xy + log_yx, (v.shape[0] ,1) )

    elif v.ndim == 1: # 1D vector
        u = v - 1/d_xy**2 * np.dot(log_xy, v) * (log_xy + log_yx)



    return u





def unsigned_angle(x, y):
    """
    Vectorized operation

    @param x is always a 1D array
    @param y is either a 1D array or 2D array of N by M

    note: "If a is an N-D array and b is a 1-D array, it is a sum product over the last axis of a and b; i.e. sum(a[i,:] * b) "
    note: "/" divide operator equivalent to np.divide, performing element-wise division
    note:  np.dot, np.linalg.norm(keepdims=False) and the return angle are 1-D array
    """
    if isinstance(x, R):
        x = canonical_quat(x.as_quat())
    if isinstance(y, R):
        y = canonical_quat(y.as_quat())

 
    if y.ndim == 1:
        y = y / np.linalg.norm(y)
    elif y.ndim == 2 and y.shape[1] > 1:
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
    else:
        y = (y / np.linalg.norm(y, axis=0, keepdims=False))[:, 0]


    if x.ndim == 2 and x.shape[1] > 1:
        dotProduct = np.sum(x * y, axis=1)
    else:
        dotProduct = np.dot(y, x) 


    angle = np.arccos(np.clip(dotProduct, -1, 1))

    return angle



def riem_log(x, y):
    """
    Vectorized operation

    @param x is the point of tangency and is always a 1D array
    @param y is either a 1D array or 2D array of N by M


    @note special cases to take care of when x=y and angle(x, y) = pi
    @note IF further normalization needed after adding perturbation?
    """
    if isinstance(x, R):
        x = canonical_quat(x.as_quat())
    if isinstance(y, R):
        y = canonical_quat(y.as_quat())

    if x.ndim == 2 and x.shape[-1] > 1:
        x = x / np.tile(np.linalg.norm(x, axis=1, keepdims=True), (1,4))
    else:
        x = x / np.linalg.norm(x)

    if y.ndim == 1:
        y = y / np.linalg.norm(y)
    elif y.ndim == 2 and y.shape[1] > 1:
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
    else:
        y = (y / np.linalg.norm(y, axis=0, keepdims=False))[:, 0]
    
    angle = unsigned_angle(x, y) 

    if y.ndim == 1:
        tanDir = y - np.dot(y, x) * x 
        if angle == 0:
            return tanDir
        elif angle == np.pi:
            tanDir = y - np.dot(y+0.001, x) * x 

        v = angle * tanDir / np.linalg.norm(tanDir)
    

    elif x.shape[0] == y.shape[0]:
        y[angle==np.pi] += 0.001

        tanDir = y - np.tile(np.sum(x * y, axis=1,keepdims=True),(1, 4)) * x
        v = np.tile(angle[:, np.newaxis],(1,4)) * tanDir / np.linalg.norm(tanDir, axis=1, keepdims=True)

        v[angle == 0] = tanDir[angle==0] 


    elif y.ndim == 2:
        y[angle==np.pi] += 0.001

        tanDir = y - np.dot(y, x)[:, np.newaxis] * x 
        v = angle[:, np.newaxis] * tanDir / np.linalg.norm(tanDir, axis=1, keepdims=True)

        v[angle == 0] = tanDir[angle==0] 
    
    return v



def riem_exp(x, v):
    """
    x is the point of tangency

    @note x is always the q_att as the point of tangency, hence 1-D
    @note v is 2-D array of single vector

    """
    if v.ndim == 2:
        v = v[:, 0]

    v_norm = np.linalg.norm(v)

    y = x * np.cos(v_norm) + v / v_norm * np.sin(v_norm)

    return y


def riem_cov(q_list, q_mean):

    q_arr = list_to_arr(q_list)
    q_mean = canonical_quat(q_mean.as_quat())

    M = 4
    

    scatter = np.zeros((M, M))
    N = len(q_list)
    for i in range(N):
        q_i = q_arr[i, :]
        log_q = riem_log(q_mean, q_i)
        # scatter  += log_q[:, np.newaxis] @ log_q[np.newaxis, :]
        scatter += np.outer(log_q, log_q)

        # print(np.linalg.eigvals(scatter))

    cov = scatter/N

    # print(np.linalg.eigvals(cov))

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