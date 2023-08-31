import numpy as np
from scipy.spatial.transform import Rotation as R



def unsigned_angle(x, y):
    """
    Vectorized operation

    @param x is always a 1D array
    @param y is either a 1D array or 2D array of N by M

    note: "If a is an N-D array and b is a 1-D array, it is a sum product over the last axis of a and b; i.e. sum(a[i,:] * b) "
    note: "/" divide operator equivalent to np.divide, performing element-wise division
    note:  np.dot, np.linalg.norm(keepdims=False) and the return angle are 1-D array
    """

    x = x / np.linalg.norm(x)
    if y.ndim == 1:
        y = y / np.linalg.norm(y)
    elif y.ndim == 2:
        y = y / np.linalg.norm(y, axis=1, keepdims=True)

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
    x = x / np.linalg.norm(x)
    if y.ndim == 1:
        y = y / np.linalg.norm(y)
    elif y.ndim == 2:
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
    
    angle = unsigned_angle(x, y) 

    if y.ndim == 1:
        tanDir = y - np.dot(y, x) * x 
        if angle == 0:
            return tanDir
        elif angle == np.pi:
            tanDir = y - np.dot(y+0.001, x) * x 

        v = angle * tanDir / np.linalg.norm(tanDir)

        
    elif y.ndim == 2:
        y[angle==np.pi] += 0.001

        tanDir = y - np.dot(y, x)[:, np.newaxis] * x 
        v = angle[:, np.newaxis] * tanDir / np.linalg.norm(tanDir, axis=1, keepdims=True)

        v[angle == 0] = tanDir[angle==0] 
    
    return v



def riem_exp(x, v):
    """
    x is the point of tangency
    """

    v_norm = np.linalg.norm(v)

    y = x * np.cos(v_norm) + v / v_norm * np.sin(v_norm)

    return y


def riem_cov(q_list, q_mean):
    M = 4

    q_mean = canonical_quat(q_mean.as_quat())
    scatter = np.zeros((M, M))
    N = len(q_list)
    for i in range(N):
        q_i = canonical_quat(q_list[i].as_quat())
        log_q = riem_log(q_mean, q_i).reshape(-1, 1)
        scatter  += log_q @ log_q.T
    
    cov = scatter/N

    return cov




def canonical_quat(q):
    """
    Force all quaternions to have positive scalar part; necessary to ensure proper propagation in DS
    """
    if (q[-1] < 0):
        return -q
    else:
        return q
    


def list_to_arr(self, q_list):

    N = len(q_list)
    M = 4

    q_arr = np.zeros((N, M))

    for i in range(N):
        q_arr[i, :] = canonical_quat(q_list[i].as_quat())

    return q_arr