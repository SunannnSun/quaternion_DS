import numpy as np
from scipy.spatial.transform import Rotation as R


def unsigned_angle(x, y):

    dotProduct = np.dot(y, x)
    cosAngle = dotProduct / (np.linalg.norm(x) * np.linalg.norm(y))
    angle = np.arccos(np.clip(cosAngle, -1, 1))

    return angle



def riem_log(x, y):
    """
    x is the point of tangency
    """

    angle = unsigned_angle(x, y).reshape(-1, 1) 
    tanDir = y - np.dot(y, x).reshape(-1, 1) * x 

    if y.ndim == 1:
        if np.linalg.norm(tanDir) == 0:
            return tanDir
        
        if angle == np.pi:
            angle = np.pi - 0.0001

    v = angle * tanDir / np.linalg.norm(tanDir, axis=1, keepdims=True)
    
    return v



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
    


class q_gmm:
    def __init__(self, *args_):

        self.q_list = args_[0]
        


    def cluster(self, dummy_arr): #dummy
        """
        Fill in the actual clustering algorithm and return the result assignment_arr
        """

        self.assignment_arr = dummy_arr



    def extract_param(self, q_list, *args_):
        N = len(q_list)
        M = 4

        if len(args_) == 1:
            assignment_arr = args_[0]
        else:
            assignment_arr = np.zeros((N, ), dtype=int)

        K = assignment_arr.max()+1

        Priors  = np.zeros((K, ))
        Mu      = np.zeros((K, M))
        Sigma   = np.zeros((K, M, M))
        q_normal_arr = np.zeros((K, ), dtype=object)

        for k in range(K):
            q_k = [q for index, q in enumerate(q_list) if assignment_arr[index]==k] 
            r_k = R.from_quat([canonical_quat(q.as_quat()) for q in q_k])
            q_k_mean = r_k.mean()
        

            Priors[k]       = len(q_k)/N
            Mu[k, :]        = canonical_quat(q_k_mean.as_quat())
            Sigma[k, :, :]  = riem_cov(q_list, q_k_mean)

            q_normal_arr[k] = q_normal(Mu[k, :], Sigma[k, :, :])
        

        self.Priors = Priors
        self.q_normal_arr = q_normal_arr

        param_dict ={
            "Priors": Priors,
            "Mu" : Mu,
            "Sigma": Sigma
        }

        return param_dict
    

    



class q_normal:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma



    
    def logProb(self, q_list):

        N = len(q_list)
        M = 4

        q_arr = list_to_arr(q_list)
        q_diff = riem_log(self.mu, q_arr)

        logProb = np.zeros((N, ))
        
        logProb += -1/2 * M * np.log(2 * np.pi)
        logProb += -1/2 * np.linalg.det(self.sigma)
        logProb += -1/2 * np.sum( q_diff.T * (self.sigma @ q_diff.T), axis= 0) #vectorize quadratic function


        return logProb

        
    




