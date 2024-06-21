import os, sys, json
import numpy as np
from scipy.spatial.transform import Rotation as R

from .util import optimize_tools, quat_tools
from .gmm_class import gmm_class



def write_json(data, path):
    with open(path, "w") as json_file:
        json.dump(data, json_file, indent=4)



def compute_ang_vel(q_k, q_kp1, dt=0.01):
    """ Compute angular velocity """

    # dq = q_k.inv() * q_kp1    # from q_k to q_kp1 in body frame
    dq = q_kp1 * q_k.inv()    # from q_k to q_kp1 in fixed frame

    dq = dq.as_rotvec() 
    w  = dq / dt
    
    return w



class quat_class:
    def __init__(self, q_in:list,  q_out:list, q_att:R, K_init:int) -> None:
        """
        Parameters:
        ----------
            q_in (list):            M-length List of Rotation objects for ORIENTATION INPUT

            q_out (list):           M-length List of Rotation objects for ORIENTATION OUTPUT

            q_att (Rotation):       Single Rotation object for ORIENTATION ATTRACTOR
            
            K_init:                 Inital number of GAUSSIAN COMPONENTS

            M:                      OBSERVATION size

            N:                      OBSERVATION dimenstion
        """

        # store parameters
        self.q_in  = q_in
        self.q_out = q_out
        self.q_att = q_att

        self.K_init = K_init
        self.M = len(q_in)
        self.N = 4

        # simulation parameters
        self.tol = 10E-3
        self.max_iter = 5000

        # define output path
        file_path           = os.path.dirname(os.path.realpath(__file__))  
        self.output_path    = os.path.join(os.path.dirname(file_path), 'output_ori.json')



    def _cluster(self):
        gmm = gmm_class(self.q_in, self.q_att, self.K_init)  

        self.gamma    = gmm.fit()  # K by M
        self.K        = gmm.K
        self.gmm      = gmm
        self.dual_gmm = gmm._dual_gmm()



    def _optimize(self):
        self.A_ori = optimize_tools.optimize_ori(self.q_in, self.q_out, self.q_att, self.gamma)



    def begin(self):
        self._cluster()
        self._optimize()
        self._logOut()



    def sim(self, q_init, dt, step_size):
        q_test = [q_init]
        gamma_test = []
        omega_test = []

        i = 0
        while np.linalg.norm((q_test[-1] * self.q_att.inv()).as_rotvec()) >= self.tol:
            if i > self.max_iter:
                print("Exceed max iteration")
                break
            
            q_in  = q_test[i]

            q_next, gamma, omega = self._step(q_in, dt, step_size)

            q_test.append(q_next)        
            gamma_test.append(gamma[:, 0])
            omega_test.append(omega)

            i += 1

        return  q_test, np.array(gamma_test), omega_test
        


    def _step(self, q_in, dt, step_size):
        """ Integrate forward by one time step """
        q_in = self._rectify(q_in)


        # read parameters
        A_ori = self.A_ori
        q_att = self.q_att
        K     = self.K
        gmm   = self.gmm

        # compute output
        q_diff  = quat_tools.riem_log(q_att, q_in)
        
        q_out_att = np.zeros((4, 1))
        gamma = gmm.logProb(q_in)   # gamma value 


        for k in range(K):
            q_out_att += gamma[k, 0] * A_ori[k] @ q_diff.T
            

        q_out_body = quat_tools.parallel_transport(q_att, q_in, q_out_att.T)
        q_out_q    = quat_tools.riem_exp(q_in, q_out_body) 
        q_out      = R.from_quat(q_out_q.reshape(4,))
        w_out      = compute_ang_vel(q_in, q_out, dt)   #angular velocity
        # q_next     = q_in * R.from_rotvec(w_out * step_size)   #compose in body frame
        q_next     = R.from_rotvec(w_out * step_size) * q_in  #compose in world frame

        return q_next, gamma, w_out
    


    def _rectify(self, q_in):
        
        """
        Rectify q_init if it lies on the unmodeled half of the quaternion space
        """
        dual_gmm    = self.dual_gmm
        gamma_dual  = dual_gmm.logProb(q_in).T

        index_of_largest = np.argmax(gamma_dual)

        if index_of_largest <= (dual_gmm.K/2 - 1):
            return q_in
        else:
            return R.from_quat(-q_in.as_quat())
        
    


    def _logOut(self): 

        Prior = self.gmm.Prior
        Mu    = self.gmm.Mu
        Mu_rollout = [q_mean.as_quat() for q_mean in Mu]
        Sigma = self.gmm.Sigma

        Mu_arr      = np.zeros((self.K, self.N)) 
        Sigma_arr   = np.zeros((self.K, self.N, self.N), dtype=np.float32)

        for k in range(self.K):
            Mu_arr[k, :] = Mu_rollout[k]
            Sigma_arr[k, :, :] = Sigma[k]

        json_output = {
            "name": "Quaternion-DS result",

            "K": self.K,
            "M": 4,
            "Prior": Prior,
            "Mu": Mu_arr.ravel().tolist(),
            "Sigma": Sigma_arr.ravel().tolist(),

            'A_ori': self.A_ori.ravel().tolist(),
            'att_ori': self.q_att.as_quat().ravel().tolist(),
            'q_init': self.q_in[0].as_quat().ravel().tolist(),
            "gripper_open": 0
        }

        write_json(json_output, self.output_path)