import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter

from . import plot_tools, optimize_tools, quat_tools
from .gmm import gmm as gmm_class
from .quat_tools import *
from .plot_tools import *



def _shift(q_list, index_list):
    """
    Note: 
        Averaging and shifting will occasionally flip the signs of all quaternions including initial points and attractors
        
        Do not output q_att_avg which could be unflipped when flip occurs. Use the last of the shifted quaternion list
        which is the common attractor for all trajectories
    """

    L = len(index_list)

    q_att      = [q_list[index_list[l][-1]]  for l in range(L)]
    q_att_quat = [q_att[l].as_quat() for l in range(L)]
    q_att_avg  = R.from_quat(q_att_quat).mean()
    # q_att_avg = quat_tools.quat_mean(q_att)           # NOT used

    q_shifted = []
    for l in range(L):
        q_diff     = q_att_avg * q_att[l].inv()
        q_shifted += [q_diff * q_list[index] for index in index_list[l]]

    return q_shifted, q_shifted[-1]



def _smooth(q_in, q_att, index_list, opt):
    """
    Note:
        The value of k in SLERP is a parameter that can be tunned according to the total number of points
    """

    q_new = []

    for l in range(len(index_list)):
        q_in_l    = [q_in[idx] for idx in index_list[l]]

        if opt == "savgol":
            q_in_att  = quat_tools.riem_log(q_att, q_in_l)

            q_new_att = savgol_filter(q_in_att, window_length=80, polyorder=2, axis=0, mode="nearest")

            q_new_arr = quat_tools.riem_exp(q_att, q_new_att)

            q_new     += [R.from_quat(q_new_arr[i, :]) for i in range(q_new_arr.shape[0])]
    
    
        elif opt == "slerp":
            k = 40
            t_list = [0.1*i for i in range(len(q_in_l))]
            
            idx_list  = np.linspace(0, len(q_in_l)-1, num=int(len(q_in_l)/k), endpoint=True, dtype=int)
            key_times = [t_list[i] for i in idx_list]
            key_rots  = R.from_quat([q_in_l[i].as_quat() for i in idx_list])
            
            slerp = Slerp(key_times, key_rots)

            idx_list  = np.linspace(0, len(q_in_l)-1, num=int(len(q_in_l)), endpoint=True, dtype=int)
            key_times = [t_list[i] for i in idx_list]

            q_interp = slerp(key_times)
            q_new    += [q_interp[i] for i in range(len(q_interp))]

    return q_new



def _filter(q_in, index_list):
    
    L = len(index_list)
    max_threshold = 0.01

    q_new = []
    q_out = []
    index_list_new = []

    for l in range(L):
        N = index_list[l].shape[0]

        threshold = np.linspace(max_threshold, 0, num=N, endpoint=True)

        q_new.append(q_in[index_list[l][0]])
        index_list_l = [index_list[l][0]]

        for i in np.arange(1, N):
            q_curr = q_new[-1]
            q_next = q_in[index_list[l][i]]
            dis    = q_next * q_curr.inv()
            if np.linalg.norm(dis.as_rotvec()) > threshold[i]:
                q_new.append(q_next)
                q_out.append(q_next)
                index_list_l.append(index_list[l][i])

        q_out.append(q_out[-1])
        index_list_new.append(np.array(index_list_l))

    return q_out, index_list_new



def pre_process(q_in_raw, index_list, opt="savgol"):

    q_in, q_att             = _shift(q_in_raw, index_list)
    # plot_tools.plot_demo(q_in, index_list, interp=True, title="q_shifted")
    
    q_in                    = _smooth(q_in, q_att, index_list, opt)
    # plot_tools.plot_demo(q_in, index_list, interp=True, title='q_smooth')

    q_out, index_list       = _filter(q_in, index_list)
    # plot_tools.plot_demo(q_in, index_list, interp=False, title='q_filter')
    

    q_init = [q_in[index_list[l][0]]  for l in range(len(index_list))]
    q_init = quat_tools.quat_mean(q_init)

    q_att = [q_in[index_list[l][-1]]  for l in range(len(index_list))]
    q_att = quat_tools.quat_mean(q_att)

    return q_in, q_out, q_init, q_att, index_list