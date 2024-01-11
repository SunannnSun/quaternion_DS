import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter

from util import plot_tools, optimize_tools, quat_tools
from util.gmm import gmm as gmm_class
from util.quat_tools import *
from util.plot_tools import *



def _smooth(q_in, q_att):
    """
    Option 1: Savgol Filter
    """
    q_in_att  = quat_tools.riem_log(q_att, q_in)

    q_new_att = savgol_filter(q_in_att, window_length=80, polyorder=2, axis=0, mode="nearest")

    q_new_arr = quat_tools.riem_exp(q_att, q_new_att)

    q_new     = [R.from_quat(q_new_arr[i, :]) for i in range(q_new_arr.shape[0])]
    

    """
    Option 2: SLERP Interpolation
    """

    # plot_tools.plot_4d_coord(q_smooth_arr, title='q_smooth_arr')

    return q_new


def _filter(q_in, q_att, index_list):

    gmm = gmm_class(q_in, q_att, index_list = index_list)
    label = gmm.begin()

    N = len(q_in)

    max_threshold = 0.03

    threshold = max_threshold * np.ones((N, ))
    threshold[label==gmm.K-1] = np.linspace(max_threshold, 0, num=np.sum(label==gmm.K-1), endpoint=True)

    # threshold = np.linspace(max_threshold, 0, num=N, endpoint=True)

    q_new  = [q_in[0]]
    q_out  = []
    for i in np.arange(1, N):
        q_curr = q_new[-1]
        q_next = q_in[i]
        dis    = q_next * q_curr.inv()
        if np.linalg.norm(dis.as_rotvec()) < threshold[i]:
            pass
        else:
            q_new.append(q_next)
            q_out.append(q_next)
    q_out.append(q_out[-1])

    index_list = [i for i in range(len(q_new))]

    return q_new, q_out, index_list



def pre_process(q_in_raw, q_att, index_list):

    q_in                    = _smooth(q_in_raw, q_att)
    q_in, q_out, index_list = _filter(q_in, q_att, index_list)


    """
    Replace the lines below with an averaging algorithm
    """
    
    q_init = q_in[0]
    q_att  = q_in[-1]



    return q_in, q_out, q_init, q_att, index_list