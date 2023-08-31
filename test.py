import numpy as np
from util.quat_tools import *



a = np.array([1, 0, 0])

b = np.array([[1, 0, 0],[0 , -1, 0], [-1, 0, 0]])

# b = np.array([-1, 0])



y = riem_log(a, b)

y =1 


c = np.array([0, 2.221, 2.221])

a = riem_exp(a, c)