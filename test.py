import numpy as np
from util.quat_tools import *



# a = np.array([1, 0, 0])

# # b = np.array([[1, 0, 0],[0 , -1, 0], [-1, 0, 0]])

# b = np.array([-1, 0, 0])



# y = riem_log(a, b)

# y =1 


# c = np.array([0, 2.221, 2.221])

# a = riem_exp(a, c)

"""
Verify the stability in Tangent Space
@note Given the attractor and initial point, denote the q_init in the tangent space

"""

# A = -1 * np.eye(4)
# dt = 0.5

# q_att = canonical_quat(R.from_euler('xyz', [52, 50, 30], degrees=True).as_quat())
# q_init= canonical_quat(R.identity().as_quat())

# q_test = [riem_log(q_att, q_init)[:, np.newaxis]]
# q_dot = []

# for i in range(10):

#     q_i = q_test[i]

#     q_dot_pred = A @ q_i  

#     d_q =q_dot_pred * dt

#     q_next = q_i + d_q

#     q_dot.append(q_dot_pred) 
#     q_test.append(q_next)



"""
Verify if dq in tangent space is equivalent to dq in quaternion space

@note you can scale the angular velocity in the tangent space by the time dt
"""

# A = -0.5 * np.eye(4)
# dt = 0.5

# # q_curr_q = canonical_quat(R.identity().as_quat())
# q_id_q = canonical_quat(R.identity().as_quat())
# q_curr_q = canonical_quat(R.from_euler('xyz', [10, 10,0], degrees=True).as_quat())
# q_att_q = canonical_quat(R.from_euler('xyz', [50, 0, 20], degrees=True).as_quat())


# q_curr_t = riem_log(q_att_q, q_curr_q)[:, np.newaxis]
# w = A @ q_curr_t  
# d_q_t = w * dt
# q_next_t = q_curr_t + d_q_t
# q_next_t = riem_exp(q_att_q, q_next_t)
# q_next_1 = R.from_quat(q_next_t).as_euler('xyz', degrees=True)


# w_new = parallel_transport(q_att_q, q_curr_q, w)
# d_q_q = w_new * dt
# q_next_2 = riem_exp(q_curr_q, d_q_q)
# q_next_2 = R.from_quat(q_next_2).as_euler('xyz', degrees=True)


# d_q_q_q = R.from_quat(riem_exp(q_curr_q, d_q_q)) * R.from_quat(q_curr_q).inv()
# q_next_3 = d_q_q_q * R.from_quat(q_curr_q)
# q_next_3 = q_next_3.as_euler('xyz', degrees=True)





"""
Verify parallel_transport in a 2-D unit sphere
"""

# a= -1
# dt = 0.5
# angle_att = np.pi/3
# angle =  np.pi/6

# q_att = np.array([np.cos(angle_att), np.sin(angle_att)])

# q_curr_q =  np.array([np.cos(angle), np.sin(angle)])
# q_curr_t = riem_log(q_att, q_curr_q)[:, np.newaxis]

# w =  a * q_curr_t 
# dq_t = w * dt

# q_next_t = q_curr_t + dq_t
# q_next_q = riem_exp(q_att, q_next_t)
# print(np.linalg.norm(q_next_q))

# dq_t_new = parallel_transport(q_att, q_curr_q, dq_t)
# q_next_q_new = riem_exp(q_curr_q, dq_t_new)
# print(np.linalg.norm(q_next_q_new))



