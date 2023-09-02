import numpy as np
from util.quat_tools import *
from util.gmm import gmm as gmm_class
from util.normal import normal as normal_class
from scipy.stats import multivariate_normal
from util import plot_tools, optimize_tools
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


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




"""
Test the redundancy in unit sphere and collinearity in covariance matrix

The problem is:
    although points in the tangent space is embeded in the ambient space, because all points resides in the same 
    hyperplane, (e.g. 1-D line in S1, 2-D plane in S2), one dimension is redundant and results in linear dependency


PCA redution: 
    we only require statistical model to provide an weighting function; hence wouldnt affect the learning
    of A matrix

Parallel transport:
    to the identity where one dimension for every point is zero, and can be eliminated?
    Define the identity 

Degenerate case of Normal:
    when cov is not full rank, allowing singularity 


Quick way to work around: <- THIS WORKS
    Define a scipy normal distribuion class of zero mean and the singular covariance
    allowing singularity, and input the log(q)
"""


# N = 11
# M = 2

# dq = np.pi/10
# angle =  0


# q_list = []

# for i in range(N):
#     q_next =  np.array([np.cos(angle+dq*i), np.sin(angle+dq*i)])
#     q_list.append(q_next)


# q_mean = np.array([0, 1])
# scatter = np.zeros((M, M))

# for i in range(N):
#     q_i = q_list[i]
#     log_q = riem_log(q_mean, q_i)
#     scatter  += log_q[:, np.newaxis] @ log_q[np.newaxis, :]

# cov = scatter/N


# rv = multivariate_normal(mean=np.zeros((M, )), cov=cov, allow_singular=True)
# log_q = [riem_log(q_mean, q) for q in q_list]

# print(rv.pdf(log_q))




"""
test if rv outputs zero postProb for quaternions
"""

# cov=np.array([[[ 0.0198658 ,  0.00406713, -0.00222208, -0.0050891 ],
#         [ 0.00406713,  0.00641748, -0.00350619, -0.00115964],
#         [-0.00222208, -0.00350619,  0.00191561,  0.00063357],
#         [-0.0050891 , -0.00115964,  0.00063357,  0.00130618]],

#        [[ 0.00027568, -0.00181571,  0.00099201,  0.00051415],
#         [-0.00181571,  0.012141  , -0.00663324, -0.00345307],
#         [ 0.00099201, -0.00663324,  0.00362407,  0.00188659],
#         [ 0.00051415, -0.00345307,  0.00188659,  0.00098334]]])


cov = np.array([[ 0.0198658 ,  0.00406713, -0.00222208, -0.0050891 ],
       [ 0.00406713,  0.00641748, -0.00350619, -0.00115964],
       [-0.00222208, -0.00350619,  0.00191561,  0.00063357],
       [-0.0050891 , -0.00115964,  0.00063357,  0.00130618]])

pca = PCA(n_components=3)


reduced_cov = pca.fit_transform(cov)



mean = np.zeros((4, 1))

normal_k = multivariate_normal(mean=np.zeros((4, )), cov=cov, allow_singular=True)


# q_mean =np.array([[ 0.24419147,  0.01574261, -0.00860098,  0.9695611 ],
#        [ 0.46123752,  0.2381517 , -0.13011426,  0.84475677]])

q_mean = np.array([ 0.24419147,  0.01574261, -0.00860098,  0.9695611 ])

q_1 = R.identity() 

q_1_log = riem_log(q_mean, canonical_quat(q_1.as_quat()))

q_2_q = np.array([ 9.06048037e-03,  9.19569112e-04, -8.17894335e-04,  9.99958196e-01])
q_2_log = riem_log(q_mean, q_mean)


prob = normal_k.pdf(q_1_log)


print(normal_k.pdf(q_2_log))

print(normal_k.pdf(np.array([0, 0, 0, 0.0001])))


q3 = np.array([-0.23040712, -0.01451531,  0.00761195,  0.05833303])
a = normal_k.logpdf(q3)
aa = multivariate_normal.logpdf(q3, mean=np.zeros((4, )), cov= normal_k.cov)



cc= 1
