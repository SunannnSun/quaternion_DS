import numpy as np
from scipy.spatial.transform import Rotation as R
from .quat_tools import *
# from .normal import normal as normal_class
from scipy.stats import multivariate_normal



class gmm:
    def __init__(self, *args_):

        self.q_list = args_[0]
        


    def cluster(self, dummy_arr): #dummy
        """
        Fill in the actual clustering algorithm and return the result assignment_arr
        """

        self.assignment_arr = dummy_arr



    def return_norma_class(self, q_list, *args_):
        """
        Return normal_class: 
            Assuming one distribution if only q_list is given;
            Output a list of normal_class if assignment_arr is provided too;
            Each normal class is defined by Mu, Sigma, (Prior is optional);
            Store the result normal class and Prior in gmm class

        @note verify later if the R.mean() yields the same results as manually computing the Krechet mean of quaternion
        """


        self.N = len(q_list)
        M = 4

        if len(args_) == 1:
            assignment_arr = args_[0]
        else:
            assignment_arr = np.zeros((self.N, ), dtype=int)

        self.K = assignment_arr.max()+1

        Prior  = np.zeros((self.K, ))
        Mu      = np.zeros((self.K, M), dtype=np.float32)
        Sigma   = np.zeros((self.K, M, M), dtype=np.float32)

        q_normal_list = []  # easier for list comprehension later

        for k in range(self.K):
            q_k = [q for index, q in enumerate(q_list) if assignment_arr[index]==k] 
            r_k = R.from_quat([canonical_quat(q.as_quat()) for q in q_k])
            q_k_mean = r_k.mean()
        

            Prior[k]       = len(q_k)/self.N
            Mu[k, :]        = canonical_quat(q_k_mean.as_quat())
            Sigma[k, :, :]  = riem_cov(q_k, q_k_mean) + 10**(-6) * np.eye(4)

            print(np.linalg.eigvals(Sigma[k, :, :]))

            q_normal_list.append(
                {
                    "prior" : Prior[k],
                    "mu"    : Mu[k, :],
                    "rv"    : multivariate_normal(np.zeros((M, )), Sigma[k, :, :], allow_singular=True)
                }
            )

        self.Prior = Prior
        self.q_normal_list = q_normal_list
        self.Sigma = Sigma


        return q_normal_list
    

    def prob(self, q_list):

        """
        @param q_list could be one quaternion or a list of quaternion

        @param log_q_k represents the logarithmic map of q_list w.r.t. mu_k
        @param k (optional) could be the number of specfic component
        """

        if isinstance(q_list, np.ndarray):
            prob = np.zeros((self.K, 1))

        else:
            prob = np.zeros((self.K, self.N))

        for k in range(self.K):
            
            _, mu_k, normal_k = tuple(self.q_normal_list[k].values())

            if isinstance(q_list, np.ndarray):
                log_q_k = riem_log(mu_k, q_list).astype(np.float16)

                                   
            else:
                # log_q_k = riem_log(mu_k, list_to_arr(q_list))
                log_q_k = [riem_log(mu_k, canonical_quat(q.as_quat())) for q in q_list]

            prob[k, :] = normal_k.pdf(log_q_k)
            
            a = normal_k.logpdf(log_q_k)
            Mean = np.zeros((4, ))
            Cov = self.Sigma[k, :, :].astype(np.float16)
            aa = multivariate_normal.logpdf(log_q_k, mean=Mean, cov= Cov, allow_singular=True)
            aaa= multivariate_normal.logpdf(log_q_k, mean=Mean, cov= normal_k.cov, allow_singular=True)


        return prob

            


    def postProb(self, q_list):

        """
        vectorized operation

        @ q_list: q_list could either be a list of roataion object, or a single quaternion array

        @param postProb is a K by N array contains the posterior probability of each q w.r.t each k_th component

        """
        if isinstance(q_list, np.ndarray):
            Prior = self.Prior
            prob  = self.prob(q_list)

            postProb     = Prior[:, np.newaxis] * prob

            postProb_sum = np.sum(postProb, axis=0, keepdims=True)

            postProb     = np.divide(postProb, np.tile(postProb_sum, (self.K, 1)))

        
        else:
            Prior = self.Prior
            prob  = self.prob(q_list)

            postProb     = np.tile(Prior[:, np.newaxis], (1, self.N)) * prob

            postProb_sum = np.sum(postProb, axis=0, keepdims=True)

            postProb     = np.divide(postProb, np.tile(postProb_sum, (self.K, 1)))


        return postProb







    def logProb(self, q_list):
        if isinstance(q_list, np.ndarray):
            logProb = np.zeros((self.K, 1))

        else:
            logProb = np.zeros((self.K, self.N))

        for k in range(self.K):
            
            _, mu_k, normal_k = tuple(self.q_normal_list[k].values())

            if isinstance(q_list, np.ndarray):
                log_q_k = riem_log(mu_k, q_list).astype(np.float16)

                                   
            else:
                # log_q_k = riem_log(mu_k, list_to_arr(q_list))
                log_q_k = [riem_log(mu_k, canonical_quat(q.as_quat())) for q in q_list]

            logProb[k, :] = normal_k.logpdf(log_q_k)
            
            # a = normal_k.logpdf(log_q_k)
            # Mean = np.zeros((4, ))
            # Cov = self.Sigma[k, :, :].astype(np.float16)
            # aa = multivariate_normal.logpdf(log_q_k, mean=Mean, cov= Cov, allow_singular=True)
            # aaa= multivariate_normal.logpdf(log_q_k, mean=Mean, cov= normal_k.cov, allow_singular=True)


        return logProb
    

    def postLogProb(self, q_list):
        if isinstance(q_list, np.ndarray):
            logPrior = np.log(self.Prior)
            logProb  = self.logProb(q_list)

            postLogProb     = logPrior[:, np.newaxis]  + logProb

            maxPostLogProb = np.max(postLogProb)
            expProb =  np.exp(postLogProb - maxPostLogProb)
            
            postProb = expProb / np.sum(expProb)



        
        else:
            logPrior = np.log(self.Prior)
            K = logPrior.shape[0]
            logProb  = self.logProb(q_list)

            postLogProb  = np.tile(logPrior[:, np.newaxis], (1, len(q_list))) + logProb

            maxPostLogProb = np.max(postLogProb, axis=0, keepdims=True)

            expProb = np.exp(postLogProb - np.tile(maxPostLogProb, (K, 1)))

            postProb = expProb / np.sum(expProb, axis = 0, keepdims=True)

            
            # prob  = self.prob(q_list)
            # postProb     = np.tile(Prior[:, np.newaxis], (1, self.N)) * prob

            # postProb_sum = np.sum(postProb, axis=0, keepdims=True)

            # postProb     = np.divide(postProb, np.tile(postProb_sum, (self.K, 1)))


        return postProb





        





    


    




        
    




