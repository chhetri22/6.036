import random
import time

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import math


def k_means(data, k, eps=1e-4, mu=None):
    """ Run the k-means algorithm
    data - an nxd ndarray
    k - number of clusters to fit
    eps - stopping criterion tolerance
    mu - an optional KxD ndarray containing initial centroids

    returns: a tuple containing
        mu - a KxD ndarray containing the learned means
        cluster_assignments - an N-vector of each point's cluster index
    """
    n, d = data.shape #here n is 250 and d is 2; k ranges from 1-5
    if mu is None:
        # randomly choose k points as initial centroids
        mu = data[random.sample(range(data.shape[0]), k)]
        mu = mu.tolist()
    
    mu_last = data[random.sample(range(data.shape[0]), k)].tolist()
    
    
    while not converged(mu, mu_last):
        mu_last = mu
        clusters_assignment = assign_clusters(data, mu)
        mu = new_centers(mu, clusters_assignment)
        
    
    return (np.array(mu), clusters_assignment['N-vector'])


def converged(mu, mu_last):

    for i in range(len(mu)):
        if set(mu[i]) != set(mu_last[i]):
            return False
    return True


def assign_clusters(data, mu):
    cluster_dict = {}
    cluster_dict['N-vector'] = []
    for x,y in data:
        dist = []
        mu_tuple = []
        for c_x,c_y in mu:
            mu_tuple.append((c_x,c_y))
            dist.append(np.linalg.norm([x-c_x,y-c_y])) #calculates eucleaden dist between the points
        index_min = dist.index(min(dist))
        
        cluster_dict['N-vector'].append(index_min)
        
        closest_centroid = mu_tuple[index_min]
        if closest_centroid not in cluster_dict:
            cluster_dict[closest_centroid] = []
            
        cluster_dict[closest_centroid].append((x,y))
        
    return cluster_dict


def new_centers(mu,clusters_assignment):
    new_centroids = []
    for centroid in clusters_assignment:
        if centroid != 'N-vector':
            new_centroids.append(mean(clusters_assignment[centroid]))
        
    return new_centroids

def mean(vals):
    sum_x, sum_y, count = 0,0,0

    for x,y in vals:
        sum_x += x
        sum_y += y
        count += 1
        
    return (float(sum_x)/count, float(sum_y)/count)


class MixtureModel(object):
    def __init__(self, k):
        self.k = k
        self.params = {
            'pi': np.random.dirichlet([1]*k),
        }

    def __getattr__(self, attr):
        if attr not in self.params:
            raise AttributeError()
        return self.params[attr]

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)


    
    
    def e_step(self, data):
        """ Performs the E-step of the EM algorithm
        data - an NxD pandas DataFrame

        returns a tuple containing
            (float) the expected log-likelihood
            (NxK ndarray) the posterior probability of the latent variables
        """
        raise NotImplementedError()

    def m_step(self, data, p_z):
        """ Performs the M-step of the EM algorithm
        data - an NxD pandas DataFrame
        p_z - an NxK numpy ndarray containing posterior probabilities

        returns a dictionary containing the new parameter values
        """
        raise NotImplementedError()

    def fit(self, data, eps=1e-4, verbose=True, max_iters=100):
        """ Fits the model to data
        data - an NxD pandas DataFrame
        eps - the tolerance for the stopping criterion
        verbose - whether to print ll every iter
        max_iters - maximum number of iterations before giving up

        returns a boolean indicating whether fitting succeeded

        if fit was successful, sets the following properties on the Model object:
          n_train - the number of data points provided
          max_ll - the maximized log-likelihood
        """
        last_ll = np.finfo(float).min
        start_t = last_t = time.time()
        i = 0
        while True:
            i += 1
            if i > max_iters:
                return False
            ll, p_z = self.e_step(data)
            new_params = self.m_step(data, p_z)
            self.params.update(new_params)
            if verbose:
                dt = time.time() - last_t
                last_t += dt
                print('iter %s: ll = %.5f  (%.2f s)' % (i, ll, dt))
                last_ts = time.time()
            if abs((ll - last_ll) / ll) < eps:
                break
            last_ll = ll

        setattr(self, 'n_train', len(data))
        setattr(self, 'max_ll', ll)
        self.params.update({'p_z': p_z})

        print('max ll = %.5f  (%.2f min, %d iters)' %
              (ll, (time.time() - start_t) / 60, i))

        return True


class GMM(MixtureModel):
    def __init__(self, k, d):
        super(GMM, self).__init__(k)
        self.params['mu'] = np.random.randn(k, d)

    def e_step(self, data):
        n, d = data.shape
        
        
        # weights
        w = self.params['pi']

        #initialize final posterior prob matrix
        post_Mat = np.zeros((n,self.k))
        
        # loop through k can calculate the multivariate normal value
        for i in range(self.k):
            post_Mat[:, i] = w[i] * multivariate_normal.pdf(data, self.mu[i], self.sigsq[i])
            
        denominator = np.sum(post_Mat, axis = 1)
        post_Mat = (post_Mat.T/denominator).T

        l_likelihood = np.sum(np.log(denominator))
        
        return (l_likelihood, post_Mat)

    def m_step(self, data, pz_x):
        n, d = data.shape
        N_k = np.sum(pz_x, axis = 0)
        new_pi = N_k/n
        
        new_mu = np.zeros(self.mu.shape)
        new_sigsq = np.zeros(self.sigsq.shape)
    
    
        for i in range(self.k):
            ## means
            for j in range(n):
                new_mu[i] += (pz_x[j][i]* data[j])/N_k[i] 
           
            ## covariances
            for j in range(n):
                new_sigsq[i] += (pz_x[j][i]* (np.linalg.norm(data[j]-new_mu[i])**2))/(2*N_k[i])            


        return {
            'pi': new_pi,
            'mu': new_mu,
            'sigsq': new_sigsq,
        }

    def fit(self, data, *args, **kwargs):
        self.params['sigsq'] = np.asarray([np.mean(data.var(0))] * self.k)
        return super(GMM, self).fit(data, *args, **kwargs)


class CMM(MixtureModel):
    def __init__(self, k, ds):
        """d is a list containing the number of categories for each feature"""
        super(CMM, self).__init__(k)
        self.params['alpha'] = [np.random.dirichlet([1]*d, size=k) for d in ds]

    def e_step(self, data):
        n, d_1 = data.shape #n here is 12
                   
        #initialize final prob matrix
        prob_Mat = np.zeros((n,self.k)) #need n by k
        
        for d in range(d_1):
            col_data = data.ix[:,d]
            
            dummies = pd.get_dummies(col_data) #n by nd
            
            alpha_t = self.alpha[d].T #nd by k

            dummies_dot_alpha_t =  dummies.dot(alpha_t)
            dummies_dot_alpha_t[dummies_dot_alpha_t == 0] = 1
            prob_Mat += np.log(dummies_dot_alpha_t) #n by k

        log_argument = np.tile(self.pi,(n,1))
        log_argument[log_argument == 0] = 1
        pi_s = np.log(log_argument)
        
        temp = prob_Mat + pi_s
        
        prob_Mat = np.exp(temp)
        
        Mat_row_sum = prob_Mat.sum(axis=1)
        
        
        prob_Mat = prob_Mat / Mat_row_sum[:,None]
        
        l_likelihood_h = np.multiply(prob_Mat, temp)
        
        l_likelihood = np.sum(l_likelihood_h)
        
        
        return (l_likelihood, prob_Mat)


    def m_step(self, data, p_z):
        
        n, D = data.shape
        new_pi = np.sum(p_z, axis = 0) / n
        new_alpha = self.alpha
        
        
        for d in range(D):
            
            col_data = data.as_matrix()[:,d]
            dummies = pd.get_dummies(col_data).as_matrix()
            
            new_p_z = p_z.T

            new_alpha[d] = np.dot(new_p_z, dummies)
            new_alpha_sum = new_alpha[d].sum(axis=1)

            new_alpha[d] = new_alpha[d] / new_alpha_sum[:,None]


        return {
            'pi': new_pi,
            'alpha': new_alpha,
        }

    @property
    def bic(self):
        
        Sum = 0
        for d in range(len(self.alpha)):
            Sum += self.alpha[d].shape[1] * self.k
            
        result = self.max_ll - (.5 * Sum ) * math.log(self.n_train)
        #self.params['bic'] = result
        return result
