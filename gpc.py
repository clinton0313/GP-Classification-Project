#%%
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.covariance import log_likelihood

from EllipticalSliceSampler import EllipticalSampler
from scipy.optimize import minimize
from typing import Callable, Sequence
#%%
class GPC():
    
    def __init__(self, optimizer:str = "Nelder-Mead"):
        self.kernel_parameters = []
        self.kernel_function = None
        self.X = None
        self.Y = None
        self.optimizer = optimizer

    def set_hyperparams(self, hyperparameters) -> None:
        assert len(hyperparameters) == len(self.kernel_parameters), \
            f"Incorrect number of hyperparameters. Expected {len(self.kernel_parameters)} instead got {len(hyperparameters)}"
        self.kernel_parameters = hyperparameters
    
    def set_kernel_function(self, kernel:Callable, hyperparameters:Sequence) -> None:
        '''Sets a kernel function that takes as arguments, X, Y, hyperparameters (as a list) and returns a real number.'''

        self.kernel_function = kernel
        self.kernel_parameters = hyperparameters

    def get_mu(self):
        assert self.X is not None, "There is no data loaded, please fit the model by calling the fit method."
        return np.zeros(len(self.X))

    def get_sigma(self, X:np.ndarray, hyperparameters:Sequence = []):
        '''Returns a variance covariance matrix using the specified kernel and hyperparameters. if no hyperparameters are
        specified then uses the class hyperparameters.'''
        
        assert self.kernel is not None, "No kernel is set"
        if hyperparameters == []:
            hyperparameters = self.kernel_parameters

        n = len(X)
        gram_matrix = np.array([np.zeros(n) for _ in range(n)])
        for i, j in itertools.product(range(n), range(n)):
            gram_matrix[i][j] = self.kernel(X[i], X[j], hyperparameters)
        return gram_matrix
    
    def loglikelihood(self, X, Y, f, hyperparameters, **kwargs) -> float:
        '''return the log likelihood'''
        f = np.random.multivariate_normal(self.get_mu, self.get_sigma(X, hyperparameters=hyperparameters))
        return - sum(np.log(1/(1 + np.exp( -f*Y))))
        #I dont think we need the X here unless we are cumputing the f every time, if the f is sampled from outside there should be no problem
        #If we sample inside of the ll we need to call the kernel here and create the MVnorm inside of the LL using X
        
    
    def sample_posterior(self, X, num_burnin:int = 100, num_samples:int = 100) -> Sequence:
        '''return n samples after an initial burn in of samples'''
        def ll(f):
            return self.loglikelihood(hyperparameters=self.kernel_parameters)

        ess = EllipticalSampler(self.get_mu(), self.get_sigma(X), ll)
        return ess.sample(num_samples, num_burnin)

    def posterior_mean(self, **kwargs) -> Sequence:
        '''Returns posterior mean'''
        return np.mean(self.sample_posterior(self.X, **kwargs), axis=0)

    def fit(self, X, y, max_iters:int = 100, **kwargs) -> None:
        '''sets hyperparameters to optimal'''
        self.X = X
        self.Y = y

        def nll(hyperparameters):
            return self.loglikelihood(self.X, self.Y, f=self.posterior_mean(**kwargs))

        res = minimize(nll, self.kernel_parameters, method=self.optimizer)
        self.set_hyperparams(res.x)
    
    def predict(self, X) -> float:
        ...



#%%

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def gaussian_kernel(x, y, theta):
    return theta[0] * np.exp(-(np.dot((x-y), (x-y))/(2*theta[1])))

def linear_kernel(x, y, theta):
    return theta[0] * np.dot(x, y) + theta[1]

X = np.random.normal(0, 1, 10)
Y = np.random.randint(1, 2, 10)

gpc = GPC()

gpc.set_kernel(linear_kernel, [100, 100])
gpc.X = X
gpc.Y = Y

#%%

fig, ax = plt.subplots()
ax.scatter(X, Y)

#%%