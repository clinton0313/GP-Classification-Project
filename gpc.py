#%%
import itertools
from django.conf import SettingsReference
import matplotlib.pyplot as plt
import numpy as np

from EllipticalSliceSampler import EllipticalSampler
from scipy.optimize import minimize
from typing import Callable, Sequence
#%%
class GPC():
    '''
    A Gaussian Process Classifier class.
    '''
    def __init__(self, kernel, hyperparameters, optimizer:str = "Nelder-Mead"):
        '''
        Args:
            kernel: A kernel function that takes three arguments: X, Y, hyperparameters. 
                hyperparameters should be a list of kernel parameters.
            hyperparameters: A sequence of hyperparmaeters that correspond to the kernel argument.
            optimizer[optional]: optimizer by scipy.optimize.minimize to fit the model. 
        '''
        self.kernel= None
        self.hyperparameters = None
        self._set_kernel(kernel, hyperparameters)

        self.X = None
        self.Y = None
        self.optimizer = optimizer

    @property
    def hyperparameters(self):
        return self._hyperparameters
    
    @property
    def kernel(self):
        return self._kernel

    def _check_is_fitted(self) -> None:
        assert self.X is not None, "There is no data loaded, please fit the model by calling the fit method."

    def _update_hyperparameters(self, hyperparameters) -> None:
        '''Updates hyperparameters of kernel or sets it if no kernel is defined'''
        if self._kernel_is_defined():
            assert len(hyperparameters) == len(self.hyperparameters), \
                f"Incorrect number of hyperparameters. Expected {len(self.hyperparameters)} instead got {len(hyperparameters)}"
        self._hyperparameters = hyperparameters

    def _set_kernel(self, kernel:Callable, hyperparameters:Sequence) -> None:
        '''Sets kernel with a list of hyperparameters'''
        self._kernel = kernel
        self._update_hyperparameters(hyperparameters)

    def _get_mu(self, X):
        '''Returns a mean vector of zeros'''
        return np.zeros(len(X))

    def _get_sigma(self, X:np.ndarray, hyperparameters:Sequence = []):
        '''Returns a variance covariance matrix using the specified kernel and hyperparameters. if no hyperparameters are
        specified then uses the class hyperparameters.'''

        if hyperparameters == []:
            hyperparameters = self.hyperparameters

        n = len(X)
        gram_matrix = np.array([np.zeros(n) for _ in range(n)])
        for i, j in itertools.product(range(n), range(n)):
            gram_matrix[i][j] = self.kernel(X[i], X[j], hyperparameters)
        return gram_matrix
    
    def _sigmoid(self, f):
        return 1/(1 + np.exp(-f))

    def _loglikelihood(self, Y, f) -> float:
        '''Returns the log likelihood for a binary classification model'''
        
        f = f.reshape(1, -1)
        Y = Y.reshape(1, -1)
        assert f.shape == Y.shape, f"f and Y are not the same shape got f: {f.shape} and Y: {Y.shape}"

        return np.sum([
            np.log(self._sigmoid(f_i)) if y == 1 
            else np.log(self._sigmoid(-f_i)) 
            for f_i, y in zip(f, Y)
            ])

    def sample_prior(self, X, num_samples:int) -> Sequence:
        '''Draw num_samples from the prior'''
        return np.random.multivariate_normal(self._get_mu(X), self._get_sigma(X), num_samples)
    
    def sample_posterior(self, X, Y, num_burnin:int = 100, num_samples:int = 300, **kwargs) -> Sequence:
        '''Draws num_samples from posterior and discards the first num_burnin samples.'''
        assert num_samples > num_burnin, f"Got {num_samples} but required to burn {num_burnin} samples"

        def log_likelihood(f):
            return self._loglikelihood(Y=Y, f=f)

        ess = EllipticalSampler(self._get_mu(X), self._get_sigma(X,**kwargs), log_likelihood)
        return ess.sample(num_samples, num_burnin)

    def posterior_mean(self, X, Y, **kwargs) -> Sequence:
        '''Returns posterior mean'''
        return np.mean(self.sample_posterior(X, Y, **kwargs), axis=0)

    def fit(self, X, y, maxiter:int = 100, **kwargs) -> None:
        '''Fits the model and finds the optimal hyperparameters'''
        self.X = X
        self.Y = y

        def nll(hyperparameters):
            return (- self._loglikelihood(Y=self.Y, f=self.posterior_mean(self.X, self.Y, hyperparameters=hyperparameters, **kwargs)))

        res = minimize(nll, self.hyperparameters, method=self.optimizer, maxiter=maxiter)
        self._update_hyperparameters(res.x)
    
    def predict(self, X) -> float:
        self._check_is_fitted()
        ...


#%%

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def gaussian_kernel(x, y, theta):
    return theta[0] * np.exp(-(np.dot((x-y), (x-y))/(2*theta[1])))

def linear_kernel(x, y, theta):
    return theta[0] * np.dot(x, y) + theta[1]

X = np.random.normal(0, 1, 10)
Y = np.random.randint(0, 2, 10)

gpc = GPC(kernel=gaussian_kernel, hyperparameters=[1,1])

gpc.sample_posterior(X, Y)
# %%
