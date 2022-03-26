#%%
import itertools
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
        self.kernel= kernel
        self.hyperparameters = hyperparameters

        self.X = None
        self.Y = None
        self.optimizer = optimizer

    def _check_is_fitted(self) -> None:
        assert self.X is not None, "There is no data loaded, please fit the model by calling the fit method."

    def _update_hyperparameters(self, hyperparameters) -> None:
        '''Updates hyperparameters of kernel'''
        assert len(hyperparameters) == len(self.hyperparameters), \
            f"Incorrect number of hyperparameters. Expected {len(self.hyperparameters)} instead got {len(hyperparameters)}"
        self.hyperparameters = hyperparameters

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
            for f_i, y in zip(f.squeeze().tolist(), Y.squeeze().tolist())
            ])

    def sample_prior(self, X, num_samples:int) -> Sequence:
        '''Draw num_samples from the prior'''
        return np.random.multivariate_normal(self._get_mu(X), self._get_sigma(X), num_samples)
    
    def sample_posterior(self, X, Y, num_burnin:int = 100, num_samples:int = 300, verbose=0, **kwargs) -> Sequence:
        '''Draws num_samples from posterior and discards the first num_burnin samples.'''
        assert num_samples > num_burnin, f"Got {num_samples} but required to burn {num_burnin} samples"

        def log_likelihood(f):
            if len(Y) < len(f): f = f[:len(Y)]
            return self._loglikelihood(Y=Y, f=f)

        ess = EllipticalSampler(self._get_mu(X), self._get_sigma(X,**kwargs), log_likelihood)
        return ess.sample(num_samples, num_burnin, verbose=verbose)

    def posterior_mean(self, X, Y, verbose=0, **kwargs) -> Sequence:
        '''Returns posterior mean'''
        return np.mean(self.sample_posterior(X, Y, verbose=verbose, **kwargs), axis=0)

    def fit(self, X, y, maxiter:int = 100, tol:float = 0.1, verbose=0, **kwargs) -> None:
        '''
        Fits the model and finds the optimal hyperparameters
        
        Args:
            maxiter: max iterations for the optimizer
            tol: threshold change in the log likelihood for convergence
            verbose: 1 shows the hyperparameters as it updates, 2 also shows progress bar for sampler.
        '''
        self.X = X
        self.Y = y

        def nll(hyperparameters):
            return (- self._loglikelihood(Y=self.Y, f=self.posterior_mean(self.X, self.Y, hyperparameters=hyperparameters, **kwargs)))
        res = minimize(
            nll, 
            self.hyperparameters, 
            method=self.optimizer, 
            options={"maxiter":maxiter, "fatol":tol}, 
            callback=lambda x: print(x) if verbose >=1 else None)
        self._update_hyperparameters(res.x)
    
    def predict(self, pred_X, verbose=0, **kwargs) -> np.ndarray:
        '''Predict function with kwargs being passed to sample_posterior'''
        self._check_is_fitted()
        pred_X = np.concatenate((self.X, pred_X))
        samples = self.sample_posterior(X, Y, verbose=verbose, **kwargs)[:,]
        return np.mean(samples, axis=0), np.var(samples, axis=0)

        # prediction = 1/(1 + np.exp(-(X - self.posterior_mean(self.X, self.Y, verbose=verbose, **kwargs))))
        # return prediction


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
gpc.fit(X, Y, verbose=0)
# %%
