#%%
import itertools
import matplotlib.pyplot as plt
import numpy as np

from EllipticalSliceSampler import EllipticalSampler
from scipy.optimize import minimize
from typing import Callable, Sequence
#%%
class GPC():
    
    def __init__(self, optimizer:str = "Nelder-Mead"):
        self.kernel_parameters = []
        self.kernel= None
        self.X = None
        self.Y = None
        self.optimizer = optimizer
        self.fig, self.ax = plt.subplots(figsize=(14,14))

    def _is_fitted(self) -> None:
        assert self.X is not None, "There is no data loaded, please fit the model by calling the fit method."

    def _kernel_is_defined(self) -> None:
        assert self.kernel is not None, "No kernel is defined"

    def set_hyperparams(self, hyperparameters) -> None:
        assert len(hyperparameters) == len(self.kernel_parameters), \
            f"Incorrect number of hyperparameters. Expected {len(self.kernel_parameters)} instead got {len(hyperparameters)}"
        self.kernel_parameters = hyperparameters
    
    def set_kernel(self, kernel:Callable, hyperparameters:Sequence) -> None:
        '''Sets a kernel function that takes as arguments, X, Y, hyperparameters (as a list) and returns a real number.'''
        self.kernel = kernel
        self.kernel_parameters = hyperparameters

    def get_mu(self, X):
        return np.zeros(len(X))

    def get_sigma(self, X:np.ndarray, hyperparameters:Sequence = []):
        '''Returns a variance covariance matrix using the specified kernel and hyperparameters. if no hyperparameters are
        specified then uses the class hyperparameters.'''
        self._kernel_is_defined()
        if hyperparameters == []:
            hyperparameters = self.kernel_parameters

        n = len(X)
        gram_matrix = np.array([np.zeros(n) for _ in range(n)])
        for i, j in itertools.product(range(n), range(n)):
            gram_matrix[i][j] = self.kernel(X[i], X[j], hyperparameters)
        return gram_matrix
    
    def loglikelihood(self, Y, f) -> float:
        '''return the log likelihood'''
        f = f.reshape(1, -1)
        Y = Y.reshape(1, -1)
        assert f.shape == Y.shape, f"f and Y are not the same shape got f: {f.shape} and Y: {Y.shape}"
        return - np.sum(np.log(1/(1 + np.exp( -f*Y))))
        #I dont think we need the X here unless we are cumputing the f every time, if the f is sampled from outside there should be no problem
        #If we sample inside of the ll we need to call the kernel here and create the MVnorm inside of the LL using X

    def sample_prior(self, X, num_samples:int) -> Sequence:
        return np.random.multivariate_normal(self.get_mu(X), self.get_sigma(X), num_samples)
    
    def sample_posterior(self, X, Y, num_burnin:int = 100, num_samples:int = 300) -> Sequence:
        '''return n samples after an initial burn in of samples'''
        assert num_samples > num_burnin, f"Got {num_samples} but required to burn {num_burnin} samples"
        def log_likelihood(f):
            return self.loglikelihood(Y=Y, f=f)

        ess = EllipticalSampler(self.get_mu(X), self.get_sigma(X), log_likelihood)
        return ess.sample(num_samples, num_burnin)

    def posterior_mean(self, X, Y, **kwargs) -> Sequence:
        '''Returns posterior mean'''
        return np.mean(self.sample_posterior(X, Y, **kwargs), axis=0)

    def fit(self, X, y, maxiter:int = 100, **kwargs) -> None:
        '''sets hyperparameters to optimal'''
        self.X = X
        self.Y = y

        def nll(hyperparameters):
            return (- self.loglikelihood(Y=self.Y, f=self.posterior_mean(self.X, self.Y, hyperparameters=hyperparameters, **kwargs)))

        res = minimize(nll, self.kernel_parameters, method=self.optimizer, maxiter=maxiter)
        self.set_hyperparams(res.x)
    
    def predict(self, X) -> float:
        self._is_fitted()
        ...

    def plot_gp(
        self, X, Y,
        dim:int = 0, 
        num_burnin:int = 100, 
        num_samples:int = 300, 
        alpha:float = 0.7, 
        plot_mean:bool = True, 
        **plot_kwargs
    ) -> None:

        if len(X.shape) == 1:
            X_dim = X
        else:
            X_dim = X[:, dim]

        self.ax.scatter(X_dim, Y, color="tab:blue")
        
        samples = self.sample_posterior(X, Y, num_burnin, num_samples)
        for f in samples:
            x, y = zip(*sorted(zip(X_dim, f), key = lambda x: x[0]))
            self.ax.plot(x, y, alpha=alpha, color="tab:purple", **plot_kwargs)
        
        if plot_mean:
            mean = np.mean(samples, axis=0)
            x, y = zip(*sorted(zip(X_dim, mean), key=lambda x: x[0]))
            self.ax.plot(x, y, color="tab:red")


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

gpc.sample_posterior(X, Y)
#%%