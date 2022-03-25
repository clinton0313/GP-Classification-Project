#%%
import itertools
import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import minimize
from typing import Callable, Sequence
#%%
class GPC():
    
    def __init__(self, optimizer:str = "Nelder-Mead"):
        self.kernel_parameters = []
        self.kernel_function = None
        self.nll = None
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
    
    def set_nll(self, X, Y, **kwargs) -> float:
        '''Builds the nll function so that nll function only takes hyperparameters as argument'''
        def nll(hyperparameters):
            '''Negative log likelihood P(f | Y)'''
            f = self.posterior_mean(**kwargs)
            ...
        self.nll = nll
    
    def sample_posterior(self, burn_in:int = 100, n_samples:int = 100) -> Sequence:
        '''return n samples after an initial burn in of samples'''
        ...

    def posterior_mean(self, **kwargs) -> Sequence:
        '''Returns posterior mean'''
        samples = np.vstack(self.sample_posterior(**kwargs))
        return np.mean(samples, axis=0)

    def fit(self, X, y, max_iters:int = 100) -> None:
        '''sets hyperparameters to optimal'''
        self.X = X
        self.Y = y

        res = minimize(self.nll, self.kernel_parameters, method=self.optimizer)
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