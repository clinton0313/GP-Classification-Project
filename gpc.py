#%%
import itertools
import numpy as np

from EllipticalSliceSampler import EllipticalSampler
from scipy.optimize import minimize
from typing import Callable, Sequence
#%%
class GPC():
    '''
    A Gaussian Process Classifier class.
    '''
    def __init__(
        self, 
        kernel:Callable, 
        hyperparameters:Sequence, 
        hyperparameter_names: Sequence = [], 
        optimizer:str = "L-BFGS-B"
    ):
        '''
        Args:
            kernel: A kernel function that takes three arguments: X, Y, hyperparameters. 
                hyperparameters should be a list of kernel parameters.
            hyperparameters: A sequence of hyperparmaeters that correspond to the kernel argument.
            hyperparameter_names[optional]: Names of the hyperparameters.
            optimizer[optional]: optimizer by scipy.optimize.minimize to fit the model. 
        '''
        self.kernel= kernel
        self.hyperparameters = hyperparameters
        self.hyperparameters_names = hyperparameter_names

        self.X = None
        self.Y = None
        self.nll = None
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
        min_eig = np.min(np.real(np.linalg.eigvals(gram_matrix)))
        if min_eig <0:
            gram_matrix -= 10*min_eig * np.eye(*gram_matrix.shape)

        return gram_matrix + 1e-12 * np.identity(n)
    
    def _sigmoid(self, f):
        return 1/(1 + np.exp(-f))

    def _list_to_array(self, x:Sequence):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        return x

    def _loglikelihood(self, Y, f) -> float:
        '''
        Returns the log likelihood for a binary classification model
        Args:
            Y: Binary labels
            f: Logits (draw from gaussian process)
        '''
        
        f = f.reshape(1, -1)
        Y = self._list_to_array(Y).reshape(1, -1)

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

    def fit(self, X, y, maxiter:int = 100, eps:float = 1e-3, tol:float = 1e-7, verbose=0, **kwargs) -> None:
        '''
        Fits the model and finds the optimal hyperparameters
        
        Args:
            maxiter: max iterations for the optimizer
            eps: step size for the optimizer
            tol: threshold change in the log likelihood for convergence
            verbose: 1 shows the hyperparameters as it updates, 2 also shows progress bar for sampler.
        '''
        self.X = X
        self.Y = y

        def nll(hyperparameters):
            neg_loglikelihood = -1 * self._loglikelihood(Y=self.Y, f=self.posterior_mean(self.X, self.Y, hyperparameters=hyperparameters, **kwargs))
            if verbose >= 1:
                print(f"Neg log likelihood is {neg_loglikelihood}")
            return neg_loglikelihood

        def callback(parameters):
            if verbose >= 1:
                if len(self.hyperparameters_names) == len(parameters):
                    parameter_strings = [str(round(p, 4)) for p in parameters]
                    output = [": ".join(name_param) for name_param in zip(self.hyperparameters_names, parameter_strings)]
                    print(" ".join(output))
                else:
                    print(parameters)

        res = minimize(
            nll, 
            self.hyperparameters, 
            method=self.optimizer, 
            options={"maxiter":maxiter, "ftol": tol, "eps": eps}, 
            callback=callback)
        self._update_hyperparameters(res.x)
        self.nll = res.fun
        print(f"Fitted with final hyperparameters: {self.hyperparameters} and neg log likelihood {res.fun}")
    
    def predict(self, pred_X, verbose=0, **kwargs) -> np.ndarray:
        '''Predict function with kwargs being passed to sample_posterior'''
        self._check_is_fitted()
        pred_X = np.concatenate((self.X, pred_X))
        samples = self.sample_posterior(pred_X, self.Y, verbose=verbose, **kwargs)[:, self.X.shape[0]:]
        return np.mean(samples, axis=0), np.var(samples, axis=0)

