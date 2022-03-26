# %%
import sys
from tqdm import tqdm
import numpy as np
from numpy.random import multivariate_normal as mn
from typing import Sequence, Callable

# %%
class EllipticalSampler:
    """
    An Elliptical Slice Sampler, as described in Murray et al. (2010).

    The ESS is an extension of slice sampling (Neal, 2003) that is designed to 
    sample from a posterior distribution in which the prior distribution is a
    Gaussian process. This property holds since log y has a Gaussian process prior. 
    The algorithm is shown to be a competitive alternative to Metropolis-Hastings (Tierney, 1994) 
    for Gaussian process priors, often times working better for real data in terms
    of a larger effective sample size of Monte Carlo output per unit compute 
    time (Murray et al., 2010). Note we allow for non-centered means.

    :param prior_µ: The prior vector mean of the latent features `f`.
    :param prior_Σ: The prior covariance matrix of the latent features `f`.
    :param ll: The observed data log likelihood function.
    """
    def __init__(
        self,
        prior_µ: np.array,
        prior_Σ: np.ndarray,
        ll: Callable,
    ) -> None:
        self.prior_µ, self.prior_Σ, self.ll = prior_µ, prior_Σ, ll
        self.n = len(self.prior_µ)
        self.f_incumbent = mn(self.prior_µ - self.prior_µ, self.prior_Σ, size=1)

    def iteration(self) -> np.ndarray:
        """
        An instance of the ESS, it selects a new location on the randomly generated 
        ellipse (see Murray et al, 2010). There are no rejections: the new state `f`
        is never equal to the current state `f` unless that is the only state on
        the ellipse with non-zero likelihood. The algorithm proposes the angle from a 
        bracket [`θ_min`, `θ_max`] which is shrunk exponentially quickly until an acceptable
        state is found. Thus the step size is effectively adapted on each iteration for
         the current `ν` and `Σ`.

        :return: An accepted proposal, characterized as an n-by-1 array.  
        """
        nu = mn(self.prior_µ - self.prior_µ, self.prior_Σ, size=1)
        log_y = self.ll(self.f_incumbent) + np.log(np.random.uniform(0,1))
        Θ = np.random.uniform(0 + sys.float_info.min, 2*np.pi)
        Θ_min, Θ_max = Θ - 2*np.pi, Θ
    
        while True:
            # try:
            f_candidate = self.f_incumbent * np.cos(Θ) + nu * np.sin(Θ)
            if self.ll(f_candidate) > log_y:
                self.f_incumbent = f_candidate
                return self.f_incumbent
            elif Θ < 0:
                Θ_min = Θ
            elif Θ > 0:
                Θ_max = Θ
            Θ = np.random.uniform(Θ_min, Θ_max)
            # except Exception as e:
            #     print(f"Error: {e}")
            #     permission = input("Exit Sampler?: [Y/N]")
            #     assert permission.lower() in ["y", "n", "yes", "no"], "Invalid response."
            #     if permission.lower in ["y", "yes"]:
            #         raise e
            #     elif permission.lower in ["n", "no"]:
            #         continue

    def sample(self, num_samples: int, num_burnin: int=0):
        """
        Performs `num_samples` iterative runs of the ESS, with the `f` state updated
        between cycles. 

        :return: A stacked array with the posterior samples.  
        """
        output = [self.iteration() + self.prior_µ for _ in tqdm(range(num_samples))]
        if num_burnin>0:
            output = output[num_burnin:]
        return np.stack(output)

