#%%

import numpy as np
import scipy

from scipy.stats import multivariate_normal
from EllipticalSliceSampler import EllipticalSampler
from utils import (
    plot_sampler,
    plot_multiple
)



# %%
# Prior on latent parameters
prior_µ = np.array([1,2,0])
prior_Σ = np.array([[ 1.13540776, -0.79362872, -0.11839691],
                    [-0.79362872,  3.39306091,  0.13178849],
                    [-0.11839691,  0.13178849,  0.27310531]])

# Likelihood parameters
lik_µ = np.array([0,0,0])
lik_Σ = np.array([[ 0.93649446,  1.03144611,  0.19181939],
                 [ 1.03144611,  3.12481816, -0.01788002],
                 [ 0.19181939, -0.01788002,  0.47977509]])

# Instantiate elliptical slice sampler
loglik = lambda f: np.log(multivariate_normal.pdf(f, mean=lik_µ, cov=lik_Σ))
ess = EllipticalSampler(prior_μ=prior_μ, prior_Σ=prior_Σ, ll=loglik)
draws = ess.sample(num_samples=10000, num_burnin=0)

# %%
# Draw to check for convergence of the MCMC routine
plot_sampler(sample=draws, ll=loglik, stepsize=50)

# %%
# Check multiple initializations converge to same likelihood
plot_multiple(ess, loglik, 1000, 5)
# %%
# Sanity check
# https://stats.stackexchange.com/questions/28744/multivariate-normal-posterior
# two gaussians, can compute posterior of multivariate
post_Σ = prior_Σ @ scipy.linalg.inv(prior_Σ + lik_Σ) @ lik_Σ

# Posterior using analytical solution
print(post_Σ)

# Posterior Covariance from the Elliptical Slice Sampler 
np.cov(draws.T)
# %%
# accounting for the mean in the sampler
# https://www.michaelchughes.com/blog/2012/08/elliptical-slice-sampling-for-priors-with-non-zero-mean/
basespace = np.linspace(0,2*np.pi, 100)
points = np.array([np.cos(basespace),np.sin(basespace)])
# %%
sd_ellipse = 2 * np.linalg.cholesky(np.cov(draws.T)) @ points
# %%
y_pred, var_pred = gpc.predict(np.array([1,3,4,5,76,4,7,78]))
prob_pred = GPC._sigmoid(y_pred)
prob_lb, prob_ub = GPC._sigmoid(y_pred-2*np.sqrt(var_pred)), GPC._sigmoid(y_pred+2*np.sqrt(var_pred))
