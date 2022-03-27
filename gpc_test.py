#%%

from gpc import GPC
from utils import gaussian_kernel, plot_gp_samples
import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt




#%%
np.random.seed(0)
x, y = make_classification(n_features=2, n_redundant=0, class_sep=5, random_state=0)


gpc = GPC(gaussian_kernel, [1, 1])

fig, axs = plt.subplots(2, 2, figsize=(12, 12))
for ax in axs.ravel():
    ax.scatter(x[:,0], y)
prior_samples = gpc.sample_prior(x, 200)
plot_gp_samples(x, prior_samples, ax=axs[0][0])
axs[0][0].set_title("Prior Samples")

posterior_samples = gpc.sample_posterior(x, y)
plot_gp_samples(x, posterior_samples, ax=axs[0][1])
axs[0][1].set_title("Posterior Samples")

gpc.fit(x, y, verbose=1, tol=1e-1, eps=1)
print(gpc.hyperparameters)
fitted_samples = gpc.sample_posterior(x, y)
plot_gp_samples(x, fitted_samples, ax=axs[1][0])
axs[1][0].set_title("Fitted Samples tol=1e-1 eps=1e-1")

gpc.fit(x, y, verbose=1, tol=1e-1, eps=2)
print(gpc.hyperparameters)
fitted_samples = gpc.sample_posterior(x, y)
plot_gp_samples(x, fitted_samples, ax=axs[1][1])
axs[1][1].set_title("Fitted Samples tol=1e-1 eps = 1")




# %%
