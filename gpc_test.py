#%%

from random import sample
from gpc import GPC
from utils import gaussian_kernel, plot_gp_samples
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#%%
np.random.seed(0)
X, Y = make_classification(n_samples = 40, n_features=2, n_redundant=0, class_sep=5, random_state=0)
x, x_test, y, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

tol1 = 1e-1
eps1 = 1e-4
tol2 = 1e-1
eps2 = 1
num_samples = 150

gpc = GPC(gaussian_kernel, [1, 1])

prior_samples = gpc.sample_prior(x, num_samples - 100)
posterior_samples = gpc.sample_posterior(x, y, num_samples = num_samples)

gpc1 = GPC(gaussian_kernel, [1,1])
gpc1.fit(x, y, verbose=1, tol=tol1, eps=eps1)
fitted_samples = gpc1.sample_posterior(x, y, num_samples = num_samples)
print(gpc1.hyperparameters)


gpc2 = GPC(gaussian_kernel, [1,1])
gpc2.fit(x, y, verbose=1, tol=tol2, eps=eps2)
fitted_samples2 = gpc2.sample_posterior(x, y, num_samples=num_samples)
print(gpc2.hyperparameters)


#%%

models = [gpc, gpc, gpc1, gpc2]

titles = [
    "Prior Samples",
    "Posterior Samples",
    f"Fitted Samples tol={tol1} eps={eps1}",
    f"Fitted samples tol={tol2} eps = {eps2}"
]

samples = [
    prior_samples, 
    posterior_samples,
    fitted_samples,
    fitted_samples2
]

predictions_samples = [
    gpc.sample_prior(np.vstack([x, x_test]), num_samples=num_samples- 100),
    gpc.sample_posterior(np.vstack([x, x_test]), y, num_samples=num_samples),
    gpc1.sample_posterior_predictions(x_test, num_samples=num_samples),
    gpc2.sample_posterior_predictions(x_test, num_samples=num_samples)
]

hyperparameters = [
    [round(p, 3) for p in model.hyperparameters] 
    for model in models
]

nlls = [
    round(np.mean([- model._loglikelihood(y, s) for s in sample]), 3)
    for model, sample in zip(models, samples)
]
#%%
def plot_example(x_dim, x, y, titles, samples, x_test = None, y_test = None):
    fig, axs = plt.subplots(2, 2, figsize=(12, 12), sharey=True, tight_layout=True)
    for i, ax in enumerate(axs.ravel()):
        if x_test is not None:
            if y_test is not None:
                ax.scatter(x_test[:, x_dim], y_test, color="black")
            x_stack = np.vstack([x, x_test])
            plot_gp_samples(x_stack, samples[i], ax=ax, x_dim=x_dim)
        else:
            plot_gp_samples(x, samples[i], ax=ax, x_dim=x_dim)
        ax.set_title(titles[i])
        ax.set_xlabel(f"Hyperparameters: {hyperparameters[i]} NLL: {nlls[i]}")
        for spine in ["top", "bottom", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.set_xticks([])
        ax.scatter(x[:,x_dim], y, color="tab:blue")
    return fig

fig = plot_example(0, x, y, titles, samples)

fig2 = plot_example(1, x, y, titles, samples)

#%%

pred_fig = plot_example(
    0, 
    x, 
    y,
    titles, 
    predictions_samples,
    x_test,
    y_test
)

pred_fig2 = plot_example(
    1, 
    x,
    y,
    titles, 
    predictions_samples,
    x_test,
    y_test
)

#%%
def gen_outofbounds_x(x, num_samples, step):
    oob_x = []
    for dim in range(x.shape[1]):
        min_x = np.array([np.min(x[:,dim]) - (i + 1) * step for i in range(num_samples)]) 
        max_x = np.array([np.max(x[:,dim]) + (i + 1) * step for i in range(num_samples)]) 
        x_dim = np.concatenate((min_x, max_x))
        oob_x.append(x_dim)
    return np.vstack(oob_x).T

new_x = gen_outofbounds_x(x, 10, 0.5)


#%%
pred_fig_oob1 = plot_example(
    0, 
    x, 
    y,
    titles, 
    predictions_samples,
    new_x
)

pred_fig_oob2 = plot_example(
    1, 
    x,
    y,
    titles, 
    predictions_samples,
    new_x
)

# %%
