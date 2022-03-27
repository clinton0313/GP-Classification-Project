import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
import numpy as np
from typing import Callable
from EllipticalSliceSampler import EllipticalSampler
from sklearn.gaussian_process.kernels import Matern
from IPython.display import clear_output
import scipy
from random import choice

cmap=colors.LinearSegmentedColormap.from_list('rg',["tab:green", "tab:red"], N=256)

#KERNEL FUNCTIONS

def white_noise_kernel(x, y, theta):
    '''Parameters: variance'''
    return theta[0]

def gaussian_white_noise_kernel(x,y,theta):
    return gaussian_kernel(x, y, [theta[0], theta[1]]) + white_noise_kernel(x, y, [theta[2]])

def linear_kernel(x, y, theta):
    '''Parameters: variance'''
    return theta[0] * np.dot(x, y) 

def gaussian_kernel(x, y, theta):
    '''Parameters: variance (amplitude), length scale'''
    return theta[0] * np.exp(-(np.dot((x-y), (x-y))/(2*theta[1])))

def periodic_kernel(x, y, theta):
    '''Parameters: variance(amplitude), length, period'''
    return theta[0] **2 * np.exp(- 2/theta[1]**2 * np.sin(np.pi * np.abs(x - y) / theta[2])**2)

def wiener_kernel(x, y, theta):
    '''Parameters: variance'''
    return theta[0] * min(x, y)

def rational_quadratic_kernel(x, y, theta):
    '''Parameters: variance(amplitude), length scale, scale mixture'''
    return theta[0] * (1 + np.dot((x-y), (x-y)) ** 2 / (2 * theta[2] * theta[1] **2)) ** (-theta[2])

def polynomial_kernel(x, y, theta):
    '''Parameters: variance, offset, degree'''
    return (theta[0] * np.dot(x, y) + theta[1]) ** theta[2]

def matern_kernel(x, y, theta):
    '''
    Parameters: variance, length-scale, nu
    nu is a smoothness parameter and as nu approaches infinity the matern kernel approximates the rbf kernel
    The matern12 kernel corresponds to nu = 1/2 making it identical to the exponential kernel
    The matern32 kernel corresponds to nu = 3/2 will be once differentiable
    The matern52 kernel corresponds to nu = 5/2 will be twice differentiable
    '''
    m = Matern(length_scale=theta[1], nu=theta[2])
    return theta[0] * m.__call__(x.reshape(1, -1), y.reshape(1,-1))

#PLOT FUNCTIONS

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def plot_gp_samples(
    X, 
    samples, 
    ax = None, 
    x_dim=0, 
    are_logits=True,
    color="tab:purple", 
    alpha=0.3, 
    plot_mean = True, 
    mean_color="tab:red", 
    **plot_kwargs
) -> None:
    '''
    Plots sample.
    Args:
        X: Design matrix (nxd)
        samples: Samples returned from GPC (num_samples x n)
        x_dim: The dimension of the the design matrix to use for 2d plotting
        are_logits: If true then the samples will be passed through a sigmoid before plotting
        ax: matplotlib ax to plot onto. If no axes is passed than a new figure will be returned.
        color: Color of sample lines.
        alpha: Alpha for sample line plots
        plot_mean: Plots the mean posterior sample if true.
        mean_color: Color of the mean line.
    '''

    if ax is None:
        fig, ax = plt.subplots(figsize=(10,10), tight_layout=True)

    if len(X.shape) != 1:
        X = X[:, x_dim].tolist()

    if are_logits:
        samples = sigmoid(samples)

    samples = samples.squeeze()
    for sample in samples:
        x, y = zip(*sorted(zip(X, sample.tolist()), key=lambda i: i[0]))
        ax.plot(x, y, color=color, alpha=alpha, **plot_kwargs)
    
    if plot_mean:
        sample_mean = np.mean(samples, axis=0)
        x, y = zip(*sorted(zip(X, sample_mean.tolist()), key=lambda i: i[0]))
        ax.plot(x, y, color=mean_color)

    try:
        return fig
    except:
        pass

def plot_sampler(sample:np.ndarray, ll:Callable, stepsize:int=25):
    """
    Plots the likelihood over posterior samples of the ESS.

    :param sample: Sample draws from a ESS.
    :param ll: Likelihood function.
    :param stepsize: The frequency of plotted draws.
    """
    points = [x for i, x in enumerate(ll(sample)) if i % stepsize == 0]
    _, ax = plt.subplots(1, 1, figsize=(12,8))
    plt.plot(np.arange(0, sample.shape[0], stepsize), points)
    ax.set_ylabel("Log-Likelihood")
    ax.set_xlabel("Sample Iteration")
    ax.set_title("Likelihood over ESS iterations")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

def plot_multiple(
    ess:Callable, 
    ll:Callable, 
    num_samples:int=10000, 
    num_init:int=5, 
    stepsize:int=25
):
    """
    Plots the likelihood over samples for multiple initializations of the ESS.
    
    :param ess: Instance of an Elliptical Slice sampler.
    :param ll: Likelihood function.
    :param num_samples: Number of samples to draw for a single initialization.
    :param num_init: Number of initializations of the ESS.
    :param stepsize: The frequency of plotted draws.
    """

    assert isinstance(ess, EllipticalSampler),  (
        "Function requires an instance of an ESS class."
    )
    samples = [ess.sample(num_samples=num_samples, num_burnin=0) for _ in range(num_init)]
    points = [[x for i, x in enumerate(ll(j)) if i % stepsize==0] for j in samples]

    _, ax = plt.subplots(1, 1, figsize=(12,8))
    for init in points:
        plt.plot(np.arange(0, samples[0].shape[0], stepsize), init)
    ax.set_ylabel("Log-Likelihood")
    ax.set_xlabel("Sample Iteration")
    ax.set_title("Likelihood over ESS iterations - Multiple Initializations")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

def plot_contour(
    train_X, 
    train_Y, 
    test_X, 
    test_Y, 
    pred_Y, 
    gpc, 
    cmap, 
    contour=True,
    **plot_kwargs
):
    X = np.concatenate([train_X, test_X])
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.5),
        np.arange(y_min, y_max, 0.5)
    )

    fig, ax = plt.subplots(1,1,figsize=(10,8))
    train_cmap = np.where(train_Y>0, "tab:red", "tab:green")
    test_cmap = np.where(test_Y>0, "tab:red", "tab:green")
    pred_cmap = np.where(pred_Y>0.5, "tab:red", "tab:green")
    
    if contour:
        Z = gpc.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z[0].reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.6, **plot_kwargs)

    ax.scatter(x=train_X[:,0], y=train_X[:,1], s=10, c=train_cmap)
    ax.scatter(x=test_X[:,0], y=test_X[:,1], s=100, c=test_cmap, edgecolors=pred_cmap, linewidth=3)

    legend_elements = [
        Line2D([0], [0], marker='o', color='black', label='True Label', markeredgecolor='w', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Predicted Label', markeredgecolor='black', markersize=10, markeredgewidth=3)
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower center", 
        ncol=2,
    )

    ax.set_title("Binary Classification using 2D observations")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def plot_ESS(log_y, f_incumbent, f_candidate, Θ_min, Θ_max, nu, loglik, i, first=True, save=True, **kwargs):
    clear_output(wait=True)
    Θ_space = np.linspace(0,2*np.pi,200)
    valid_Θ = [t for t in Θ_space if (Θ_max >= t) | (t >= 2*np.pi + Θ_min)]
    disc_Θ = [t for t in Θ_space if (Θ_max <= t) | (t <= 2*np.pi + Θ_min)]
    if first:
        valid_Θ = Θ_space
    valid_space = np.stack([(f_incumbent * np.cos(x) + nu * np.sin(x)).squeeze() for x in valid_Θ])
    disc_space = np.stack([(f_incumbent * np.cos(x) + nu * np.sin(x)).squeeze() for x in disc_Θ])
    higher_cmap = np.where(loglik(valid_space) > log_y, "tab:green", "black")

    fig, ax = plt.subplots(1,1, **kwargs)
    ax.scatter(x=valid_space[:,0], y=valid_space[:,1],s=10, c=higher_cmap, alpha=1)
    ax.scatter(x=disc_space[:,0], y=disc_space[:,1],s=10, c="tab:gray", alpha=0.2)
    ax.plot(f_incumbent[0], f_incumbent[1], 'k^', markersize=10, label="$f_t$")
    ax.plot(f_candidate[0], f_candidate[1], 'b^', markersize=10, label="$f_{t+1}$")
    ax.legend(loc="upper left", prop={'size': 16})
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title(
        "$\log y=$" +
        str(round(log_y,2)) +
        "$\quadL(f_{t+1})=$" +
        str(round(loglik(f_candidate),2)) +
        "$\quad[Θ_{min},\;Θ_{max}]=$" + "[" + str(round(Θ_min,2)) + ", " + str(round(Θ_max,2)) +"]"
    )
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()
    if save:
        fig.savefig(f'fig/mean_error_uniform_{i}.png', bbox_inches='tight', dpi=400)

# INFORMATION CRITERIA

def loglikelihood(Y, f) -> float:
        '''
        Returns the log likelihood for a binary classification model
        Args:
            Y: Binary labels
            f: Logits (draw from gaussian process)
        '''
        
        f = np.array(f).reshape(1, -1)
        Y = np.array(Y).reshape(1, -1)

        return np.sum([
            np.log(sigmoid(f_i)) if y == 1 
            else np.log(sigmoid(-f_i)) 
            for f_i, y in zip(f.squeeze().tolist(), Y.squeeze().tolist())
            ])

def AIC(f, Y):
    return 4 - loglikelihood(Y,f)

def BIC(f,Y):
    return 2*np.log(len(Y)) - loglikelihood(Y,f)

def WAIC(Y,posterior_samples):
   return -2*np.mean([loglikelihood(Y, f) for f in posterior_samples]) + 2*loglikelihood(Y, np.mean(posterior_samples, axis= 0))    

def plot_live(points, prior_Σ, sd_ellipse, sd2_ellipse):
    clear_output(wait=True)
    Σ_hat = np.cov(points.T)
    post_Σ = prior_Σ @ scipy.linalg.inv(2*prior_Σ) @ prior_Σ

    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(sd_ellipse[0,:], sd_ellipse[1,:], label="1 SD")
    ax.plot(sd2_ellipse[0,:], sd2_ellipse[1,:], label="2 SD")
    ax.scatter(x=points[:,0], y=points[:,1], s=15, c="k", alpha=0.8)
    ax.legend(loc="lower left")
    sigma_cfs_txt = '$\Sigma^{cfs}_{11} = %.2f$, $\Sigma^{cfs}_{12} = %.2f$, $\Sigma^{cfs}_{22} = %.2f$' % (post_Σ[0,0], post_Σ[0,1], post_Σ[1,1])
    sigma_post_txt = '$\hat{\Sigma}^{post}_{11} = %.2f$, $\hat{\Sigma}^{post}_{12} = %.2f$, $\hat{\Sigma}^{post}_{22} = %.2f$' % (Σ_hat[0,0], Σ_hat[0,1], Σ_hat[1,1])
    ax.text(0, 3, sigma_cfs_txt, fontsize = 16)
    ax.text(0, 2.5, sigma_post_txt, fontsize = 16)
    ax.set_xlim(-3,3)
    ax.set_ylim(-3,3.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()

def label_colors(labels, col1="tab:orange", col2="tab:blue"):
    return [col1 if label == 1 else col2 for label in labels]

def proba_alphas(probas):
    return [np.abs(0.5 - proba) * 2 for proba in probas]

def plot_2d_gp(x_train, y_labels, y_probas, ax):
    ax.scatter(x_train[:,0], x_train[:,1], alpha=proba_alphas(y_probas), c=label_colors(y_labels))

def get_labels(sample):
    return [1 if x >= 0.5 else 0 for x in sigmoid(sample)]

def mean_sample(samples):
    return np.mean(np.vstack(samples), axis=0)

def plot_mean_samples(x_train, y_train, pop_sample, sample_size=100, ncol=4, nrow=4, **kwargs):
    mean_samples = [mean_sample([choice(pop_sample) for _ in range(sample_size)]) for _ in range(nrow * ncol - 1)]
    fig = compare_samples_with_original(x_train, y_train, mean_samples, nrow=nrow, ncol=ncol, **kwargs)
    return fig

def compare_samples_with_original(x_train, y_train, samples, nrow = 4, ncol = 4, plot_proba = True):
    fig, axs = plt.subplots(nrow, ncol, figsize=(12, 12), tight_layout=True, sharey=True, sharex=True)
    plot_2d_gp(x_train, y_train, y_train, axs.ravel()[0])
    for ax in axs.ravel()[1:]:
        sample = choice(samples)
        sample_labels = get_labels(sample)
        sample_alpha = sigmoid(sample) if plot_proba else sample_labels
        plot_2d_gp(x_train, sample_labels, sample_alpha, ax)
    return fig

def plot_1d(x_dim, x, y, sample:list, ax, title, num_samples = 50):
    samples = np.array([choice(sample) for _ in range(num_samples)])
    plot_gp_samples(x, samples, ax=ax, x_dim=x_dim)
    pred_labels = get_labels(mean_sample(samples))
    pred_colors = ["tab:blue" if pred == true else "black" for pred, true in zip(pred_labels, y)]
    ax.scatter(x[:,x_dim], y, c=pred_colors)
    ax.axhline(0.5, color="black", linestyle="dashed")
    ax.set_title(title)
