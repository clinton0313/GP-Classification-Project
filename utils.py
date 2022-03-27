import matplotlib.pyplot as plt
import numpy as np
from typing import Callable
from EllipticalSliceSampler import EllipticalSampler
from sklearn.gaussian_process.kernels import Matern

#KERNEL FUNCTIONS

def white_noise_kernel(x, y, theta):
    '''Parameters: variance'''
    return theta[0]

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
