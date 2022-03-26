# %%
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable
from EllipticalSliceSampler import EllipticalSampler
# %%
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

# %%
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
