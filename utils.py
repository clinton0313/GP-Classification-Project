import matplotlib.pyplot as plt
import numpy as np

def plot_gp_samples(
    X, 
    samples, 
    ax = None, 
    x_dim=0, 
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