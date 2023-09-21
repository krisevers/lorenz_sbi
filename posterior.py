import torch
import sbi.utils as utils
from sbi.analysis import conditional_pairplot

from utils import pairplot, marginal_correlation

if __name__=="__main__":

    import numpy as np
    import pylab as plt

    from lorenz import lorenz, statistics
    from scipy.integrate import odeint

    # Set the Lorenz parameters.
    sigma = 10.0
    beta  = 8.0 / 3.0
    rho   = 28.0

    # Set the initial condition.
    x0 = np.array([0.0, 1.0, 1.05])

    # Set the time sample.
    t = np.linspace(0.0, 100.0, 1000)

    # Solve the ODE.
    x_t = odeint(lorenz, x0, t, args=(sigma, beta, rho))

    mean, cov, cor, eigvals, eigvecs, lyap = statistics(x_t)

    obs_x = np.concatenate([mean, cov.flatten(), cor.flatten()])
    obs_theta = [sigma, beta, rho]

    # Load posterior.
    posterior = torch.load("models/posterior.pt")

    num_samples = 100000

    posterior.set_default_x(obs_x)
    posterior_samples = posterior.sample((num_samples,))

    fig, ax = pairplot(samples=posterior_samples, labels=[r"$\sigma$", r"$\beta$", r"$\rho$"], figsize=(10, 10))

    plt.savefig("png/pairplot.png")

    fig, ax = marginal_correlation(samples=posterior_samples, labels=[r"$\sigma$", r"$\beta$", r"$\rho$"], figsize=(10, 10))

    plt.savefig("png/marginal_correlation.png")



    import IPython; IPython.embed();