import numpy as np
from scipy.integrate import odeint
from lorenz import lorenz, statistics

num_sims = 1000

X = np.empty((num_sims, 3 + 3**2 + 3**2 + 3 + 3))
theta = np.empty((num_sims, 3))

for i in range(num_sims):
    print("Simulation %d" % i, end="\r")

    # Set the Lorenz parameters.
    sigma = np.random.uniform(0.0, 20.0)
    beta  = np.random.uniform(0.0, 10.0)
    rho   = np.random.uniform(0.0, 50.0)

    # Set the initial condition.
    x0 = np.array([0.0, 1.0, 1.05])

    # Set the time sample.
    t = np.linspace(0.0, 100.0, 1000)

    # Solve the ODE.
    x_t = odeint(lorenz, x0, t, args=(sigma, beta, rho))

    mean, cov, cor, eigvals, eigvecs, lyap = statistics(x_t)

    X[i, 0:3]   = mean
    X[i, 3:12]  = cov.flatten()
    X[i, 12:21] = cor.flatten()
    X[i, 21:24] = eigvals
    X[i, 24:27] = lyap

    theta[i, 0] = sigma
    theta[i, 1] = beta
    theta[i, 2] = rho

# Normalize X per column.
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

np.save("data/X.npy", X)
np.save("data/theta.npy", theta)

import IPython; IPython.embed()
