"""
Implement the Lorenz attractor using the ODE solver from scipy.integrate.
"""

import numpy as np
from scipy.integrate import odeint

def lorenz(x, t, sigma, beta, rho):
    """
    The Lorenz equations.
    Arguments
    ---------
    x : 3D vector of floats
        The current state.
    t : float
        The current time.
    sigma : float
        The Lorenz parameter sigma.
    beta : float
        The Lorenz parameter beta.
    rho : float
        The Lorenz parameter rho.
    Returns
    -------
    x_dot : 3D vector of floats
        The time-derivative of the Lorenz state.
    """
    x_dot = np.zeros(3)
    x_dot[0] = sigma * (x[1] - x[0])
    x_dot[1] = x[0] * (rho - x[2]) - x[1]
    x_dot[2] = x[0] * x[1] - beta * x[2]
    return x_dot

def statistics(x_t):
    """
    Obtain summary statistics from Lorenz attractor time series data.
    """
    # Compute the mean.
    mean = np.mean(x_t, axis=0)

    # Compute the covariance matrix.
    cov = np.cov(x_t, rowvar=False)

    # Compute the correlation matrix.
    cor = np.corrcoef(x_t, rowvar=False)

    # Compute the eigenvalues and eigenvectors.
    eigvals, eigvecs = np.linalg.eig(cov)

    # Compute the Lyapunov exponents.
    lyap = np.zeros(3)
    for i in range(3):
        lyap[i] = np.mean(np.log(np.abs(np.dot(eigvecs[:,i], x_t.T))))

    return mean, cov, cor, eigvals, eigvecs, lyap

if __name__=="__main__":

    # Set the Lorenz parameters.
    sigma = 10.0
    beta = 8.0 / 3.0
    rho = 28.0

    # Set the initial condition.
    x0 = np.array([0.0, 1.0, 1.05])

    # Set the time sample.
    t = np.linspace(0.0, 100.0, 10000)

    # Solve the ODE.
    x_t = odeint(lorenz, x0, t, args=(sigma, beta, rho))

    mean, cov, cor, eigvals, eigvecs, lyap = statistics(x_t)

    import pylab as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Plot the Lorenz attractor using a Matplotlib 3D projection.
    fig = plt.figure()
    plt.plot(x_t[:,0], x_t[:,1])
    plt.show()

    print("Mean: ", mean)
    print("Covariance matrix: ", cov)
    print("Correlation matrix: ", cor)
    print("Eigenvalues: ", eigvals)
    print("Eigenvectors: ", eigvecs)
    print("Lyapunov exponents: ", lyap)

