import numpy as np
from scipy.integrate import odeint
from lorenz import lorenz, statistics

"""
Create worker and run a set of simulations in parallel on the CPU.
"""

def worker(params):

    # Set the Lorenz parameters.
    sigma = params[0]
    beta  = params[1]
    rho   = params[2]

    # Set the initial condition.
    x0 = np.array([0.0, 1.0, 1.05])

    # Set the time sample.
    t = np.linspace(0.0, 100.0, 1000)

    # Solve the ODE.
    x_t = odeint(lorenz, x0, t, args=(sigma, beta, rho))

    mean, cov, cor, eigvals, eigvecs, lyap = statistics(x_t)

    X = np.concatenate([mean, cov.flatten(), cor.flatten(), params])

    return X


if __name__=="__main__":

    import mpi4py.MPI as MPI
    import tqdm as tqdm

    PATH = "data/"

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    num_simulations = 10000*size

    theta = np.empty((num_simulations, 3))
    theta[:, 0] = np.random.uniform(0.0, 20.0, size=num_simulations)
    theta[:, 1] = np.random.uniform(0.0, 10.0, size=num_simulations)
    theta[:, 2] = np.random.uniform(0.0, 30.0, size=num_simulations)

    # Split the simulations across the workers.
    theta = np.array_split(theta, size)[rank]

    num_simulations_per_worker = int(num_simulations / size)

    X = []
    for i in tqdm.tqdm(range(num_simulations_per_worker), disable=not rank==0):
        X_ = worker(theta[i])
        X.append(X_)

    # Gather the results.
    X = comm.allgather(X)

    if rank == 0:

        X = np.reshape(X, (num_simulations, -1))

        # Save the data.
        np.save(PATH + "X.npy", X)
