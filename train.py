import numpy as np
import pylab as plt

import torch
import sbi.utils as utils
from sbi.inference import SNPE, SNLE, SNRE
from sbi.analysis import pairplot, conditional_pairplot

def train(num_simulations,
          x, 
          theta,
          num_threads=1,
          method="SNPE",
          device="cpu",
          density_estimator="maf"):
    
    torch.set_num_threads(num_threads)

    if (len(x.shape) == 1):
        x = x[:, None]
    if (len(theta.shape) == 1):
        theta = theta[:, None]


    if (method == "SNPE"):
        inference = SNPE(
            density_estimator=density_estimator, device=device
        )
    elif (method == "SNLE"):
        inference = SNLE(
            density_estimator=density_estimator, device=device
        )
    elif (method == "SNRE"):
        inference = SNRE(
            density_estimator=density_estimator, device=device
        )
    else:
        raise ValueError("Unknown inference method")
    
    inference = inference.append_simulations(theta, x)
    _density_estimator = inference.train()
    posterior = inference.build_posterior(_density_estimator)

    return posterior

def infer(obs_stats,
          num_samples,
          posterior):
    return posterior.sample((num_samples,), x=obs_stats)

if __name__=="__main__":

    import argparse 

    parser = argparse.ArgumentParser(description="Train a density estimator on the Lorenz system.")
    parser.add_argument("--data", type=str, default="data/X.npy", help="Path to the data file.")
    parser.add_argument("--method", type=str, default="SNPE", help="Inference method.")
    parser.add_argument("--density_estimator", type=str, default="maf", help="Density estimator.")
    parser.add_argument("--num_threads", type=int, default=1, help="Number of threads.")
    parser.add_argument("--device", type=str, default="cpu", help="Device.")

    args = parser.parse_args()

    X = np.load(args.data, allow_pickle=True)

    params = X[:, -3:]
    stats  = X[:, :-3]

    num_simulations = X.shape[0]

    theta = torch.from_numpy(params).float()
    x = torch.from_numpy(stats).float()

    posterior = train(num_simulations,
                        x,
                        theta,
                        num_threads         = args.num_threads,
                        method              = args.method,
                        device              = args.device,
                        density_estimator   = args.density_estimator
                        )
    
    # save posterior
    torch.save(posterior, "models/posterior.pt")
