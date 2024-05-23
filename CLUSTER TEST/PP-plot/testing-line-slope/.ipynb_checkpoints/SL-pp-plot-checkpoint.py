#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import emcee
import warnings
import corner
import pandas as pd
import bilby
warnings.filterwarnings("ignore")


NRES = 200
N = 50

M_LOW = -5
M_HIGH = -2
B_LOW = 1
B_HIGH = 5

priors = {"m": bilby.core.prior.Uniform(M_LOW, M_HIGH, "m"), 
          "b": bilby.core.prior.Uniform(B_LOW, B_HIGH, "b")}

RESULTS = []
def log_likelihood(theta, x, y, yerr):
    m, b = theta
    model = m * x + b
    sigma2 = yerr**2
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

def log_prior(theta):
    m, b = theta
    if -5.0 < m < 0.5 and 0.0 < b < 10.0:
        return 0.0
    return -np.inf

def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)


for ii in range(NRES):
    injections = dict()
    posterior = dict()
    
    for key, prior in priors.items():
        injections[key] = prior.sample()

    x = np.sort(10 * np.random.rand(N))
    yerr = 0.1 + np.random.rand(N)
    y = injections["m"] * x + injections["b"]
    y += yerr * np.random.randn(N)

    pos = np.array([injections["m"], injections["b"]]) + 0.1 * np.random.randn(2) + 1e-4 * np.random.randn(10, 2)
    nwalkers, ndim = pos.shape
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, yerr), pool=pool)
        sampler.run_mcmc(pos, 3000, progress=True)
    
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    posterior = {"m": flat_samples[:,0], "b": flat_samples[:,1]}

    posterior = pd.DataFrame(dict(posterior))
    
    posterior.to_csv(f'samples/run{ii+1}.txt', sep='\t')
        
    result = bilby.result.Result(
        label="PPtest",
        injection_parameters=injections,
        posterior=posterior,
        search_parameter_keys=injections.keys(),
        priors=priors)
    
    RESULTS.append(result)

    fig = corner.corner(flat_samples, labels=["m", "b"], truths=[injections["m"], injections["b"]], title_fmt=".2E", show_titles=True, title_kwargs={'fontsize':12})
    plt.savefig('corner-plot-EMCEE-line/run_'+str(ii+1)+'.png')
    plt.close()
        
bilby.result.make_pp_plot(RESULTS, filename=f"PP-Plot.png", confidence_interval=[0.68, 0.95, 0.997])