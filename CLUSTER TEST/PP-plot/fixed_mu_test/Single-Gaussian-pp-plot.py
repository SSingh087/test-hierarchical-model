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
MU_P_LOW = 0
MU_P_HIGH = 5
# SIGAM_P_LOW = 0
# SIGMA_P_HIGH = 1
true_parent_scale = 0.5

priors = {"mu_p": bilby.core.prior.Uniform(MU_P_LOW, MU_P_HIGH, "mu_p")}

N_obs = 500
size_child = 100
true_child_scale = 0.3

NRES = 200

RESULTS = []
def log_hyperprior(params):
    parent_mean = params
    if (0 < parent_mean < 5):
        return 0.0
    return -np.inf

def log_likelihood(params):
    parent_mean = params
    return np.log(np.prod(size_child**-1 * np.sum((2 * np.pi * true_parent_scale**2)**-.5 * np.exp(-(true_child_gaussian - parent_mean)**2 / (2 * true_parent_scale**2)), axis=1)))

def log_probability(params):
    log_prior_val = log_hyperprior(params)
    if not np.isfinite(log_prior_val):
        return -np.inf
    return log_prior_val + log_likelihood(params)


for ii in range(NRES):
    injections = dict()
    posterior = dict()
    
    for key, prior in priors.items():
        injections[key] = prior.sample()
    
    true_parent_gaussian = np.random.normal(loc=injections["mu_p"], scale=true_parent_scale, size=N_obs)
    err_on_mean_child = np.expand_dims(np.random.randn(N_obs)*true_child_scale, axis=1)
    true_child_gaussian = np.random.normal(loc=true_parent_gaussian[:, np.newaxis] + err_on_mean_child, scale=true_child_scale, size=(N_obs, size_child))

    pos = np.array([injections["mu_p"]]) + 0.1 * np.random.randn(1) + 1e-6 * np.random.randn(30, 1)
    nwalkers, ndim = pos.shape

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)
        sampler.run_mcmc(pos, 3000, progress=True)

    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    posterior = {"mu_p": flat_samples[:,0]}
    
    posterior = pd.DataFrame(dict(posterior))
    posterior.to_csv(f'samples/run_{ii+1}.txt', sep='\t')
        
    result = bilby.result.Result(
        label="PPtest",
        injection_parameters=injections,
        posterior=posterior,
        search_parameter_keys=injections.keys(),
        priors=priors)
    
    RESULTS.append(result)

    fig = corner.corner(flat_samples, labels=["mu_p"], truths=[injections["mu_p"]], title_fmt=".2E", show_titles=True, title_kwargs={'fontsize':12})
    plt.savefig('corner-plot/run_'+str(ii+1)+'.png')
    plt.close()
        
bilby.result.make_pp_plot(RESULTS, filename=f"PP-Plot", confidence_interval=[0.68, 0.95, 0.997])