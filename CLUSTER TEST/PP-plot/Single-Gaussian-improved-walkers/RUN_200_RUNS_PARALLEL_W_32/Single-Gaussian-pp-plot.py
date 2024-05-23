#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import emcee
import warnings
import corner
import pandas as pd
import bilby
import h5py
import configparser
import argparse

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument("--n_event")
parser.add_argument("--mu_p_low")
parser.add_argument("--mu_p_high")
parser.add_argument("--sigma_p_low")
parser.add_argument("--sigma_p_high")

args = parser.parse_args()


# Get event number
n_event = np.int32(args.n_event)

# get injection value specific to the event number
hf = h5py.File('injections.h5', 'r')
injection_group = hf.get('injections')
injection_mu_p = np.array(injection_group.get('mu_p'))[n_event]
injection_sigma_p = np.array(injection_group.get('sigma_p'))[n_event]

# Get the prior values
MU_P_LOW = np.float64(args.mu_p_low)
MU_P_HIGH = np.float64(args.mu_p_high)
SIGMA_P_LOW = np.float64(args.sigma_p_low)
SIGMA_P_HIGH = np.float64(args.sigma_p_high)

# set the priors
priors = {"mu_p": bilby.core.prior.Uniform(MU_P_LOW, MU_P_HIGH, "mu_p"), 
          "sigma_p": bilby.core.prior.Uniform(SIGMA_P_LOW, SIGMA_P_HIGH, "sigma_p")}


N_obs = 500
size_child = 100
true_child_scale = 0.5
N_chains = 32

def log_hyperprior(params):
    parent_mean, parent_scale = params
    if (MU_P_LOW < parent_mean < MU_P_HIGH and 
        SIGMA_P_LOW < parent_scale < SIGMA_P_HIGH):
        return 0.0
    return -np.inf

def log_likelihood(params, data):
    parent_mean, parent_scale = params
    return np.log(np.prod(np.mean((2 * np.pi * parent_scale**2)**-.5 * np.exp(-(data - parent_mean)**2 / (2 * parent_scale**2)), axis=1)))

def log_probability(params, data):
    log_prior_val = log_hyperprior(params)
    if not np.isfinite(log_prior_val):
        return -np.inf
    return log_prior_val + log_likelihood(params, data)


def get_walker_pos(N_chains):
    # Assigning random posoition around the true
    walker_mu = []
    walker_sigma = []
    for _ in range(N_chains):
        walker_mu.append(priors["mu_p"].sample())
        walker_sigma.append(priors["sigma_p"].sample())
    
    return np.array(walker_mu), np.array(walker_sigma)


true_parent_gaussian = np.random.normal(loc=injection_mu_p, scale=injection_sigma_p, size=N_obs)
err_on_mean_child = np.expand_dims(np.random.randn(N_obs)*true_child_scale, axis=1)
true_child_gaussian = np.random.normal(loc=true_parent_gaussian[:, np.newaxis] + err_on_mean_child, scale=true_child_scale, size=(N_obs, size_child))

pos = np.array(get_walker_pos(N_chains)).transpose()
nwalkers, ndim = pos.shape

# with Pool() as pool:
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=[true_child_gaussian])
sampler.run_mcmc(pos, 10000)

flat_samples = sampler.get_chain(discard=500, thin=15, flat=True)

hf = h5py.File(f'data/run_{n_event}.h5', 'w')

g_data = hf.create_group('data')
g_data.create_dataset('data', data=true_child_gaussian)
g_data.create_dataset('error_on_data', data=err_on_mean_child)
g_data.create_dataset('hyper_data', data=true_parent_gaussian)

g_sample = hf.create_group('samples')
g_sample.create_dataset('mu_p', data=flat_samples[:, 0])
g_sample.create_dataset('sigma_p', data=flat_samples[:,1])

hf.close()

del hf

fig = corner.corner(flat_samples, labels=["mu_p", "sigma_p"], truths=[injection_mu_p, injection_sigma_p], title_fmt=".2E", show_titles=True, title_kwargs={'fontsize':12})
plt.savefig('corner-plot/run_'+str(n_event)+'.png')
plt.close()

fig, axes = plt.subplots(2, figsize=(10, 4), sharex=True)
samples = sampler.get_chain()
labels = ["mu_p", "sigma_p"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)
axes[0].hlines(injection_mu_p, 0, len(samples), label="true value")
axes[1].hlines(injection_sigma_p, 0, len(samples))
axes[-1].set_xlabel("step number")
axes[0].legend()
plt.savefig('chains/run_'+str(n_event)+'.png')
plt.close()
