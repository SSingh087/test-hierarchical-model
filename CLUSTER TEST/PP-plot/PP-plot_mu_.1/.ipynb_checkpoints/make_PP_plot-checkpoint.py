#!/usr/bin/env python

import numpy as np
import bilby
import h5py
import configparser
import argparse
import pandas as pd

RESULTS = []

parser = argparse.ArgumentParser()

parser.add_argument("--tot_events")
parser.add_argument("--mu_p_low")
parser.add_argument("--mu_p_high")
parser.add_argument("--sigma_p_low")
parser.add_argument("--sigma_p_high")

args = parser.parse_args()

tot_events = np.int32(args.tot_events)
mu_p_low = np.float64(args.mu_p_low)
mu_p_high = np.float64(args.mu_p_high)
sigma_p_low = np.float64(args.sigma_p_low)
sigma_p_high = np.float64(args.sigma_p_high)

priors = {"mu_p": bilby.core.prior.Uniform(mu_p_low, mu_p_high, "mu_p"), 
          "sigma_p": bilby.core.prior.Uniform(sigma_p_low, sigma_p_high, "sigma_p")}


hf = h5py.File('injections.h5', 'r')
injection_group = hf.get('injections')
injection_mu_p = np.array(injection_group.get('mu_p'))
injection_sigma_p = np.array(injection_group.get('sigma_p'))

hf.close()

for i in range(tot_events):
    injections = dict()
    sf = h5py.File(f'data/run_{i}.h5', 'r')
    g_sample = sf.get('samples')
    samples_mu_p = np.array(g_sample.get('mu_p'))
    samples_sigma_p = np.array(g_sample.get('sigma_p'))
    
    posterior = pd.DataFrame(dict({"mu_p": samples_mu_p, "sigma_p": samples_sigma_p}))
    injections["mu_p"] = injection_mu_p[i]
    injections["sigma_p"] = injection_sigma_p[i]
    
    result = bilby.result.Result(
        label="PPtest",
        injection_parameters=injections,
        posterior=posterior,
        search_parameter_keys=injections.keys(),
        priors=priors)

    RESULTS.append(result)
    sf.close()
    
bilby.result.make_pp_plot(RESULTS, filename=f"PP-Plot.png", confidence_interval=[0.68, 0.95, 0.997])