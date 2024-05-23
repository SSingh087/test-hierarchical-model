#!/usr/bin/env python

import numpy as np
import bilby
import configparser
import argparse
import h5py

parser = argparse.ArgumentParser()

parser.add_argument("--tot_events")
parser.add_argument("--mu_p_low")
parser.add_argument("--mu_p_high")
parser.add_argument("--sigma_p_low")
parser.add_argument("--sigma_p_high")

args = parser.parse_args()

tot_events = np.int64(args.tot_events)
mu_p_low = np.float64(args.mu_p_low)
mu_p_high = np.float64(args.mu_p_high)
sigma_p_low = np.float64(args.sigma_p_low)
sigma_p_high = np.float64(args.sigma_p_high)


print("tot_events: ", tot_events)
print("mu_p_low: ", mu_p_low)
print("mu_p_high: ", mu_p_high)
print("sigma_p_low: ", sigma_p_low)
print("sigma_p_high: ", sigma_p_high)

priors = {"mu_p": bilby.core.prior.Uniform(mu_p_low, mu_p_high, "mu_p"), 
          "sigma_p": bilby.core.prior.Uniform(sigma_p_low, sigma_p_high, "sigma_p")}

injections_mu_p = []
injections_sigma_p = []
for i in range(tot_events):
    injections_mu_p.append(priors["mu_p"].sample())
    injections_sigma_p.append(priors["sigma_p"].sample())

injections_mu_p = np.array(injections_mu_p)
injections_sigma_p = np.array(injections_sigma_p)

hf = h5py.File('injections.h5', 'w')

g_inject = hf.create_group('injections')
g_inject.create_dataset('mu_p', data=injections_mu_p)
g_inject.create_dataset('sigma_p', data=injections_sigma_p)

hf.close()

print("Injection complete !")