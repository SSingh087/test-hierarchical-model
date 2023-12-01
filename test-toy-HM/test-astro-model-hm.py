import pymc3 as pm
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import arviz as az

SEEDS = 2000
Z_MAX = 5

# DEFINING DISTRIBUTIONS 

ALPHA_DISTRIBUTION_POP = 5
BETA_DISTRIBUTION_POP = 1
MU_DISTRIBUTION_POP = 1
SIGMA_DISTRIBUTION_POP = 6

def population_dist(z, alpha):
    return BETA_DISTRIBUTION_POP*st.invgamma.pdf(x=z, a=alpha, loc=MU_DISTRIBUTION_POP, scale=SIGMA_DISTRIBUTION_POP)

def merger_rate(z, alpha):
    # merger rate depends on the population distribution
    # the parameters of merger rate also depend on the 
    # parameter of the population distribution
    
    return population_dist(z, alpha)
    
z_events = np.random.choice(np.linspace(0, Z_MAX, SEEDS//2), size=SEEDS//2, p=merger_rate(np.linspace(0, Z_MAX, SEEDS//2), ALPHA_DISTRIBUTION_POP/2) / np.sum(merger_rate(np.linspace(0, Z_MAX, SEEDS//2), ALPHA_DISTRIBUTION_POP/2)))

LEN_DATA = 500
Z_SRC = np.empty((SEEDS//2, LEN_DATA))
for i in range(SEEDS//2):
    Z_SRC[i] = st.norm(loc=z_events[i], scale=0.5).rvs(size=LEN_DATA)
    
with pm.Model() as model:
    alpha_merger_rate = pm.Uniform('ALPHA', lower=ALPHA_DISTRIBUTION_POP/2-1, upper=ALPHA_DISTRIBUTION_POP/2+1)
    MU = pm.Uniform('MU', lower=MU_DISTRIBUTION_POP-1, upper=MU_DISTRIBUTION_POP+1)
    SIGMA = pm.Uniform('SIGMA', lower=SIGMA_DISTRIBUTION_POP-1, upper=SIGMA_DISTRIBUTION_POP+1)
    prior = pm.InverseGamma('mu', beta=BETA_DISTRIBUTION_POP, alpha=alpha_merger_rate, mu=MU, sigma=SIGMA)  # prior draws data points from hyperprior
    obs = pm.Normal('obs', mu=prior, sigma=0.5, observed=Z_SRC)  # likelihood draws data points from prior
    step = pm.Metropolis()
 
    # sample with 3 independent Markov chains
    trace = pm.sample(draws=500, chains=2, step=step, return_inferencedata=True, cores=4)    
    
az.summary(trace, var_names=["MU", "SIGMA", "ALPHA"])["mean"]
az.show   
