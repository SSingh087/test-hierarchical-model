import numpy as np
import matplotlib.pyplot as plt
import emcee 
import warnings
import bilby
import corner
warnings.filterwarnings("ignore")


true_parent_mean = 3.0 # hyperparameter
true_parent_scale = .5 # hyperparameter

true_child_scale = 0.3

UPPER_BOUND = 4
LOWER_BOUND = 2

MU_P_LOW = 0
MU_P_HIGH = 5
SIGMA_P_LOW = 0.2
SIGMA_P_HIGH = 0.9

N_chains = 32

N_inj = 500
size_child = 500


true_parent_gaussian = np.random.normal(loc=true_parent_mean, scale=true_parent_scale, size=N_inj)

err_on_mean_child = np.expand_dims(np.random.randn(N_inj)*true_child_scale, axis=1)
true_child_gaussian = np.random.normal(loc=true_parent_gaussian[:, np.newaxis] + err_on_mean_child, scale=true_child_scale, size=(N_inj, size_child))

# adding selection effect
true_child_gaussian_observed = []
for i in range(len(true_parent_gaussian)):
    if (true_child_gaussian[i].mean() > LOWER_BOUND and 
        true_child_gaussian[i].mean() < UPPER_BOUND):
        true_child_gaussian_observed.append(true_child_gaussian[i])
        
true_child_gaussian_observed = np.array(true_child_gaussian_observed)
N_obs = true_child_gaussian_observed.shape[0]

priors = {"mu_p": bilby.core.prior.Uniform(MU_P_LOW, MU_P_HIGH, "mu_p"), 
          "sigma_p": bilby.core.prior.Uniform(SIGMA_P_LOW, SIGMA_P_HIGH, "sigma_p")}

def get_walker_pos(N_chains):
    walker_mu = []
    walker_sigma = []
    for _ in range(N_chains):
        walker_mu.append(priors["mu_p"].sample())
        walker_sigma.append(priors["sigma_p"].sample())
    return np.array(walker_mu), np.array(walker_sigma)

def model(params, data):
    parent_mean, parent_scale = params
    return np.mean((2 * np.pi * parent_scale**2)**-.5 * np.exp(-(data - parent_mean)**2 / (2 * parent_scale**2)), axis=1)

def alpha(params):
    parent_mean, parent_scale = params
    SAMPLES = 10**4
    distribution = np.random.normal(loc=parent_mean, scale=parent_scale, size=SAMPLES)
    population = distribution + np.random.randn(SAMPLES) * true_child_scale
    return np.count_nonzero((population > LOWER_BOUND) & (population < UPPER_BOUND)) / SAMPLES

def log_hyperprior(params):
    parent_mean, parent_scale = params
    if (0 < parent_mean < 5 and 
        0.2 < parent_scale < 0.9):
        return 0.0
    return -np.inf

def log_likelihood(params, data):
    return np.log(np.prod(model(params, data) / alpha(params)))

def log_probability(params, data):
    log_prior_val = log_hyperprior(params)
    if not np.isfinite(log_prior_val):
        return -np.inf
    return log_prior_val + log_likelihood(params, data) - N_inj + N_obs * np.log(N_inj)

# pos = np.array([true_parent_mean, true_parent_scale]) + 0.1 * np.random.randn(2) + 1e-6 * np.random.randn(N_chains, 2)
pos = np.array(get_walker_pos(N_chains)).transpose()
nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=[true_child_gaussian_observed])
sampler.run_mcmc(pos, 10000)

fig, axes = plt.subplots(2, figsize=(10, 4), sharex=True)
samples = sampler.get_chain()
labels = ["mu_p", "sigma_p"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)
axes[0].hlines(true_parent_mean, 0, len(samples), label="true value")
axes[1].hlines(true_parent_scale, 0, len(samples))
axes[-1].set_xlabel("step number")
axes[0].legend()
plt.savefig('chains.png')

plt.close()

flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
fig = corner.corner(
    flat_samples, labels=labels, truths=[true_parent_mean, true_parent_scale], title_fmt=".2E", show_titles=True, title_kwargs={'fontsize':12}
)
plt.savefig('corner.png')
plt.close()