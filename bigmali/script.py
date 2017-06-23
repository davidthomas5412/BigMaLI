from bigmali.grid import Grid
from bigmali.prior import TinkerPrior
import numpy as np
import pandas as pd
from scipy.stats import lognorm
from hyperparameter import get


prior = TinkerPrior(Grid())
print prior.pdf(12760674392.2, 0)
import sys
sys.exit()

data = pd.read_csv('/Users/user/Code/PanglossNotebooks/MassLuminosityProject/mock_data.csv')

prior = TinkerPrior(Grid())

idx = 0
true_mass = data.mass.ix[idx]
true_z = data.z.ix[idx]
true_lum = data.lum.ix[idx]
true_lum_obs = data.lum_obs.ix[idx]
true_lum_obs_collection = data.lum_obs
sigma = 0.05

def log10(arr):
    return np.log(arr) / np.log(10)

def p1(lobs, lum, sigma):
    return fast_lognormal(lum, sigma, lobs)


def p2(lum, mass, a1, a2, a3, a4, S, z):
    mu_lum = np.exp(a1) * ((mass / a3) ** a2) * ((1 + z) ** (a4))
    return fast_lognormal(mu_lum, S, lum)


def p3(mass, z):
    return prior.fetch(z).pdf(mass)


def q1(lum, lobs, sigma):
    return fast_lognormal(lobs, sigma, lum)


def q2(mass, lum, a1, a2, a3, a4, S, z):
    mu_mass = a3 * (lum / (np.exp(a1) * (1 + z) ** a4)) ** (1 / a2)
    return fast_lognormal(mu_mass, S, mass)

def midpoints(arr):
    n = len(arr)-1
    ret = np.zeros(n)
    for i in xrange(n):
        ret[i] = (arr[i+1] + arr[i]) / 2.
    return ret

def fast_lognormal(mu, sigma, x):
    return  (1/(x * sigma * np.sqrt(2 * np.pi))) * np.exp(- 0.5 * (np.log(x) - np.log(mu)) ** 2 / sigma ** 2)

@profile
def numerical_integration(a1, a2, a3, a4, S, nsamples=10**3):
    masses = midpoints(prior.fetch(true_z).mass[1:])
    delta_masses = np.diff(prior.fetch(true_z).mass[1:])
    lums_tmp = np.logspace(log10(np.min(data.lum_obs)), log10(np.max(data.lum_obs)), nsamples)
    lums = midpoints(lums_tmp)
    delta_lums = np.diff(lums_tmp)
    integral = 0
    for i,lum in enumerate(lums):
        integral += np.sum(delta_masses * delta_lums[i] * p1(true_lum_obs, lum, sigma) * \
            p2(lum, masses, a1, a2, a3, a4, S, true_z) * p3(masses, true_z))
    return integral

@profile
def simple_monte_carlo_integration(a1, a2, a3, a4, S, nsamples=10**6):
    masses = prior.fetch(true_z).rvs(nsamples)
    mu_lum = np.exp(a1) * ((masses / a3) ** a2) * ((1 + true_z) ** (a4))
    lums = lognorm(S, scale=mu_lum).rvs()
    return np.sum(p1(true_lum_obs, lums, sigma)) / (nsamples)

@profile
def importance_sampling_integration(a1, a2, a3, a4, S, nsamples=10**6):
    rev_S = 5.6578015811698101 * S
    lums = lognorm(sigma, scale=true_lum_obs).rvs(size=nsamples)
    mu_mass = a3 * (lums / (np.exp(a1) * (1 + true_z) ** a4)) ** (1 / a2)
    masses = lognorm(rev_S, scale=mu_mass).rvs()
    pp1 = p1(true_lum_obs, lums, sigma)
    pp2 = p2(lums, masses, a1, a2, a3, a4, S, true_z)
    pp3 = p3(masses, true_z)
    qq1 = q1(lums, true_lum_obs, sigma)
    qq2 = q2(masses, lums, a1, a2, a3, a4, rev_S, true_z)
    integral = np.sum((pp1 * pp2 * pp3) / (qq1 * qq2)) / len(lums)
    return integral


a1,a2,a3,a4,S = get()

# numerical_integration(a1,a2,a3,a4,S)
# simple_monte_carlo_integration(a1,a2,a3,a4,S, nsamples=10**3)
importance_sampling_integration(a1,a2,a3,a4,S, nsamples=10**7)