from scipy.stats import norm
import numpy as np


def get(seed=0):
    np.random.seed(seed)
    alpha1 = norm(10.709, 0.022).rvs()
    alpha2 = norm(0.359, 0.009).rvs()
    alpha3 = 2.35e14
    alpha4 = norm(1.10, 0.06).rvs()
    S = S = norm(0.155, 0.0009).rvs()
    return [alpha1, alpha2, alpha3, alpha4, S]
