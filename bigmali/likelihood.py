import numpy as np
from scipy.misc import logsumexp


class BiasedLikelihood(object):
    """
    Class uses importance sampling to compute mass-luminosity likelihood.
    """
    def __init__(self, grid, prior, lum_obs, z):
        self.grid = grid
        self.prior = prior
        self.lum_obs = lum_obs
        self.z = z
        self.sigma_l = 0.05
        self.s_m_mult = 5.6578015811698101

    @staticmethod
    def fast_log_lognorm(mu, sigma, x):
        return -np.log(x * sigma * np.sqrt(2 * np.pi)) \
               - 0.5 * ((np.log(x) - np.log(mu)) ** 2 / sigma ** 2)

    def evaluate(self, alpha1, alpha2, alpha3, alpha4, s, nsamples=100):
        s_m = self.s_m_mult * s
        out = 0
        for i, (lum_obs, snapz) in enumerate(zip(self.lum_obs, self.z)):
            lum_samples = np.random.lognormal(mean=np.log(lum_obs), sigma=self.sigma_l,
                                              size=nsamples)
            mu_mass = alpha3 * (lum_samples / (np.exp(alpha1) * (1 + snapz) ** alpha4)) ** \
                               (1 / alpha2)
            mass_samples = np.maximum(np.minimum(np.random.lognormal(mean=np.log(mu_mass),
                                      sigma=s_m, size=nsamples), self.prior.max_mass - 1),
                                      self.prior.min_mass + 1)
            mu_lum = np.exp(alpha1) * ((mass_samples / alpha3) ** alpha2) * ((1 + snapz) ** (
                alpha4))
            v1 = BiasedLikelihood.fast_log_lognorm(lum_samples, self.sigma_l, lum_obs)
            v2 = BiasedLikelihood.fast_log_lognorm(mu_lum, s, lum_samples)
            v3 = self.prior.logpdf(mass_samples, snapz)
            v4 = BiasedLikelihood.fast_log_lognorm(lum_obs, self.sigma_l, lum_samples)
            v5 = BiasedLikelihood.fast_log_lognorm(mu_mass, s_m, mass_samples)
            out += logsumexp(v1 + v2 + v3 - v4 - v5) - np.log(nsamples)
        return out
