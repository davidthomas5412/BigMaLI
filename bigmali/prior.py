import numpy as np
import scipy.interpolate as interpolate
import hmf


class MassPrior(object):
    """
    Class to represent mass prior distribution.
    Initialize with mass and probability arrays.
    Can both evaluate and sample from distribution.
    """

    def __init__(self, mass, prob):
        self.mass = mass
        self.prob = prob
        self.min = mass.min()
        self.max = mass.max()

        # have to add 0,1 samples for interpolation bounds
        cumsum = np.append(np.append(np.array([0]), np.cumsum(prob)), np.array([1]))
        masses = np.append(np.append(np.array([self.min - 1]), self.mass), np.array([self.max + 1]))
        self.inv_cdf = interpolate.interp1d(cumsum, masses)

    def logpdf(self, mass):
        if np.any(self.pdf(mass) <= 0):
            print 'foo'
        return np.log(self.pdf(mass))

    def pdf(self, mass):
        """
        Note: Assumes the mass is within range of the prior.
        """
        right_ind = np.minimum(np.searchsorted(self.mass, mass), len(self.mass) - 1)
        left_ind = right_ind - 1
        # find where we fall in interval between masses
        f = (mass - self.mass[left_ind]) / (self.mass[right_ind] - self.mass[left_ind])
        return f * self.prob[right_ind] + (1 - f) * self.prob[left_ind]

    def rvs(self, size=1):
        return self.inv_cdf(np.random.rand(size))


class TinkerPrior(object):
    """
    Class that assigns tinker prior (Tinker10 from hmf) to each redshift bin in grid.
    """
    def __init__(self, grid, h=0.73, mmin=10.2358590918, mmax=14.3277327776):
        self.grid = grid
        self.grid_to_prior = dict()
        self.min_mass = float("inf")
        self.max_mass = 0
        for z in grid.redshifts:
            mf = hmf.MassFunction(z=z, Mmin=mmin, Mmax=mmax,
                                  cosmo_model=hmf.cosmo.WMAP5,
                                  hmf_model=hmf.fitting_functions.Tinker10)
            self.grid_to_prior[z] = MassPrior(mf.m * h, mf.dndm / np.trapz(mf.dndm, x=mf.m * h))
            self.min_mass = min(np.min((mf.m * h).min()), self.min_mass)
            self.max_mass = max((mf.m * h).max(), self.max_mass)

    def fetch(self, z):
        return self.grid_to_prior[self.grid.snap(z)]

    def pdf(self, mass, z):
        return self.fetch(z).pdf(mass)

    def logpdf(self, mass, z):
        return self.fetch(z).logpdf(mass)

    def rvs(self, z, size=1):
        return self.fetch(z).inv_cdf(np.random.rand(size))
