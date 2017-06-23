import numpy as np
import scipy.interpolate as interpolate
import hmf

# class DataPrior(object):
#     def __init__(self, mass):
#         space = np.logspace(mass.min(), mass.max(), 1000)
#         x,y = np.histogram(mass, space, normed=True)
#         self.mass = np.append(np.append(, mass))
#         probex = np.append(np.append(, y))
#         self.prob = probex / np.trapz(probex, x=self.mass)
#         cumsum = np.cumsum(self.prob) / np.sum(self.prob)
#         self.inv_cdf = interpolate.interp1d(cumsum, self.mass)
#
#     def logpdf(self, mass):
#         return np.log(self.pdf(mass))
#
#     def pdf(self, mass):
#         """
#         Note: Assumes the mass is within range of the prior.
#         """
#         right_ind = np.minimum(np.searchsorted(self.mass, mass), len(self.mass) - 1)
#         left_ind = right_ind - 1
#         # find where we fall in interval between masses (similar to trapezoidal integration)
#         f = (mass - self.mass[left_ind]) / (self.mass[right_ind] - self.mass[left_ind])
#         return f * self.prob[right_ind] + (1 - f) * self.prob[left_ind]
#
#     def rvs(self, size=1):
#         return self.inv_cdf(np.random.rand(size))

class MassPrior(object):
    """
    Class to represent mass prior distribution.
    Initialize with mass and probability arrays.
    Can both evaluate and sample from distribution.
    """
    def __init__(self, mass, prob):
        self.prob = prob
        # add epsilon background
        epsilon = 1e-30
        self.mass = np.append(np.append([1, mass.min()-1], mass), [mass.max()+1, 100 * mass.max()])
        deps = [epsilon, epsilon]
        probex = np.append(np.append(deps, prob), deps)
        self.prob = probex / np.trapz(probex, x=self.mass)
        # need to normalize by prob values for interpolation because normalized against
        # trapezoidal integration
        cumsum = np.cumsum(self.prob) / np.sum(self.prob)
        self.inv_cdf = interpolate.interp1d(cumsum, self.mass)

    def logpdf(self, mass):
        return np.log(self.pdf(mass))

    def pdf(self, mass):
        """
        Note: Assumes the mass is within range of the prior.
        """
        right_ind = np.minimum(np.searchsorted(self.mass, mass), len(self.mass) - 1)
        left_ind = right_ind - 1
        # find where we fall in interval between masses (similar to trapezoidal integration)
        f = (mass - self.mass[left_ind]) / (self.mass[right_ind] - self.mass[left_ind])
        return f * self.prob[right_ind] + (1 - f) * self.prob[left_ind]

    def rvs(self, size=1):
        return self.inv_cdf(np.random.rand(size))

    @staticmethod
    def default(h=0.73, mmin=10.2358590918, mmax=14.3277327776, z=0):
        mf = hmf.MassFunction(z=z, Mmin=mmin, Mmax=mmax,
                              cosmo_model=hmf.cosmo.WMAP5,
                              hmf_model=hmf.fitting_functions.Tinker10)
        return MassPrior(mf.m * h, mf.dndm)


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
        return self.fetch(z).rvs(size=size)
