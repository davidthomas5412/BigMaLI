import numpy as np


class Grid(object):
    """
    Manages redshift bins.
    """

    def __init__(self, mmin=0, mmax=3.5, nbins=20):
        self.redshifts = np.linspace(mmin, mmax, nbins)
        self.nbins = nbins

    def snap(self, z):
        ind = np.minimum(self.nbins - 1, np.searchsorted(self.redshifts, z))
        return self.redshifts[ind]
