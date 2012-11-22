"""
datasources.py
Implementation of some common procedurally-generated datasets
"""

from __future__ import division
from numpy import *
from scaffold import DataSource, ParameterException
import util

class Cluster:
    """
    A Gaussian cluster parameterized by a mean vector and covariance matrix
    """
    def __init__(self, mu=None, cov=None):
        self.mu = None
        self.cov = None
        if mu is not None:
            self.mu = asarray(mu, double)
        if cov is not None:
            self.cov = asarray(cov, double)

    def dim(self):
        return len(self.mu)

class FiniteMixture(DataSource):
    """
    A Gaussian finite mixture model
    """
    def load(self):
        try:
            self.n_points = self.params['n_points']
            self.clusters = self.params['clusters']
            self.weights = asarray(self.params['weights'])
        except KeyError as error:
            raise ParameterException("Required finite mixture parameter not passed in: %r" % error)
        dim = self.clusters[0].dim()
        self.c = util.sample(self.weights, self.n_points, self.rng)
        self.data = empty((self.n_points, dim))
        for i, cluster in enumerate(self.clusters):
            idx = self.c==i
            cluster = self.clusters[i]
            n_in_cluster = int(sum(idx))
            self.data[idx] = self.rng.multivariate_normal(cluster.mu, cluster.cov,  size=n_in_cluster)

