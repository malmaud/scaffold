"""
Implementation of some common procedurally-generated datasets
"""

from __future__ import division
from matplotlib.patches import Ellipse
from matplotlib.pylab import *
from scaffold import ProceduralDataSource, ParameterException
import helpers

class Cluster:
    """
    A Gaussian cluster parameterized by a mean vector and covariance matrix
    """
    def __init__(self, mu=None, cov=None):
        self.mu = None
        self.cov = None
        if mu is not None:
            self.mu = asarray(mu, 'd')
        if cov is not None:
            self.cov = asarray(cov, 'd')

    def dim(self):
        """
        :return: The cluster dimensionality
        """
        return len(self.mu)

    def sample_points(self, n, rng=random):
        """
        Sample points from the cluster

        :param n: The number of points
        :param rng: The random-number generator
        :type n: int
        :return: An *n* x *dim* array. Each row is a point; each column is a dimension.
        """
        return rng.multivariate_normal(self.mu, self.cov, size=int(n))

    def __hash__(self):
        return hash(str(self.mu)+str(self.cov))

    def __str__(self):
        return str(self.mu) + "," + str(self.cov)


class FiniteMixture(ProceduralDataSource):

    """
    A Gaussian finite mixture model
    """

    def __init__(self, **kwargs):
        super(FiniteMixture, self).__init__(**kwargs)

    def load_data(self):
        """
        Loads the latent variables and data implicitly given by the class's parameters (in *self.param*)

        Expected parameter keys:

        n_points
         Number of points in the dataset

        clusters
         A list of clusters of type :py:class:`Cluster`

        weights
         A list of mixing weights for each cluster in *clusters*
        """
        try:
            self.n_points = self.params['n_points']
            self.clusters = self.params['clusters']
            self.weights = asarray(self.params['weights'])
        except KeyError as error:
            raise ParameterException("Required finite mixture parameter not passed in: %r" % error)
        dim = self.clusters[0].dim()
        self.c = helpers.discrete_sample(self.weights, self.n_points, self.rng)
        self.data = empty((self.n_points, dim))
        for i, cluster in enumerate(self.clusters):
            idx = self.c==i
            n_in_cluster = int(sum(idx))
            self.data[idx] = cluster.sample_points(n_in_cluster, self.rng)

    def points_in_cluster(self, c):
        return self.data[self.c==c]

    def show(self):
        colors = helpers.circlelist(['red', 'blue', 'orange', 'green', 'yellow'])
        for c in range(len(self.clusters)):
            cluster = self.clusters[c]
            x = self.points_in_cluster(c)
            scatter(x[:, 0], x[:, 1], color=colors[c])
            width = cluster.cov[0, 0] * 2
            height = cluster.cov[1, 1] * 2
            e = Ellipse(cluster.mu, width, height, alpha=.5, color=colors[c])
            gca().add_artist(e)

    def llh_pred(self, x):
        pass #todo: implement this

class EmptyData(ProceduralDataSource):
    def __init__(self, **kwargs):
        super(EmptyData, self).__init__(**kwargs)

    def load_data(self):
        self.data = empty(0)

