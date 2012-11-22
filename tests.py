"""
tests.py
Nose unit tests
"""

from __future__ import division
from numpy import *
from nose.tools import assert_almost_equal

def test_namespace():
    """
    Is the top-level namespace exposing the right symbols?
    Can the key classes be created?
    """
    import scaffold
    scaffold.State()
    scaffold.History()
    scaffold.Chain()
    scaffold.DataSource()
    scaffold.Experiment()

def test_datasource():
    from datasources import FiniteMixture, Cluster
    c1 = Cluster([1, 1], [[1, 0], [0, 1]])
    c2 = Cluster([-5, -3], [[4, 1], [1, 3]])
    d = FiniteMixture()
    d.init(seed=1, clusters=[c1,c2], n_points=50, weights=[.7, .3], test_fraction=.3)
    n_total = d.size()
    n_train = len(d.train_data())
    n_test = len(d.test_data())
    assert n_total == 50
    assert n_total == n_train+n_test
    assert int(.3*n_total)==n_test

def test_sample():
    from util import sample
    w = asarray([3, 6, 2], double)
    r = w/sum(w)
    rng = random.RandomState(0)
    samples = sample(w, 1e5, rng)
    w_sampled = bincount(samples)/len(samples)
    delta = .05
    assert_almost_equal(w_sampled[0], r[0], delta=delta)
    assert_almost_equal(w_sampled[1], r[1], delta=delta)
    assert_almost_equal(w_sampled[2], r[2], delta=delta)
