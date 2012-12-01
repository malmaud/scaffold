"""
test_suite.py

Nose unit tests
"""

from __future__ import division
from numpy import *
from matplotlib.pyplot import *
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

def test_discrete_sample():
    from helpers import discrete_sample
    w = asarray([3, 6, 2], 'd')

    def bin_samples(samples):
        w_sampled = bincount(samples, minlength=len(w)) / len(samples)
        return w_sampled

    r = w/sum(w)
    rng = random.RandomState(0)
    samples = discrete_sample(w, 1e5, rng=rng)
    w_sampled = bin_samples(samples)
    delta = .05
    assert_almost_equal(w_sampled[0], r[0], delta=delta)
    assert_almost_equal(w_sampled[1], r[1], delta=delta)
    assert_almost_equal(w_sampled[2], r[2], delta=delta)

    samples = discrete_sample(log(w), 1e5, rng=rng, log_mode=True)
    w_sampled = bin_samples(samples)
    delta = .05
    assert_almost_equal(w_sampled[0], r[0], delta=delta)
    assert_almost_equal(w_sampled[1], r[1], delta=delta)
    assert_almost_equal(w_sampled[2], r[2], delta=delta)

    samples = discrete_sample(w, 1e5, rng, temperature=10000)
    r = repeat(1/len(w), len(w))
    w_sampled = bin_samples(samples)
    delta = .05
    assert_almost_equal(w_sampled[0], r[0], delta=delta)
    assert_almost_equal(w_sampled[1], r[1], delta=delta)
    assert_almost_equal(w_sampled[2], r[2], delta=delta)

    samples = discrete_sample(w, 1e5, rng, temperature=.01)
    w_sampled = bin_samples(samples)
    r = zeros_like(w)
    r[argmax(w)] = 1
    delta = .05
    assert_almost_equal(w_sampled[0], r[0], delta=delta)
    assert_almost_equal(w_sampled[1], r[1], delta=delta)
    assert_almost_equal(w_sampled[2], r[2], delta=delta)

def test_data_store():
    from storage import LocalStore, CloudStore
    obj_in = [1, 2, ('a', 'b'), 3]
    local = LocalStore()
    local.store(obj_in, 'test')
    local.close()
    local = LocalStore()
    obj_out = local.load('test')
    assert obj_in==obj_out
    cloud_store = CloudStore()
    cloud_store.store(obj_in, 'test')
    obj_out = cloud_store.load('test')
    assert obj_in==obj_out

def test_remote_figures():
    from scaffold import History
    from helpers import save_fig_to_str
    h = History()
    ioff()
    plot([1,2],[3,4])
    s = save_fig_to_str()
    h.summary = dict(test_figure=s)
    h.show_fig('test_figure')
