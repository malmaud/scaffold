from __future__ import division
import sys

from numpy import *
from scipy import stats
from scipy import special
import datasources
from datasources import BinomialCluster
import scaffold
import helpers
import runner


gammaln = special.gammaln
betaln = special.betaln


class State(scaffold.State):
    __slots__ = ['alpha', 'beta', 'c']


class BinoChain(scaffold.Chain):
    def start_state(self, params, data_params, rng):
        return self.sample_latent(params, data_params, rng)

    def sample_alpha(self, state, params, n, rng):
        n_clusters = len(unique(state.c))

        def calc_alpha_llh(alpha):
            prior = stats.gamma.logpdf(alpha, params['alpha_shape'], scale=params['alpha_scale'])
            lh = gammaln(alpha) + n_clusters * log(alpha) - gammaln(alpha + n)
            return prior + lh

        grid = linspace(.1, 10, 1000)
        alpha_llh = calc_alpha_llh(grid)
        alpha = grid[helpers.discrete_sample(alpha_llh, rng=rng, log_mode=True)]
        return alpha

    def transition(self, state, params, data, rng):
        n = len(data)
        s = State()
        s.alpha = self.sample_alpha(state, params, n, rng)
        c = copy(state.c)
        for i in range(n):
            c[i] = self.sample_c(i, c, params, data, s.alpha, state.beta, rng)
        s.c = c
        s.beta = self.sample_beta(state, rng)
        return s

    def sample_beta(self, state, rng):
        return state.beta

    def sample_c(self, i, c, params, data, dp_alpha, beta, rng, debug=False):
        c_diff = delete(c, i)
        cluster_ids = unique(c_diff)
        n_clusters = len(cluster_ids)
        p = zeros(n_clusters + 1)
        x = data[i].astype(int)
        alpha_set = []
        beta_set = []
        count_set = []
        for j, cluster_id in enumerate(cluster_ids):
            count = sum(c_diff == cluster_id)
            prior = log(count)
            c_in = (c == cluster_id)
            c_in[i] = False
            alpha = beta + sum(data[c_in] == True, 0)
            beta = beta + sum(data[c_in] == False, 0)
            lh = sum(betaln(alpha + x, beta + (1-x)) - betaln(alpha, beta))
            p[j] = lh + prior
        prior = log(dp_alpha)
        lh =sum(betaln(beta + x, beta - x + 1) - betaln(beta, beta))
        p[-1] = prior+lh
        if debug:
            p_conv = exp(p)
            p_conv /= sum(p_conv)
        idx = helpers.discrete_sample(p, rng=rng, log_mode=True)
        if idx == len(p) - 1:
            c_return= cluster_ids[-1] + 1
        else:
            c_return = cluster_ids[idx]
        if debug:
            return c_return, (p_conv, alpha_set, beta_set, x, count_set)
        else:
            return c_return


    def sample_data(self, state, params, data_params, rng):
        _, c = unique(state.c, return_inverse=True)
        n_clusters = len(unique(c))
        clusters = rng.beta(state.beta, state.beta, size=n_clusters)
        n = data_params['n']
        dim = data_params['dim']
        x = zeros((n, dim), bool)
        for i in range(n):
            cluster = clusters[c[i]]
            x[i] = rng.random_sample(size=dim) < cluster
        return x

    def sample_latent(self, params, data_params, rng):
        s = State()
        s.alpha = rng.gamma(params['alpha_shape'], scale=params['alpha_scale'])
        #s.beta = rng.gamma(params['beta_shape'], scale=params['beta_scale'])
        s.beta = 1
        n = data_params['n']
        c = zeros(n, int)
        for i in range(1, n):
            c_before = c[:i]
            cluster_ids = unique(c_before)
            p = zeros(len(cluster_ids)+1)
            for j, cluster_id in enumerate(cluster_ids):
                p[j] = sum(cluster_ids==cluster_id)
            p[-1] = s.alpha
            c[i] = helpers.discrete_sample(p, rng=rng)
        s.c = c
        return s


cluster1 = BinomialCluster([.1, .8, .1, .8])
cluster2 = BinomialCluster([.8, .1, .8, .1])

chain = BinoChain(alpha=1, beta=1, alpha_shape=1, alpha_scale=1, beta_shape=2, beta_scale=1/2)
dp = dict(n=10, dim=5)

fm = datasources.FiniteMixture()

expt = runner.Experiment()
expt.data_seeds = [0]
expt.method_seeds =[0]
expt.data_srcs = [dict(data_class='FiniteMixture', clusters=(cluster1, cluster2),
                               n_points=20,
                               weights=(.5, .5))]
expt.methods = [dict(chain_class='BinoChain', alpha=1, beta=1, alpha_shape=1, alpha_scale=1, beta_shape=2, beta_scale=1/2, max_iters=1000)]

def run_tests(n=1000):
    tests = [lambda state: state.alpha, lambda state: len(unique(state.c)),
             lambda state: state.beta, lambda state: state.alpha*state.beta,
             lambda state: state.alpha**2]
    z = chain.geweke_test(n, dp, tests)
    return z

def run_expt():
    expt.run()
    results = expt.fetch_results()
    h = results[0]
    return h

if __name__=="__main__":
    if len(sys.argv)>1:
        n = int(sys.argv[1])
    else:
        n = 1000
    run(n)