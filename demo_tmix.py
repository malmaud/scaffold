from __future__ import division

import scaffold
import runner
import helpers
import datasources
from scipy import stats
from numpy import *


class State(scaffold.State):
    __slots__ = ['s', 'omega', 'var', 'p', 'mu']


class Chain(scaffold.Chain):
    def start_state(self, params, data_params, rng):
        s = self.sample_latent(params, data_params, rng)
        return s

    def transition(self, state, params, data, rng):
        s = State()
        s.p = rng.beta(params['alpha'] + sum(state.s == 1), params['beta'] + sum(state.s == 0))
        n = len(data)
        s.s = empty(n)
        for i in range(n):
            p_cluster = empty(2)
            for j in range(2):
                if j==0:
                    p = s.p
                else:
                    p = 1-s.p
                lh = stats.norm(state.mu[j], sqrt(state.var[j])).pdf(data[i])
                prior = p
                p_cluster[j] = lh * prior
            s.s[i] = helpers.discrete_sample(p_cluster)

        s.mu = empty(2)
        for j in range(2):
            n_j = sum(s.s==j)
            mu_prime = params['mu0'] / params['sigma0'] ** 2 + sum(data[s.s == j]) / state.var[j]
            mu_prime /= (1 / params['sigma0'] ** 2 + n_j / state.var[j])
            prec_prime = 1/params['sigma0']**2 + n_j/state.var[j]
            s.mu[j] = rng.normal(mu_prime, sqrt(1/prec_prime))
            assert not isnan(s.mu[j])
        s.omega = state.omega
        s.var = state.var

        return s


    def sample_data(self, state, params, data_params, rng):
        n = data_params['n']
        x = empty(n)
        for i in range(n):
            mu = state.mu[state.s[i]]
            var = state.var[state.s[i]]
            x[i] = rng.normal(mu, sqrt(var/state.omega[i]))
        return x

    def sample_latent(self, params, data_params, rng):
        s = State()
        s.p = rng.beta(params['alpha'], params['beta'])
        s.mu = rng.normal(params['mu0'], params['sigma0'], size=2)
        #s.var = 1 / rng.gamma(params['sigma_s'], params['sigma_c'], size=2)
        #s.omega = rng.chisquare(params['ups'], size=data_params['n'])
        s.var = ones(2)
        s.omega = ones(data_params['n'])
        s.s = (rng.rand(data_params['n']) < s.p).astype(int)
        return s


chain = Chain(mu0=0, sigma0=3, alpha=5, beta=5, ups=5, sigma_s=1, sigma_c=1, max_iters=10)
dp = dict(n=6)

x = empty(100)
rng = random.RandomState(0)
x[0:50] = rng.normal(10, 1, size=50)
x[50:] = rng.normal(-5, 2, size=50)
chain.data = x


def run():
    z = chain.geweke_test(1000, dict(n=5), [
        lambda state: mean(state.mu),
        lambda state: sum(state.s==0),
        lambda state: state.p
    ])
    return z


if __name__=="__main__":
    run()
