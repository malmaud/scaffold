"""
demo.py

A simple, illustrative example of using the scaffold.
Trivial case of beta-bernoulli model. There is only one unknown quantity (the true coin weight) which we do 'gibbs' sampling on (e.g., make iid draws from the beta posterior).

 In this demo, the true coin weight is .4.
"""

from __future__ import division
import scaffold
from scaffold import ParameterException
import helpers
from matplotlib.pylab import *

class CoinState(scaffold.State):
    __slots__ = ['p_heads']

    def __init__(self):
        super(CoinState, self).__init__()

    def __str__(self):
        s = "P(heads)=%.2f" % self.p_heads
        return s

    def get_state(self):
        return dict(p_heads=self.p_heads)

class CoinData(scaffold.DataSource):
    def __init__(self, **kwargs):
        super(CoinData, self).__init__(**kwargs)
        self.register("CoinData")

    def load(self):
        self.p_heads = self.params['p_heads']
        self.n_flips = self.params['n_flips']
        self.data = self.rng.rand(self.n_flips)<self.p_heads

CoinData.register("CoinData")

class CoinChain(scaffold.Chain):
    def __init__(self, **kwargs):
        super(CoinChain, self).__init__(**kwargs)
        try:
            self.n_iters = self.params['n_iter']
            self.prior_heads = self.params['prior_heads']
            self.prior_tails = self.params['prior_tails']
            self.start_mode = self.params.get('start_mode', 'from_prior')
        except KeyError:
            raise ParameterException("Chain missing needed parameters")

    def start_state(self):
        s = CoinState()
        if self.start_mode=='from_prior':
            s.p_heads = self.rng.beta(self.prior_heads, self.prior_tails)
        else:
            s.p_heads = .5
        return s

    def do_stop(self, state):
        return state.iter > self.n_iters #todo: off by one?

    def transition(self, prev_state):
        s = CoinState()
        coin_data = self.data
        n_heads = self.prior_heads + sum(coin_data==True)
        n_tails = self.prior_tails + sum(coin_data==False)
        s.p_heads = self.rng.beta(n_heads, n_tails)
        return s

    def summarize(self, history):
        trace = history.get_traces('p_heads')
        figure()
        trace.hist()
        posterior = helpers.save_fig_to_str()
        return dict(posterior=posterior)

    def __str__(self):
        s = []
        s.append("Start mode: %r" % self.start_mode)
        s.append("# of iterations: %r" % self.n_iters)
        s.append("Beta prior: (%r, %r)" % (self.prior_heads, self.prior_tails))
        return "\n".join(s)

CoinChain.register("CoinChain")

expt = scaffold.Experiment(run_mode = 'local')

expt.data_srcs = [dict(
    data_class='CoinData',
    p_heads=.4,
    n_flips=10)]

expt.methods = [dict(
    chain_class='CoinChain',
    n_iter=100,
    prior_heads=1,
    prior_tails = 1,
    start_mode='from_prior')]

expt.data_seeds = [0, 2]
expt.method_seeds = [0]

if __name__=="__main__":
    expt.run() #todo: this may not work because of pickling namespace issues