"""
demo.py

A simple, illustrative example of using the scaffold.
Trivial case of beta-bernoulli model. There is only one unknown quantity (the true coin weight) which we do 'gibbs' sampling on (e.g., make iid draws from the beta posterior).

 In this demo, the true coin weight is .4.
"""

from __future__ import division
import scaffold
from scaffold import ParameterException, State
import helpers
import matplotlib.pyplot as plt
import numpy as np

class CoinState(scaffold.State):
    __slots__ = ['p_heads', 'p_tails']

    def summarize(self):
        self.p_tails = 1-self.p_heads

class CoinData(scaffold.DataSource):
    def __init__(self, **kwargs):
        super(CoinData, self).__init__(**kwargs)

    def load(self):
        self.p_heads = self.params['p_heads']
        self.n_flips = self.params['n_flips']
        self.data = self.rng.rand(self.n_flips)<self.p_heads

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
        p_heads = [state.p_heads for state in history.states]

        plt.figure()
        plt.hist(p_heads, normed=True)
        plt.title('Posterior distribution on P(heads)')
        plt.xlabel('P(heads)')
        plt.ylabel('Frequency')
        p_heads_hist = helpers.save_fig_to_str()

        plt.figure()
        plt.plot([state.iter for state in history.states], p_heads)
        plt.title('P(heads) trace')
        plt.xlabel('Iteration')
        plt.ylabel('P(heads) value')
        p_heads_trace = helpers.save_fig_to_str()

        p_heads_mean = np.mean(p_heads)

        return dict(p_heads_trace=p_heads_trace, p_heads_hist=p_heads_hist, p_heads_mean=p_heads_mean)

expt = scaffold.Experiment(run_mode = 'local')
expt.data_srcs = [dict(data_class='CoinData', p_heads=.4, n_flips=100)]
expt.methods = [dict(chain_class='CoinChain', n_iter=100, prior_heads=1, prior_tails = 1, start_mode='from_prior')]
expt.data_seeds = [0, 2]
expt.method_seeds = [0, 1]
expt.data_src_classes['CoinData'] = CoinData
expt.chain_classes['CoinChain'] = CoinChain

if __name__=="__main__":
    expt.run()
