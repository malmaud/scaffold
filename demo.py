"""
demo.py

A simple, illustrative example of using the scaffold.
Trivial case of beta-bernoulli model. There is only one unknown quantity (the true coin weight) which we do 'gibbs' sampling
on (e.g., make iid draws from the beta posterior.) 
"""

from __future__ import division
import scaffold
from scaffold import ParameterException

class CoinData(scaffold.DataSource):
    def __init__(self, **kwargs):
        super(CoinData, self).__init__(**kwargs)

    def load(self):
        self.p_heads = self.params['p_heads']
        self.n_flips = self.params['n_flips']
        self.data = self.rng.rand(self.n_flips)>self.p_heads

class State(scaffold.State):
    def __init__(self):
        super(State, self).__init__()

class Chain(scaffold.Chain):
    def __init__(self, **kwargs):
        super(Chain, self).__init__(**kwargs)
        try:
            self.n_iters = self.params['n_iter']
            self.prior_heads = self.params['prior_heads']
            self.prior_tails = self.params['prior_tails']
            self.start_mode = self.params.get('start_mode', 'from_prior')
        except KeyError:
            raise ParameterException("Chain missing needed parameters")

    def start_state(self):
        s = State()
        if self.start_mode=='from_prior':
            s.p_heads = self.rng.beta(self.prior_heads, self.prior_tails)
        else:
            s.p_heads = .5
        return s

    def do_stop(self, state):
        return state.iter > self.n_iters

    def transition(self, prev_state):
        s = State()
        coin_data = self.data()
        n_heads = self.prior_heads + sum(coin_data.flips==True)
        n_tails = self.prior_tails + sum(coin_data.flips==False)
        s.p_heads = self.rng.beta(n_heads, n_tails)
        return s

if __name__=="__main__":
    expt = scaffold.Experiment()
    expt.run()