"""
demo.py

A simple, illustrative example of using the scaffold.
Trivial case of beta-bernoulli model. There is only one unknown quantity (the true coin weight)
which we do 'gibbs' sampling on (e.g., make iid draws from the beta posterior).

"""

from __future__ import division
import pandas
from runner import Experiment
import scaffold
from scaffold import ParameterException
import helpers
from matplotlib.pylab import *
import datasources


class CoinState(scaffold.State):
    __slots__ = ['p_heads', 'p_tails']

    def __init__(self):
        super(CoinState, self).__init__()

    def __str__(self):
        s = "P(heads)=%.2f" % self.p_heads
        return s

    def summarize(self):
        self.p_tails = 1 - self.p_heads # Trivial example of computing summary information on the cloud

    def show(self, **kwargs):
        s = pandas.Series([self.p_heads, self.p_tails], index=['Heads', 'Tails'])
        s.plot(kind='bar')
        ylabel('Probability')
        yticks(arange(6) * .2)

    def sample_data(self, n_data, rng):
        self.data = rng.rand(n_data) < self.p_heads


class CoinData(scaffold.DataSource):
    def __init__(self, **kwargs):
        super(CoinData, self).__init__(**kwargs)
        try:
            self.p_heads = kwargs['p_heads']
            self.n_flips = kwargs['n_flips']
        except KeyError as err:
            raise ParameterException('CoinData is missing needed parameters: %r' % err)

    def load(self, params, rng):
        #todo: make more explicit
        self.data = rng.rand(params['n_flips']) < params['p_heads']

    def __str__(self):
        s = []
        s.append("Binomial data source, p=%.2f, n=%d" % (self.p_heads, self.n_flips))
        if self.loaded:
            s.append("Data: %r" % self.data)
        return "\n".join(s)

    def show(self, **kwargs):
        show_latents = kwargs.get('show_latents', False)
        plot_args = kwargs.get('plot_args', {})
        if show_latents:
            fig, (ax_counts, ax_latents) = subplots(2, 1)
        else:
            fig, ax_counts = subplots(1, 1)
        n_heads = sum(self.data == True)
        n_tails = sum(self.data == False)
        s = pandas.Series([n_heads, n_tails], index=['# Heads', '# Tails'])
        sca(ax_counts)
        s.plot(kind='bar', **plot_args)
        if show_latents:
            sca(ax_latents)
            s_latent = pandas.Series([self.p_heads], index=['P(heads)'])
            s_latent.plot(kind='bar', **plot_args)
            yticks(linspace(0, 1, 5))
        fig.subplots_adjust(hspace=1)


class CoinChain(scaffold.Chain):
    def __init__(self, **kwargs):
        super(CoinChain, self).__init__(**kwargs)
        try:
            self.start_mode = self.params.get('start_mode', 'from_prior')
        except KeyError as err:
            raise ParameterException("Chain missing needed parameters: %r" % err)

    def start_state(self, params, rng):
        s = CoinState()
        start_mode = params['start_mode']
        if start_mode == 'from_prior':
            s.p_heads = self.rng.beta(params['prior_heads'], params['prior_tails'])
        elif start_mode == 'fixed':
            s.p_heads = .5
        else:
            raise ParameterException("Start mode %r not understood" % self.start_mode)
        return s

    def transition(self, prev_state, params, coin_data, rng):
        prior_heads, prior_tails = params['prior_heads'], params['prior_tails']
        s = CoinState()
        n_heads = prior_heads + sum(coin_data == True)
        n_tails = prior_tails + sum(coin_data == False)
        s.p_heads = rng.beta(n_heads, n_tails)
        return s

    def summarize(self, history):
        trace = history.get_traces('p_heads')
        figure()
        trace.hist()
        posterior = helpers.save_fig_to_str()
        return dict(posterior=posterior)

    def show(self, **kwargs):
        pass


expt = Experiment()

#todo: also take cartesian product to generate data sources
expt.data_srcs = [
    dict(
        data_class='CoinData',
        p_heads=.4,
        n_flips=10),

    dict(
        data_class="CoinData",
        p_heads=.8,
        n_flips=100
    )
]


method = dict(chain_class="CoinChain", n_iters=100, prior_heads=1, prior_tails=1, start_mode='from_prior')

expt.methods = [method]
expt.data_seeds = [0, 1]
expt.method_seeds = [0, 1]

def run():
    expt.run(run_mode='local')
    results = expt.fetch_results(iters=5)
    fig, ax = subplots(len(results), 2)
    for i, result in enumerate(results):
        trace = result.get_traces(['p_heads'])
        sca(ax[i, 0])
        plot(trace['p_heads'])
        sca(ax[i, 1])
        hist(trace['p_heads'])


