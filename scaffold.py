"""
scaffold.py
Top-level module for accessing scaffold classes.
These classes are meant to be inherited from as needed.
"""
from __future__ import division
import cloud
from numpy import *

class State:
    """
    Represents all state variables of the algorithm at a particular iteration
    """
    def __init__(self):
        self.iter = None
        self.time = None

    def latents(self):
        pass

    def summarize(self):
        pass

    def bucket_id(self):
        pass

class History:
    """
    The complete history of a single run of a particular algorithm on a particular dataset.
    For MCMC algorithms, corresponds to the 'trace' as used in the R MCMC package.
    A history includes the state of an algorithm at each iteration, as well as summary statistics that
    have been pre-computed.
    """
    def __init__(self):
        self.chain = None
        self.states = []

class Chain:
    """
    Provides the actual implementation of a Markovan  algorithm.
    """
    def __init__(self, seed=0, **kwargs):
        self.params = kwargs
        self.seed = seed
        self.rng = random.RandomState(seed)

    def transition(self, state):
        pass

    def do_stop(self, state):
        pass

class DataSource:
    """
    Represents datasets that has been procedurally generated.
    """
    def __init__(self, seed=0, **kwargs):
        self.params = kwargs
        self.seed = seed
        self.rng = random.RandomState(seed)

    def load(self):
        pass

    def train_data(self):
        pass

    def test_data(self):
        pass

class Experiment:
    """
    Encodes the parameters and results of an experiment.
    An experiment is the running of difference algorithms on different datasets.
    """
    def __init__(self):
        self.methods = []
        self.data_srcs = []
        self.method_seeds = []
        self.data_seeds = []

    def run(self):
        pass

