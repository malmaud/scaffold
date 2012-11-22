"""
scaffold.py
Top-level module for accessing scaffold classes.
These classes are meant to be inherited from as needed.
"""
from __future__ import division
import cloud
from numpy import *
import time
import itertools
from copy import deepcopy
import util

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
        self.datasource_id = None

    def bucket_id(self):
        pass

class Chain:
    """
    Provides the actual implementation of a Markovan  algorithm.
    """
    def __init__(self, **kwargs):
        self.params = kwargs
        self.seed = kwargs.get('seed', 0)
        self.rng = random.RandomState(self.seed)

    def transition(self, state):
        pass

    def do_stop(self, state):
        pass

    def start_state(self):
        return

    def run(self):
        states = []
        state = self.start_state()
        for iter in itertools.count():
            state.iter = iter
            state.time = time.time()
            states.append(state)
            new_state = self.transition(state)
            if self.do_stop(new_state): #todo: should last state be included?
                break
            state = new_state
        for state in states:
            state.summarize()
        history = History()
        history.states = states
        history.chain = deepcopy(self)
        return history

class DataSource:
    """
    Represents datasets that have been procedurally generated.
    """
    def __init__(self):
        return

    def init(self, **kwargs):
        self.seed = kwargs.get('seed', 0)
        self.rng = random.RandomState(self.seed)
        test_fraction = kwargs.get('test_fraction', .2)
        self.params = kwargs
        self.load()
        self.split_data(test_fraction)

    def load(self):
        """
        Load/generate the data into memory
        """
        pass

    def train_data(self):
        return self.data[self.train_idx]

    def test_data(self):
        return self.data[self.test_idx]

    def size(self):
        return len(self.data)

    def split_data(self, test_fraction):
        n = self.size()
        n_test = int(test_fraction*n)
        idx = arange(n)
        self.rng.shuffle(idx)
        self.test_idx = idx[0:n_test]
        self.train_idx = idx[n_test:]

    def pred_lh(self):
        """
        P(test data|train data) under procedure that generated data
        """
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

class ParameterException(BaseException):
    pass