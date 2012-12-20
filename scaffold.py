"""
Classes for representing the state and operators of an interative algorithm.

These classes are meant to be inherited from as needed.
"""
from __future__ import division
import time
import itertools
from copy import deepcopy
from numpy import *
from helpers import ParameterException
import abc


class JLogger:
    """
    Hack because picloud is complaining about pickling the standard python logger
    """
    def debug(self, str):
        print str

    def info(self, str):
        print str

logger = JLogger()

class State(object):
    """
    Represents all state variables of the algorithm at a particular iteration.

    Derived classes must use slots for storing their instance variables, rather than relying on :py:attr:`self.__dict__`.

    At a minimum, a state object will have the following attributes:

    iter
     The iteration number of the algorithm that this state corresponds to. State 0 corresponds to the initial state of the algorithm, before any transitions have been applied. The last state is the state that caused :py:meth:`Chain.do_stop` to return *True*.

    time
     The time (in seconds since epoch) that this state was created. Mainly used to assess runtime of algorithms.
    """

    __slots__ = ['iter', 'time', 'data']
    # Slots are used for memory efficiency
    # The 'data' slot is only used for follow-the-prior testing

    def __init__(self):
        self.iter, self.time, self.data = None, None, None

    def summarize(self):
        """
        Perform any work on computing summary statistics or visualizations of this iteration. Typically executing at end of an MCMC run.

        Main purpose is to allow for remote computation of state summary, rather than having the state pulled back to the client and then having the client create visualizations.

        **Warning**: Should depend *only* on the instance variables defined in the state object.
        """
        pass

    def __getstate__(self):
        d = dict(iter=self.iter, time=self.time, data=self.data)
        state = {}
        for k in self.__slots__:
            state[k] = getattr(self, k)
        d.update(state)
        return d

    def __setstate__(self, state):
        for k, v in state.iteritems():
            setattr(self, k, v)

    def show(self, **kwargs):
        pass

    def sample_data(self, n_data):
        pass

registry = {}

class RegisteredClass(abc.ABCMeta):
    def __init__(cls, name, bases, attrs):
        abc.ABCMeta.__init__(cls, name, bases, attrs)
        if name in registry:
            logger.debug("Name conflict in registered class")
        registry[name] = cls

class Chain(object):
    """
    Provides the actual implementation of a Markovian  algorithm.
    """

    __metaclass__ = RegisteredClass

    def __init__(self, **kwargs):
        """
        :param kwargs: A set of parameters controlling the inference algorithm. Keys:

        seed (Default 0)
         The random seed used for the iterative algorithm

        follow_prior (Default *False*)
         A boolean value indicating whether the 'observed' variables should be resampled after each iteration for debugging the transition operator, or instead should be clamped to the data vector assigned to the chain

        n_data (Default 10)
         If *follow_prior* is **True**, this is how many data points of data to train the model on. Otherwise, has no effect.

        All other keys are passed through to the derived class.
        """
        self.params = kwargs
        self.seed = kwargs.get('seed', 0)
        self.max_runtime = kwargs.get('max_runtime', 60*30)
        self.max_iters = kwargs.get('max_iters', inf)
        self.rng = random.RandomState(self.seed)
        self.data = None
        self.follow_prior = kwargs.get('follow_prior', False)
        if self.follow_prior:
            self.n_data_prior = kwargs.get('n_data', 10)
        self.start_time = None
        self.end_time = None

    def get_n_data(self):
        if self.follow_prior:
            return self.n_data_prior
        else:
            return len(self.data)

    n_data = property(get_n_data)

    @abc.abstractmethod
    def transition(self, state):
        """
        Implementation of the transition operator. Expected to be implemented in a user-derived subclass.

        :param state: The current state of the Markov algorithm

        :return: The next state of the Markov Algorithm
        """
        pass

    def get_net_runtime(self):
        return time.time() - self.start_time

    net_runtime = property(get_net_runtime)

    def _should_stop(self, state):
        if self.do_stop(state):
            return True
        if self.net_runtime > self.max_runtime:
            return True
        if state.iter >= self.max_iters:
            return True
        return False


    def do_stop(self, state):
        """
        method that decides when the iterative algorithm should terminate

        :param state: Current state

        :return: *True* if the algorithm should terminate. *False* otherwise.
        """
        return False

    @abc.abstractmethod
    def start_state(self):
        """
        :return: The initial state of the algorithm

        """
        pass

    def attach_state_metadata(self, state, iter):
        state.iter = iter
        state.time = time.time()

    def sample_data(self, state):
        pass

    def run(self):
        """
        Actually executes the algorithm. Starting with the state returned  by :py:meth:`start_state`, continues to call :py:meth:`transition` to retrieve subsequent states of the algorithm, until :py:meth:`do_stop` indicates the algorithm should terminate.

        :return: A list of :py:class:`State` objects, representing the state of the algorithm at the start of each iteration. **Exception**: The last state is the list is the state at the end of the last iteration.
        """
        logger.debug('Running chain')
        if self.data is None:
            raise ParameterException("Data source not set when trying to run chain")
        states = []
        self.start_time = time.time()
        state = self.start_state()
        self.attach_state_metadata(state, 0)
        if self.follow_prior:
            self.sample_data(state)
        for iter in itertools.count():
            if iter%50==0:
                logger.debug("Chain running iteration %d" % iter)
            states.append(state)
            new_state = self.transition(state)
            if self.follow_prior:
                self.sample_data(new_state)
                #new_state.sample_data(self.n_data, self.rng)
            self.attach_state_metadata(new_state, iter+1)
            if self._should_stop(new_state):
                states.append(new_state)
                break
            state = new_state
        logger.debug("Chain complete, now summarizing states")
        for state in states:
            state.summarize()
        logger.debug("States summarized")
        self.end_time = time.time()
        return states

    def get_data(self, state):
        if self.follow_prior:
            return state.data
        else:
            return self.data

    def summarize(self, history):
        """
        Return a summary of *history*, which will be computed on the cloud and then cached for local use.
        """
        pass

    def show(self, **kwargs):
        pass

class DataSource(object):
    """
    Represents datasets that have been procedurally generated. Intended to be inherited from by users.
    """

    __metaclass__ = RegisteredClass

    def __init__(self, **kwargs):
        """
        Initializes the data source by setting its parameters. Note that data is not actually generated until *load*
        is called. This division is meant to allow for a client to set parameters, while the actual data is generated
        on the cloud rather than uploaded.

        :param kwargs: A set of parameters controlling the data source. At a minimum, keys should include

        seed
         An integer specifying the random seed

        test_fraction
         What fraction of the data in the dataset should be used as held-out test data, as opposed to training data
          for the inference algorithms

        """
        self.seed = kwargs.get('seed', 0)
        self.rng = random.RandomState(self.seed)
        self.data = None
        self.test_fraction = kwargs.get('test_fraction', .2)
        self.params = kwargs
        self.loaded = False


    @abc.abstractmethod
    def load_data(self):
        pass

    def load(self):
        """
        Load/generate the data into memory
        """
        if self.loaded:
            logger.debug("Dataset is trying to load after already being loaded")
            return
        self.load_data()
        if self.data is None:
            raise BaseException("Datasouce 'load_data' method failed to create data attribute")
        self.split_data(self.test_fraction)
        self.loaded = True

    def get_train_data(self):
        """

        :return: Training data
        """
        return self.data[self.train_idx]

    def get_test_data(self):
        """

        :return: Held-out test data
        """
        return self.data[self.test_idx]

    train_data = property(get_train_data)
    test_data = property(get_test_data)

    def size(self):
        """

        :return: The number of data points currently in the dataset
        """
        return len(self.data)

    def split_data(self, test_fraction):
        """
        Splits the data into a training dataset and test dataset. Meant for internal use only.

        :param test_fraction: Fraction of data to put in the test training set. 1-test_fraction is put into the training set.
        :type test_fraction: float
        """
        n = self.size()
        n_test = int(test_fraction*n)
        idx = arange(n)
        self.rng.shuffle(idx)
        self.test_idx = idx[0:n_test]
        self.train_idx = idx[n_test:]

class ProceduralDataSource(DataSource):
    def __init__(self, **kwargs):
        super(ProceduralDataSource, self).__init__(**kwargs)

    @abc.abstractmethod
    def llh_pred(self, x):
        """
        For procedural datasets, the log-likelihood of generating the data in *x* given the latent variables of the model
        :param x:
        :return:
        """
        pass

    def branch(self, seed): #untested
        params = self.params.copy()
        params['seed']= seed
        new_src = type(self)(**params)
        return new_src





