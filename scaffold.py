"""
scaffold.py

Top-level module for accessing scaffold classes.
These classes are meant to be inherited from as needed.
"""
from __future__ import division
import cStringIO
import cloud
from numpy import *
import pandas
from pandas import DataFrame, Index
import time
import tempfile
import itertools
import subprocess
from copy import deepcopy
import helpers
from helpers import VirtualException, ParameterException
from matplotlib.pyplot import *
import storage

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

    __slots__ = ['iter', 'time'] #Slots are used for memory efficiency

    def __init__(self):
        pass

    def summarize(self):
        """
        Perform any work on computing summary statistics or visualizations of this iteration. Typically executing at end of an MCMC run.

        Main purpose is to allow for remote computation of state summary, rather than having the state pulled back to the client and then having the client create visualizations.

        **Warning**: Should depend *only* on the instance variables defined in the state object.
        """
        pass

    def copy(self):
        return deepcopy(self)

    def get_state(self):
        return {}

    def __getstate__(self):
        d = dict(iter=self.iter, time=self.time)
        d.update(self.get_state())
        return d

    def __setstate__(self, state):
        for k, v in state.iteritems():
            setattr(self, k, v)

    def show(self, **kwargs):
        pass

class History(object):
    """
    The complete history of a single run of a particular algorithm on a particular dataset.

    For MCMC algorithms, corresponds to the 'trace' as used in the R MCMC package.

    A history includes the state of an algorithm at each iteration, as well as summary statistics that have been pre-computed using the :py:meth:`State.summarize` methods and :py:meth:`Chain.summarize` method.

    **Attributes**:

    states
     A list of *State* objects

    job
     A job description
    """
    def __init__(self):
        self.job = None
        self.states = []
        self.summary = []

    def relative_times(self):
        times = array([state.time for state in self.states], 'd')
        times -= times[0]
        return times

    def get_traces(self, attr_names):
        """
        Returns traces of specific state variables in a computationally convenient form

        :param attr_names: A list of names of names to return traces for, or a string identifying a single variable.

        :return: A numeric dataframe where each column corresponds to one of the variables in *attr_names* and row corresponds to one iteration. If *attr_names* is a string instead of a list, returns instead a 1d data series that is the trace of that one variable.
        """
        collapse = False
        if isinstance(attr_names, str):
            attr_names = [attr_names]
            collapse = True
        x = empty((len(self.states), len(attr_names)))
        for i, name in enumerate(attr_names):
            x[:, i] = array([getattr(state, name) for state in self.states], 'd')
        index = pandas.Index([state.iter for state in self.states], name='Iteration')
        traces = DataFrame(x, columns = attr_names, index=index)
        if collapse:
            return traces.ix[:, 0]
        else:
            return traces

class ModelRegistry(object):
    data_src_classes = {}
    chain_classes = {}

    def register_data_src(self, name, klass):
        self.data_src_classes[name] = klass

    def register_chain(self, name, klass):
        self.chain_classes[name] = klass

registry = ModelRegistry()

class Chain(object):
    """
    Provides the actual implementation of a Markovan  algorithm.
    """
    def __init__(self, **kwargs):
        """
        :param kwargs: A set of parameters controlling the inference algorithm. Expected keys:

        seed
         The random seed used for the iterative algorithm

        All other keys are passed through to the derived class.
        """
        self.params = kwargs
        self.seed = kwargs['seed']
        self.rng = random.RandomState(self.seed)
        self.data = None

    @classmethod
    def register(cls, name):
        registry.register_chain(name, cls)

    def transition(self, state):
        """
        Implementation of the transition operator. Expected to be implemented in a user-derived subclass.

        :param state: The current state of the Markov algorithm

        :return: The next state of the Markov Algorithm
        """
        raise VirtualException()

    def do_stop(self, state):
        """
        Virtual method that decides when the iterative algorithm should terminate

        :param state: Current state

        :return: *True* if the algorithm should terminate. *False* otherwise.
        """
        raise VirtualException()

    def start_state(self):
        """
        :return: The initial state of the algorithm

        """
        raise VirtualException()

    def attach_state_metadata(self, state, iter):
        state.iter = iter
        state.time = time.time()

    def run(self):
        """
        Actually executes the algorithm. Starting with the state returned  by :py:meth:`start_state`, continues to call :py:meth:`transition` to retrieve subsequent states of the algorithm, until :py:meth:`do_stop` indicates the algorithm should terminate.

        :return: A list of :py:class:`State` objects, representing the state of the algorithm at the start of each iteration. Exception: The last state is the list is the state at the end of the last iteration.
        """
        logger.debug('Running chain')
        if self.data is None:
            raise ParameterException("Data source not set when trying to run chain")
        states = []
        state = self.start_state()
        self.attach_state_metadata(state, 0)
        for iter in itertools.count():
            if iter%50==0:
                logger.debug("Chain running iteration %d" % iter)
            states.append(state)
            new_state = self.transition(state)
            self.attach_state_metadata(new_state, iter+1)
            if self.do_stop(new_state):
                states.append(new_state)
                break
            state = new_state
        logger.debug("Chain complete, now summarizing states")
        for state in states:
            state.summarize()
        logger.debug("States summarized")
        return states

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
    def __init__(self):
        return

    @classmethod
    def register(cls, name):
        registry.register_data_src(name, cls)

    def init(self, **kwargs):
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
        self.seed = kwargs['seed']
        logger.debug("dataset created with seed %d" % self.seed)
        self.rng = random.RandomState(self.seed)
        self.data = None
        test_fraction = kwargs.get('test_fraction', .2)
        self.params = kwargs
        self.load() #todo: load should not be called here, as per the docstring
        if self.data is None:
            raise BaseException("Datasouce 'load' method failed to create data attribute")
        self.split_data(test_fraction)

    def load(self):
        """
        Load/generate the data into memory
        """
        raise VirtualException()

    def train_data(self):
        """

        :return: Training data
        """
        return self.data[self.train_idx]

    def test_data(self):
        """

        :return: Held-out test data
        """
        return self.data[self.test_idx]

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

    def pred_lh(self):
        """
        E(P(test data|train data)) under procedure that generated data. Estimate of predictive entropy.
        """
        raise VirtualException()

class Job(object):
    method = None
    data_src = None
    method_seed = 0
    data_seed = 0

    def __init__(self, method=None, data_src=None, method_seed=0, data_seed=0):
        self.method, self.data_src, self.method_seed, self.data_seed = \
        method, data_src, method_seed, data_seed

    def get_data(self):
        """
        :rtype: DataSource
        """
        cls = registry.data_src_classes[self.data_src['data_class']]
        data = cls()
        data.init(seed=self.data_seed, **self.data_src)
        return data

    def get_chain(self):
        """
        :rtype : Chain
        """
        cls = registry.chain_classes[self.method['chain_class']]
        chain = cls(seed=self.method_seed, **self.method)
        return chain

    def __str__(self):
        s = cStringIO.StringIO()
        print >>s, "Method: %r" % self.method
        print >>s, "Data source: %r" % self.data_src
        print >>s, "Seeds: (Method %r, Data %r)" % (self.method_seed, self.data_seed)
        return s.getvalue()

    def fetch_results(self, iters=None, via_remote=False, run_mode='local'):

        def f():
            if run_mode=='cloud':
                store = storage.CloudStore()
            else:
                store = storage.LocalStore()
            full_history = store[self]
            partial_history = History()
            if iters is None:
                partial_history.states = full_history.states
            else:
                partial_history.states = [state for state in full_history.states if state.iter in iters]
            partial_history.job = self
            partial_history.summary = full_history.summary
            return partial_history
        if via_remote:
            job_id = cloud.call(f, _env="malmaud") #todo: don't hard-code environment
            return cloud.result(job_id)
        else:
            return f()

    def run(self, run_mode):
        use_cache = False
        def f():
            if run_mode == "cloud":
                store = storage.CloudStore()
            elif run_mode=="local":
                store = storage.LocalStore()
            else:
                raise BaseException("Run mode %r not recognized" % run_mode)
            if use_cache and (self in store):
                return
            logger.debug("Running job")
            chain = self.get_chain()
            data = self.get_data()
            chain.data = data.train_data()
            ioff()
            states = chain.run()
            logger.debug('Job chain completed')
            history = History()
            history.states = states
            history.job = self
            logger.debug("Summarizing chain")
            history.summary = chain.summarize(history)
            logger.debug("Chain summarized")
            store[history.job] = history
            store.close()
        if run_mode=='local':
            return f()
        elif run_mode=='cloud':
            job_id = cloud.call(f, _env="malmaud")
            return job_id

class Experiment(object):
    """
    Encodes the parameters and results of an experiment.
    An experiment is the running of difference algorithms on different datasets.
    """
    def __init__(self, run_mode='cloud'):
        self.methods = []
        self.data_srcs = []
        self.method_seeds = []
        self.data_seeds = []
        self.run_mode = run_mode
        self.results = None

    def iter_jobs(self):
        for job_parms in itertools.product(self.methods, self.data_srcs, self.method_seeds, self.data_seeds):
            job = Job(*job_parms)
            yield job

    def run(self):
        """
        Runs the experiment, storing results in the cache. If same job has already been run, will overwrite results.
        """
        logger.debug('Running experiment')
        cloud_job_ids = []
        for job in self.iter_jobs():
            result = job.run(self.run_mode)
            if self.run_mode == 'cloud':
                cloud_job_ids.append(result)
        if self.run_mode=='cloud':
            logger.info("Waiting for cloud jobs to finish")
            cloud.join(cloud_job_ids)
            logger.info("Cloud jobs finished")
        self.jobs = list(self.iter_jobs())

    def fetch_results(self):
        self.results = []
        for job in self.jobs:
            self.results.append(job.fetch_results())
        return self.results


