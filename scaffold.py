"""
scaffold.py

Top-level module for accessing scaffold classes.
These classes are meant to be inherited from as needed.
"""
from __future__ import division
import cloud
from numpy import *
import time
import tempfile
import itertools
import subprocess
from copy import deepcopy
import util
from util import logger
from numpy import *
from matplotlib.pyplot import *

class VirtualException(BaseException):
    """
    Error raised when a method of a superclass is called directly
    when it was  intended that a child class override that method
    """
    pass

class ParameterException(BaseException):
    """
    Exception type for when an expected key is missing from the parameter dictionary of a parameterized algorithm
    """
    pass

class State(object):
    """
    Represents all state variables of the algorithm at a particular iteration.

    At a minimum, a state object will have the following attributes:

    iter
     The iteration number of the algorithm that this state corresponds to. State 0 corresponds to the initial state of the algorithm, before any transitions have been applied. The last state is state that caused *do_stop* to return *True*.

    time
     The time (in seconds since epoch) that this state was created. Mainly used to assess runtime of algorithms.
    """
    iter = None
    time = None

    def __init__(self):
        pass

    def latents(self):
        pass

    def summarize(self):
        """
        Perform any work on computing summary statistics or visualizations of this iteration. Typically executing at
         end of an MCMC run.

        Main purpose is to allow for remote computation of state summary, rather than having the state pulled back to
        the client and then having the client create visualizations.

        Should depend *only* on the instance variables defined in the state object. No guarentee is made on when this
         will be executed.
        """
        pass

    def copy(self):
        return deepcopy(self)

class History(object):
    """
    The complete history of a single run of a particular algorithm on a particular dataset.

    For MCMC algorithms, corresponds to the 'trace' as used in the R MCMC package.

    A history includes the state of an algorithm at each iteration, as well as summary statistics that
    have been pre-computed using the *State.summarize* method.

    Attributes:

    states
     A list of *State* objects

    chain
     An instance of a *Chain* object, containing all the parameters of the algorithm.

    data_source_params
     A dict of all the parameters used for generating the dataset.
    """
    def __init__(self):
        self.chain = None
        self.states = []
        self.summary = []

    def relative_times(self):
        times = array([state.time for state in self.states], 'd')
        times -= times[0]
        return times

    def show_fig(self, key):
        fig = self.summary[key]
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.pdf', delete=False)
        f.write(fig)
        f.close()
        subprocess.call(['open', f.name]) #todo: only works on OS X

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
        self.seed = kwargs.get('seed', 0)
        self.rng = random.RandomState(self.seed)
        self.data = None

    def transition(self, state):
        """
        Implementation of the transition operator. Expected to be implemented in a user-derived subclass.

        :param state: The current state of the Markov algorithm
        :raise:
        :return: The next state of the Markov Algorithm
        """
        raise VirtualException()

    def do_stop(self, state):
        """
        Virtual method that decides when the iterative algorithm should terminate

        :param state: Current state
        :raise:
        :return: *True* if the algorithm should terminate. *False* otherwise.
        """
        raise VirtualException()

    def start_state(self):
        """
        :return: The initial state of the algorithm
        :raise:
        """
        raise VirtualException()

    def attach_state_metadata(self, state, iter):
        state.iter = iter
        state.time = time.time()

    def run(self):
        """
        Actually executes the algorithm. Starting with the state retunred  by *start_state*, continues to call *transition* to retrieve subsequent states of the algorithm, until *do_stop* indicates the algorithm should terminate.

        :return: A *History* object that contains a complete history of the state parameters at each iteration, as well as any pre-computed summary statistics and visualizations as computed by *State.summarize*
        """
        logger.debug('Running chain')
        if self.data is None:
            raise ParameterException("Data source not set when trying to run chain")
        states = []
        state = self.start_state()
        self.attach_state_metadata(state, 0)
        for iter in itertools.count():
            logger.debug("Chain running iteration %d" % iter)
            states.append(state)
            new_state = self.transition(state)
            self.attach_state_metadata(new_state, iter+1)
            if self.do_stop(new_state):
                states.append(new_state)
                break
            state = new_state
        for state in states:
            state.summarize()
        return states

    def summarize(self, history):
        pass

class DataSource(object):
    """
    Represents datasets that have been procedurally generated. Intended to be inherited from by users.
    """
    def __init__(self):
        return

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
        self.seed = kwargs.get('seed', 0)
        self.rng = random.RandomState(self.seed)
        self.data = None
        test_fraction = kwargs.get('test_fraction', .2)
        self.params = kwargs
        self.load()
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

        :param test_fraction: Fraction of data to put in the test traing set. 1-test_fraction is put into the training set.
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

    def iter_jobs(self):
        for job_parms in \
           itertools.product(self.methods, self.data_srcs, self.method_seeds, self.data_seeds):
            yield job_parms

    def run(self):
        """
        Runs the experiment, storing results in the local cache
        """
        logger.debug('Running experiment')
        ioff()
        jobs = []
        for job_params in self.iter_jobs():
            method, data_src_params, method_seed, data_seed = job_params
            def f():
                chain = method['chain_class'](seed=method_seed, **method)
                data_source = data_src_params['data_class']()
                data_source.init(seed=data_seed, **data_src_params)
                chain.data = data_source.train_data()
                states = chain.run()
                history = History()
                history.chain = chain
                history.states = states
                history.data_source = data_source
                history.summary = chain.summarize(history)
                return history
            if self.run_mode=='local':
                history_cache(job_params, f())
            elif self.run_mode=='cloud':
                job_id = cloud.call(f, _env='malmaud')
                jobs.append(job_id)
        if self.run_mode=='cloud':
            logger.info("Waiting for cloud jobs to finish")
            cloud.join(jobs)
            logger.info("Cloud jobs finished")
            for job_param, job in itertools.izip(self.iter_jobs(), jobs):
                history_cache(job_param, cloud.result(job))
        results = [history_cache(job_param) for job_param in self.iter_jobs()]
        return list(self.iter_jobs()), results


@util.memory.cache(ignore=['results'])
def history_cache(job_params, results=None):
    """
    Provides read/write access to the local cache.

    :param job_params: A key into the cache. Typically a dict that uniquely defined a computational job.
    :param results: If this is non-None, it is interpreted as the value associated with the key *job_params* and the local cache is updated. Otherwise, this call is interpreted as a read request and the results previously stored with *job_params* are returned.
    """
    if results is None: #todo: support dynamic computation of results
        raise ParameterException("Tried to access cache of unrun job")
    return results


