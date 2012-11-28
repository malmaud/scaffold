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
import shelve
import cPickle
import StringIO

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
     The iteration number of the algorithm that this state corresponds to. State 0 corresponds to the initial state of
     the algorithm, before any transitions have been applied. The last state is state that caused *do_stop* to return
     *True*.

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
        self.data_source_params = None

    def data_source(self):
        """

        :return: The data source that this run executed on
        """
        return DataSource(**self.data_source_params)

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
        self.data_source = None

    def set_datasource(self, **source_params):
        """
        Sets the data source for this chain.
        :param source_params: A dict of parameters that implicitly specify the data source.
        """
        self.data_source = source_params

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

    def run(self):
        """
        Actually executes the algorithm. Starting with the state retunred  by *start_state*, continues to call
        *transition* to retrieve subsequent states of the algorithm, until *do_stop* indicates the algorithm
        should terminate.

        :return: A *History* object that contains a complete history of the state parameters at each iteration,
        as well as any pre-computed summary statistics and visualizations as computed by *State.summarize*
        """
        if self.data_source is None:
            raise ParameterException("Data source not set when trying to run chain")
        states = []
        state = self.start_state()
        for iter in itertools.count():
            state.iter = iter
            state.time = time.time()
            states.append(state)
            new_state = self.transition(state)
            if self.do_stop(new_state): #todo: make last state be included
                break
            state = new_state
        for state in states:
            state.summarize()
        history = History()
        history.states = states
        history.chain = deepcopy(self)
        history.data_source_params = self.data_source.params.copy()
        return history #todo: implement cloud storage of history

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
        test_fraction = kwargs.get('test_fraction', .2)
        self.params = kwargs
        self.load()
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
        :param test_fraction:
        :return:
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
        jobs = []
        job_params_list = []
        for job_params in self.iter_jobs():
            method, data_src_params, method_seed, data_seed = job_params
            def f():
                chain = method['chain_class'](seed=method_seed, **method)
                data_source = data_src_params['data_class']()
                data_source.init(seed=data_seed, **data_src_params)
                chain.set_datasource(data_source)
                history = chain.run()
                return history
            if self.run_mode=='local':
                history_cache(job_params, f())
            elif self.run_mode=='cloud':
                job_id = cloud.call(f, _env='malmaud')
                jobs.append(job_id)
                job_params_list.append(job_params)
        if self.run_mode=='cloud':
            util.logging.debug("Waiting for cloud jobs to finish")
            cloud.join(jobs)
            util.logging.debug("Cloud jobs finished")
            for job_param, job in zip(job_params_list, jobs):
                history_cache(job_param, cloud.result(job))

@util.memory.cache(ignore=['results'])
def history_cache(job_params, results=None):
    """
    Provides read/write access to the local cache.

    :param job_params: A key into the cache. Typically a dict that uniquely defined a computational job.
    :param results: If this is non-None, it is interpreted as the value associated with the key *job_params* and the
    local cache is updated. Otherwise, this call is interpreted as a read request and the results previously stored with
    *job_params* are returned.
    """
    if results is None: #todo: support dynamic computation of results
        raise ParameterException("Tried to access cache of unrun job")
    return results


class DataStore(object):
    """
    Simple abstract data persistence interface, based around storing and retrieving
    Python objects by a string label. Provides a dict-like interface
    """
    def store(self, object, key):
        raise VirtualException()

    def load(self, key):
        raise VirtualException()

    def __getitem__(self, key):
        return self.load(key)

    def __setitem__(self, key, value):
        self.store(value, key)

class LocalStore(DataStore):
    """
    Implements a data store using the Python *shelve* library, where the shelf is stored locally. Useful for debugging.
    """
    def __init__(self, filename='data/data.shelve'):
        self.filename = filename
        self.shelve = shelve.open(filename, writeback=True, protocol=2)

    def store(self, object, key):
        self.shelve[key] = object

    def load(self, key):
        return self.shelve[key]

    def close(self):
        self.shelve.close()

class CloudStore(DataStore):
    """
    Implements a data store using the picloud *bucket* system. The objects must be serialiable via the *cPickle* library.

    Note that picloud charges both for storage and for transmitting data, so maybe best not to store gigantic objects
    with this system.
    """
    def store(self, object, key):
        data = cPickle.dumps(object, protocol=2)
        file_form = StringIO.StringIO(data)
        cloud.bucket.putf(file_form, key)

    def load(self, key):
        file_form = cloud.bucket.getf(key)
        data = file_form.read()
        return cPickle.loads(data)