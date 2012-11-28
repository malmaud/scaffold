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
    pass

class State(object):
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

    def copy(self):
        return deepcopy(self)

class History(object):
    """
    The complete history of a single run of a particular algorithm on a particular dataset.
    For MCMC algorithms, corresponds to the 'trace' as used in the R MCMC package.
    A history includes the state of an algorithm at each iteration, as well as summary statistics that
    have been pre-computed.
    """
    def __init__(self):
        self.chain = None
        self.states = []
        self.data_source_params = None

    def data_source(self):
        return DataSource(**self.data_source_params)

class Chain(object):
    """
    Provides the actual implementation of a Markovan  algorithm.
    """
    def __init__(self, **kwargs):
        self.params = kwargs
        self.seed = kwargs.get('seed', 0)
        self.rng = random.RandomState(self.seed)
        self.data_source = None

    def set_datasource(self, **source_params):
        self.data_source = source_params

    def transition(self, state):
        raise VirtualException()

    def do_stop(self, state):
        raise VirtualException()

    def start_state(self):
        raise VirtualException()

    def run(self):
        if self.data_source is None:
            raise ParameterException("Data source not set when trying to run chain")
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
        history.data_source_params = self.data_source.params.copy()
        return history #todo: implement cloud storage of history

class DataSource(object):
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
        raise VirtualException()

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
        E(P(test data|train data)) under procedure that generated data. Estimate of entropy
        """
        raise VirtualException()

@util.memory.cache(ignore=['results'])
def history_cache(job_params, results=None): #todo: support dynamic computation of results
    if results is None:
        raise ParameterException("Tried to access cache of unrun job")
    return results

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


class DataStore(object):
    """
    Simple data persistance interface, based around storing and retrieving
    Python objects by a string label
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
    def store(self, object, key):
        data = cPickle.dumps(object, protocol=2)
        file_form = StringIO.StringIO(data)
        cloud.bucket.putf(file_form, key)

    def load(self, key):
        file_form = cloud.bucket.getf(key)
        data = file_form.read()
        return cPickle.loads(data)