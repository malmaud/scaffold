"""
Classes for executing an iterative algorithm on the cloud.
"""

from __future__ import division
import cStringIO
import itertools
import cloud
from matplotlib.pyplot import ioff
from numpy import array, empty
import pandas
import helpers
from helpers import frozendict
from scaffold import registry, logger
import storage

class Job(object):
    """
    Encodes the parameters of a single run of an algorithm on a single dataset.
    """

    method = None
    data_src = None
    method_seed = 0
    data_seed = 0

    def get_params(self):
        method = frozendict(self.method)
        data_src = frozendict(self.data_src)
        return (method, data_src, self.method_seed, self.data_seed)

    params = property(get_params)

    def __init__(self, method=None, data_src=None, method_seed=0, data_seed=0):
        self.method, self.data_src, self.method_seed, self.data_seed = \
        method, data_src, method_seed, data_seed
        self.job_id = None

    def __hash__(self):
        return hash(self.params)

    def __eq__(self, other):
        return self.params==other.params #untested

    def get_data(self):
        """
        :rtype: DataSource
        """
        cls = registry[self.data_src['data_class']]
        data = cls(seed=self.data_seed, **self.data_src)
        #data.load()
        #data.init(seed=self.data_seed, **self.data_src)
        return data

    def get_chain(self):
        """
        :rtype : Chain
        """
        cls = registry[self.method['chain_class']]
        chain = cls(seed=self.method_seed, **self.method)
        return chain

    data = property(get_data)
    chain = property(get_chain)

    def __str__(self):
        s = cStringIO.StringIO()
        print >>s, "Method: %r" % self.method
        print >>s, "Data source: %r" % self.data_src
        print >>s, "Seeds: (Method %r, Data %r)" % (self.method_seed, self.data_seed)
        return s.getvalue()

    def fetch_results(self, iters=None, via_remote=False, run_mode='cloud'):
        def f():
            if run_mode=='cloud':
                store = storage.CloudStore()
            else:
                store = storage.LocalStore()
            full_history = store[self.params]
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
        chain = self.chain
        data = self.data
        if run_mode == 'cloud':
            store = storage.CloudStore()
        else:
            store = storage.LocalStore()
        self.key = store.hash_key(self.params)
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
            #chain = self.get_chain()
            #data = self.get_data()
            data.load()
            chain.data = data.train_data
            chain.data_source = data
            ioff()
            states = chain.run()
            logger.debug('Chain completed')
            history = History()
            history.states = states
            history.job = self
            logger.debug("Summarizing chain")
            history.summary = chain.summarize(history)
            logger.debug("Chain summarized")
            logger.debug("Job params: %r" % (self.params,))
            #logger.debug("Raw hash: %r" % hash(self.params))
            #logger.debug("Hash value: %r" % store.hash_key(self.params))
            store.raw_store(self.key, history)
            #store[self.params] = history
            store.close()
        if run_mode=='local':
            return f()
        elif run_mode=='cloud':
            job_id = cloud.call(f, _env="malmaud")
            self.job_id = job_id
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
        self.jobs = None

    def iter_jobs(self):
        for job_parms in itertools.product(self.methods, self.data_srcs, self.method_seeds, self.data_seeds):
            job = Job(*job_parms)
            yield job

    def run(self):
        """
        Runs the experiment, storing results in the global cache. If same job has already been run, will overwrite results.
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

    def fetch_results(self, **kwargs):
        self.results = []
        kwargs['run_mode'] = self.run_mode
        for job in self.jobs:
            self.results.append(job.fetch_results(**kwargs))
        return self.results

    def iteritems(self):
        if self.jobs is None:
            return
        if self.results is None:
            self.fetch_results()
        for job, result in zip(self.jobs, self.results):
            yield (job, result)


class History(object):
    """
    The complete history of a single run of a particular algorithm on a particular dataset.

    For MCMC algorithms, corresponds to the 'trace' as used in the R MCMC package.

    A history includes the state of an algorithm at each iteration, as well as summary statistics and graphs that have been pre-computed using the :py:meth:`State.summarize` methods and :py:meth:`Chain.summarize` method.

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

    def get_traces(self, attr_names, include_time=False):
        """
        Returns traces of specific state variables in a computationally convenient form

        :param attr_names: A list of names of names to return traces for, or a string identifying a single variable.

        :return: A numeric dataframe where each column corresponds to one of the variables in *attr_names* and row corresponds to one iteration. If *attr_names* is a string instead of a list, returns instead a 1d data series that is the trace of that one variable.
        """
        collapse = False
        if isinstance(attr_names, str):
            attr_names = [attr_names]
            collapse = True
        if include_time:
            collapse = False
        x = empty((len(self.states), len(attr_names)))
        for i, name in enumerate(attr_names):
            x[:, i] = array([getattr(state, name) for state in self.states], 'd')
        index = pandas.Index([state.iter for state in self.states], name='Iteration')
        traces = pandas.DataFrame(x, columns = attr_names, index=index)
        if include_time:
            traces['time'] = self.relative_times()
        if collapse:
            return traces.ix[:, 0]
        else:
            return traces

    data = property(lambda self: self.job.data)
    chain = property(lambda self: self.job.chain)