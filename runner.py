"""
Classes for executing an iterative algorithm on the cloud.
"""

from __future__ import division
import cStringIO
import itertools
import cloud
from matplotlib.pyplot import ioff, ion
from numpy import array, empty
import pandas
import helpers
from helpers import frozendict
from scaffold import registry, logger
import storage

picloud_env = "malmaud" #todo: this should configurable somewhere


def get_chain(params, seed):
    cls = registry[params['chain_class']]
    chain = cls(seed=seed, **params)
    return chain


class Job(object):
    """
    Represent a single run of an algorithm on a given dataset, for fixed seeds.
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
        self.method, self.data_src, self.method_seed, self.data_seed =\
        method, data_src, method_seed, data_seed
        self.job_id = None

    def __hash__(self):
        return hash(self.params)

    def __eq__(self, other):
        return self.params == other.params #untested

    def get_data(self):
        """
        """
        cls = registry[self.data_src['data_class']]
        data = cls(seed=self.data_seed, **self.data_src)
        data._load_data() #todo: maybe dont call this automatically
        return data

    def get_chain(self):
        """
        """
        return get_chain(self.method, self.method_seed)

    data = property(get_data)
    chain = property(get_chain)

    def __str__(self):
        s = cStringIO.StringIO()
        print >> s, "Method: %r" % self.method
        print >> s, "Data source: %r" % self.data_src
        print >> s, "Seeds: (Method %r, Data %r)" % (self.method_seed, self.data_seed)
        return s.getvalue()

    def fetch_results(self, iters=None, via_remote=True, run_mode='cloud'):
        """
        Returns the result of the job that has already been run as a :py:class:`History` object. Typically you would call :py:meth:`run` first, then call :py:meth:`fetch_results` to get the resutlts. The method has various methods to control how much of the job is returned, to avoid excessive memory usage and data transfer between the cloud and local machine.

        :param iters: If *iters* is an iterable, returns only the iterations of the chain in *iters*. If *iters* is a scalar, return every *iters* state (the stride). If None, returns all states.
        :param via_remote: If *True*, executes the state filtering on the cloud before transferring the data to the local machine. If false, filter the state on the local machine.
        :param run_mode: Controls whether to search for the results on the local macine or on the cloud. Can be *local* or *cloud*.
        :return: A :py:class:`History` object that contains a filtered version the states of the Markov chain visited when this job ran.
        """

        def f():
            if run_mode == 'cloud':
                store = storage.CloudStore()
            else:
                store = storage.LocalStore()
            full_history = store[self.params]
            partial_history = History()
            if iters is None:
                partial_history.states = full_history.states
            else:
                if isinstance(iters, int): #iters interpreted as stride
                    iter_set = range(0, len(full_history.states), iters)
                else:
                    iter_set = iters
                partial_history.states = [state for state in full_history.states if state.iter in iter_set]
            partial_history.job = self
            partial_history.summary = full_history.summary
            return partial_history

        if via_remote:
            job_id = cloud.call(f, _env=picloud_env)
            return cloud.result(job_id)
        else:
            return f()

    def run(self, run_mode='local', use_cache=False):
        """
        Runs the job, storing all the states of the Markov chain in a datastore.

        :param run_mode: A string that controls how and where the job is run. Currently has two allowable values:

         local
          Runs the job locally and stores the data locally. Useful for debugging.

         cloud
           Runs the job on picloud and stores the data in a picloud bucket.


        :param use_cache: If *True* and this job has already been run at a previous time, return the results of that job. If *False*, rerun the job.
        :return: If run_mode is 'cloud', returns the picloud job id in a non-blocking way. If run_mode is 'local', does not return anything and will not return until the job is completed.
        :raise:
        """
        chain = self.chain
        data = self.data
        if run_mode == 'cloud':
            store = storage.CloudStore()
        else:
            store = storage.LocalStore()
            # Calculating the key outside of the cloud is necessary since the hashing functions on the cloud may not
        # agree with the local hashing functions (might be a 32-bit vs 64-bit python issue).
        self.key = store.hash_key(self.params)

        def f():
            if run_mode == "cloud":
                store = storage.CloudStore()
            elif run_mode == "local":
                store = storage.LocalStore()
            else:
                raise BaseException("Run mode %r not recognized" % run_mode)
            store.auto_hash = False
            if use_cache and (self.key in store):
                logger.debug("Cache hit")
                return
            logger.debug("Cache miss")
            logger.debug("Running job")
            data._load_data()
            chain.data = data.train_data
            chain.data_source = data
            ioff()
            states = chain._run()
            ion()
            logger.debug('Chain completed')
            history = History()
            history.states = states
            history.job = self
            logger.debug("Summarizing chain")
            history.summary = chain.summarize(history)
            logger.debug("Chain summarized")
            logger.debug("Job params: %r" % (self.params,))
            store[self.key] = history
            store.close()

        if run_mode == 'local':
            return f()
        elif run_mode == 'cloud':
            job_id = cloud.call(f, _env=picloud_env)
            self.job_id = job_id
            return job_id


class Experiment(object):
    """
    Encodes the parameters and results of an experiment.
    An experiment is the running of difference algorithms on different datasets.
    More precisely, it is the Cartesian product of four sets:
    {Algorithms}*{Data source}*{Seeds for algorithm}*{Seeds for data sources}

    :ivar methods: A list of method descriptors
    :ivar data_srcs: A list of data source descriptors
    :ivar method_seeds: A list of method seeds
    :ivar data_seeds: A list of data source seeds
    """

    def __init__(self):
        self.methods = []
        self.data_srcs = []
        self.method_seeds = []
        self.data_seeds = []
        self.run_mode = None
        self.results = None
        self.jobs = None

    def iter_jobs(self):
        for job_parms in itertools.product(self.methods, self.data_srcs, self.method_seeds, self.data_seeds):
            job = Job(*job_parms)
            yield job

    def run(self, **kwargs):
        """
        Runs the experiment. If the experiment is run on the cloud, blocks until all jobs complete.
        """
        logger.debug('Running experiment')
        cloud_job_ids = []
        self.run_params = kwargs.copy()
        run_mode = kwargs.get('run_mode', 'local')
        self.run_mode = run_mode
        for job in self.iter_jobs():
            result = job.run(**kwargs)
            if run_mode == 'cloud':
                cloud_job_ids.append(result)
        if run_mode == 'cloud':
            logger.info("Waiting for cloud jobs to finish")
            cloud.join(cloud_job_ids)
            logger.info("Cloud jobs finished")
        self.jobs = list(self.iter_jobs())

    def fetch_results(self, **kwargs):
        if self.jobs is None:
            raise BaseException("Must run experiment before results can be fetched")
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

    A history includes the state of an algorithm at each iteration, as well as summary statistics and graphs
    that have been pre-computed using the :py:meth:`State.summarize` methods and :py:meth:`Chain.summarize` method.

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
        Returns traces of specific state variables in a computationally convenient form.

        :param attr_names: A list of names of names to return traces for, or a string identifying a single variable.

        :return: A numeric dataframe where each column corresponds to one of the variables in *attr_names* and row corresponds to one iteration. If *attr_names* is a string instead of a list, returns instead a 1d data series that is the trace of that one variable.
        """
        collapse = False
        if (not hasattr(attr_names, '__getitem__')) or isinstance(attr_names, str):
            attr_names = [attr_names]
            collapse = True
        if include_time:
            collapse = False
        x = empty((len(self.states), len(attr_names)), object)
        for i, name in enumerate(attr_names):
            if hasattr(name, '__call__'):
                x[:, i] = array([name(state) for state in self.states], object)
            else:
                x[:, i] = array([getattr(state, name) for state in self.states], object)
        index = pandas.Index([state.iter for state in self.states], name='Iteration')
        traces = pandas.DataFrame(x, columns=attr_names, index=index)
        if include_time:
            traces['time'] = self.relative_times()
        if collapse:
            return traces.ix[:, 0]
        else:
            return traces

    def show_fig(self, fig_name):
        helpers.show_fig(self.summary[fig_name])

    def get_train_data(self):
        source = self.data
        source._load_data()
        return source.train_data


    data = property(lambda self: self.job.data)
    chain = property(lambda self: self.job.chain)