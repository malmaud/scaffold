"""
A set of utility functions used in various modules.
"""

from __future__ import division
from numpy import *
from pdb import set_trace
import logging
import joblib
import cStringIO
import cPickle
import matplotlib.pylab as plt
import hashlib

logger = logging.getLogger('scaffold')
[logger.removeHandler(h) for h in logger.handlers]
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s: %(message)s',
datefmt = '%H:%M:%S')
handler.setFormatter(formatter)
logger.addHandler(handler)

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

def discrete_sample(w, n=1, rng=random, log_mode=False, temperature=None):
    """
    Sample from a general  discrete distribution.

    :param w: A list of weights of each discrete outcome. Does not need to be normalized.
    :param  n: The number of samples to return.
    :param rng: The random number generator to use (e.g. as returned by *random.RandomState*)
    :param log_mode: If *True*, interpret *w* as the log of the true weights. If *False*, interpret *w* as the literal weights. Default *False*.
    :param temperature: The soft-max annealing temperature (:math:`\\tau`). *None* indicates not to use soft-max.

    In :math:`\lim \\tau\\to 0`, returns :math:`\\text{argmax}(w)`. In :math:`\lim \\tau\\to\infty`, returns :math:`\sim\\text{DiscreteUniform}(|w|)`

    :return: A list of *n* integers, corresponding to the indices of *w* that were chosen.
    """
    w = asarray(w, 'd')
    softmax = temperature is not None #todo: in softax mode, we are not robust to overflow
    seterr(over='raise', under='raise')
    if log_mode:
        if softmax:
            raise BaseException("softmax in logmode not implemented")
        c = logaddexp.accumulate(w)
        c -= c[-1]
        r = log(rng.rand(n))
        return searchsorted(c, r)
    else:
        if softmax:
            w = exp(w/temperature)
        c = cumsum(w)
        c /= c[-1]
        r = rng.rand(n)
        return searchsorted(c, r)

def save_fig_to_str():
    """
    Returns a string representing the current figure, in PDF format. Useful for creating a figure on a remote process and marshaling it back to the client.

    Example::

        plot([1,2], [3,4])
        s = save_fig_to_str()
        f = open('myfile.pdf', 'wb')
        f.write(s)


    :return: A string of bytes in PDF format.
    """
    buffer = cStringIO.StringIO()
    plt.savefig(buffer, format='pdf')
    buffer.seek(0)
    return buffer.read()

def hash_robust(obj):
    """
    Returns a string hash of *obj*, even if *obj* is not hashable. Mainly useful for hashing dictionaries.

    :param obj: Any picklable Python object
    :return: A printable string

    Example::

        x = dict(name='jon', research='AI')
        hash(x) # Will throw an exception since dictionaries are unhashable
        hash_robust(x) # Will work

    **Warning**: It is possible for two objects to compare equal according to the == operator, and yet hash to different strings.

    """
    hash_length = 20 # The longer it is, the less likely collisions.
    key = cPickle.dumps(obj)
    key_hash = hashlib.sha1(key).hexdigest()
    return key_hash[:hash_length]