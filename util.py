"""
util.py

A set of utility functions used in various modules.
"""

from __future__ import division
from numpy import *
from pdb import set_trace
import logging
import joblib

memory = joblib.Memory('./data', mmap_mode='r', verbose=1)

def discrete_sample(w, n, rng=random, log_mode=False, temperature=None):
    """
    Sample from a general  discrete distribution.

    :param w: A list of weights of each discrete outcome. Does not need to be normalized.
    :param  n: The number of samples to return.
    :param rng: The random number generator to use (e.g. as returned by *random.RandomState)
    :param log_mode: If *True*, interpret *w* as the log of the true weights. If *False*, interpret *w* as the literal
    weights. Default *False*.
    :param temperature: The soft-max annealing temperature. *None* indicates not to use soft-max. A temperature of 1 corresponds to no modification to *w*.

     In $lim w\to 0, returns $argmax(w)$. In $lim w\to\infty$, returns *Uniform(len(w))*
    :return: A list of *n* integers, corresponding to the indices of *w* that were chosen.
    """
    w = asarray(w, 'd')
    softmax = temperature is not None
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
