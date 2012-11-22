"""
A set of utility functions used in various modules
"""

from __future__ import division
from numpy import *
from pdb import set_trace

def sample(w, n, rng=random):
    """
    Sample from a general  discrete distribution
    """
    w = asarray(w, double)
    c = cumsum(w)
    r = rng.rand(n) * c[-1]
    return searchsorted(c, r)
