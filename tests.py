"""
tests.py
Nose unit tests
"""

from __future__ import division

def test_namespace():
    """
    Is the top-level namespace exposing the right symbols?
    Can the key classes be created?
    """
    import scaffold
    scaffold.State()
    scaffold.History()
    scaffold.Chain()
    scaffold.DataSource()
    scaffold.Experiment()
