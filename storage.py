"""
storage.py

Provides persistent storage classes.

Meant for storing histories of algorithm runs, as well as any cached analysis or visualizations.
"""

from __future__ import division
import os
import shelve
from helpers import VirtualException, hash_robust
import cloud
import cPickle
import cStringIO
import helpers

class DataStore(object):
    """
    Simple abstract data persistence interface, based around storing and retrieving
    Python objects by a string label. Provides a dict-like interface
    """
    def store(self, object, key):
        raise VirtualException()

    def load(self, key):
        raise VirtualException()

    def close(self):
        pass

    def __getitem__(self, key):
        return self.load(key)

    def __setitem__(self, key, value):
        self.store(value, key)

class LocalStore(DataStore):
    """
    Implements a data store using the Python *shelve* library, where the shelf is stored locally.

    Useful for debugging. Write-storage is not thread-safe.
    """
    def __init__(self, filename='data/data.shelve'):
        self.filename = filename
        self.shelve = shelve.open(filename, writeback=False, protocol=-1)

    def store(self, object, key):
        key_str = helpers.hash_robust(key)
        self.shelve[key_str] = object

    def load(self, key):
        key_str = helpers.hash_robust(key)
        return self.shelve[key_str]

    def close(self):
        self.shelve.close()

    def __in__(self, key):
        return helpers.hash_robust(key) in self.shelve.keys()

class CloudStore(DataStore):
    """
    Implements a data store using the picloud *bucket* system. The objects must be serialiable via the *cPickle* library.

    Note that picloud charges both for storage and for transmitting data, so maybe best not to store gigantic objects
    with this system.
    """
    def __init__(self, path='scaffold'):
        self.path = path

    def hash_key(self, key):
        return os.path.join(self.path, hash_robust(key))

    def store(self, object, key):
        data = cPickle.dumps(object, protocol=-1)
        file_form = cStringIO.StringIO(data)
        cloud.bucket.putf(file_form, self.hash_key(key))

    def load(self, key):
        file_form = cloud.bucket.getf(self.hash_key(key))
        data = file_form.read()
        return cPickle.loads(data)

    def __delitem__(self, key):
        cloud.bucket.remove(self.hash_key(key)) #untested

    

