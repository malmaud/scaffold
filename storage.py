"""
storage.py

Provides persistent storage classes.

Meant for storing histories of algorithm runs, as well as any cached analysis or visualizations.
"""

from __future__ import division
import hashlib
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
    def __init__(self):
        self.path = ''

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

    def hash_key(self, key):
        raw_hash = hash(key)
        s = hashlib.sha1(str(raw_hash)).hexdigest()
        s = s[0:20]
        return os.path.join(self.path, s)

class LocalStore(DataStore):
    """
    Implements a data store using the Python *shelve* library, where the shelf is stored locally.

    Useful for debugging. Write-storage is not thread-safe.
    """
    def __init__(self, filename='data/data.shelve'):
        DataStore.__init__(self)
        self.filename = filename
        self.shelve = shelve.open(filename, writeback=False, protocol=-1)

    def store(self, object, key):
        key_str = self.hash_key(key)
        self.shelve[key_str] = object

    def raw_store(self, key, object):
        self.shelve[key] = object

    def load(self, key):
        key_str = self.hash_key(key)
        return self.shelve[key_str]

    def close(self):
        self.shelve.close()

class CloudStore(DataStore):
    """
    Implements a data store using the picloud *bucket* system. The objects must be serialiable via the *cPickle* library.

    Note that picloud charges both for storage and for transmitting data, so maybe best not to store gigantic objects
    with this system.
    """
    def __init__(self, path='scaffold'):
        DataStore.__init__(self)
        self.path = path

    def store(self, object, key):
        data = cPickle.dumps(object, protocol=-1)
        file_form = cStringIO.StringIO(data)
        cloud.bucket.putf(file_form, self.hash_key(key))

    def load(self, key):
        file_form = cloud.bucket.getf(self.hash_key(key))
        data = file_form.read()
        return cPickle.loads(data)

    def raw_store(self, raw_key, object):
        data = cPickle.dumps(object, protocol=-1)
        file_form = cStringIO.StringIO(data)
        cloud.bucket.putf(file_form, raw_key)

    def __delitem__(self, key):
        cloud.bucket.remove(self.hash_key(key)) #untested

    

