Storage classes
================

Introduction
--------------
.. py:module:: storage

Provides classes for storing the results of a run of an algorithm. Currently supports two classes: :py:class:`LocalStore` stores data locally, using the Python shelve library. :py:class:`CloudStore` stores data on the cloud, via picloud buckets (which are ultimately a thin abstraction of S3 buckets). You probably will not need to use these classes directly, as the :py:class:`runner.Job` class manages them.


Full documentation
-------------------


.. autoclass:: DataStore
   :members:

.. autoclass:: LocalStore
   :members:

.. autoclass:: CloudStore
   :members: