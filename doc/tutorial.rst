Tutorial
============

.. py:module:: scaffold

Summary of usage
---------------------

To use the scaffold module, you will in general write at least three classes that inherit from base classes provided by the module.

* Inherit from :py:class:`State`, defining a class whose instance variables are the state of the algorithm at the start of each iteration.
* Inherit from :py:class:`DataSource`, defining a class which will load in the dataset that the algorithm will be trained and tested on. The dataset can either be procedurely generated or come from an arbitrary corpus. Some common types of synthetic data are already defined in the :py:mod:`datasources` module, in which case you do not need to write your own DataSoure.
* Inherit from :py:class:`Chain`, implementing methods that implement the transition operator that takes the state at one iteration to the state at the next iteration, as well as a method that returns a starting state.

To execute the algorithm, you will

* Create an :py:class:`Experiment` instance and then tell it about your custom classes (the state and a set of data sources of chains).
* Tell your *Experiment* instance how many times to execute each (data source, chain) combination
* Call :py:meth:`Experiment.run`, passing in a parameter that controls whether you want the experiment to execute locally or remote
* Fetch the history of the job executions into local RAM. Histories are stored in the :py:class:`History` object. In the simplest case of loading all results at once, you will call the :py:meth:`Experiment.fetch_results` method. More fine-grained control is available to only load part of the results (useful if the execution traces are too large to fit in memory or would take too long to transfer from the remote server).
* Use methods defined on the :py:class:`History` to analyze the results.

Demo overview
-----------------
(Will go through basic demo)