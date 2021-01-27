sempler.ANM
===========

The `sempler.ANM` class allows to define and sample from general
additive noise models. Any assignment function is possible, as are the
noise distributions. The underlying graph is assumed to be acyclic.

The ANM is represented by (1) the adjacency matrix of the underlying
graph, (2) the assignment functions of each variable, and (3) the
distributions of each variable's noise term.


.. autoclass:: sempler.ANM
   :members:
