sempler.DRFSCM
==============

The :class:`sempler.DRFSCM` class implements a procedure to generate semi-synthetic data for causal discovery. The procedure is described in detail in appendix <TODO> of [1]. Please cite [1] if you use this procedure for your work.

As input, the procedure takes a directed acyclic graph over some variables and a dataset consisting of their observations under different environments. The procedure then fits a non-parametric structural causal model, where the conditional distributions entailed by the graph are approximated via distributional random forests [2]. Once fitted, you can sample from this collection of forests to produce a new, semi-synthetic dataset that respects the conditional independence relationships entailed by the graph, while its marginal and conditional distributions closely match those of the original dataset.

Additional dependencies
-----------------------

To run this procedure you will need additional dependencies, which are not required for the rest of sempler's functionality. In particular,

- you will need an installation of `R`
- the `R` package <TODO>, which you can install with
- the `rpy2` python package, which you can install with

.. autoclass:: sempler.DRFSCM
   :members:

References
----------
[1] <TODO>
[2] <TODO>
