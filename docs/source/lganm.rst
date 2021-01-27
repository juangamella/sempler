sempler.LGANM
=============

The `sempler.LGANM` class allows to define and sample from linear SCMs
with Gaussian additive noise (i.e. a Gaussian Bayesian
network).

Additionally, the LGANM class allows sampling "in the
population setting", i.e. by returning a symbolic gaussian
distribution, sempler.NormalDistribution, which allows for
manipulation such as conditioning, marginalization and regression in
the population setting.

The SCM is represented by the connectivity (weights) matrix and the
noise term means and variances. The underlying graph is assumed to be acyclic.


.. autoclass:: sempler.LGANM
   :members:
