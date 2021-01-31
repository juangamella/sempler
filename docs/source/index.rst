Welcome to sempler's documentation!
===================================

Sempler allows you to generate observational and interventional data from general structural causal models.

Two classes are defined for this purpose.

- :class:`sempler.ANM` is for general (acyclic) additive noise SCMs. Any assignment function is possible, as are the distributions of the noise terms.
- :class:`sempler.LGANM` is for linear Gaussian SCMs. While this is also possible with :class:`sempler.ANM`, this class simplifies the interface and offers the additional functionality of sampling "in the population setting", i.e. by returning a symbolic gaussian distribution (see :func:`sempler.LGANM.sample` and :class:`sempler.NormalDistribution`).

To allow for random generation of SCMs and interventional distributions, the module :class:`sempler.generators` contains functions to sample random DAGs and intervention targets.

Versioning
----------

Sempler is still at its infancy and its API is subject to change. Non backward-compatible changes to the API are reflected by a change to the minor or major version number, e.g.

    *code written using sempler==0.1.2 will run with sempler==0.1.3, but may not run with sempler==0.2.0.*

License
-------

Sempler is open-source and shared under a BSD 3-Clause License. You can find the source code in the `GitHub repository <https://github.com/juangamella/sempler>`__.

Feedback
--------

Feedback is most welcome! You can add an issue in sempler's `repository <https://github.com/juangamella/sempler>`__ or send an `email <mailto:juan.gamella@stat.math.ethz.ch>`__.
    
About the Name
--------------

*Structural causal models* are sometimes referred to in the literature as *structural equation models*, or SEMs. The name sempler comes from *SEM sampler*.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   self
   anm
   lganm
   normal_distribution
   generators
   


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
