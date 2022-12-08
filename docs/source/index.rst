Welcome to sempler's documentation!
===================================

Sempler allows you to generate generate semi-synthetic data with known causal ground truth but distributions closely resembling those of a real data set of choice. It is one of the software contributions of the paper `"Characterization and Greedy Learning of Gaussian Structural Causal Models under Unknown Interventions" <https://arxiv.org/abs/2211.14897>`__ by Juan L. Gamella, Armeen Taeb, Christina Heinze-Deml and Peter Bühlmann. You can find more details in Appendix F of the `paper <https://arxiv.org/pdf/2211.14897.pdf>`__.

If you find this code useful, please consider citing:::

  @article{gamella2022characterization,
    title={Characterization and Greedy Learning of Gaussian Structural Causal Models under Unknown Interventions},
    author={Gamella, Juan L. and Taeb, Armeen and Heinze-Deml, Christina and B\"uhlmann, Peter},
    journal={arXiv preprint arXiv:2211.14897},
    year={2022}
  }

Overview
--------

The semi-synthetic data generation procedure is implemented in the class :class:`sempler.DRFNet`. For now, only an `R` implementation of distributional random forests [2] is available. Thus, to run the procedure you will additionally need

- an `R` installation; you can find an installation guide `here <https://rstudio-education.github.io/hopr/starting.html>`__
- the `R` package ``drf``, which you can install by typing ``install.packages("drf")`` in an R terminal

Sempler also allows you can generate purely synthetic data from general structural causal models with additive noise. Two classes are defined for this purpose.

- :class:`sempler.ANM` is for general (acyclic) additive noise SCMs. Any assignment function is possible, as are the distributions of the noise terms.
- :class:`sempler.LGANM` is for linear Gaussian SCMs. While this is also possible with :class:`sempler.ANM`, this class simplifies the interface and offers the additional functionality of sampling "in the population setting", i.e. by returning a symbolic gaussian distribution (see :func:`sempler.LGANM.sample` and :class:`sempler.NormalDistribution`).

To allow for random generation of SCMs and interventional distributions, the module :class:`sempler.generators` contains functions to sample random DAGs and intervention targets.
 
------

.. _references main:

References
**********

[1] Gamella, J.L, Taeb, A., Heinze-Deml, C., & Bühlmann, P. (2022). Characterization and greedy learning of Gaussian structural causal models under unknown noise interventions. *arXiv preprint arXiv:2211.14897*, 2022.

Versioning
**********

Sempler is still at its infancy and its API is subject to change. Non backward-compatible changes to the API are reflected by a change to the minor or major version number, e.g.

    *code written using sempler==0.1.2 will run with sempler==0.1.3, but may not run with sempler==0.2.0.*

License
*******

Sempler is open-source and shared under a BSD 3-Clause License. You can find the source code in the `GitHub repository <https://github.com/juangamella/sempler>`__.

Feedback
********

Feedback is most welcome! You can add an issue in sempler's `repository <https://github.com/juangamella/sempler>`__ or send an `email <mailto:juan.gamella@stat.math.ethz.ch>`__.
    
About the Name
**************

*Structural causal models* are sometimes referred to in the literature as *structural equation models*, or SEMs. The name sempler comes from *SEM sampler*.




.. toctree::
   :maxdepth: 0
   :caption: Contents:

   self
   anm
   lganm
   drfnet
   normal_distribution
   generators
   


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
