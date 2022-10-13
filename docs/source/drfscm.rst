sempler.DRFSCM
==============

The :class:`sempler.DRFSCM` class implements a procedure to generate semi-synthetic data for causal discovery, from a given graph and dataset. The procedure is described in detail in :ref:`[1, appendix F]<references class>`. **Please cite** :ref:`[1]<references class>` **if you use this procedure for your work.**

As input, the procedure takes a directed acyclic graph over some variables and a dataset consisting of their observations under different environments. The procedure then fits a non-parametric structural causal model, where the conditional distributions entailed by the graph are approximated via distributional random forests :ref:`[2]<references class>`. Once fitted, you can sample from this collection of forests to produce a new, semi-synthetic dataset that respects the conditional independence relationships entailed by the graph, while its marginal and conditional distributions closely match those of the original dataset.

**Additional dependencies**

To run this procedure you will need additional dependencies, which are not required for the rest of sempler's functionality. In particular,

- you will need an installation of `R`; you can find a guide `here <https://rstudio-education.github.io/hopr/starting.html>`__
- the `R` package ``drf``, which you can install by typing ``install.packages("drf")`` in an R terminal
- the python packages ``pandas`` and ``rpy2``, which you can install by executing ``pip install rpy2 pandas`` in a suitable shell
  
.. autoclass:: sempler.DRFSCM
   :members:

.. _references class:

References
----------

[1] Gamella, J.L, Taeb, A., Heinze-Deml, C., & Bühlmann, P. (2022). Characterization and greedy learning of Gaussian structural causal models under unknown noise interventions . arXiv preprint arXiv:2005.14458.

[2] Ćevid, D., Michel, L., Näf, J., Meinshausen, N., & Bühlmann, P. (2020). Distributional random forests: Heterogeneity adjustment and multivariate distributional regression. `arXiv preprint arXiv:2005.14458 <https://arxiv.org/abs/2005.14458>`__.
