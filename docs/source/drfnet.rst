sempler.DRFNet
==============

The :class:`sempler.DRFNet` class implements a procedure to generate realistic semi-synthetic data for causal discovery. The procedure is described in detail in Appendix F of :ref:`[1]<references class>`.

**If you use this procedure for your work, please consider citing**:::
  
  @article{gamella2022characterization,
    title={Characterization and Greedy Learning of Gaussian Structural Causal Models under Unknown Interventions},
    author={Gamella, Juan L. and Taeb, Armeen and Heinze-Deml, Christina and B\"uhlmann, Peter},
    journal={arXiv preprint arXiv:2211.14897},
    year={2022}
  }

As input, the procedure takes a directed acyclic graph over some variables and a dataset consisting of their observations under different environments. The data can also be "observational", that is, from a single environment. The procedure then fits a non-parametric Bayesian network, where the conditional distributions entailed by the graph are approximated via distributional random forests :ref:`[2]<references class>`. Once fitted, you can sample from this collection of forests to produce a new, semi-synthetic dataset that respects acyclicity, causal sufficiency, and the conditional independence relationships entailed by the given graph, while its marginal and conditional distributions closely resemble those of the original dataset :ref:`[1, figure 4]<references class>`.

**Additional R dependencies**

For now, only an `R` implementation of distributional random forests [2] is available. Thus, to run the procedure you will additionally need

- an `R` installation; you can find an installation guide `here <https://rstudio-education.github.io/hopr/starting.html>`__
- the `R` package ``drf``, which you can install by typing ``install.packages("drf")`` in an R terminal

The class is documented below.
  
.. autoclass:: sempler.DRFNet
   :members:

.. _references class:

References
----------

[1] Gamella, J.L, Taeb, A., Heinze-Deml, C., & Bühlmann, P. (2022). Characterization and greedy learning of Gaussian structural causal models under unknown noise interventions. `arXiv preprint arXiv:2211.14897 <https://arxiv.org/abs/2211.14897>`__, 2022.

[2] Ćevid, D., Michel, L., Näf, J., Meinshausen, N., & Bühlmann, P. (2020). Distributional random forests: Heterogeneity adjustment and multivariate distributional regression. `arXiv preprint arXiv:2005.14458 <https://arxiv.org/abs/2005.14458>`__, 2020.
