# Sempler: generate synthetic and realistic semi-synthetic data with known ground truth for causal discovery

[![PyPI version](https://badge.fury.io/py/sempler.svg)](https://badge.fury.io/py/sempler)
[![Downloads](https://static.pepy.tech/badge/sempler)](https://pepy.tech/project/sempler)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

![Real and semi-synthetic data produced from the Sachs dataset](./docs/marginals.png)

[Documentation at https://sempler.readthedocs.io/en/latest/]

Sempler allows you to generate synthetic data from SCMs and semi-synthetic data with known causal ground truth but distributions closely resembling those of a real data set of choice. It is one of the software contributions of the paper [*"Characterization and Greedy Learning of Gaussian Structural Causal Models under Unknown Interventions"*](https://arxiv.org/abs/2211.14897) by Juan L. Gamella, Armeen Taeb, Christina Heinze-Deml and Peter Bühlmann. You can find more details in Appendix E of the paper.

If you find this code useful, please consider citing:

```
@article{gamella2022characterization,
  title={Characterization and greedy learning of Gaussian structural causal models under unknown interventions},
  author={Gamella, Juan L and Taeb, Armeen and Heinze-Deml, Christina and B{\"u}hlmann, Peter},
  journal={arXiv preprint arXiv:2211.14897},
  year={2022}
}
```

## Overview

The semi-synthetic data generation procedure is implemented in the class `sempler.DRFNet` (see [docs](https://sempler.readthedocs.io/en/latest/.)). A detailed explanation of the procedure can be found in Appendix E of the [paper](https://arxiv.org/abs/2211.14897).

Additionally, you can generate purely synthetic data from general additive-noise models. Two classes are defined for this purpose.

- `sempler.ANM` is for general (acyclic) additive noise SCMs. Any assignment function is possible, as are the distributions of the noise terms.
- `sempler.LGANM` is for linear Gaussian SCMs. While this is also possible with `sempler.ANM`, this class simplifies the interface and offers the additional functionality of sampling "in the population setting", i.e. by returning a symbolic gaussian distribution (see `sempler.LGANM.sample` and `sempler.NormalDistribution`).

To allow for random generation of SCMs and interventional distributions, the module `sempler.generators` contains functions to sample random DAGs and intervention targets.

## Installation

You can clone this repo or install using pip. To install sempler in its most basic form, i.e. to generate purely synthetic data with `sempler.ANM` and `sempler.LGANM`, simply run
```
pip install sempler
```

To install the additional dependencies needed for the semi-synthetic data generation procedure, run

```
pip install sempler[DRFNet]
```

which will install sempler with the additional `rpy2` dependency. You will also need:
- an `R` installation; you can find an installation guide [here](https://rstudio-education.github.io/hopr/starting.html)
- the `R` package `drf`, which you can install by typing `install.packages("drf")` in an R terminal


### Versioning

Sempler is still at its infancy and its API is subject to change. Non backward-compatible changes to the API are reflected by a change to the minor or major version number,

> e.g. *code written using sempler==0.1.2 will run with sempler==0.1.3, but may not run with sempler==0.2.0.*

## Documentation

You can find the full documentation at https://sempler.readthedocs.io/en/latest/.
  
## Feedback

Feedback is most welcome! You can add an issue  or send an [email](mailto:juan.gamella@stat.math.ethz.ch>).
