# Sempler: Sampling observational and interventional data from general structural equation models (SEMs)

You can find the full docs at https://sempler.readthedocs.io/en/latest/.

### Installation
You can clone this repo or install using pip:
```
pip install sempler
```

Sempler is still at its infancy and its API is subject to change. Non backward-compatible changes to the API are reflected by a change to the minor or major version number,

> e.g. *code written using sempler==0.1.2 will run with sempler==0.1.3, but may not run with sempler==0.2.0.*

## Overview

Sempler allows you to generate observational and interventional data from general structural causal models.

Two classes are defined for this purpose.

- `sempler.ANM` is for general (acyclic) additive noise SCMs. Any assignment function is possible, as are the distributions of the noise terms.
- `sempler.LGANM` is for linear Gaussian SCMs. While this is also possible with `sempler.ANM`, this class simplifies the interface and offers the additional functionality of sampling "in the population setting", i.e. by returning a symbolic gaussian distribution (see `sempler.LGANM.sample` and `sempler.NormalDistribution`).

To allow for random generation of SCMs and interventional distributions, the module `sempler.generators` contains functions to sample random DAGs and intervention targets.

## Documentation

You can find the docs at https://sempler.readthedocs.io/en/latest/.
  
## Feedback

Feedback is most welcome! You can add an issue  or send an [email](mailto:juan.gamella@stat.math.ethz.ch>).
