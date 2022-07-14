# Copyright 2021 Juan L Gamella

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Main module containing the sempler.LGANM class, to represent and
sample from a linear Gaussian additive noise model.
"""

import numpy as np
import sempler.utils as utils
from sempler.normal_distribution import NormalDistribution

# ---------------------------------------------------------------------
# LGANM class


class LGANM:
    """Represents a linear model with Gaussian additive noise.

    Parameters
    ----------
    W : array_like
        Connectivity (weights) matrix representing a DAG.
    variances : numpy.ndarray or tuple
        The variances of the noise terms, or a tuple representing
        the lower/upper bounds to sample them from a uniform
        distribution.
    means : numpy.ndarray or tuple
        The means of the noise terms, or a tuple representing
        the lower/upper bounds to sample them from a uniform
        distribution.

    Raises
    ------
    ValueError
        If the given connectivity does not correspond to a DAG.

    Examples
    --------

    Constructing a linear Gaussian SCM.

    >>> import sempler
    >>> import numpy as np

    (1) Define the connectivity matrix:

    >>> W = np.array([[0, 0, 0, 0.1, 0],
    ...               [0, 0, 2.1, 0, 0],
    ...               [0, 0, 0, 3.2, 0],
    ...               [0, 0, 0, 0, 5.0],
    ...               [0, 0, 0, 0, 0  ]])

    (2a) With explicit means and variances:

    >>> means = np.array([0,1,2,3,4])
    >>> variances = np.array([1,1,1,1,1])
    >>> lganm = sempler.LGANM(W, means, variances)

    (2b) With randomly sampled means and variances:

    >>> lganm = sempler.LGANM(W, (0,1), (0,1))

    An exception is thrown when the connectivity matrix does not correspond to a DAG:

    >>> A = [[0,1,0],[0,0,1],[1,0,0]]
    >>> sempler.LGANM(A, (0,0), (1,1))
    Traceback (most recent call last):
      ...
    ValueError: The given graph is not a DAG.

    Attributes
    ----------
    W : array_like
        Connectivity (weights) matrix representing a DAG.
    variances : numpy.ndarray
        The variances of the noise terms.
    means : numpy.ndarray
        The means of the noise terms.
    p : int
        The number of variables (size) of the SCM.

    """

    def __init__(self, W, means, variances, random_state=None):
        # Set connectivity matrix
        W = np.atleast_2d(W)
        if not utils.is_dag(W):
            raise ValueError("The given graph is not a DAG.")
        self.W = W.copy()
        self.p = len(W)

        rng = np.random.default_rng(random_state)

        # Set variances
        if isinstance(variances, tuple) and len(variances) == 2:
            self.variances = rng.uniform(
                variances[0], variances[1], size=self.p)
        elif type(variances) == np.ndarray and len(variances) == self.p:
            self.variances = variances.copy()
        else:
            raise ValueError(
                "Unexpected value for variances. Expected a two-element tuple or numpy.ndarray of length p.")
        # Set means
        if isinstance(means, tuple) and len(means) == 2:
            self.means = rng.uniform(means[0], means[1], size=self.p)
        elif type(means) == np.ndarray and len(means) == self.p:
            self.means = means.copy()
        else:
            raise ValueError(
                "Unexpected value for means. Expected a two-element tuple or numpy.ndarray of length p.")

    def sample(self, n=100, population=False, do_interventions={}, shift_interventions={}, noise_interventions={}, random_state=None):
        """Generates n observations from the linear Gaussian SCM, under the
        given do, shift or noise interventions. If none are given,
        sample from the observational distribution.

        Parameters
        ----------
        n : int, optional
            The size of the sample (i.e. number of
            observations). Defaults to 100.
        population : bool, optional
            If True, the function returns a symbolic normal
            distribution instead of samples (see
            sempler.NormalDistribution). Defaults to False.
        do_interventions : dict, optional
            A dictionary where keys correspond to the intervened
            variables, and the values are tuples representing the new
            mean/variance of the intervened variable, e.g. {1: (1,2)}.
        shift_interventions : dict, optional
            A dictionary where keys correspond to the intervened
            variables, and the values are tuples representing the
            mean/variance of the noise which is added to the
            intervened variables, e.g. {1: (1,2)}.
        noise_interventions : dict, optional
            A dictionary where keys correspond to the intervened
            variables, and the values are tuples representing the
            mean/variance of the new noise, e.g. {1: (1,2)}.
        random_state: int, optional
            To set the random state for reproducibility. Succesive
            calls with the same random state will return the same
            sample.

        Returns
        -------
        numpy.ndarray or sempler.NormalDistribution
            An array containing the sample, where each column
            corresponds to a variable; or, if population=True, a
            symbolic normal distribution (see
            sempler.NormalDistribution).

        Examples
        --------

        Sampling the observational environment in the "population setting"

        >>> distribution = lganm.sample(population = True)

        Sampling under a shift intervention on variable 1 with standard gaussian noise

        >>> samples = lganm.sample(100, shift_interventions = {1: (0,1)})

        Sampling under a noise intervention on variable 0 and a do intervention on variable 2:

        >>> samples = lganm.sample(100,
        ...                       noise_interventions = {0: (1,2)},
        ...                       do_interventions = {2 : (3,4)})

        Interventions can also be deterministic, i.e. setting a variable/noise term to a fixed value:

        >>> samples = lganm.sample(5, do_interventions = {2 : (99,0)})
        >>> samples[:,2]
        array([99., 99., 99., 99., 99.])

        """
        # Must copy as they can be changed by interventions, but we
        # still want to keep the observational SEM
        W = self.W.copy()
        variances = self.variances.copy()
        means = self.means.copy()

        # Perform shift interventions
        if shift_interventions:
            shift_interventions = _parse_interventions(shift_interventions)
            targets = shift_interventions[:, 0].astype(int)
            means[targets] += shift_interventions[:, 1]
            variances[targets] += shift_interventions[:, 2]

        # Perform noise interventions. Note that they take preference
        # i.e. "override" shift interventions
        if noise_interventions:
            noise_interventions = _parse_interventions(noise_interventions)
            targets = noise_interventions[:, 0].astype(int)
            means[targets] = noise_interventions[:, 1]
            variances[targets] = noise_interventions[:, 2]

        # Perform do interventions. Note that they take preference
        # i.e. "override" shift and noise interventions
        if do_interventions:
            do_interventions = _parse_interventions(do_interventions)
            targets = do_interventions[:, 0].astype(int)
            means[targets] = do_interventions[:, 1]
            variances[targets] = do_interventions[:, 2]
            W[:, targets] = 0

        # Sampling by building the joint distribution
        A = np.linalg.inv(np.eye(self.p) - W.T)
        mean = A @ means
        covariance = A @ np.diag(variances) @ A.T
        distribution = NormalDistribution(mean, covariance)
        if not population:
            return distribution.sample(n, random_state=random_state)
        else:
            return distribution


def _parse_interventions(interventions_dict):
    """Used internally by LGANM.sample. Transforms the interventions from
    a dictionary to an array"""
    interventions = []
    for (target, params) in interventions_dict.items():
        # Mean and variance provided
        if type(params) == tuple and len(params) == 2:
            interventions.append([target, params[0], params[1]])
        # Only mean provided, assume we're setting the variable to a deterministic value
        elif type(params) in [float, int]:
            interventions.append([target, params, 0])
        else:
            raise ValueError("Wrongly specified intervention")
    return np.array(interventions)


# To run the LGANM.sample doctests
if __name__ == '__main__':
    import doctest
    # Build LGANM
    W = np.array([[0, 0, 0, 0.1, 0],
                  [0, 0, 2.1, 0, 0],
                  [0, 0, 0, 3.2, 0],
                  [0, 0, 0, 0, 5.0],
                  [0, 0, 0, 0, 0]])
    lganm = LGANM(W, (0, 1), (0, 1))
    doctest.testmod(extraglobs={'lganm': lganm}, verbose=True)
