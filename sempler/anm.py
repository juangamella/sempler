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

"""
Main module containing the SCM classes (sempler.ANM and
sempler.LGANM), the sempler.NormalDistribution class and additional
functions to generate random graphs.
"""

import numpy as np
from copy import deepcopy
import sempler.utils as utils
import sempler.functions as functions

# ---------------------------------------------------------------------
# ANM class


class ANM:
    """Class to represent a general (acyclic) additive noise model.

    Parameters
    ----------
    A : numpy.ndarray
        The `p x p` adjacency matrix specifying the functional
        dependencies, where `p` is the number of variables and `A[i,j]
        != 0` if `i` appears in the assignment of `j` (i.e. `i -> j`).
    assignments : list of (function or None)
        A list of p functions representing the functional assignments
        of each variable. Each function must take a vector of
        observations where each column corresponds to a parent as
        specified by the adjacency matrix `A`, or be `None` or
        `sempler.functions.null` if the variable has no parents.
    noise_distributions : list of function
        A list of `p` functions that generate samples from each
        variable's noise distribution (see sempler.noise for
        details).

    Raises
    ------
    ValueError
        If the given adjacency does not correspond to a DAG.

    Example
    -------

    Constructing an ANM.

    >>> import sempler
    >>> import sempler.noise as noise
    >>> import numpy as np

    (1) Define the connectivity matrix:

    >>> A = np.array([[0, 0, 0, 1, 0],
    ...               [0, 0, 1, 0, 0],
    ...               [0, 0, 0, 1, 0],
    ...               [0, 0, 0, 0, 1],
    ...               [0, 0, 0, 0, 0]])

    (2) Define the noise distributions (see sempler.noise):

    >>> noise_distributions = [noise.normal(0,1)] * 5

    (3) Define the variable assignments:

    >>> assignments = [None, None, np.sin, lambda x: np.exp(x[:,0]) + 2*x[:,1], lambda x: 2*x]

    Putting it all together:

    >>> anm = sempler.ANM(A, assignments, noise_distributions)

    An exception is raised if the adjacency matrix does not belong to a DAG:

    >>> A[4,0] = 1
    >>> sempler.LGANM(A, (0,0), (1,1))
    Traceback (most recent call last):
      ...
    ValueError: The given graph is not a DAG.

    Attributes
    ----------
    A : numpy.ndarray
        The adjacency matrix specifying the functional dependencies.
    p : int
        The number of variables (size) of the SCM.
    assignments : list of function
        The assignment functions of the variables.
    noise_distributions : list of function
        A list of functions representing the noise term distribution
        of each variable (see sempler.noise).

    """

    def __init__(self, A, assignments, noise_distributions):
        self.ordering = utils.topological_ordering(A)
        self.p = len(A)
        self.A = deepcopy(A)
        self.assignments = [functions.null if fun is None else deepcopy(fun) for fun in assignments]
        self.noise_distributions = deepcopy(noise_distributions)

    def sample(self, n, do_interventions={}, shift_interventions={}, noise_interventions={}, random_state=None):
        """Generates i.i.d. observations from the ANM, under the given do, shift or
        noise interventions. If no interventions are given, sample from the observational
        distribution.

        Parameters
        ----------
        n : int
            The size of the sample (i.e. number of observations).
        do_interventions : dict, optional
            A dictionary where keys correspond to the intervened
            variables, and the values are the distribution functions
            (see sempler.noise) to generate samples for each
            intervened variable.
        shift_interventions : dict, optional
            A dictionary where keys correspond to the intervened
            variables, and the values are the distribution functions
            (see sempler.noise) to generate noise samples which are
            added to the intervened variables.
        noise_interventions : dict, optional
            A dictionary where keys correspond to the intervened
            variables, and the values are the distribution functions
            (see sempler.noise) of the new noise.
        random_state: int, optional
            To set the random state for reproducibility,
            i.e. successive calls will return the same sample.

        Returns
        -------
        numpy.ndarray
            An array containing the sample, with each column
            corresponding to a variable.

        Examples
        --------

        Sampling from the observational setting:

        >>> samples = anm.sample(100)

        Sampling under a shift intervention on variable 1:

        >>> import sempler.noise as noise
        >>> samples = anm.sample(100, shift_interventions = {1: noise.normal(0,1)})

        Sampling under a noise intervention on variable 0 and a do intervention on variable 2:

        >>> samples = anm.sample(100,
        ...                      noise_interventions = {0: noise.normal()},
        ...                      do_interventions = {2 : noise.uniform()})

        """
        # Set random state (if requested)
        np.random.seed(random_state) if random_state is not None else None
        # Sample according to a topological ordering of the connectivity matrix
        X = np.zeros((n, self.p))
        for i in self.ordering:
            # If i is do intervened, sample from the corresponding
            # interventional distribution
            if i in do_interventions:
                X[:, i] = do_interventions[i](n)
            # Otherwise maintain dependence on parents
            else:
                assignment = np.transpose(self.assignments[i](X[:, self.A[:, i] != 0]))
                # Shift-intervention: add noise from given distribution
                if i in shift_interventions:
                    noise = self.noise_distributions[i](n) + shift_interventions[i](n)
                # Noise-intervention: sample noise from given distribution
                elif i in noise_interventions:
                    noise = noise_interventions[i](n)
                # No intervention: sample noise from original distribution
                else:
                    noise = self.noise_distributions[i](n)
                X[:, i] = assignment + noise
        return X


# To run the ANM.sample doctests
if __name__ == '__main__':
    import doctest
    import sempler.noise
    # Build ANM
    A = np.array([[0, 0, 0, 1, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0]])
    noise_distributions = [sempler.noise.normal(0, 1)] * 5
    assignments = [None, None, np.sin, lambda x: np.exp(x[:, 0]) + 2 * x[:, 1], lambda x: 2 * x]
    anm = ANM(A, assignments, noise_distributions)
    doctest.testmod(extraglobs={'anm': anm}, verbose=True)
