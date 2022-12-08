# Copyright 2020 Juan Luis Gamella Martin

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

"""Module containing the procedure for semi-synthetic data
generation, as described in the paper


Please cite it if you use this procedure in your work.

<TODO: Dependencies>
"""

import numpy as np
import drf
import sempler.utils
import copy
import pandas as pd
import time

_GRAPH_TYPE_ERROR = "graph must be a 2-dimensional numpy.ndarray"
_DATA_TYPE_ERROR = "data must be a list of 2-dimensional numpy.ndarray"
_N_TYPE_ERROR = "n must be a positive int or list of positive ints"

# --------------------------------------------------------------------
# Auxiliary functions


def _bootstrap(data, n=None, random_state=None):
    """Generate a bootstrap sample from the given data.

    Parameters
    ----------
    data : numpy.ndarray
        The matrix containing the original sample, with the first axis
        corresponding to observations.
    n : int or NoneType, default=None
        The size of the bootstrap sample. If `None` (default), it is
        set to be the size of the original sample.
    random_state : int or NoneType, default=None
        To set the random seed for reproducibility.

    Returns
    -------
    sample : numpy.ndarray
        The resulting bootstrap sample.

    Examples
    --------

    >>> rng = np.random.default_rng(42)
    >>> data = rng.uniform(size=(10,10))
    >>> sample = _bootstrap(data)
    >>> sample.shape
    (10, 10)

    >>> data = rng.uniform(size=5)
    >>> sample = _bootstrap(data)
    >>> sample.shape
    (5,)
    >>> _bootstrap(data, n=3, random_state=42)
    array([0.90858069, 0.96917638, 0.96917638])
    """
    # Sample with replacement
    n = len(data) if n is None else n
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(data), n, replace=True)
    sample = data[idx]
    return sample


class BayesianNetwork:
    """Parent class implementing the basic input checks on behalf of the
    inheriting classes.

    Parameters
    ----------
    graph : numpy.ndarray
        Two dimensional array representing a DAG connectivity, where
        `graph[i,j] != 0` implies the edge `i -> j`.
    p : int
        The number of variables in the graph and data.
    e : int
        The number of environments in the data.
    Ns : list of int
        The sizes of the samples from each environment.
    N : int
        The total number of observations (across all environments).
    """

    def __init__(self, graph, data, verbose=False):
        # Check inputs: graph
        if not isinstance(graph, np.ndarray):
            raise TypeError(_GRAPH_TYPE_ERROR)
        elif graph.ndim != 2:
            raise ValueError(_GRAPH_TYPE_ERROR)
        elif not sempler.utils.is_dag(graph):
            raise ValueError("graph is not a DAG.")

        # Check inputs: data
        if not isinstance(data, list):
            raise TypeError(_DATA_TYPE_ERROR)
        else:
            for sample in data:
                # Check every sample is a numpy.ndarray
                if not isinstance(sample, np.ndarray):
                    raise TypeError(_DATA_TYPE_ERROR)
                # with two dimensions
                elif sample.ndim != 2:
                    raise ValueError(_DATA_TYPE_ERROR)
                # and the number of variables matches that in the graph
                elif sample.shape[1] != graph.shape[1]:
                    raise ValueError(
                        "graph and data have different number of variables"
                    )

        # Set parameters
        self.graph = (graph != 0).astype(int)
        self._data = copy.deepcopy(data)
        self.p = graph.shape[1]
        self.e = len(self._data)
        self.Ns = [len(sample) for sample in self._data]
        self.N = np.sum(self.Ns)
        self._ordering = sempler.utils.topological_ordering(self.graph)

    def sample(self, n):
        # Check input: n
        if n is not None and type(n) not in [int, list]:
            raise TypeError(_N_TYPE_ERROR)
        elif type(n) == int and n <= 0:
            raise ValueError(_N_TYPE_ERROR)
        elif type(n) == list:
            for i in n:
                if type(i) != int:
                    raise TypeError(_N_TYPE_ERROR)
                elif i <= 0:
                    raise ValueError(_N_TYPE_ERROR)
        return None


# --------------------------------------------------------------------
# DRFNet class


class DRFNet(BayesianNetwork):
    """Fit a non-parametric Bayesian network with the given adjacency to the
    data, modelling the conditionals through distributional random
    forests.

    Parameters
    ----------
    graph : numpy.ndarray
        Two dimensional array representing the desired ground truth,
        where `graph[i,j] != 0` implies the edge `i -> j`.
    data : list of numpy.ndarray
        The data to which the Bayesian network will be fitted, as a list
        of two-dimensional arrays containing the samples from each
        environment, where rows correspond to observations and columns
        to variables.
    verbose : bool, default=False
        If debugging traces should be printed.

    Raises
    ------
    TypeError :
        If the graph or data are of the wrong type.
    ValueError :
        If the given adjacency is not a DAG or the `samples` in
        the data are of the wrong size.

    Examples
    --------

    Fitting to some random data (from two environments) and a random graph.

    >>> rng = np.random.default_rng(42)
    >>> data = [rng.uniform(size=(100, 5)) for _ in range(2)]
    >>> graph = sempler.generators.dag_avg_deg(p=5, k=2, random_state=42)
    >>> network = DRFNet(graph, data)
    >>> network.graph
    array([[0, 0, 1, 1, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 1, 0, 1],
           [0, 0, 0, 0, 0]])

    """

    def __init__(self, graph, data, verbose=False):
        super().__init__(graph, data, verbose)

        # Fit distributional random forests, i.e. one per (node with
        # parents) x environment
        if verbose:
            start = time.time()
            print("-------------------------------------") if verbose else None
            print("Fitting distributional random forests") if verbose else None
        self._random_forests = np.empty((self.p, self.e), dtype=object)
        for i in range(self.p):
            parents = sempler.utils.pa(i, self.graph)
            print(
                "  node %d/%d - parents %s     " % (i + 1, self.p, parents)
            ) if verbose else None
            # Don't fit a DRF if node is a source node
            if parents != set():
                for k in range(self.e):
                    print(
                        "    fitting environment %d/%d" % (k + 1, self.e), end="  \r"
                    ) if verbose else None
                    # Using default values from DRF repository
                    DRF = drf.drf(
                        min_node_size=15, num_trees=2000, splitting_rule="FourierMMD"
                    )
                    Y = pd.DataFrame(self._data[k][:, i])
                    X = pd.DataFrame(self._data[k][:, sorted(parents)])
                    DRF.fit(X, Y)
                    # print(DRF.info())
                    self._random_forests[i, k] = DRF
        print(
            "Done in %0.2f seconds           " % (time.time() - start)
        ) if verbose else None

    def sample(self, n=None, random_state=None):
        """Generate a sample from the fitted Bayesian network.

        Parameters
        ----------
        n : int or list of ints or NoneType, default=None
            The desired sample size. If `None` the sample size from
            each environment matches that of the original
            data. Otherwise: either a single value (same number of
            observations per environment) or a list specifying the
            size of the sample from each environment.
        random_state : int or NoneType, default=None
            To set the random seed for reproducibility.

        Returns
        -------
        sample : list of numpy.ndarray
            The resulting samples, one per environment.

        Raises
        ------
        TypeError :
            If `n` is of the wrong type.
        ValueError :
            If `n` is non-positive or has the wrong length (number of
            environments).

        Examples
        --------

        If not specifying a sample size, the sample sizes in the new
        data match those of the original:

        >>> new_data = network.sample()
        >>> len(new_data)
        2
        >>> [len(sample) for sample in new_data]
        [100, 100]

        Specifying the sample sizes:

        >>> new_data = network.sample(3)
        >>> [len(sample) for sample in new_data]        
        [3, 3]
        >>> new_data = network.sample([2, 3])
        >>> [len(sample) for sample in new_data]
        [2, 3]
        """
        # Checks inputs
        super().sample(n)
        # Set sample sizes
        if n is None:
            n = self.Ns
        elif type(n) == int:
            n = [n] * self.e
        # Generate a sample for each environment
        sampled_data = []
        for k in range(self.e):
            sample = np.zeros((n[k], self.p), dtype=float)
            for i in self._ordering:
                if self._random_forests[i, k] is None:
                    # Node has no parents, generate a sample using bootstrapping
                    sample[:, i] = _bootstrap(
                        self._data[k][:, i], n[k], random_state=random_state
                    )
                else:
                    parents = sempler.utils.pa(i, self.graph)
                    new_data = pd.DataFrame(sample[:, sorted(parents)])
                    forest = self._random_forests[i, k]
                    output = forest.predict(n=1, functional="sample", newdata=new_data)
                    sample[:, i] = output.sample[:, 0, 0]
            sampled_data.append(sample)
        return sampled_data


# To run the doctests
if __name__ == "__main__":
    import doctest
    import sempler.generators

    rng = np.random.default_rng(42)
    data = [rng.uniform(size=(100, 4)) for _ in range(2)]
    graph = sempler.generators.dag_avg_deg(4, 2, 1, 1, random_state=42)
    network = DRFNet(graph, data)
    doctest.testmod(
        extraglobs={"rng": rng, "data": data, "graph": graph, "network": network}, verbose=True
    )
