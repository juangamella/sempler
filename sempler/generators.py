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
This module contains functions to generate random graphs, which can then be used together with :class:`sempler.ANM` or :class:`sempler.LGANM` to produce random SCMs.
"""

import numpy as np

# ---------------------------------------------------------------------
# DAG Generating Functions


def dag_avg_deg(p, k, w_min=1, w_max=1, return_ordering=False, random_state=None, debug=False):
    """Generate an Erdos-Renyi graph with p nodes and average degree k,
    and orient edges according to a random ordering. Sample the edge
    weights from a uniform distribution.

    Parameters
    ----------
    p : int
        The number of nodes in the graph.
    k : float
        The desired average degree.
    w_min : float, optional
        The lower bound on the sampled weights. Defaults to 1.
    w_max : float, optional
        The upper bound on the sampled weights. Defaults to 1.
    return_ordering: bool, optional
        If the topological ordering used to orient the edges should be
        returned.
    random_state : int,optional
        To set the random state for reproducibility.
    debug : bool, optional
        If debug traces should be printed.

    Returns
    -------
    W : numpy.ndarray
       The connectivity (weights) matrix of the generated DAG.
    ordering : numpy.ndarray, optional
       If return_ordering = True, a topological ordering of the graph.

    Example
    -------

    >>> from sempler.generators import dag_avg_deg
    >>> dag_avg_deg(5, 2, random_state = 42)
    array([[0., 0., 1., 1., 0.],
           [0., 0., 1., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 1., 0., 1.],
           [0., 0., 0., 0., 0.]])

    Optionally, the ordering used to orient the edges can be returned

    >>> dag_avg_deg(5, 2, return_ordering = True, random_state = 42)
    (array([[0., 0., 1., 1., 0.],
           [0., 0., 1., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 1., 0., 1.],
           [0., 0., 0., 0., 0.]]), array([0, 3, 1, 4, 2]))


    """
    rng = np.random.default_rng(random_state)
    # Generate adjacency matrix as if top. ordering is 1..p
    prob = k / (p - 1)
    print("p = %d, k = %0.2f, P = %0.4f" % (p, k, prob)) if debug else None
    A = rng.uniform(size=(p, p))
    A = (A <= prob).astype(float)
    A = np.triu(A, k=1)
    weights = rng.uniform(w_min, w_max, size=A.shape)
    W = A * weights

    # Permute rows/columns according to random topological ordering
    permutation = rng.permutation(p)
    # Note the actual topological ordering is the "conjugate" of permutation eg. [3,1,2] -> [2,3,1]
    print("avg degree = %0.2f" % (np.sum(A) * 2 / len(A))) if debug else None
    if return_ordering:
        return (W[permutation, :][:, permutation], np.argsort(permutation))
    else:
        return W[permutation, :][:, permutation]


def dag_full(p, w_min=1, w_max=1, return_ordering=False, random_state=None):
    """Create a fully connected DAG, sampling the weights from a uniform
    distribution.

    Parameters
    ----------
    p : int
        The number of nodes in the graph.
    w_min : float, optional
        The lower bound on the sampled weights. Defaults to 1.
    w_max : float, optional
        The upper bound on the sampled weights. Defaults to 1.
    return_ordering: bool, optional
        If the topological ordering used to orient the edges should be
        returned.
    random_state : int,optional
        To set the random state for reproducibility.
    debug : bool, optional
        If debug traces should be printed.

    Returns
    -------
    W : numpy.ndarray
       The connectivity (weights) matrix of the generated DAG.
    ordering : numpy.ndarray, optional
       If return_ordering = True, a topological ordering of the graph.

    Example
    -------

    >>> from sempler.generators import dag_full
    >>> dag_full(4, random_state = 42)
    array([[0., 0., 1., 1.],
           [1., 0., 1., 1.],
           [0., 0., 0., 0.],
           [0., 0., 1., 0.]])


    Optionally, the ordering used to orient the edges can be returned

    >>> dag_full(4, return_ordering = True, random_state = 42)
    (array([[0., 0., 1., 1.],
           [1., 0., 1., 1.],
           [0., 0., 0., 0.],
           [0., 0., 1., 0.]]), array([1, 0, 3, 2]))



    """
    rng = np.random.default_rng(random_state)
    # Build a triangular matrix
    A = np.triu(np.ones((p, p)), k=1)
    weights = rng.uniform(w_min, w_max, size=A.shape)
    W = A * weights
    # Permute rows/columns according to random topological ordering
    permutation = rng.permutation(p)
    # Note the actual topological ordering is the "conjugate" of permutation eg. [3,1,2] -> [2,3,1]
    if return_ordering:
        return (W[permutation, :][:, permutation], np.argsort(permutation))
    else:
        return W[permutation, :][:, permutation]


def intervention_targets(p, K, size, replace=True, random_state=None):
    """Sample a set of intervention targets.

    Parameters
    ----------
    p : int
        The number of variables, i.e. targets will be sampled from
        `[0,p-1]`.
    K : int
        The total number of interventions.
    size : int or tuple
        Specifies the size of each intervention, i.e. the number of
        targets / intervention. If a two-element tuple, the number of
        targets is sampled uniformly at random from `[size[0],
        size[1]]`.
    replace : bool, default=True
        Wether the intervention targets should be sampled with
        replacement, i.e. if repeated targets are allowed across
        environments.
    random_state : int or None
        To set the random state for reproducibility.

    Returns
    -------
    interventions : list of list of int
        The sampled intervention targets.

    Raises
    ------
    ValueError :
        If the size of each intervention (i.e. number of targets) is
        larger than the actual number of variables, or if the tuple
        passed as size does not have length 2.

    Examples
    --------

    Generating a set of single-variable interventions:

    >>> from sempler.generators import intervention_targets
    >>> intervention_targets(10, 5, 1, random_state=42)
    [[0], [7], [6], [4], [4]]

    Without replacement:

    >>> intervention_targets(10, 5, 1, replace=False, random_state=42)
    [[0], [7], [6], [4], [3]]

    Generating a set of interventions with random number of targets:

    >>> intervention_targets(10, 5, (1,3), random_state=42)
    [[8], [2, 6, 0], [8, 7], [6, 7], [8, 1]]

    Without replacement:

    >>> intervention_targets(10, 5, (1,2), replace=False, random_state=42)
    [[8], [6, 0], [1, 4], [7], [9]]

    An exception is raised if `size > p`:

    >>> intervention_targets(4, 5, 5)
    Traceback (most recent call last):
      ...
    ValueError: The (max.) intervention size cannot be larger than the number of variables.

    Or if `size` is a tuple with size different than two:

    >>> intervention_targets(4, 5, (0,1,2))
    Traceback (most recent call last):
      ...
    ValueError: The intervention size must be a positive integer or two-element tuple.

    If sampling targets without replacement, the maximum intervention
    size and number of interventions must be set accordingly,
    i.e. `max_size x K <= p`. Otherwise an exception is raised:

    >>> intervention_targets(10, 5, (0,3), replace=False)
    Traceback (most recent call last):
      ...
    ValueError: Cannot sample targets without replacement for the given intervention size and number of interventions.


    """
    rng = np.random.default_rng(random_state)
    # Build intervention sizes
    if isinstance(size, tuple) and len(size) == 2:
        min_size, max_size = size
        sizes = rng.integers(size[0], size[1] + 1, K)
    elif isinstance(size, tuple):
        raise ValueError("The intervention size must be a positive integer or two-element tuple.")
    else:
        max_size = size
        sizes = [size] * K
    # If sampling without
    if not replace:
        if max_size * K > p:
            raise ValueError(
                "Cannot sample targets without replacement for the given intervention size and number of interventions.")
    # Check max size condition
    if max_size > p:
        raise ValueError(
            "The (max.) intervention size cannot be larger than the number of variables.")
    # Sample the targets
    if replace:
        interventions = []
        targets = list(range(p))
        for i, k in enumerate(range(K)):
            intervention = list(rng.choice(targets, size=sizes[i], replace=False))
            interventions.append(intervention)
    else:
        interventions = []
        remaining_targets = set(range(p))
        for i, k in enumerate(range(K)):
            intervention = list(rng.choice(list(remaining_targets), size=sizes[i], replace=False))
            remaining_targets -= set(intervention)
            interventions.append(intervention)
    return interventions


# To run the method's doctests
if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=True)
