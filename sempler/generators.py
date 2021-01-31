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

#---------------------------------------------------------------------
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
    array([[0., 0., 0., 1., 0.],
           [0., 0., 1., 1., 0.],
           [0., 0., 0., 1., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.]])

    Optionally, the ordering used to orient the edges can be returned

    >>> dag_avg_deg(5, 2, return_ordering = True, random_state = 42)
    (array([[0., 0., 0., 1., 0.],
           [0., 0., 1., 1., 0.],
           [0., 0., 0., 1., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.]]), array([0, 4, 1, 2, 3]))

    """
    np.random.seed(random_state) if random_state is not None else None
    # Generate adjacency matrix as if top. ordering is 1..p
    prob = k / (p-1)
    print("p = %d, k = %0.2f, P = %0.4f" % (p,k,prob)) if debug else None
    A = np.random.uniform(size = (p,p))
    A = (A <= prob).astype(float)
    A = np.triu(A, k=1)
    weights = np.random.uniform(w_min, w_max, size=A.shape)
    W = A * weights
    
    # Permute rows/columns according to random topological ordering
    permutation = np.random.permutation(p)
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
    array([[0., 1., 1., 1.],
           [0., 0., 0., 1.],
           [0., 1., 0., 1.],
           [0., 0., 0., 0.]])

    Optionally, the ordering used to orient the edges can be returned

    >>> dag_full(4, return_ordering = True, random_state = 42)
    (array([[0., 1., 1., 1.],
           [0., 0., 0., 1.],
           [0., 1., 0., 1.],
           [0., 0., 0., 0.]]), array([0, 2, 1, 3]))


    """
    np.random.seed(random_state) if random_state is not None else None
    # Build a triangular matrix
    A = np.triu(np.ones((p,p)), k=1)
    weights = np.random.uniform(w_min, w_max, size=A.shape)
    W = A * weights
    # Permute rows/columns according to random topological ordering
    permutation = np.random.permutation(p)
    # Note the actual topological ordering is the "conjugate" of permutation eg. [3,1,2] -> [2,3,1]
    if return_ordering:
        return (W[permutation, :][:, permutation], np.argsort(permutation))
    else:
        return W[permutation, :][:, permutation]

# To run the method's doctests
if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=True)