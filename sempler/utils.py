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
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools


def argmin(array):
    """Return the index of the minimum element of an array.

    Examples
    --------
    >>> _argmin(np.array([1,-1,2,3]))
    (1,)
    >>> _argmin(np.array([[0,1],[-1,1]]))
    (1, 0)
    """
    return np.unravel_index(np.argmin(array), array.shape)


def argmax(array):
    """Return the index of the minimum element of an array.

    Examples
    --------
    >>> _argmin(np.array([1,-1,2,3]))
    (3,)
    >>> _argmin(np.array([[0,1],[2,1]]))
    (1, 0)
    """
    return np.unravel_index(np.argmax(array), array.shape)


def matrix_block(M, rows, cols):
    """
    Select a block of a matrix given by the row and column indices
    """
    return M[rows, :][:, cols]


def sampling_matrix(W):
    """Given the weighted adjacency matrix of a DAG, return
    the matrix A such that the DAG generates samples
      A @ diag(var)^1/2 @ Z + mu
    where Z is an isotropic normal, and var/mu are the variances/means
    of the noise variables of the graph.
    """
    p = len(W)
    return np.linalg.inv(np.eye(p) - W.T)


def all_but(k, p):
    """Return [0,...,p-1] without k"""
    k = np.atleast_1d(k)
    return [i for i in range(p) if i not in k]


def combinations(p, target, empty=True):
    """Return all possible subsets of the set {0...p-1} \\ {target}"""
    base = set(range(p)) - {target}
    sets = []
    for size in range(0 if empty else 1, p):
        sets += [set(s) for s in itertools.combinations(base, size)]
    return sets


def nonzero(A, tol=1e-12):
    """Return the indices of the nonzero (up to tol) elements in A"""
    return np.where(np.abs(A) > tol)[0]


def graph_info(i, W, interventions=None):
    """Returns the parents, children, parents of children and markov
    blanket of variable i in DAG W, using the graph structure
    """
    G = nx.from_numpy_array(W, create_using=nx.DiGraph)
    parents = set(G.predecessors(i))
    children = set(G.successors(i))
    parents_of_children = set()
    for child in children:
        parents_of_children.update(G.predecessors(child))
    if len(children) > 0:
        parents_of_children.remove(i)
    mb = parents.union(children, parents_of_children)
    return (parents, children, parents_of_children, mb)


def stable_blanket(i, W, interventions=set()):
    """Return the stable blanket using the graph structure"""
    G = nx.from_numpy_array(W, create_using=nx.DiGraph)
    parents = set(G.predecessors(i))
    children = set(G.successors(i))
    unstable_descendants = set()
    for j in interventions:
        if j in children:
            unstable_descendants.update({j})
            unstable_descendants.update(nx.algorithms.dag.descendants(G, j))
    stable_children = set.difference(children, unstable_descendants)
    parents_of_stable_children = set()
    for child in stable_children:
        parents_of_stable_children.update(G.predecessors(child))
    if len(stable_children) > 0:
        parents_of_stable_children.remove(i)
    sb = set.union(parents, stable_children, parents_of_stable_children)
    return sb


def descendants(i, W):
    """Return the descendants of a node using the graph structure"""
    G = nx.from_numpy_array(W, create_using=nx.DiGraph)
    return nx.algorithms.dag.descendants(G, i)


def ancestors(i, W):
    """Return the ancestors of a node using the graph structure"""
    G = nx.from_numpy_array(W, create_using=nx.DiGraph)
    return nx.algorithms.dag.ancestors(G, i)


def is_dag(A):
    """Checks wether the given adjacency matrix corresponds to a DAG.

    Parameters
    ----------
    A : np.array
        the adjacency matrix of the graph, where A[i,j] != 0 => i -> j.

    Returns
    -------
    is_dag : bool
        if the adjacency corresponds to a DAG

    """
    try:
        topological_ordering(A)
        return True
    except ValueError:
        return False


def topological_ordering(A):
    """Return a topological ordering for the DAG with adjacency matrix A,
    using Kahn's 1962 algorithm.

    Raises a ValueError exception if the given adjacency does not
    correspond to a DAG.

    Parameters
    ----------
    A : np.array
        The adjacency matrix of the graph, where A[i,j] != 0 => i -> j.

    Returns
    -------
    ordering : list of ints
        A topological ordering for the DAG.

    Raises
    ------
    ValueError
        If the given adjacency does not correspond to a DAG.

    """
    # Check that there are no undirected edges
    if (np.logical_and(A, A.T)).sum() > 0:
        raise ValueError("The given graph is not a DAG")
    # Run the algorithm from the 1962 paper "Topological sorting of
    # large networks" by AB Kahn
    A = A.copy()
    sinks = list(np.where(A.sum(axis=0) == 0)[0])
    ordering = []
    while len(sinks) > 0:
        i = sinks.pop()
        ordering.append(i)
        for j in ch(i, A):
            A[i, j] = 0
            if len(pa(j, A)) == 0:
                sinks.append(j)
    # If A still contains edges there is at least one cycle
    if A.sum() > 0:
        raise ValueError("The given graph is not a DAG")
    else:
        return ordering


def edge_weights(W):
    """Return the weights of all the edges of W in a dictionary, i.e. with
    keys (i,j) for values W[i,j] when W[i,j] != 0."""
    # TODO: Test
    fro, to = np.where(W != 0)
    edges = list(zip(fro, to))
    weights = [W[i, j] for i, j in edges]
    edge_weights = dict(zip(edges, weights))
    return edge_weights


def plot_graph(W, labels=None, weights=False, block=False):
    """Plot a graph with weight matrix W."""
    G = nx.from_numpy_array(W, create_using=nx.DiGraph)
    pos = nx.drawing.layout.shell_layout(G, scale=0.5)
    p = len(W)
    if labels is None:
        node_labels = dict(zip(np.arange(p), map(lambda i: "$X_{%d}$" % i, range(p))))
    else:
        node_labels = dict(zip(np.arange(p), labels))
    # Plot
    fig = plt.figure()
    params = {'node_color': 'white',
              'edgecolors': 'black',
              'node_size': 900,
              'linewidths': 1.5,
              'width': 1.5,
              'arrowsize': 20,
              'arrowstyle': '->',
              'min_target_margin': 10,
              'labels': node_labels}
    nx.draw(G, pos, **params)
    # Edge weights
    if weights:
        formatted = dict((e, "%0.3f" % w) for (e, w) in edge_weights(W).items())
        nx.draw_networkx_edge_labels(G, pos, formatted, font_color='red')
    fig.set_facecolor("white")
    plt.show(block=block)


def plot_matrix(A, ax=None, vmin=-3, vmax=3, formt="%0.2f", thresh=1e-16, block=False):
    """Plot a heatmap for the given matrix A.

    Parameters
    ----------
    A : numpy.ndarray
        The matrix to plot.
    ax : matplotlib.pyplot.axis, optional
        The axis to plot on; if None, create a new figure.
    vmin : float, default=-3
        The lower threshold for color saturation.
    vmax : float, default=3
        The upper threshold for color saturation.
    formt : string, default="%0.2f"
        The format with which to print the values of the matrix on top
        of the corresponding cell.
    thresh : float, default=1e-16
        Elements of the matrix which are lower than the threshold in
        absolute value are plotted as a zero (i.e. white, no text)
    """
    if ax is None:
        plt.figure()
        ax = plt.gca()
    ax.imshow(A, vmin=vmin, vmax=vmax, cmap='bwr')
    for i in range(len(A)):
        for j in range(len(A)):
            if A[i, j] != 0:
                ax.text(j, i, formt % A[i, j], ha='center', va='center')
    plt.show(block=block)


def allclose(A, B, rtol=1e-5, atol=1e-8):
    """Use np.allclose to compare, but relative tolerance is relative to
    the smallest element compared
    """
    return np.allclose(np.maximum(A, B), np.minimum(A, B), rtol, atol)

# --------------------------------------------------------------------
# Graph functions for PDAGS


def na(y, x, A):
    """Return all neighbors of y which are adjacent to x in A.

    Parameters
    ----------
    y : int
        the node's index
    x : int
        the node's index
    A : np.array
        the adjacency matrix of the graph, where A[i,j] != 0 => i -> j
        and A[i,j] != 0 & A[j,i] != 0 => i - j.

    Returns
    -------
    nodes : set of ints
        the resulting nodes

    """
    return neighbors(y, A) & adj(x, A)


def neighbors(i, A):
    """The neighbors of i in A, i.e. all nodes connected to i by an
    undirected edge.

    Parameters
    ----------
    i : int
        the node's index
    A : np.array
        the adjacency matrix of the graph, where A[i,j] != 0 => i -> j
        and A[i,j] != 0 & A[j,i] != 0 => i - j.

    Returns
    -------
    nodes : set of ints
        the neighbor nodes

    """
    return set(np.where(np.logical_and(A[i, :] != 0, A[:, i] != 0))[0])


def adj(i, A):
    """The adjacent nodes of i in A, i.e. all nodes connected by a
    directed or undirected edge.
    Parameters
    ----------
    i : int
        the node's index
    A : np.array
        the adjacency matrix of the graph, where A[i,j] != 0 => i -> j
        and A[i,j] != 0 & A[j,i] != 0 => i - j.

    Returns
    -------
    nodes : set of ints
        the adjacent nodes

    """
    return set(np.where(np.logical_or(A[i, :] != 0, A[:, i] != 0))[0])


def pa(i, A):
    """The parents of i in A.

    Parameters
    ----------
    i : int
        the node's index
    A : np.array
        the adjacency matrix of the graph, where A[i,j] != 0 => i -> j
        and A[i,j] != 0 & A[j,i] != 0 => i - j.

    Returns
    -------
    nodes : set of ints
        the parent nodes

    """
    return set(np.where(np.logical_and(A[:, i] != 0, A[i, :] == 0))[0])


def ch(i, A):
    """The children of i in A.

    Parameters
    ----------
    A : np.array
        the adjacency matrix of the graph, where A[i,j] != 0 => i -> j
        and A[i,j] != 0 & A[j,i] != 0 => i - j.

    Returns
    -------
    nodes : set of ints
        the children nodes

    """
    return set(np.where(np.logical_and(A[i, :] != 0, A[:, i] == 0))[0])


def same_normal(sample_a, sample_b, atol=5e-2, debug=False):
    """
    Test (crudely, by L1 dist. of means and covariances) if samples
    from two distributions come from the same Gaussian
    """
    mean_a, mean_b = np.mean(sample_a, axis=0), np.mean(sample_b, axis=0)
    cov_a, cov_b = np.cov(sample_a, rowvar=False), np.cov(sample_b, rowvar=False)
    print("MEANS\n%s\n%s\n\nCOVARIANCES\n%s\n%s" %
          (mean_a, mean_b, cov_a, cov_b)) if debug else None
    means = np.allclose(mean_a, mean_b, atol=atol)
    covariances = np.allclose(cov_a, cov_b, atol=atol)
    return means and covariances

# Example graphs


def eg1():
    W = np.array([[0, 1, -1, 0, 0],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0]])
    markov_blankets = [[1, 2],
                       [0, 3, 2],
                       [0, 3, 1],
                       [1, 2, 4],
                       [3]]
    parents = [[],
               [0],
               [0],
               [1, 2],
               [3]]
    return W, parents, markov_blankets


def eg2():
    W = np.array([[0, 1, -1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0]])
    markov_blankets = [[1, 2],
                       [0, 3, 2],
                       [0, 3, 1],
                       [1, 2, 4, 5],
                       [3, 5],
                       [3, 4]]
    parents = [[],
               [0],
               [0],
               [1, 2],
               [3, 5],
               []]
    return W, parents, markov_blankets


def eg3():
    W = np.array([[0, 1, -1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 1, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0]])
    markov_blankets = [[1, 2],
                       [0, 3, 2],
                       [0, 3, 1],
                       [1, 2, 4, 5],
                       [3, 5, 7],
                       [3, 4, 6],
                       [5],
                       [4]]
    parents = [[],
               [0],
               [0],
               [1, 2],
               [3, 5],
               [],
               [5],
               [4]]
    return W, parents, markov_blankets


def eg4():
    W = np.array([[0, 0, 1, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, 0]])
    markov_blankets = [[1, 2],
                       [0, 2],
                       [0, 1, 3],
                       [2]]
    parents = [[],
               [],
               [0, 1],
               [2]]
    return W, parents, markov_blankets


def eg5():
    W = np.array([[0., 1., 0., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0., 0., 0.],
                  [0., 1., 0., 1., 0., 1., 0., 0.],
                  [0., 1., 0., 0., 0., 1., 1., 0.],
                  [1., 0., 0., 0., 0., 0., 1., 0.],
                  [0., 0., 0., 0., 0., 0., 0., 0.],
                  [1., 0., 0., 0., 0., 0., 0., 0.],
                  [1., 0., 0., 0., 1., 1., 0., 0.]])
    ordering = np.array([7, 4, 2, 3, 6, 5, 0, 1])
    parents = [[4, 6, 7],
               [0, 2, 3],
               [],
               [2],
               [7],
               [2, 3, 7],
               [3, 4],
               []]
    children = [[1],
                [],
                [1, 3, 5],
                [1, 5, 6],
                [0, 6],
                [],
                [0],
                [0, 4, 5]]
    markov_blankets = [[4, 6, 7, 1, 2, 3],
                       [0, 2, 3],
                       [1, 3, 5, 0, 7],
                       [0, 1, 2, 4, 5, 6, 7],
                       [7, 0, 6, 7, 3],
                       [2, 3, 7],
                       [3, 4, 0, 7],
                       [0, 4, 5, 2, 3, 6]]
    return W, parents, markov_blankets


def eg6():
    W = np.array([[0, 0, 1, 0, 1],
                  [0, 0, 1, 0, 1],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0]])
    parents = [[],
               [],
               [0, 1],
               [2],
               [0, 1, 3]]
    children = [[2],
                [2],
                [3],
                [4],
                []]
    markov_blankets = [[1, 2, 3, 4],
                       [0, 2, 3, 4],
                       [0, 1, 3],
                       [2, 4, 0, 1],
                       [0, 1, 3]]
    return W, parents, markov_blankets
