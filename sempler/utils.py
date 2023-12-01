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
import itertools
from functools import reduce

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


def ancestors(i, A):
    # TODO: This will break if Python's max stack depth is reached
    """The ancestors of i in A.

    Parameters
    ----------
    A : np.array
        the adjacency matrix of the graph, where A[i,j] != 0 => i -> j
        and A[i,j] != 0 & A[j,i] != 0 => i - j.

    Returns
    -------
    nodes : set of ints
        the ancestor nodes
    """
    anc = pa(i, A)
    for j in pa(i, A):
        anc |= ancestors(j, A)
    return anc


def descendants(i, A):
    # TODO: This will break if Python's max stack depth is reached
    """The descendants of i in A.

    Parameters
    ----------
    A : np.array
        the adjacency matrix of the graph, where A[i,j] != 0 => i -> j
        and A[i,j] != 0 & A[j,i] != 0 => i - j.

    Returns
    -------
    nodes : set of ints
        the descendant nodes
    """
    desc = {i}
    for j in ch(i, A):
        desc |= descendants(j, A)
    return desc


def transitive_closure(A):
    """Perform the transitive closure on the dag A, i.e. a new graph where
    there is an edge from every node to all its descendants.

    """
    if not is_dag(A):
        raise ValueError("The given graph is not a DAG.")
    closure = np.zeros_like(A)
    for i in range(len(A)):
        desc = list(descendants(i, A) - {i})
        closure[i, desc] = 1
    return closure


def allclose(A, B, rtol=1e-5, atol=1e-8):
    """Use np.allclose to compare, but relative tolerance is relative to
    the smallest element compared
    """
    return np.allclose(np.maximum(A, B), np.minimum(A, B), rtol, atol)

def same_normal(sample_a, sample_b, atol=5e-2, debug=False):
    """
    Test (crudely, by L1 dist. of means and covariances) if samples
    from two distributions come from the same Gaussian
    """
    mean_a, mean_b = np.mean(sample_a, axis=0), np.mean(sample_b, axis=0)
    cov_a, cov_b = np.cov(sample_a, rowvar=False), np.cov(sample_b, rowvar=False)
    print(
        "MEANS\n%s\n%s\n\nCOVARIANCES\n%s\n%s" % (mean_a, mean_b, cov_a, cov_b)
    ) if debug else None
    means = np.allclose(mean_a, mean_b, atol=atol)
    covariances = np.allclose(cov_a, cov_b, atol=atol)
    return means and covariances

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


def an(i, A):
    # TODO: This will break if Python's max stack depth is reached
    """The ancestors of i in A.

    Parameters
    ----------
    A : np.array
        the adjacency matrix of the graph, where A[i,j] != 0 => i -> j
        and A[i,j] != 0 & A[j,i] != 0 => i - j.

    Returns
    -------
    nodes : set of ints
        the ancestor nodes
    """
    ancestors = pa(i, A)
    for j in pa(i, A):
        ancestors |= an(j, A)
    return ancestors


def desc(i, A):
    # TODO: This will break if Python's max stack depth is reached
    """The descendants of i in A.

    Parameters
    ----------
    A : np.array
        the adjacency matrix of the graph, where A[i,j] != 0 => i -> j
        and A[i,j] != 0 & A[j,i] != 0 => i - j.

    Returns
    -------
    nodes : set of ints
        the descendant nodes
    """
    descendants = {i}
    for j in ch(i, A):
        descendants |= desc(j, A)
    return descendants


def is_clique(S, A):
    """Check if the subgraph of A induced by nodes S is a clique.

    Parameters
    ----------
    S : set of ints
        set containing the nodes' indices
    A : np.array
        the adjacency matrix of the graph, where A[i,j] != 0 => i -> j
        and A[i,j] != 0 & A[j,i] != 0 => i - j.

    Returns
    -------
    is_clique : bool
        if the subgraph induced by S is a clique in A

    """
    S = list(S)
    subgraph = A[S, :][:, S]
    subgraph = skeleton(subgraph)  # drop edge orientations
    no_edges = np.sum(subgraph != 0)
    n = len(S)
    return no_edges == n * (n - 1)


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


def is_complete(P):
    """Checks wether the underlying undirected graph is the complete graph."""
    no_edges = ((P + P.T) != 0).sum() / 2
    p = len(P)
    return no_edges == (p * (p - 1) / 2)


def chain_graph(p):
    """Generate the adjacency of a chain DAG with p variables."""
    A = np.zeros((p, p))
    ix = np.arange(p - 1)
    A[ix, ix + 1] = 1
    return A

def is_chain_graph(A):
    """Check whether the given DAG adjacency corresponds to a chain
graph.
    """
    p = len(A)
    return (A == chain_graph(p)).all()


def mec(A, check_chain=True):
    """Given a DAG adjacency A, enumerate all the members in its
    Markov-equivalence class.

    Parameters
    ----------
    A : numpy.ndarray
        The adjacency matrix of the graph, where A[i,j] != 0 => i -> j
        and A[i,j] != 0 & A[j,i] != 0 => i - j.
    check_chain : bool, default=True
        If True, checks whether the given graph is a chain graph, and
        if so, uses a more efficient way to enumerate the MEC (see
        chain_graph_mec).

    Returns
    -------
    mec : numpy.ndarray
        A three dimensional array, with axis=0 corresponding to the
        graphs in the MEC.

    Raises
    ------
    ValueError :
        If the given adjacency does not correspond to a DAG.

    """
    # Check input
    if not is_dag(A):
        raise ValueError("The given adjacency does not correspond to a DAG.")
    if check_chain and is_chain_graph(A):
        return chain_graph_MEC(len(A))
    else:
        cpdag = dag_to_cpdag(A)
        return all_dags(cpdag)


def imec(A, I, check_chain=True):
    """Given a DAG adjacency A and a set of intervention targets I,
    enumerate all the members in the corresponding I-equivalence
    class.

    Parameters
    ----------
    A : numpy.ndarray
        The adjacency matrix of the graph, where A[i,j] != 0 => i -> j
        and A[i,j] != 0 & A[j,i] != 0 => i - j.
    I : set
        A set containing the intervention targets.
    check_chain : bool, default=True
        If True, checks whether the given graph is a chain graph, and
        if so, uses a more efficient way to enumerate the MEC (see
        chain_graph_mec).

    Returns
    -------
    mec : numpy.ndarray
        A three dimensional array, with axis=0 corresponding to the
        graphs in the MEC.

    Raises
    ------
    ValueError :
        If the given adjacency does not correspond to a DAG.

    """
    # Check inputs
    if not is_dag(A):
        raise ValueError("The given adjacency does not correspond to a DAG.")
    if not I <= set(range(len(A))):
        raise ValueError("Targets I must be a subset of [p].")

    if check_chain and is_chain_graph(A):
        return chain_graph_IMEC(A, I)
    else:
        icpdag = dag_to_icpdag(A, I)
        return all_dags(icpdag)


def chain_graph_MEC(p):
    """Enumerate the MEC of a chain graph of p-variables. Implemented as a
    separate function because enumerating all graphs from the CPDAG is
    extremely costly for chain graphs."""
    MEC = []
    for i in range(p):
        A = np.zeros((p, p))
        # Add "backward pointing" edges
        for j in range(i, 0, -1):
            A[j, j - 1] = 1
        # Add "forward pointing" edges
        for j in range(i, p - 1):
            A[j, j + 1] = 1
        MEC.append(A)
    return np.array(MEC)


def chain_graph_IMEC(A, I):
    """Given a chain graph and some intervention targets, enumerate all the members of its
    interventional equivalence class.

    Raises
    ------
    ValueError :
        If the given adjacency does not correspond to a chain graph.
    """
    # Check input
    if not is_chain_graph(A):
        raise ValueError("The given adjacency is not a chain graph.")
    p = len(A)
    MEC = chain_graph_MEC(p)
    IMEC = []
    I = list(I)
    for me in MEC:
        # If parents of intervened variables match, keep in I-MEC
        if (me[:, I] == A[:, I]).all():
            IMEC.append(me)
    return np.array(IMEC)


def topological_ordering(A):
    """Return a topological ordering for the DAG with adjacency matrix A,
    using Kahn's 1962 algorithm.

    Raises a ValueError exception if the given adjacency does not
    correspond to a DAG.

    Parameters
    ----------
    A: np.array
        the adjacency matrix of the graph, where A[i, j] != 0 = > i -> j.

    Returns
    -------
    ordering: list of ints
        a topological ordering for the DAG

    Raises
    ------
    ValueError
        If the given graph is not a DAG.

    """
    # Check that there are no undirected edges
    if only_undirected(A).sum() > 0:
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


def semi_directed_paths(fro, to, A):
    """Return all paths from i to j in A. Note: a path is a sequence
    (a_1, ..., a_n) of non-repeating nodes where either a_i -> a_i+1 or
    a_i - a_i+1 are edges in the PDAG A.

    Parameters
    ----------
    fro: int
        the index of the starting node
    to: int
        the index of the target node
    A: np.array
        the adjacency matrix of the graph, where A[i, j] != 0 = > i -> j.

    Returns
    -------
    paths: list of lists
        all the paths between the two nodes

    """
    # NOTE: Implemented non-recursively to avoid issues with Python's
    # stack size limit
    stack = [(fro, [], list(ch(fro, A) | neighbors(fro, A)))]
    paths = []
    # Precompute the nodes that are accessible from each node for a
    # significant increase in speed
    accessible = dict((i, ch(i, A) | neighbors(i, A)) for i in range(len(A)))
    while len(stack) > 0:
        current_node, visited, to_visit = stack[0]
        if current_node == to:
            paths.append(visited + [current_node])
            stack = stack[1:]
        elif to_visit == []:
            stack = stack[1:]
        else:
            next_node = to_visit.pop()
            next_to_visit = list(accessible[next_node] - set(visited) - {current_node})
            stack = [(next_node, visited + [current_node], next_to_visit)] + stack
    return paths

def separates(S, A, B, G):
    """Returns true if the set S separates A from B in G, i.e. if all
    paths in G from nodes in A to nodes in B contain a node in
    S. Exception is raised if S, A and B are not pairwise disjoint.

    Parameters
    ----------
    S: set of ints
        a set of nodes in G
    A: set of ints
        a set of nodes in G
    B: set of ints
        a set of nodes in G
    G: np.array
        the adjacency matrix of the graph, where G[i, j] != 0 = > i -> j
        and G[i, j] != 0 and G[j, i] != 0 = > i - j.

    Returns
    -------
    separated: bool
        if S separates A from B in G

    """
    # Check that sets are pairwise disjoint
    if len(A & B) or len(A & S) or len(B & S):
        raise ValueError("The sets S=%s,A=%s and B=%s are not pairwise disjoint" % (S, A, B))
    for a in A:
        for b in B:
            for path in semi_directed_paths(a, b, G):
                if set(path) & S == set():
                    return False
    return True

def chain_component(i, G):
    """Return all nodes in the connected component of node i after
    dropping all directed edges in G.

    Parameters
    ----------
    i: int
        the node's index
    G: np.array
        the adjacency matrix of the graph, where G[i, j] != 0 = > i -> j
        and G[i, j] != 0 and G[j, i] != 0 = > i - j

    Returns
    -------
    visited: set of ints
        the nodes in the chain component of i

    """
    A = only_undirected(G)
    visited = set()
    to_visit = {i}
    # NOTE: Using a breadth-first search
    while len(to_visit) > 0:
        for j in to_visit:
            visited.add(j)
            to_visit = (to_visit | neighbors(j, A)) - visited
    return visited


def induced_subgraph(S, G):
    """Remove all edges which are not between nodes in S.

    Parameters
    ----------
    S: set of ints
        a set of node indices
    G: np.array
        the adjacency matrix of the graph, where G[i, j] != 0 = > i -> j
        and G[i, j] != 0 and G[j, i] != 0 = > i - j

    Returns
    -------
    subgraph: np.array
       the adjacency matrix of the resulting graph where all edges
       between nodes not in S are removed. Note that this is not
       really the subgraph, as the nodes not in S still appear as
       disconnected nodes.
    """
    mask = np.zeros_like(G, dtype=bool)
    mask[list(S), :] = True
    mask = np.logical_and(mask, mask.T)
    subgraph = np.zeros_like(G)
    subgraph[mask] = G[mask]
    return subgraph


def vstructures(A):
    """
    Return the v-structures of a DAG or PDAG, given its adjacency matrix.

    Parameters
    ----------
    A: np.array
        The adjacency of the(P)DAG, where A[i, j] != 0 = > i -> j

    Returns
    -------
    vstructs: set()
        the set of v-structures, where every v-structure is a three
        element tuple, e.g. (i, j, k) represents the v-structure
        i -> j < - k, where i < j for consistency.

    """
    # Construct the graph with only the directed edges
    dir_A = only_directed(A)
    # Search for colliders in the graph with only directed edges
    colliders = np.where((dir_A != 0).sum(axis=0) > 1)[0]
    # For each collider, and all pairs of parents, check if the
    # parents are adjacent in A
    vstructs = []
    for c in colliders:
        for (i, j) in itertools.combinations(pa(c, A), 2):
            if A[i, j] == 0 and A[j, i] == 0:
                # Ordering might be defensive here, as
                # itertools.combinations already returns ordered
                # tuples; motivation is to not depend on their feature
                vstruct = (i, c, j) if i < j else (j, c, i)
                vstructs.append(vstruct)
    return set(vstructs)


def moral_graph(A):
    # TODO: Test
    """Return the moral graph of DAG A.

    Parameters
    ----------
    A: np.array
        The adjacency of the DAG, where A[i, j] != 0 = > i -> j

    Returns
    -------
    moral: np.array
       The adjacency of the (undirected) moral graph of A.
    """
    moral = skeleton(A)
    for (i, _, j) in vstructures(A):
        moral[i, j] = 1
        moral[j, i] = 1
    return moral


def degrees(A):
    # TODO: Test
    """Return the degrees of the nodes of a graph.

    Parameters
    ----------
    A: np.array
        The adjacency of the graph, where A[i, j] != 0 = > i -> j

    Returns
    -------
    degrees: np.array
        A vector containing the degree of each vertex in the graph.
    """
    return np.sum(skeleton(A), axis=0)


def only_directed(P):
    """
    Return the graph with the same nodes as P and only its directed edges.

    Parameters
    ----------
    P: np.array
        adjacency matrix of a graph

    Returns
    -------
    G: np.array
        adjacency matrix of the graph with the same nodes as P and
        only its directed edges

    """
    mask = np.logical_and(P != 0, P.T == 0)
    G = np.zeros_like(P)
    # set to the same values in case P is a weight matrix and there is
    # interest in maintaining the weights
    G[mask] = P[mask]
    return G


def only_undirected(P):
    """
    Return the graph with the same nodes as P and only its undirected edges.

    Parameters
    ----------
    P: np.array
        adjacency matrix of a graph

    Returns
    -------
    G: np.array
        adjacency matrix of the graph with the same nodes as P and
        only its undirected edges

    """
    mask = np.logical_and(P != 0, P.T != 0)
    G = np.zeros_like(P)
    # set to the same values in case P is a weight matrix and there is
    # interest in maintaining the weights
    G[mask] = P[mask]
    return G


def undirected_edges(P):
    """
    Return a list with the undirected edges in the PDAG P.

    Parameters
    ----------
    P: np.array
        Adjacency matrix of a PDAG.

    Returns
    -------
    edges: list
        List of tuples containing nodes connected by an undirected
        edge. For an edge i - j, the tuple `(i, j)` is returned if `i >
        j`, and `(j, i)` otherwise.
    """
    fro, to = np.where(only_undirected(P))
    undirected_edges = filter(lambda e: e[0] > e[1], zip(fro, to))
    return list(undirected_edges)


def directed_edges(A):
    """Return a list with the undirected edges in a graph A.

    Parameters
    ----------
    A : np.array
        Adjacency matrix of a graph.

    Returns
    -------
    edges : list of tuples
        List of tuples, where (i,j) simbolizes an edge i->j in A.
    """
    fro, to = np.where(only_directed(A))
    return list(zip(fro, to))


def edge_weights(W):
    """Return the weights of all the edges of W in a dictionary, i.e. with
    keys (i,j) for values W[i,j] when W[i,j] != 0."""
    # TODO: Test
    fro, to = np.where(W != 0)
    edges = list(zip(fro, to))
    weights = [W[i, j] for i, j in edges]
    edge_weights = dict(zip(edges, weights))
    return edge_weights


def skeleton(A):
    """Return the skeleton of a given graph.

    Parameters
    ----------
    A: np.array
        adjacency matrix of a graph

    Returns
    -------
    S: np.array
        adjacency matrix of the skeleton, i.e. the graph resulting
        from dropping all edge orientations

    """
    return ((A + A.T) != 0).astype(int)


def is_consistent_extension(G, P, debug=False):
    """Returns True if the DAG G is a consistent extension of the PDAG
    P. Will raise a ValueError exception if the graph G is not a DAG
    (i.e. cycles or undirected edges).

    Parameters
    ----------
    G: np.array
        the adjacency matrix of DAG
    P: np.array
        the adjacency matrix of PDAG
    debug: bool
        if debugging traces should be outputted

    Returns
    -------
    consistent: bool
        True if G is a consistent extension of P(see below)

    """
    # G is a consistent extension of P iff
    #   0. it is a DAG
    #   1. they have the same v-structures,
    #   2. they have the same skeleton, and
    #   3. every directed edge in P has the same direction in G
    if not is_dag(G):
        raise ValueError("G is not a DAG")
    # v-structures
    same_vstructures = vstructures(P) == vstructures(G)
    # skeleton
    same_skeleton = (skeleton(P) == skeleton(G)).all()
    # same orientation
    directed_P = only_directed(P)
    same_orientation = G[directed_P != 0].all()  # No need to check
    # transpose as G is
    # guaranteed to have
    # no undirected edges
    if debug:
        print("v-structures (%s) (P,G): " % same_vstructures, vstructures(P), vstructures(G))
        print("skeleton (%s) (P,G): " % same_skeleton, skeleton(P), skeleton(G))
        print("orientation (%s) (P,G): " % same_orientation, P, G)
    return same_vstructures and same_orientation and same_skeleton


def has_consistent_extension(pdag):
    """
    Tests whether the given pdag has a consistent extension.

    Parameters
    ----------
    pdag: np.array
        The adjacency matrix of PDAG.

    Returns
    -------
    bool
       Wether the PDAG admits a consistent extension.

    """
    # TODO: Test
    try:
        pdag_to_dag(pdag)
        return True
    except ValueError:
        return False


def are_forward_neighbors(P1, P2, x, y):
    """Check that there is a DAG consistent with P2 such that removing
    any edge between x and y results in a dag consistent with P1.
    """
    dags = all_dags(P2)
    dags[:, x, y] = 0
    dags[:, y, x] = 0
    for dag in dags:
        if is_consistent_extension(dag, P1):
            # print("  neighbors")
            return True
    # print("  NOT neighbors")
    return False


def are_backward_neighbors(P1, P2, x, y):
    """
    Check that there is a DAG consistent with P2 such that removing
    any edge between x and y results in a dag consistent with P1.
    """
    return are_forward_neighbors(P2, P1, x, y)


def to_factorization(G):
    """Return the given graph in a factorization representation, i.e. a
    list of the parents of each node."""
    # TODO: Test, docstring
    if not is_dag(G):
        raise ValueError("G is not a DAG")
    return tuple(tuple(sorted(pa(j, G))) for j in range(len(G)))


def is_supergraph(sup, A):
    """Return True if sup is a supergraph of A.
    """
    return (np.logical_and(sup, A) == A).all()


def has_subgraph(L1, L2):
    # TODO: Test
    """Test if every graph in L1 has a subgraph in L2"""
    for A1 in L1:
        has_subgraph = np.array([is_supergraph(A1, A2) for A2 in L2]).any()
        if not has_subgraph:
            return False
    return True


def has_supergraph(L1, L2):
    # TODO: Test
    """Test if every graph in L1 has a supergraph in L2"""
    for A1 in L1:
        has_supergraph = np.array([is_supergraph(A2, A1) for A2 in L2]).any()
        if not has_supergraph:
            return False
    return True


def remove_edges(A, no_edges, random_state=42):
    """Remove `no_edges` at random from A."""
    A = A.astype(bool).astype(int)
    rng = np.random.default_rng(random_state)
    edges = directed_edges(A)
    if len(edges) < no_edges:
        raise ValueError("There are not enough edges to remove.")
    pruned = A.copy()
    for (fro, to) in rng.choice(edges, no_edges, replace=False):
        pruned[fro, to] = 0
    return pruned


def add_edges(A, no_edges, random_state=42):
    """Add `no_edges` at random to A while preserving a DAG structure."""
    A = A.astype(bool).astype(int)
    # Edges between non-adjacent nodes
    fro, to = np.where((A + A.T + np.eye(len(A))) == 0)
    rng = np.random.default_rng(random_state)
    edges = list(zip(fro, to))
    rng.shuffle(edges)
    # Check inputs
    p = len(A)
    can_add = int(p * (p - 1) / 2 - A.sum())
    if no_edges > can_add:
        raise ValueError("It is not possible to add %d edges to the given DAG." % no_edges)
    # Add edges at random
    supergraph = A.copy()
    i = 0
    while (supergraph.sum() - A.sum()) < no_edges and i < len(edges):
        next_supergraph = supergraph.copy()
        next_supergraph[edges[i]] = 1
        i += 1
        if is_dag(next_supergraph):
            supergraph = next_supergraph
    assert (supergraph.sum() - A.sum() == no_edges)
    return supergraph


# --------------------------------------------------------------------
# Functions for PDAG to CPDAG conversion (from the GES algorithm)

    # The following functions implement the conversion from PDAG to
    # CPDAG that is carried after each transition to a different
    # equivalence class, after the selection and application of the
    # highest scoring insert/delete/turn operator. It consists of the
    # succesive application of three algorithms, all described in
    # Appendix C (pages 552,553) of Chickering's 2002 GES paper
    # (www.jmlr.org/papers/volume3/chickering02b/chickering02b.pdf).
    #
    # The algorithms are:

    #   1. Obtaining a consistent extension of a PDAG, implemented in
    #   the function pdag_to_dag.
    #
    #   2. Obtaining a total ordering of the edges of the extension
    #   resulting from (1). It is summarized in Fig. 13 of
    #   Chickering's paper and implemented in the function
    #   order_edges.
    #
    #   3. Labelling the edges as compelled or reversible, by which we
    #   can easily obtain the CPDAG. It is summarized in Fig. 14 of
    #   Chickering's paper and implemented in the function label_edges.

    # The above are put together in the function pdag_to_cpdag

    # NOTE!!!: Algorithm (1) is from the 1992 paper "A simple
    # algorithm to construct a consistent extension of a partially
    # oriented graph" by Dorit Dor and Michael Tarsi. There is an
    # ERROR in the summarized version in Chickering's paper. In
    # particular, the condition that N_x U Pa_x is a clique is not
    # equivalent to the condition from Dor & Torsi that every neighbor
    # of X should be adjacent to all of X's adjacent nodes. The
    # condition summarized in Chickering is more restrictive (i.e. it
    # also asks that the parents of X are adjacent to each other), but
    # this only results in an error for some graphs, and was only
    # uncovered during exhaustive testing.

# The complete pipeline: pdag -> dag -> ordered -> labelled -> cpdag


def pdag_to_cpdag(pdag):
    """
    Transform a PDAG into its corresponding CPDAG. Returns a ValueError
    exception if the given PDAG does not admit a consistent extension.

    Parameters
    ----------
    pdag: np.array
        the adjacency matrix of a given PDAG where pdag[i, j] != 0 if i
        -> j and i - j if also pdag[j, i] != 0.

    Returns
    -------
    cpdag: np.array
        the adjacency matrix of the corresponding CPDAG

    """
    # 1. Obtain a consistent extension of the pdag
    dag = pdag_to_dag(pdag)
    # 2. Recover the cpdag
    return dag_to_cpdag(dag)

# dag -> ordered -> labelled -> cpdag


def dag_to_cpdag(G):
    """
    Return the completed partially directed acyclic graph(CPDAG) that
    represents the Markov equivalence class of a given DAG. Returns a
    ValueError exception if the given graph is not a DAG.

    Parameters
    ----------
    G: np.array
        the adjacency matrix of the given graph, where G[i, j] != 0 iff i -> j

    Returns
    -------
    cpdag: np.array
        the adjacency matrix of the corresponding CPDAG

    """
    # 1. Perform a total ordering of the edges
    ordered = order_edges(G)
    # 2. Label edges as compelled or reversible
    labelled = label_edges(ordered)
    # 3. Construct CPDAG
    cpdag = np.zeros_like(labelled)
    # set compelled edges
    cpdag[labelled == 1] = labelled[labelled == 1]
    # set reversible edges
    fros, tos = np.where(labelled == -1)
    for (x, y) in zip(fros, tos):
        cpdag[x, y], cpdag[y, x] = 1, 1
    return cpdag


def pdag_to_dag(P, debug=False):
    """
    Find a consistent extension of the given PDAG. Return a

    Parameters
    ----------
    P: np.array
        adjacency matrix representing the PDAG connectivity, where
        P[i, j] = 1 = > i -> j
    debug: bool, optional
        if debugging traces should be printed

    Returns
    -------
    G: np.array
        the adjacency matrix of a DAG which is a consistent extension
        (i.e. same v-structures and skeleton) of P.

    Raises
    ------
    ValueError
        if the PDAG does not admit a consistent extension.
    """
    oP = P.copy()
    G = only_directed(P)
    indexes = list(range(len(P)))  # To keep track of the real variable
    # indexes as we remove nodes from P
    while P.size > 0:
        print(P) if debug else None
        print(indexes) if debug else None
        # Select a node which
        #   1. has no outgoing edges in P (i.e. childless, is a sink)
        #   2. all its neighbors are adjacent to all its adjacent nodes
        found = False
        i = 0
        while not found and i < len(P):
            # Check condition 1
            sink = len(ch(i, P)) == 0
            # Check condition 2
            n_i = neighbors(i, P)
            adj_i = adj(i, P)
            adj_neighbors = np.all([adj_i - {y} <= adj(y, P) for y in n_i])
            print("   i:", i, ": n=", n_i, "adj=", adj_i, "ch=", ch(i, P)) if debug else None
            found = sink and adj_neighbors
            # If found, orient all incident undirected edges and
            # remove i from the subgraph
            if found:
                print("  Found candidate %d (%d)" % (i, indexes[i])) if debug else None
                # Orient all incident undirected edges
                real_i = indexes[i]
                real_neighbors = [indexes[j] for j in n_i]
                for j in real_neighbors:
                    G[j, real_i] = 1
                # Remove i and its incident (directed and undirected edges)
                all_but_i = list(set(range(len(P))) - {i})
                P = P[all_but_i, :][:, all_but_i]
                indexes.remove(real_i)  # to keep track of the real
                # variable indices
            else:
                i += 1
        # A node which satisfies conditions 1,2 exists iff the
        # PDAG admits a consistent extension
        if not found:
            raise ValueError("PDAG %s does not admit consistent extension" % oP)
    return G


def order_edges(G):
    """
    Find a total ordering of the edges in DAG G, as an intermediate
    step to obtaining the CPDAG representing the Markov equivalence class to
    which it belongs. Raises a ValueError exception if G is not a DAG.

    Parameters
    ----------
    G: np.array
        the adjacency matrix of a graph G, where G[i, j] != 0 iff i -> j.

    Returns
    -------
    ordered: np.array
       the adjacency matrix of the graph G, but with labelled edges,
       i.e. i -> j is has label x iff ordered[i, j] = x.

    """
    if not is_dag(G):
        raise ValueError("The given graph is not a DAG")
    # i.e. if i -> j, then i appears before j in order
    order = topological_ordering(G)
    # You can check the above by seeing that np.all([i == order[pos[i]] for i in range(p)]) is True
    # Unlabelled edges as marked with -1
    ordered = (G != 0).astype(int) * -1
    i = 1
    while (ordered == -1).any():
        # let y be the lowest ordered node that has an unlabelled edge
        # incident to it
        froms, tos = np.where(ordered == -1)
        with_unlabelled = np.unique(np.hstack((froms, tos)))
        y = sort(with_unlabelled, reversed(order))[0]
        # let x be the highest ordered node s.t. the edge x -> y
        # exists and is unlabelled
        unlabelled_parents_y = np.where(ordered[:, y] == -1)[0]
        x = sort(unlabelled_parents_y, order)[0]
        ordered[x, y] = i
        i += 1
    return ordered


def label_edges(ordered):
    """Given a DAG with edges labelled according to a total ordering,
    label each edge as being compelled or reverisble.

    Parameters
    ----------
    ordered: np.array
        the adjacency matrix of a graph, with the edges labelled
        according to a total ordering.

    Returns
    -------
    labelled: np.array
        the adjacency matrix of G but with labelled edges, where
          - labelled[i, j] = 1 iff i -> j is compelled, and
          - labelled[i, j] = -1 iff i -> j is reversible.

    """
    # Validate the input
    if not is_dag(ordered):
        raise ValueError("The given graph is not a DAG")
    no_edges = (ordered != 0).sum()
    if sorted(ordered[ordered != 0]) != list(range(1, no_edges + 1)):
        raise ValueError("The ordering of edges is not valid:", ordered[ordered != 0])
    # define labels: 1: compelled, -1: reversible, -2: unknown
    COM, REV, UNK = 1, -1, -2
    labelled = (ordered != 0).astype(int) * UNK
    # while there are unknown edges
    while (labelled == UNK).any():
        # print(labelled)
        # let (x,y) be the unknown edge with lowest order
        # (i.e. appears last in the ordering, NOT has smalles label)
        # in ordered
        unknown_edges = (ordered * (labelled == UNK).astype(int)).astype(float)
        unknown_edges[unknown_edges == 0] = -np.inf
        # print(unknown_edges)
        (x, y) = np.unravel_index(np.argmax(unknown_edges), unknown_edges.shape)
        # print(x,y)
        # iterate over all edges w -> x which are compelled
        Ws = np.where(labelled[:, x] == COM)[0]
        end = False
        for w in Ws:
            # if w is not a parent of y, label all edges into y as
            # compelled, and finish this pass
            if labelled[w, y] == 0:
                labelled[list(pa(y, labelled)), y] = COM
                end = True
                break
            # otherwise, label w -> y as compelled
            else:
                labelled[w, y] = COM
        if not end:
            # if there exists an edge z -> y such that z != x and z is
            # not a parent of x, label all unknown edges (this
            # includes x -> y) into y with compelled; label with
            # reversible otherwise.
            z_exists = len(pa(y, labelled) - {x} - pa(x, labelled)) > 0
            unknown = np.where(labelled[:, y] == UNK)[0]
            assert x in unknown
            labelled[unknown, y] = COM if z_exists else REV
    return labelled


# --------------------------------------------------------------------
# Implementation of the Meek rules, see maximally_orient below

def rule_1(i, j, A):
    # If there is at least one parent of i which is not adjacent to j, orient the edge i->j
    if len(pa(i, A)) > 0 and not pa(i, A) <= adj(j, A):
        return True
    else:
        return False


def rule_2(i, j, A):
    # If there is a path i -> k -> j, then orient i ->j
    return len(ch(i, A) & pa(j, A)) > 0


def rule_3(i, j, A):
    # If i is a neighbor of at least two parents of j, and these are
    # NOT adjacent, then orient i -> j
    intersection = neighbors(i, A) & pa(j, A)
    # Check that there are at least two parents which are not adjacent
    if len(intersection) >= 2:
        for k in intersection:
            for l in intersection - {k}:
                if k not in adj(l, A):
                    return True
    return False


def rule_4(i, j, A):
    pa_j = pa(j, A)
    n_i = neighbors(i, A)
    # if i is a neighbor of a parent k of j, and a neighbor of a
    # parent h of l, and j and h are not adjacent, orient i -> j
    Ks = pa_j & n_i
    if len(Ks) > 0:
        Hs = n_i & set(reduce(lambda acc, k: acc | pa(k, A), Ks, set()))
        if len(Hs) > 0:
            # Check that h and j are not adjacent
            adj_j = adj(j, A)
            for h in Hs:
                if h not in adj_j:
                    return True
    return False


def maximally_orient(P, debug=False):
    """Maximally orient the PDAG P by successive application of the meek
    rules(see functions rule_{1, 2, 3, 4})

    Parameters
    ----------
    P: np.array
        Adjacency matrix of a PDAG.
    debug : bool
        If debugging traces should be printed.

    Returns
    -------
    maximally_oriented: np.array
        the maximally oriented PDAG

    """
    # Ensure that the given PDAG does admit a consistent extension
    try:
        pdag_to_dag(P)
    except ValueError as e:
        raise e
    P = P.copy()
    # Repeatedly apply meek rules until no edges can be oriented
    oriented_edges = True
    while oriented_edges:
        oriented_edges = False
        for (i, j) in undirected_edges(P):
            if rule_1(i, j, P) or rule_2(i, j, P) or rule_3(i, j, P) or rule_4(i, j, P):
                # orient i -> j
                oriented_edges = True
                if debug:
                    rules = [rule_1(i, j, P), rule_2(i, j, P), rule_3(i, j, P), rule_4(i, j, P)]
                    print('Rules: %s => Oriented %d -> %d' % (rules, i, j))
                P[j, i] = 0
            elif rule_1(j, i, P) or rule_2(j, i, P) or rule_3(j, i, P) or rule_4(j, i, P):
                oriented_edges = True
                # orient j -> i
                if debug:
                    rules = [rule_1(j, i, P), rule_2(j, i, P), rule_3(j, i, P), rule_4(j, i, P)]
                    print('Rules: %s => Orjented %d -> %d' % (rules, j, i))
                P[i, j] = 0
    return P

# --------------------------------------------------------------------
# Implementation of the modified completion algorithm for GnIES


def pdag_to_icpdag(P, I):
    """Completion algorithm used in the inner procedure of the GnIES
    algorithm. Given a PDAG representing some graphs in the I-equivalence class,
    returns the I-CPDAG representing all of the graphs in the class.

    Parameters
    ----------
    P: np.array
        Adjacency matrix of the PDAG, where `P[i, j] = 1` and `P[j,i]
        = 0` implies i -> j and `P[i, j] = P[j,i] = 1` implies `i -
        j`.
    I : set of ints
        The intervention targets.

    Returns
    -------
    icpdag : np.array
        The adjacency matrix of the resulting I-CPDAG.

    Raises
    ------
    ValueError :
        If any variable in I has undirected edges in the given PDAG.

    """
    # Check that all edges around variables in I are oriented
    for i in I:
        if len(neighbors(i, P)) > 0:
            msg = "Invalid PDAG: has undirected edges around %d for I=%s"
            raise ValueError(msg % (i, I))
    G = pdag_to_dag(P)
    return dag_to_icpdag(G, I)


def dag_to_icpdag(G, I, debug=False):
    """Given a DAG adjacency G and intervention targets I, return the
    I-CPDAG corresponding to its I-equivalence class.

    """
    P = dag_to_cpdag(G)
    # Construct edges that must be oriented
    directed_edges = []
    for i in I:
        directed_edges += [(i, j) for j in ch(i, G)]
        directed_edges += [(j, i) for j in pa(i, G)]
    while len(directed_edges) > 0:
        print(directed_edges) if debug else None
        (x, y) = directed_edges.pop()
        P[y, x] = 0
        P = maximally_orient(P, debug)
    assert is_consistent_extension(G, P)
    return P


# --------------------------------------------------------------------
# General utilities

def all_dags(pdag, max_combinations=None):
    """
    Given a PDAG, enumerate all its consistent extensions(DAGs).

    Parameters
    ----------
    pdag: np.array
        Adjacency matrix representing the PDAG connectivity, where
        P[i, j] != 0 = > i -> j.

    Returns
    -------
    dags: np.array
        The adjacencies of the consistent extensions.
    """
    fro, to = np.where(only_undirected(pdag))
    undirected_edges = np.array(list(filter(lambda e: e[0] > e[1], zip(fro, to))))
    if max_combinations is not None and 2**len(undirected_edges) > max_combinations:
        raise ValueError("The number of different edge orientations (%d) is over the value given for max_combinations (%d)" % (2**len(undirected_edges), max_combinations))
    assert len(undirected_edges) == np.sum(only_undirected(pdag)) / 2
    if len(undirected_edges) == 0:
        return np.array([pdag.copy()])
    # All possible orientations for the undirected edges
    combinations = cartesian([np.array([True, False])] * len(undirected_edges), dtype=bool)
    assert len(combinations) == 2**(np.sum(only_undirected(pdag)) / 2)
    dags = []
    # Iterate over all possible combinations, flipping the edges
    oriented_edges = np.zeros_like(undirected_edges)
    for flipped in combinations:
        oriented_edges[flipped, :] = undirected_edges[:, [1, 0]][flipped]
        oriented_edges[flipped == False, :] = undirected_edges[:, [0, 1]][flipped == False]
        A = pdag.copy()
        A[oriented_edges[:, 1], oriented_edges[:, 0]] = 0
        dags.append(A)
    # NOTE: Done in two steps to check assertion
    assert len(np.unique(dags, axis=0)) == len(combinations)
    # Filter graphs with cycles and inconsistent extensions
    dags = [A for A in dags if is_dag(A) and is_consistent_extension(A, pdag)]
    return np.array(dags)


# Very fast way to generate a cartesian product of input arrays
# Credit: https://gist.github.com/hernamesbarbara/68d073f551565de02ac5


def cartesian(arrays, out=None, dtype=np.byte):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays: list of array-like
        1-D arrays to form the cartesian product of.
    out: ndarray
        Array to place the cartesian product in .

    Returns
    -------
    out: ndarray
        2-D array of shape(M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >> > cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """
    arrays = [np.asarray(x) for x in arrays]

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
    return out


def sort(L, order=None):
    """Sort the elements in an iterable according to its pre-defined
    'sorted' function, or according to a given order: i will precede j
    if i precedes j in the order.

    Parameters
    ----------
    L: iterable
        the iterable to be sorted
    order: iterable or None, optional
        a given ordering. In the sorted result, i will precede j if i
        precedes j in order. If None, the predefined 'sorted' function
        of the iterator will be used. Defaults to None.

    Returns
    -------
    ordered: list
        a list containing the elements of L, sorted from lesser to
        greater or according to the given order.

    """
    L = list(L)
    if order is None:
        return sorted(L)
    else:
        order = list(order)
        pos = np.zeros(len(order), dtype=int)
        pos[order] = range(len(order))
        positions = [pos[l] for l in L]
        return [tup[1] for tup in sorted(zip(positions, L))]


def subsets(S):
    """
    Return an iterator with all possible subsets of the set S.

    Parameters
    ----------
    S: set
        a given set

    Returns
    -------
    subsets: iterable
        an iterable over the subsets of S, in increasing size

    """
    subsets = []
    for r in range(len(S) + 1):
        subsets += [set(ss) for ss in itertools.combinations(S, r)]
    return subsets


def member(L, A):
    """
    Return the index of the first appearance of array A in L.

    Parameters
    ----------
    L: list of np.array
        list on which to perform the search
    A: np.array
        the target array

    Returns
    -------
    position: int or None
        the index of the first appearance of array A in list L, or
        None if A is not in L.

    """
    for i, B in enumerate(L):
        if (A == B).all():
            return i
    return None


def delete(array, mask, axis=None):
    """Wrapper for numpy.delete, which adapts the call depending on the
    numpy version(the API changed on 1.19.0)

    Parameters
    ----------
    array: array_like
        Input array.
    mask: boolean array
        Specifies the sub-arrays to remove along the given axis
    axis: int
        The axis along which to delete the subarrays specified by mask

    Returns
    -------
    out: ndarray
        a copy of array with the elements specified by mask removed

    """
    if np.version.version < '1.19.0':
        idx = np.where(mask)[0]
        return np.delete(array, idx, axis)
    else:
        return np.delete(array, mask, axis)


def split_data(data, ratios, random_state=42):
    """Split the given data into several folds for training/validation.

    Parameters
    ----------
    data : list of numpy.ndarray
        A list containing the sample from each environment.
    ratios : list of float
        A list with as many elements as folds, each indicating the
        proportion of the data allocated to it. The elements of the
        list must sum to 1.

    Returns
    -------
    splits : list of list of numpy.ndarray

    Raises
    ------
    ValueError :
        If the elements in ratios do not sum up to 1.

    """
    # Check inputs
    if np.sum(ratios) != 1:
        raise ValueError("The elements in ratios must add up to 1.")
    # Split the data into folds
    n_folds = len(ratios)
    folds = dict((i, []) for i in range(n_folds))
    rng = np.random.default_rng(random_state)
    for sample in data:
        n = len(sample)
        sample = sample.copy()
        rng.shuffle(sample)
        start = 0
        for i, ratio in enumerate(ratios):
            if i < n_folds:
                fold_size = round(n * ratio)
                fold_sample = sample[start:start + fold_size]
            else:
                fold_sample = sample[start::]
            folds[i].append(fold_sample)
            start += fold_size
    assert len(folds) == n_folds
    assert len(folds[0]) == len(data)
    return list(folds.values())


def sorted_tuple(iterable):
    """Given an iterable which implements tuple and sorted, return the
    corresponding tuple representation where all elements are sorted.
    TODO: Write a proper docstring."""
    return tuple(sorted(tuple(iterable)))


def all_but(k, p):
    """Return [0,...,p-1] without k"""
    k = np.atleast_1d(k)
    return [i for i in range(p) if i not in k]


# Example graphs


def eg1():
    W = np.array(
        [
            [0, 1, -1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ]
    )
    markov_blankets = [[1, 2], [0, 3, 2], [0, 3, 1], [1, 2, 4], [3]]
    parents = [[], [0], [0], [1, 2], [3]]
    return W, parents, markov_blankets


def eg2():
    W = np.array(
        [
            [0, 1, -1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
        ]
    )
    markov_blankets = [[1, 2], [0, 3, 2], [0, 3, 1], [1, 2, 4, 5], [3, 5], [3, 4]]
    parents = [[], [0], [0], [1, 2], [3, 5], []]
    return W, parents, markov_blankets


def eg3():
    W = np.array(
        [
            [0, 1, -1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    markov_blankets = [
        [1, 2],
        [0, 3, 2],
        [0, 3, 1],
        [1, 2, 4, 5],
        [3, 5, 7],
        [3, 4, 6],
        [5],
        [4],
    ]
    parents = [[], [0], [0], [1, 2], [3, 5], [], [5], [4]]
    return W, parents, markov_blankets


def eg4():
    W = np.array([[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
    markov_blankets = [[1, 2], [0, 2], [0, 1, 3], [2]]
    parents = [[], [], [0, 1], [2]]
    return W, parents, markov_blankets


def eg5():
    W = np.array(
        [
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        ]
    )
    ordering = np.array([7, 4, 2, 3, 6, 5, 0, 1])
    parents = [[4, 6, 7], [0, 2, 3], [], [2], [7], [2, 3, 7], [3, 4], []]
    children = [[1], [], [1, 3, 5], [1, 5, 6], [0, 6], [], [0], [0, 4, 5]]
    markov_blankets = [
        [4, 6, 7, 1, 2, 3],
        [0, 2, 3],
        [1, 3, 5, 0, 7],
        [0, 1, 2, 4, 5, 6, 7],
        [7, 0, 6, 7, 3],
        [2, 3, 7],
        [3, 4, 0, 7],
        [0, 4, 5, 2, 3, 6],
    ]
    return W, parents, markov_blankets


def eg6():
    W = np.array(
        [
            [0, 0, 1, 0, 1],
            [0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ]
    )
    parents = [[], [], [0, 1], [2], [0, 1, 3]]
    children = [[2], [2], [3], [4], []]
    markov_blankets = [[1, 2, 3, 4], [0, 2, 3, 4], [0, 1, 3], [2, 4, 0, 1], [0, 1, 3]]
    return W, parents, markov_blankets
