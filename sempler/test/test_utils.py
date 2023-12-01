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

# ---------------------------------------------------------------------
# Unit testing for module utils

import unittest
import numpy as np

import sempler
import sempler.generators
import itertools

# Tested functions
import sempler.utils as utils

import networkx as nx

class UtilsTests(unittest.TestCase):
    def test_combinations(self):
        for p in range(2, 10):
            target = np.random.choice(p)
            combinations = utils.combinations(p, target)
            self.assertEqual(int(2 ** (p - 1)), len(combinations))
            [self.assertTrue(target not in s) for s in combinations]

    def test_is_dag_1(self):
        # Should be correct
        A = np.array(
            [
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
            ]
        )
        self.assertTrue(utils.is_dag(A))

    def test_is_dag_2(self):
        # DAG with a cycle
        A = np.array(
            [
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 1],
                [1, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
            ]
        )
        self.assertFalse(utils.is_dag(A))

    def test_is_dag_3(self):
        # PDAG
        A = np.array(
            [
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0],
            ]
        )
        self.assertFalse(utils.is_dag(A))

    def test_topological_sort_1(self):
        A = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 1, 0],
            ]
        ).T
        order = utils.topological_ordering(A)
        possible_orders = [[0, 1, 2, 3, 4], [1, 0, 2, 3, 4]]
        self.assertIn(order, possible_orders)

    def test_topological_sort_2(self):
        A = np.array(
            [
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 1, 0],
            ]
        ).T
        try:
            utils.topological_ordering(A)
            self.fail()
        except:
            pass

    def test_topological_sort_3(self):
        G = 100
        p = 30
        for i in range(G):
            A = sempler.generators.dag_avg_deg(p, 3, 1, 1)
            ordering = utils.topological_ordering(A)
            fro, to = np.where(A != 0)
            # Test that the ordering is correct, i.e. for every edge x
            # -> y in the graph, x appears before in the ordering
            for x, y in zip(fro, to):
                pos_x = np.where(np.array(ordering) == x)[0][0]
                pos_y = np.where(np.array(ordering) == y)[0][0]
                self.assertLess(pos_x, pos_y)
        print("Checked topological sorting for %d DAGs" % (i + 1))

    def test_matrix_block(self):
        M = np.array(
            [[11, 12, 13, 14], [21, 22, 23, 24], [31, 32, 33, 34], [41, 42, 43, 44]]
        )
        # Tests
        tests = [
            (range(4), range(4), M),
            ([1, 2], [3], np.array([[24, 34]]).T),
            (range(4), [1], M[:, [1]]),
            ([2], range(4), M[[2], :]),
            ([0, 1], [0, 1], np.array([[11, 12], [21, 22]])),
            ([0, 1], [1, 3], np.array([[12, 14], [22, 24]])),
            # Test order of indices is also respected
            (range(3, -1, -1), range(3, -1, -1), M[::-1, ::-1]),
            (range(3, -1, -1), range(4), M[::-1, :]),
            (range(3, -1, -1), [1], M[::-1, [1]]),
            ([2], range(3, -1, -1), M[[2], ::-1]),
            ([1, 0], [0, 1], np.array([[21, 22], [11, 12]])),
            ([0, 1], [3, 1], np.array([[14, 12], [24, 22]])),
        ]
        for test in tests:
            (A, B, truth) = test
            # print(A, B, truth, matrix_block(M, A, B))
            self.assertTrue((utils.matrix_block(M, A, B) == truth).all())

    def test_sampling_matrix(self):
        W = np.array([[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
        truth = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]])
        self.assertTrue((truth == utils.sampling_matrix(W)).all())

    def test_nonzero(self):
        tol = 1e-12
        A = np.array([0, 1, tol, -1, tol / 2, -tol * 2])
        self.assertTrue((np.array([1, 3, 5]) == utils.nonzero(A)).all())

    def test_all_but(self):
        self.assertTrue([0, 1, 3, 4] == utils.all_but(2, 5))
        self.assertTrue([0, 3, 4] == utils.all_but([1, 2], 5))

    def test_closure(self):
        p = 10
        # Empty graph
        A = np.zeros((p, p))
        self.assertTrue((utils.transitive_closure(A) == A).all())
        # Fully connected
        A = sempler.generators.dag_full(p)
        self.assertTrue((utils.transitive_closure(A) == A).all())
        # Chain graph
        A = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        closure = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
        self.assertTrue((utils.transitive_closure(A) == closure).all())
        # Common cause
        A = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
        self.assertTrue((utils.transitive_closure(A) == A).all())
        # Common cause
        A = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
        self.assertTrue((utils.transitive_closure(A) == A).all())


class UtilsTests2(unittest.TestCase):

    # ------------------------
    # Tests of graph functions

    def test_subsets(self):
        # Test 1
        self.assertEqual([set()], utils.subsets(set()))
        # Test 2
        S = {0, 1}
        subsets = utils.subsets(S)
        self.assertEqual(4, len(subsets))
        for s in [set(), {0}, {1}, {0, 1}]:
            self.assertIn(s, subsets)
        # Test 3
        S = {1, 2, 3, 4}
        subsets = utils.subsets(S)
        self.assertEqual(16, len(subsets))

    def test_is_dag_1(self):
        # Should be correct
        A = np.array([[0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 1],
                      [0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0]])
        self.assertTrue(utils.is_dag(A))

    def test_is_dag_2(self):
        # DAG with a cycle
        A = np.array([[0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 1],
                      [1, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0]])
        self.assertFalse(utils.is_dag(A))

    def test_is_dag_3(self):
        # PDAG
        A = np.array([[0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 1],
                      [0, 0, 0, 0, 1],
                      [0, 0, 0, 1, 0]])
        self.assertFalse(utils.is_dag(A))

    def test_topological_sort_1(self):
        A = np.array([[0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [1, 1, 0, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 1, 1, 0]]).T
        order = utils.topological_ordering(A)
        possible_orders = [[0, 1, 2, 3, 4], [1, 0, 2, 3, 4]]
        self.assertIn(order, possible_orders)

    def test_topological_sort_2(self):
        A = np.array([[0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0],
                      [1, 1, 0, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 1, 1, 0]]).T
        try:
            utils.topological_ordering(A)
            self.fail()
        except ValueError:
            pass

    def test_topological_sort_3(self):
        G = 100
        p = 30
        for i in range(G):
            A = sempler.generators.dag_avg_deg(p, 3, 1, 1)
            ordering = utils.topological_ordering(A)
            fro, to = np.where(A != 0)
            # Test that the ordering is correct, i.e. for every edge x
            # -> y in the graph, x appears before in the ordering
            for (x, y) in zip(fro, to):
                pos_x = np.where(np.array(ordering) == x)[0][0]
                pos_y = np.where(np.array(ordering) == y)[0][0]
                self.assertLess(pos_x, pos_y)
        print("Checked topological sorting for %d DAGs" % (i + 1))

    def test_neighbors(self):
        p = 4
        A = np.zeros((p, p))
        # Test 1
        for i in range(p):
            self.assertEqual(set(), utils.neighbors(i, A))
        # Test 2
        A = np.array([[0, 1, 1, 0],
                      [0, 0, 1, 0],
                      [0, 1, 0, 1],
                      [0, 0, 1, 0]])
        self.assertEqual(set(), utils.neighbors(0, A))
        self.assertEqual({2}, utils.neighbors(1, A))
        self.assertEqual({1, 3}, utils.neighbors(2, A))
        self.assertEqual({2}, utils.neighbors(3, A))

    def test_adj(self):
        p = 4
        A = np.zeros((p, p))
        # Test 1
        for i in range(p):
            self.assertEqual(set(), utils.adj(i, A))
        # Test 2
        A = np.array([[0, 1, 1, 0],
                      [0, 0, 1, 0],
                      [0, 1, 0, 1],
                      [0, 0, 1, 0]])
        self.assertEqual({1, 2}, utils.adj(0, A))
        self.assertEqual({0, 2}, utils.adj(1, A))
        self.assertEqual({0, 1, 3}, utils.adj(2, A))
        self.assertEqual({2}, utils.adj(3, A))

    def test_na(self):
        A = np.array([[0, 1, 1, 0],
                      [0, 0, 1, 0],
                      [0, 1, 0, 1],
                      [0, 0, 1, 0]])
        # 00
        self.assertEqual(set(), utils.na(0, 0, A))
        # 01
        self.assertEqual(set(), utils.na(0, 1, A))
        # 02
        self.assertEqual(set(), utils.na(0, 2, A))
        # 03
        self.assertEqual(set(), utils.na(0, 3, A))

        # 10
        self.assertEqual({2}, utils.na(1, 0, A))
        # 11
        self.assertEqual({2}, utils.na(1, 1, A))
        # 12
        self.assertEqual(set(), utils.na(1, 2, A))
        # 13
        self.assertEqual({2}, utils.na(1, 3, A))

        # 20
        self.assertEqual({1}, utils.na(2, 0, A))
        # 21
        self.assertEqual(set(), utils.na(2, 1, A))
        # 22
        self.assertEqual({1, 3}, utils.na(2, 2, A))
        # 23
        self.assertEqual(set(), utils.na(2, 3, A))

        # 30
        self.assertEqual({2}, utils.na(3, 0, A))
        # 31
        self.assertEqual({2}, utils.na(3, 1, A))
        # 32
        self.assertEqual(set(), utils.na(3, 2, A))
        # 33
        self.assertEqual({2}, utils.na(3, 3, A))

    def test_pa(self):
        p = 4
        A = np.zeros((p, p))
        # Test 1
        for i in range(p):
            self.assertEqual(set(), utils.pa(i, A))
        # Test 2
        A = np.array([[0, 1, 1, 0],
                      [0, 0, 1, 0],
                      [0, 1, 0, 1],
                      [0, 0, 1, 0]])
        self.assertEqual(set(), utils.pa(0, A))
        self.assertEqual({0}, utils.pa(1, A))
        self.assertEqual({0}, utils.pa(2, A))
        self.assertEqual(set(), utils.pa(3, A))

    def test_ch(self):
        p = 4
        A = np.zeros((p, p))
        # Test 1
        for i in range(p):
            self.assertEqual(set(), utils.ch(i, A))
        # Test 2
        A = np.array([[0, 1, 1, 0],
                      [0, 0, 1, 0],
                      [0, 1, 0, 1],
                      [0, 0, 1, 0]])
        self.assertEqual({1, 2}, utils.ch(0, A))
        self.assertEqual(set(), utils.ch(1, A))
        self.assertEqual(set(), utils.ch(2, A))
        self.assertEqual(set(), utils.ch(3, A))

    def test_is_clique(self):
        A = np.array([[0, 1, 1, 0],
                      [0, 0, 1, 0],
                      [0, 1, 0, 1],
                      [0, 0, 1, 0]])
        # |S| = 1
        for i in range(len(A)):
            self.assertTrue(utils.is_clique({i}, A))
        # |S| = 2
        self.assertTrue(utils.is_clique({0, 1}, A))
        self.assertTrue(utils.is_clique({0, 2}, A))
        self.assertFalse(utils.is_clique({0, 3}, A))
        self.assertTrue(utils.is_clique({1, 2}, A))
        self.assertFalse(utils.is_clique({1, 3}, A))
        self.assertTrue(utils.is_clique({2, 3}, A))
        # |S| = 3
        self.assertTrue(utils.is_clique({0, 1, 2}, A))
        self.assertFalse(utils.is_clique({0, 1, 3}, A))
        self.assertFalse(utils.is_clique({0, 2, 3}, A))
        self.assertFalse(utils.is_clique({1, 2, 3}, A))
        # |S| = 4
        self.assertFalse(utils.is_clique({0, 1, 2, 3}, A))

    def test_semi_directed_paths_1(self):
        A = np.array([[0, 1, 1, 0],
                      [0, 0, 1, 0],
                      [0, 1, 0, 1],
                      [0, 0, 1, 0]])
        # 0 to 1
        paths = utils.semi_directed_paths(0, 1, A)
        self.assertEqual(2, len(paths))
        self.assertTrue([0, 1] in paths)
        self.assertTrue([0, 2, 1] in paths)
        # 1 to 0
        paths = utils.semi_directed_paths(1, 0, A)
        self.assertEqual(0, len(paths))

        # 0 to 2
        paths = utils.semi_directed_paths(0, 2, A)
        self.assertEqual(2, len(paths))
        self.assertTrue([0, 2] in paths)
        self.assertTrue([0, 1, 2] in paths)
        # 2 to 0
        paths = utils.semi_directed_paths(2, 0, A)
        self.assertEqual(0, len(paths))

        # 0 to 3
        paths = utils.semi_directed_paths(0, 3, A)
        self.assertEqual(2, len(paths))
        self.assertTrue([0, 2, 3] in paths)
        self.assertTrue([0, 1, 2, 3] in paths)
        # 3 to 0
        paths = utils.semi_directed_paths(3, 0, A)
        self.assertEqual(0, len(paths))

        # 1 to 2
        paths = utils.semi_directed_paths(1, 2, A)
        self.assertEqual(1, len(paths))
        self.assertTrue([1, 2] in paths)
        # 2 to 1
        paths = utils.semi_directed_paths(2, 1, A)
        self.assertEqual(1, len(paths))
        self.assertTrue([2, 1] in paths)

        # 1 to 3
        paths = utils.semi_directed_paths(1, 3, A)
        self.assertEqual(1, len(paths))
        self.assertTrue([1, 2, 3] in paths)
        # 3 to 1
        paths = utils.semi_directed_paths(3, 1, A)
        self.assertEqual(1, len(paths))
        self.assertTrue([3, 2, 1] in paths)

        # 2 to 3
        paths = utils.semi_directed_paths(2, 3, A)
        self.assertEqual(1, len(paths))
        self.assertTrue([2, 3] in paths)
        # 3 to 2
        paths = utils.semi_directed_paths(3, 2, A)
        self.assertEqual(1, len(paths))
        self.assertTrue([3, 2] in paths)

    def test_semi_directed_paths_2(self):
        # Test vs. networkx implementation
        G = 100
        p = 30
        for i in range(G):
            A = sempler.generators.dag_avg_deg(p, 3, 1, 1)
            cpdag = utils.pdag_to_cpdag(A)
            G = nx.from_numpy_array(cpdag, create_using=nx.DiGraph)
            for (x, y) in itertools.combinations(range(p), 2):
                # From x to y
                paths_own = utils.semi_directed_paths(x, y, cpdag)
                paths_nx = list(nx.algorithms.all_simple_paths(G, x, y))
                self.assertEqual(sorted(paths_nx), sorted(paths_own))
                # From y to x
                paths_own = utils.semi_directed_paths(y, x, cpdag)
                paths_nx = list(nx.algorithms.all_simple_paths(G, y, x))
                self.assertEqual(sorted(paths_nx), sorted(paths_own))
        print("Checked path enumeration for %d PDAGs" % (i + 1))

    def test_semi_directed_paths_3(self):
        A = np.array([[0, 1, 0, 0],
                      [1, 0, 1, 1],
                      [0, 1, 0, 1],
                      [0, 1, 1, 0]])
        G = nx.from_numpy_array(A, create_using=nx.DiGraph)
        for (x, y) in itertools.combinations(range(len(A)), 2):
            # From x to y
            paths_own = utils.semi_directed_paths(x, y, A)
            paths_nx = list(nx.algorithms.all_simple_paths(G, x, y))
            self.assertEqual(sorted(paths_nx), sorted(paths_own))
            # From y to x
            paths_own = utils.semi_directed_paths(y, x, A)
            paths_nx = list(nx.algorithms.all_simple_paths(G, y, x))
            self.assertEqual(sorted(paths_nx), sorted(paths_own))

    def test_semi_directed_paths_4(self):
        # Test that there exist no paths between a node and those who
        # appear before it in the topological ordering of a DAG
        for i in range(100):
            A = sempler.generators.dag_avg_deg(20, 3)
            ordering = utils.topological_ordering(A)
            ordering.reverse()
            for i, fro in enumerate(ordering):
                for to in ordering[(i+1):]:
                    self.assertEqual([], utils.semi_directed_paths(fro, to, A))
                    
    def test_skeleton(self):
        # Test utils.skeleton
        skeleton = np.array([[0, 1, 1, 0],
                             [1, 0, 1, 0],
                             [1, 1, 0, 1],
                             [0, 0, 1, 0]])
        # Test 0
        self.assertTrue((utils.skeleton(skeleton) == skeleton).all())
        # Test 1
        A1 = np.array([[0, 1, 1, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1],
                       [0, 0, 0, 0]])
        self.assertTrue((utils.skeleton(A1) == skeleton).all())
        # Test 2
        A2 = np.array([[0, 1, 1, 0],
                       [1, 0, 1, 0],
                       [0, 0, 0, 1],
                       [0, 0, 1, 0]])
        self.assertTrue((utils.skeleton(A2) == skeleton).all())
        # Test 3
        A3 = np.array([[0, 1, 1, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1],
                       [0, 0, 1, 0]])
        self.assertTrue((utils.skeleton(A3) == skeleton).all())

    def test_only_directed(self):
        # Test utils.only_directed
        # Undirected graph should result in empty graph
        A = np.array([[0, 1, 1, 0],
                      [1, 0, 1, 0],
                      [1, 1, 0, 1],
                      [0, 0, 1, 0]])
        self.assertTrue((utils.only_directed(A) == np.zeros_like(A)).all())
        # Directed graph should return the same graph (maintaining weights)
        A = np.array([[0, 1, 1, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [0, 0, 0, 0]]) * np.random.uniform(size=A.shape)
        self.assertTrue((utils.only_directed(A) == A).all())
        # Mixed graph should result in graph with only the directed edges
        A = np.array([[0, .5, 1, 0],
                      [0, 0, 1, 0],
                      [0, 1, 0, 1],
                      [0, 0, 1, 0]])
        truth = np.zeros_like(A)
        truth[0, 1], truth[0, 2] = 0.5, 1
        self.assertTrue((utils.only_directed(A) == truth).all())

    def test_only_undirected(self):
        # Test utils.only_undirected
        # Undirected graph should result in the same graph (maintaining weights)
        A = np.array([[0, 1, 1, 0],
                      [1, 0, 1, 0],
                      [1, 1, 0, 1],
                      [0, 0, 1, 0]]) * np.random.uniform(size=(4, 4))
        self.assertTrue((utils.only_undirected(A) == A).all())
        # Directed graph should return an empty graph
        A = np.array([[0, 1, 1, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [0, 0, 0, 0]])
        self.assertTrue((utils.only_undirected(A) == np.zeros_like(A)).all())
        # Mixed graph should result in graph with only the directed edges
        A = np.array([[0, 1, 1, 0],
                      [0, 0, 1, 0],
                      [0, 1, 0, 1],
                      [0, 0, 1, 0]])
        truth = np.array([[0, 0, 0, 0],
                          [0, 0, 1, 0],
                          [0, 1, 0, 1],
                          [0, 0, 1, 0]])
        self.assertTrue((utils.only_undirected(A) == truth).all())
        # Undirected and directed should be disjoint
        union = np.logical_xor(utils.only_directed(A), utils.only_undirected(A))
        self.assertTrue((union == A).all())

    def test_vstructures(self):
        # Test utils.vstructures
        # TODO: These tests do not contain any cases where (i,c,j)
        # with i > j and is saved as (j,c,i) instead
        # Undirected graph should yield no v_structures
        A = np.array([[0, 1, 1, 0],
                      [1, 0, 1, 0],
                      [1, 1, 0, 1],
                      [0, 0, 1, 0]])
        self.assertEqual(set(), utils.vstructures(A))
        # Fully directed graph 1
        A = np.array([[0, 1, 1, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [0, 0, 0, 0]])
        self.assertEqual(set(), utils.vstructures(A))
        # Fully directed graph 2
        A = np.array([[0, 1, 1, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 0],
                      [0, 0, 1, 0]])
        self.assertEqual({(1, 2, 3), (0, 2, 3)}, utils.vstructures(A))
        # Fully directed graph 3
        A = np.array([[0, 0, 1],
                      [0, 0, 1],
                      [0, 0, 0]])
        self.assertEqual({(0, 2, 1)}, utils.vstructures(A))
        # Mixed graph 1
        A = np.array([[0, 1, 1, 0],
                      [1, 0, 1, 0],
                      [0, 0, 0, 0],
                      [0, 0, 1, 0]])
        self.assertEqual({(1, 2, 3), (0, 2, 3)}, utils.vstructures(A))
        # Mixed graph 2
        A = np.array([[0, 1, 1, 0],
                      [1, 0, 1, 0],
                      [0, 0, 0, 1],
                      [0, 0, 0, 0]])
        self.assertEqual(set(), utils.vstructures(A))
        # Mixed graph 3
        A = np.array([[0, 1, 1, 0],
                      [1, 0, 1, 0],
                      [0, 0, 0, 1],
                      [0, 0, 1, 0]])
        self.assertEqual(set(), utils.vstructures(A))
        # Mixed graph 4
        A = np.array([[0, 0, 1],
                      [0, 0, 1],
                      [0, 1, 0]])
        self.assertEqual(set(), utils.vstructures(A))

    def test_is_consistent_extension_precondition(self):
        P = np.array([[0, 1, 1, 0],
                      [1, 0, 1, 0],
                      [0, 1, 0, 1],
                      [0, 0, 1, 0]])
        # Exception if A is not a DAG (has cycle)
        A = P.copy()
        A[1, 2], A[0, 1] = 0, 0
        try:
            utils.is_consistent_extension(A, P)
            self.fail()
        except ValueError as e:
            print("OK:", e)
        # Exception if A is not a DAG (has undirected edges)
        A = P.copy()
        A[2, 1], A[1, 0] = 0, 0
        try:
            utils.is_consistent_extension(A, P)
            self.fail()
        except ValueError as e:
            print("OK:", e)

    def test_is_consistent_extension_1(self):
        P = np.array([[0, 1, 1, 0],
                      [1, 0, 1, 0],
                      [0, 1, 0, 1],
                      [0, 0, 1, 0]])
        # Should return True
        A = P.copy()
        A[2, 1], A[1, 0], A[3, 2] = 0, 0, 0
        self.assertTrue(utils.is_consistent_extension(A, P))
        # Should return False (vstructs (0,2,3) and (1,2,3))
        A = P.copy()
        A[2, 1], A[1, 0], A[2, 3] = 0, 0, 0
        self.assertFalse(utils.is_consistent_extension(A, P))
        # Should return False (vstructs (0,2,3))
        A = P.copy()
        A[1, 2], A[1, 0], A[2, 3] = 0, 0, 0
        self.assertFalse(utils.is_consistent_extension(A, P))
        # Should return False (different skeleton)
        A = P.copy()
        A[2, 1], A[1, 0], A[3, 2] = 0, 0, 0
        A[1, 3] = 1
        self.assertFalse(utils.is_consistent_extension(A, P))
        # Should return False (different orientation)
        A = P.copy()
        A[2, 1], A[3, 2] = 0, 0
        A[0, 1] = 0
        A[0, 2], A[2, 0] = 0, 1
        self.assertFalse(utils.is_consistent_extension(A, P))

    def test_is_consistent_extension_2(self):
        P = np.array([[0, 1, 1, 0],
                      [1, 0, 1, 0],
                      [0, 0, 0, 1],
                      [0, 0, 1, 0]])
        # There are four extensions, two of which are consistent (same v-structures)
        # Extension 1 (consistent)
        A = np.array([[0, 0, 1, 0],
                      [1, 0, 1, 0],
                      [0, 0, 0, 1],
                      [0, 0, 0, 0]])
        self.assertTrue(utils.is_consistent_extension(A, P))
        # Extension 2 (consistent)
        A = np.array([[0, 1, 1, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [0, 0, 0, 0]])
        self.assertTrue(utils.is_consistent_extension(A, P))
        # Extension 3 (not consistent)
        A = np.array([[0, 0, 1, 0],
                      [1, 0, 1, 0],
                      [0, 0, 0, 0],
                      [0, 0, 1, 0]])
        self.assertFalse(utils.is_consistent_extension(A, P))
        # Extension 4 (not consistent)
        A = np.array([[0, 1, 1, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 0],
                      [0, 0, 1, 0]])
        self.assertFalse(utils.is_consistent_extension(A, P))

    def test_separates_preconditions(self):
        A = np.array([[0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [1, 1, 0, 1, 1],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]])
        # S and A are not disjoint
        try:
            utils.separates({1}, {1, 2}, {3, 4}, A)
            self.fail("Exception should have been thrown")
        except ValueError as e:
            print("OK:", e)
        # S and B are not disjoint
        try:
            utils.separates({0, 1}, {2}, {0, 3, 4}, A)
            self.fail("Exception should have been thrown")
        except ValueError as e:
            print("OK:", e)
        # A and B are not disjoint
        try:
            utils.separates({0, 1}, {2, 3}, {3, 4}, A)
            self.fail("Exception should have been thrown")
        except ValueError as e:
            print("OK:", e)
        # None are disjoint
        try:
            utils.separates({0, 1}, {0, 2, 3}, {1, 3, 4}, A)
            self.fail("Exception should have been thrown")
        except ValueError as e:
            print("OK:", e)

    def test_separates_1(self):
        A = np.array([[0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [1, 1, 0, 1, 1],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]])
        self.assertTrue(utils.separates({2}, {0, 1}, {3, 4}, A))
        self.assertTrue(utils.separates({2}, {3, 4}, {0, 1}, A))
        self.assertFalse(utils.separates(set(), {0, 1}, {3, 4}, A))
        self.assertTrue(utils.separates(set(), {3, 4}, {0, 1}, A))
        self.assertTrue(utils.separates(set(), {3}, {0}, A))
        self.assertFalse(utils.separates(set(), {0}, {3}, A))
        self.assertTrue(utils.separates({2}, {0}, {3}, A))

    def test_separates_2(self):
        A = np.array([[0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 1],
                      [1, 1, 0, 1, 1],
                      [0, 0, 1, 0, 1],
                      [0, 0, 1, 1, 0]])
        self.assertFalse(utils.separates({2}, {0, 1}, {3, 4}, A))
        self.assertTrue(utils.separates({2}, {0}, {3, 4}, A))
        self.assertTrue(utils.separates({2, 1}, {0}, {3, 4}, A))
        self.assertTrue(utils.separates({2, 4}, {1}, {3}, A))

    def test_chain_component_1(self):
        G = np.array([[0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 1],
                      [0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0]])
        chain_components = [(0, {0}),
                            (1, {1}),
                            (2, {2, 3, 4}),
                            (3, {2, 3, 4}),
                            (4, {2, 3, 4})]
        for (i, truth) in chain_components:
            self.assertEqual(truth, utils.chain_component(i, G))

    def test_chain_component_2(self):
        G = np.array([[0, 1, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0],
                      [1, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 1, 1, 1],
                      [0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 1, 1, 0]])
        chain_components = [(0, {0, 1, 3}),
                            (1, {0, 1, 3}),
                            (2, {2}),
                            (3, {0, 1, 3}),
                            (4, {4, 5, 7}),
                            (5, {4, 5, 7}),
                            (6, {6}),
                            (7, {4, 5, 7})]
        for (i, truth) in chain_components:
            self.assertEqual(truth, utils.chain_component(i, G))

    def test_chain_component_3(self):
        # Check the following property in random graphs:
        # if i is in the chain component of j, then the chain
        # component of i is equal to that of j
        G = 500
        p = 20
        for i in range(G):
            A = sempler.generators.dag_avg_deg(p, 3, 1, 1)
            cpdag = utils.dag_to_cpdag(A)
            for j in range(p):
                chain_component = utils.chain_component(j, cpdag)
                for h in chain_component:
                    self.assertEqual(chain_component, utils.chain_component(h, cpdag))

    def test_chain_component_4(self):
        # Check that in a directed graph, the chain component of each
        # node is itself
        G = 500
        p = 20
        for i in range(G):
            A = sempler.generators.dag_avg_deg(p, 3, 1, 1)
            for j in range(p):
                self.assertEqual({j}, utils.chain_component(j, A))

    def test_induced_graph_1(self):
        # Test that
        # 1. The subgraph induced by an empty set of nodes should always
        # be a disconnected graph
        # 2. When the set is not empty, the returned "subgraph" is correct
        G = 500
        p = 20
        for i in range(G):
            A = sempler.generators.dag_avg_deg(p, 3, 1, 1)
            G = utils.dag_to_cpdag(A)
            # Test 1
            self.assertTrue((np.zeros_like(G) == utils.induced_subgraph(set(), G)).all())
            # Test 2
            for _ in range(10):
                S = set(np.random.choice(range(p), size=np.random.randint(0, p)))
                truth = G.copy()
                Sc = set(range(p)) - S
                truth[list(Sc), :] = 0
                truth[:, list(Sc)] = 0
                self.assertTrue((truth == utils.induced_subgraph(S, G)).all())

    def test_induced_graph_2(self):
        G = np.array([[0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 1],
                      [0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0]])
        # Test 0: Sets which return a graph with no edges
        for S in [set(), {0}, {1}, {2}, {3}, {4}, {0, 1}, {0, 3}, {0, 4}, {3, 4}, {1, 3}, {1, 4}]:
            self.assertTrue((np.zeros_like(G) == utils.induced_subgraph(S, G)).all())
        # Test 1
        S = {0, 1, 2}
        truth = np.array([[0, 0, 1, 0, 0],
                          [0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0]])
        self.assertTrue((truth == utils.induced_subgraph(S, G)).all())
        # Test 2
        S = {0, 2, 3}
        truth = np.array([[0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0],
                          [0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0]])
        self.assertTrue((truth == utils.induced_subgraph(S, G)).all())
        # Test 3
        S = {1, 2, 3, 4}
        truth = np.array([[0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0],
                          [0, 0, 0, 1, 1],
                          [0, 0, 1, 0, 0],
                          [0, 0, 1, 0, 0]])
        self.assertTrue((truth == utils.induced_subgraph(S, G)).all())

    # Tests for other auxiliary functions

    def test_sort_1(self):
        # Test that without order, behaviour is identical as python's
        # sorted
        for _ in range(10):
            L = np.random.permutation(range(10))
            self.assertEqual(utils.sort(L), sorted(L))
        # Test that (1) when the order is a permutation of the list,
        # the result is the order itself, and (2) when applied to an
        # empty list, sorted is the identity
        for _ in range(10):
            L = np.random.permutation(100)
            order = list(np.random.permutation(100))
            self.assertEqual(order, utils.sort(L, order))
            self.assertEqual([], utils.sort([], order))

    def test_sort_2(self):
        # Check that ordering works as expected when order is specified
        # Test 1
        L = [3, 6, 1, 2, 0]
        order = [0, 2, 4, 6, 1, 3, 5]
        self.assertEqual([0, 2, 6, 1, 3], utils.sort(L, order))
        # Test 2, with duplicated elements
        L = [0, 1, 0, 6, 1, 9, 9, 4]
        order = [0, 2, 4, 6, 8, 1, 3, 5, 7, 9]
        self.assertEqual([0, 0, 4, 6, 1, 1, 9, 9], utils.sort(L, order))
        # Test 3, with different order
        L = [8, 8, 1, 9, 7, 1, 3, 0, 2, 4, 0, 1, 3, 7, 5]
        order = [7, 3, 6, 5, 0, 4, 1, 2, 6, 8, 9]
        truth = [7, 7, 3, 3, 5, 0, 0, 4, 1, 1, 1, 2, 8, 8, 9]
        self.assertEqual(truth, utils.sort(L, order))

    def test_all_dags_0(self):
        # PBT: Running all_dags on a DAG should return the DAG itself
        cases = 1
        p = 10
        k = 3
        for i in range(cases):
            A = sempler.generators.dag_avg_deg(p, k, 1, 1)
            dags = utils.all_dags(A)
            self.assertEqual(len(dags), 1)
            self.assertTrue((dags[0] == A).all())
            self.assertIsInstance(dags, np.ndarray)

    def test_all_dags_limit(self):
        A = np.array([[0, 1, 0],
                      [0, 0, 1],
                      [0, 0, 0]])
        cpdag = utils.dag_to_cpdag(A)
        try:
            utils.all_dags(cpdag, max_combinations=1)
            self.fail()
        except ValueError as e:
            print("OK :", e)
            

    def test_all_dags_1(self):
        # PBT:
        # For a PDAG P
        # 0. Return type should be an np.array()
        # 1. There should be leq than 2 ^ u DAGs(where u is the
        #                                        number of undirected edges in P)
        # 2. No. of DAGs > 0 <= > PDAG admits a consistent extension
        # 3. All DAGs should be DAGs and consistent extensions
        # 4. All DAGs should be unique
        # 5. All DAGs should have the same number of edges
        cases = 1
        p = 10
        k = 3
        for i in range(cases):
            A = sempler.generators.dag_avg_deg(p, k, 1, 1)
            # Unorient some edges at random
            random_pdag = A.copy()
            mask = np.random.uniform(size=A.shape) > 0.5
            random_pdag = A + A.T * mask
            for pdag in [utils.dag_to_cpdag(A), random_pdag]:
                u = np.sum(utils.only_undirected(pdag)) / 2
                dags = utils.all_dags(pdag)
                # Check type
                self.assertIsInstance(dags, np.ndarray)
                # Check size
                self.assertLessEqual(len(dags), 2**u)
                try:
                    utils.pdag_to_dag(pdag)
                    self.assertGreater(len(dags), 0)
                    # Check uniqueness
                    self.assertEqual(len(dags), len(np.unique(dags, axis=0)))
                    # Check consistency
                    for dag in dags:
                        self.assertTrue(utils.is_dag(dag))
                        self.assertTrue(utils.is_consistent_extension(dag, pdag))
                    # Check number of edges
                    no_edges = np.sum(dags, axis=(1, 2))
                    expected_no_edges = utils.only_directed(
                        pdag).sum() + utils.only_undirected(pdag).sum() / 2
                    self.assertTrue((no_edges == expected_no_edges).all())
                except ValueError:
                    self.assertEqual(len(dags), 0)
        print("Checked enumeration size and consistency for %d PDAGS" % ((i + 1) * 2))

    # Causaldag implementation is wrong(see notebook example), decided
    # not to test against it

    # def test_all_dags_vs_causaldag(self):
    #     cases = 50
    #     p = 10
    #     k = 3
    #     for i in range(cases):
    #         A = sempler.generators.dag_avg_deg(p, k, 1, 1)
    #         # Unorient some edges at random
    #         random_pdag = A.copy()
    #         mask = np.random.uniform(size=A.shape) > 0.5
    #         random_pdag = A + A.T * mask
    #         for pdag in [utils.dag_to_cpdag(A), random_pdag]:
    #             print(i)
    #             PDAG = cd.PDAG.from_amat(pdag)
    #             dags_cd = PDAG.all_dags()
    #             dags = utils.all_dags(pdag)
    #             if len(dags) != len(dags_cd):
    #                 print(pdag, dags, dags_cd)
    #             self.assertEqual(len(dags), len(dags_cd))
    #     print("Checked vs causaldag implementation for %d PDAGS" % ((i + 1) * 2))

    def test_all_dags_2(self):
        # X0 - X1 - X2
        pdag = np.array([[0, 1, 0],
                         [1, 0, 1],
                         [0, 1, 0]])
        true_dags = [
            # X0 <- X1 -> X2
            np.array([[0, 0, 0],
                      [1, 0, 1],
                      [0, 0, 0]]),
            # X0 -> X1 -> X2
            np.array([[0, 1, 0],
                      [0, 0, 1],
                      [0, 0, 0]]),
            # X0 <- X1 <- X2
            np.array([[0, 0, 0],
                      [1, 0, 0],
                      [0, 1, 0]])
        ]
        dags = utils.all_dags(pdag)
        self.assertEqual(len(dags), len(true_dags))
        self.assertEqual(len(np.unique(dags, axis=0)), len(true_dags))
        for dag in dags:
            self.assertTrue(utils.member(true_dags, dag) is not None)

    def test_all_dags_3(self):
        # X0 -> X1 <- X2
        pdag = np.array([[0, 1, 0],
                         [0, 0, 0],
                         [0, 1, 0]])
        dags = utils.all_dags(pdag)
        self.assertEqual(len(dags), 1)
        self.assertTrue((dags[0] == pdag).all())

    def test_all_dags_4(self):
        # X0 -> X1 - X2
        pdag = np.array([[0, 1, 0],
                         [0, 0, 1],
                         [0, 1, 0]])
        true_dag = np.array([[0, 1, 0],
                             [0, 0, 1],
                             [0, 0, 0]])
        dags = utils.all_dags(pdag)
        self.assertEqual(len(dags), 1)
        self.assertTrue((dags[0] == true_dag).all())

    def test_all_dags_5(self):
        pdag = np.array([[0, 0, 1, 0, 0],
                         [0, 0, 1, 0, 0],
                         [0, 1, 0, 1, 1],
                         [0, 0, 1, 0, 1],
                         [0, 0, 1, 1, 0]])
        true_dags = [np.array([[0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 0],
                               [0, 1, 0, 1, 1],
                               [0, 0, 0, 0, 1],
                               [0, 0, 0, 0, 0]]),
                     np.array([[0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 0],
                               [0, 1, 0, 1, 1],
                               [0, 0, 0, 0, 0],
                               [0, 0, 0, 1, 0]])]
        dags = utils.all_dags(pdag)
        self.assertEqual(len(dags), len(true_dags))
        self.assertEqual(len(np.unique(dags, axis=0)), len(true_dags))
        for dag in dags:
            self.assertTrue(utils.member(true_dags, dag) is not None)

    def test_is_complete_1(self):
        p = 5
        G = np.ones((p, p)) - np.eye(p)
        self.assertTrue(utils.is_complete(G))
        G = np.triu(np.ones((p, p)), k=1)
        self.assertTrue(utils.is_complete(G))
        for i in range(10):
            G = sempler.generators.dag_full(p)
            self.assertTrue(utils.is_complete(G))

    def test_degrees_1(self):
        """Test that for random generated DAGs, the returned degrees satisty
        the degree formula, i.e. sum-of-degrees = 2 * #edges"""
        G = 50
        p = 10
        for i in range(G):
            A = sempler.generators.dag_avg_deg(p, 2.5, random_state=i)
            no_edges = A.sum()
            degrees = utils.degrees(A)
            self.assertEqual(degrees.sum(), 2 * no_edges)
            # Check that the degrees are the same for the CPDAG and skeleton of A
            cpdag = utils.dag_to_cpdag(A)
            skeleton = utils.dag_to_cpdag(A)
            self.assertTrue((degrees == utils.degrees(cpdag)).all())
            self.assertTrue((degrees == utils.degrees(skeleton)).all())
            # Check that result matches when we construct the degree
            # through other functions
            alt_degrees = []
            for j in range(p):
                alt_degrees.append(len(utils.pa(j, A)) + len(utils.ch(j, A)))
            self.assertTrue((alt_degrees == degrees).all())

    def test_degrees_2(self):
        A = np.array([[0, 1, 0, 0],
                      [0, 0, 1, 1],
                      [0, 0, 0, 1],
                      [0, 0, 0, 0]])
        degrees = utils.degrees(A)
        true_degrees = np.array([1, 3, 2, 2])
        self.assertTrue((true_degrees == degrees).all())
        # Check that same holds for skeleton and cpdag of the graph
        cpdag = utils.dag_to_cpdag(A)
        skeleton = utils.dag_to_cpdag(A)
        self.assertTrue((degrees == utils.degrees(cpdag)).all())
        self.assertTrue((degrees == utils.degrees(skeleton)).all())

    def test_moral_graph_1(self):
        """Test that for random generated DAGs, the moral graph of the DAG and
        the CPDAG are the same."""
        G = 50
        p = 10
        for i in range(G):
            A = sempler.generators.dag_avg_deg(p, 2.5, random_state=i)
            cpdag = utils.dag_to_cpdag(A)
            self.assertTrue(
                (utils.moral_graph(A) == utils.moral_graph(cpdag)).all())

    def test_moral_graph_2(self):
        """Test moralization of basic X -> Y <- Z"""
        A = np.array([[0, 1, 0],
                      [0, 0, 0],
                      [0, 1, 0]])
        true_moral = np.array([[0, 1, 1],
                               [1, 0, 1],
                               [1, 1, 0]])
        self.assertTrue((utils.moral_graph(A) == true_moral).all())

    def test_moral_graph_3(self):
        """Test moralization of basic X -> Y <- Z"""
        A = np.array([[0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 1],
                      [0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0]])
        true_moral = np.array([[0, 1, 1, 0, 0],
                               [1, 0, 1, 0, 0],
                               [1, 1, 0, 1, 1],
                               [0, 0, 1, 0, 1],
                               [0, 0, 1, 1, 0]])
        self.assertTrue((utils.moral_graph(A) == true_moral).all())

    def test_split_data_1(self):
        p = 10
        n_obs = [10, 20, 30, 40, 50]
        X = [np.ones((n_obs[i], p)) * i for i in range(len(n_obs))]
        # Check that for n_folds everything remains the same
        for i, env in enumerate(utils.split_data(X, ratios=[1])[0]):
            self.assertTrue((env == X[i]).all())
            # Check for other fold sizes
        for n_folds in [2, 5, 10]:
            ratios = [1 / n_folds] * n_folds
            folds = utils.split_data(X, ratios)
            # print("\n----------------------")
            # print("n_folds =", n_folds)
            # print(folds)
            for fold in folds:
                # Check dimensions
                self.assertEqual(len(fold), len(n_obs))
                for i, env_sample in enumerate(fold):
                    # Check dimensions
                    self.assertEqual(len(env_sample), n_obs[i] / n_folds)
                    # Check that only data from the correct environment ended here
                    self.assertTrue((env_sample == i).all())

    def test_split_data_2(self):
        """Check that when splitting the data, each fold actually ends up with
        different observations"""
        W = sempler.generators.dag_avg_deg(4, 2.5, 0.5, 1)
        scm = sempler.LGANM(W, (0, 0), (3, 4))
        data = [scm.sample(20), scm.sample(20, noise_interventions={0: (1, 1)})]
        [fold_1, fold_2] = utils.split_data(data, ratios=[0.5, 0.5])
        flattened_1 = np.array(fold_1).flatten()
        flattened_2 = np.array(fold_2).flatten()
        self.assertFalse((flattened_1 == flattened_2).any())

    def test_split_data_3(self):
        """Check that when splitting the data, each fold actually ends up with
        different observations"""
        W = sempler.generators.dag_avg_deg(4, 2.5, 0.5, 1)
        scm = sempler.LGANM(W, (0, 0), (3, 4))
        n = 99
        data = [scm.sample(99), scm.sample(99, noise_interventions={0: (1, 1)})]
        # Check that all datapoints are contained in one of the folds
        [fold_1, fold_2] = utils.split_data(data, ratios=[0.5, 0.5])
        for (s1, s2) in zip(fold_1, fold_2):
            self.assertEqual(len(s1) + len(s2), n)
            # Check that the observations in each fold are disjoint
        flattened_1 = set(np.array(fold_1).flatten())
        flattened_2 = set(np.array(fold_2).flatten())
        self.assertEqual(set(), flattened_1 & flattened_2)
        # Check that the correct number of observations are assigned
        # to each fold
        n_obs = [100, 200]
        data = [scm.sample(n_obs[0]), scm.sample(n_obs[1], noise_interventions={0: (1, 1)})]
        ratios = [.1, .2, .3, .4]
        folds = utils.split_data(data, ratios=ratios)
        for i, fold in enumerate(folds):
            for j, sample in enumerate(fold):
                self.assertEqual(len(sample), ratios[i] * n_obs[j])

        def test_directed_edges(self):
            p = 15
            k = 2.7
            for i in range(100):
                A = sempler.generators.dag_avg_deg(p, k)
                edges = utils.directed_edges(A)
                # Check number is correct
                self.assertEqual(len(edges), np.sum(A))
                # Check that original graph can be reconstructed
                new_A = np.zeros_like(A)
                for (i, j) in edges:
                    new_A[i, j] = 1
                    self.assertTrue((new_A == A).all())

        def test_supergraph_1(self):
            A = np.array([[0, 1, 0],
                          [0, 0, 0],
                          [0, 0, 0]])
            B1, B2, B3 = A.copy(), A.copy(), A.copy()
            # Same graph should return true
            # Add edge
            B1[0, 2] = 1
            self.assertTrue(utils.is_supergraph(B1, A))
            self.assertFalse(utils.is_supergraph(A, B1))
            # Add edge
            B2[1, 2] = 1
            self.assertTrue(utils.is_supergraph(B2, A))
            self.assertFalse(utils.is_supergraph(A, B2))
            # Flip
            self.assertFalse(utils.is_supergraph(A, A.T))
            B3[0, 2] = 1
            B3[1, 2] = 1
            self.assertTrue(utils.is_supergraph(B3, A))
            self.assertTrue(utils.is_supergraph(B3, B1))
            self.assertTrue(utils.is_supergraph(B3, B2))
            self.assertFalse(utils.is_supergraph(A, B3))
            self.assertFalse(utils.is_supergraph(B1, B3))
            self.assertFalse(utils.is_supergraph(B2, B3))

        def test_supergraph_2(self):
            p = 15
            k = 2.7
            for i in range(10):
                A = sempler.generators.dag_avg_deg(p, k)
                self.assertTrue(utils.is_supergraph(A, A))

        def test_remove_edges(self):
            p = 15
            k = 2.7
            rem = 10
            for i in range(50):
                A = sempler.generators.dag_avg_deg(p, k)
                subgraph = utils.remove_edges(A, rem, random_state=i)
                self.assertEqual(A.sum() - rem, subgraph.sum())
                self.assertTrue(utils.is_supergraph(A, subgraph))
                # Check that removing as many edges as there are returns
                # the empty graph
                subgraph = utils.remove_edges(A, int(A.sum()))
                self.assertTrue((np.zeros_like(A) == subgraph).all())
                # Check that error is returned when attempting to remove
                # more edges than possible
                try:
                    utils.remove_edges(A, A.sum() + 1)
                    self.fail()
                except ValueError:
                    pass

        def test_add_edges(self):
            p = 15
            k = 2.7
            add = 10
            for i in range(50):
                A = sempler.generators.dag_avg_deg(p, k)
                supergraph = utils.add_edges(A, add, random_state=i)
                self.assertEqual(A.sum() + add, supergraph.sum())
                self.assertTrue(utils.is_supergraph(supergraph, A))
                # Check that adding all possible edges leads to the fully
                # connected graph
                can_add = int(p * (p - 1) / 2 - A.sum())
                supergraph = utils.add_edges(A, can_add)
                self.assertTrue((np.ones_like(A) - np.eye(p) ==
                                 utils.skeleton(supergraph)).all())
                # Check that error is returned when attempting to add more
                # edges than is possible
                try:
                    utils.add_edges(A, can_add + 1)
                    self.fail()
                except ValueError:
                    pass

    def test_chain_graph(self):
        for p in range(1, 20):
            A = utils.chain_graph(p)
            self.assertTrue(utils.is_chain_graph(A))
            self.assertEqual(p - 1, np.sum(A))

    def test_chain_graph_MEC(self):
        # For smaller chain graphs computing the MEC from the
        # CPDAG is still fast; check the outputs match
        for p in range(1, 12):
            # start = time.time()
            MEC_a = utils.chain_graph_MEC(p)
            A = utils.chain_graph(p)
            cpdag = utils.dag_to_cpdag(A)
            MEC_b = utils.all_dags(cpdag)
            MEC_c = utils.mec(A)
            # print("p=%d - %0.2f seconds" % (p, time.time() - start))
            # Check both sets are the same
            self.assertTrue(same_elements(MEC_a, MEC_b))
            self.assertTrue(same_elements(MEC_a, MEC_c))

    def test_chain_graph_IMEC(self):
        # For smaller chain graphs computing the MEC from the
        # CPDAG is still fast; check the outputs match
        rng = np.random.default_rng(42)
        for p in range(1, 12):
            for k in range(p + 1):
                A = utils.chain_graph(p)
                I = set(rng.choice(range(p), size=k, replace=False))
                # start = time.time()
                IMEC_a = utils.chain_graph_IMEC(A, I)
                icpdag = utils.dag_to_icpdag(A, I)
                IMEC_b = utils.all_dags(icpdag)
                IMEC_c = utils.imec(A, I)
                # print("p=%d |I|=%d - %0.2f seconds" % (p, len(I), time.time() - start))
                # Check both sets are the same
                self.assertTrue(same_elements(IMEC_a, IMEC_b))
                self.assertTrue(same_elements(IMEC_a, IMEC_c))

    def test_imec(self):
        G = 50
        p = 12
        for i in range(G):
            A = sempler.generators.dag_avg_deg(p, 3, 1, 1)
            # Check that MEC and I-MEC are the same for I = set()
            mec = utils.mec(A)
            imec = utils.imec(A, set())
            self.assertTrue(same_elements(mec, imec))
            # Check that I-MEC is a singleton when I = [p]
            I = set(range(p))
            imec = utils.imec(A, I)
            self.assertEqual(1, len(imec))
            self.assertTrue((A == imec[0]).all())

    def test_ancestors_1(self):
        # Test that
        #   - a node is not contained in its ancestors
        #   - the nodes that come after a node in the ordering
        #     do not form part of its ancestors
        #   - the parents of a node are contained in its ancestors
        #   - its children are not
        #   - a node is a descendant of its ancestors
        G = 50
        p = 20
        for i in range(G):
            A = sempler.generators.dag_avg_deg(p, 3, 1, 1)
            ordering = utils.topological_ordering(A)
            for i, node in enumerate(ordering):
                ancestors = utils.an(node, A)
                self.assertNotIn(node, ancestors)
                self.assertEqual(set(), ancestors & set(ordering[i:]))
                self.assertTrue(utils.pa(node, A) <= ancestors)
                self.assertEqual(set(), utils.ch(node, A) & ancestors)
                for j in ancestors:
                    self.assertTrue(node in utils.desc(j, A))

    def test_ancestors_2(self):
        A = np.array([[0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 1, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 0]])
        ancestors = [set(),
                     set(),
                     {0,1},
                     {0,1,2},
                     set(),
                     {4},
                     {0,1,2,3,4,5}]
        for i, truth in enumerate(ancestors):
            self.assertEqual(truth, utils.an(i, A))

    def test_descendants_1(self):
        # Test that
        #   - a node is always contained in its descendants
        #   - the children of a node are contained in its descendants
        #   - its parents are not
        #   - a node is an ancestor of its descendants
        G = 50
        p = 20
        for i in range(G):
            A = sempler.generators.dag_avg_deg(p, 3, 1, 1)
            ordering = utils.topological_ordering(A)
            for i, node in enumerate(ordering):
                descendants = utils.desc(node, A)
                self.assertIn(node, descendants)
                self.assertTrue(utils.ch(node, A) <= descendants)
                self.assertEqual(set(), utils.pa(node, A) & descendants)
                for j in descendants:
                    if node != j:
                        self.assertTrue(node in utils.an(j, A))

    def test_descendants_2(self):
        A = np.array([[0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 1, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 0]])
        descendants = [{0,2,3,6},
                       {1,2,3,6},
                       {2,3,6},
                       {3,6},
                       {4,5,6},
                       {5,6},
                       {6}]
        for i, truth in enumerate(descendants):
            self.assertEqual(truth, utils.desc(i, A))

def same_elements(A, B):
    """Check that two arrays have the same elements (in same or different
    order) along the zero axis."""
    if len(np.unique(A, axis=0)) != len(np.unique(B, axis=0)):
        return False
    else:
        for a in A:
            if utils.member(B, a) is None:
                return False
        return True
