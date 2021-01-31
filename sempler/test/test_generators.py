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

#---------------------------------------------------------------------
# Unit testing for module utils

import unittest
import numpy as np

import sempler.generators
import sempler.utils as utils

# Tests for the DAG generation
class GeneratorTests(unittest.TestCase):
    def test_avg_degree(self):
        p = 1000
        for k in range(1,5):
            W = sempler.generators.dag_avg_deg(p, k, 1, 2)
            av_deg = np.sum(W > 0) * 2 / p
            self.assertEqual(len(W), p)
            self.assertTrue(av_deg - k < 0.5)
            self.assertTrue(utils.is_dag(W))

    def test_disconnected_graph(self):
        W = sempler.generators.dag_avg_deg(10, 0, 1, 1)
        self.assertEqual(np.sum(W), 0)

    def test_full_dag(self):
        for p in range(10):
            W = sempler.generators.dag_full(p)
            self.assertEqual(p*(p-1)/2, W.sum())

    def test_intervention_targets(self):
        possible_targets = set(range(10))
        # Test random-sized interventions
        interventions = sempler.generators.intervention_targets(10, 100, (0,3))
        for intervention in interventions:
            self.assertLessEqual(len(intervention), 3)
            self.assertGreaterEqual(len(intervention), 0)
            self.assertEqual(len(intervention), len(set(intervention) & possible_targets))
        # Test empty-sized interventions
        interventions = sempler.generators.intervention_targets(10, 100, (0,0))
        for intervention in interventions:
            self.assertEqual(len(intervention), 0)
        # Test single-target interventions
        interventions = sempler.generators.intervention_targets(10, 100, 1)
        for intervention in interventions:
            self.assertEqual(len(intervention), 1)
            self.assertEqual(len(intervention), len(set(intervention) & possible_targets))
