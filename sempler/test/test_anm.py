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
# Unit testing for module sampling.py

import unittest
import numpy as np

import sempler
import sempler.noise
import sempler.utils
import sempler.generators

# Tests that the interface behaves as expected
class api_Tests(unittest.TestCase):
        
    def test_constructor(self):
        functions = [None, lambda x: x, lambda x: x]
        noises = [sempler.noise.normal()] * 3
        # Should raise exception for graph with cycle
        A = np.array([[0,1,0],
                      [0,0,1],
                      [1,0,0]])
        try:
            sempler.ANM(A, functions, noises)
            self.fail("Should have raised an exception")
        except ValueError as e:
            print("OK;",e)
        # Should raise exception for PDAG
        A = np.array([[0,1,0],
                      [1,0,1],
                      [0,0,0]])
        try:
            sempler.ANM(A, functions, noises)
            self.fail("Should have raised an exception")
        except ValueError as e:
            print("OK;",e)
        # Should work
        A = np.array([[0,1,0],
                      [0,0,1],
                      [0,0,0]])
        sempler.ANM(A, [None, None, None], noises)
        # Test attributes
        anm = sempler.ANM(A, functions, noises)
        anm.A
        anm.p
        anm.assignments
        anm.noise_distributions
        # Test sampling
        # Should fail if n not specified
        try:
            anm.sample()
            self.fail("Should have raised an exception")
        except Exception as e:
            print("OK;",e)
        anm.sample(0)
        anm.sample(10)
        # Test empty interventions
        anm.sample(10, do_interventions = {})
        anm.sample(10, noise_interventions = {})
        anm.sample(10, shift_interventions = {})
        # Test non-empty interventions
        anm.sample(10, do_interventions = {0: sempler.noise.uniform()})
        anm.sample(10, noise_interventions = {0: sempler.noise.uniform()})
        anm.sample(10, shift_interventions = {0: sempler.noise.uniform()})

class sampling_Tests(unittest.TestCase):

    def test_gaussian_sampling(self):
        # Test 100 interventions
        K = 50
        W = np.array([[0, 0, 0, 0.2, 0],
                      [0, 0, 0.4, 0, 0],
                      [0, 0, 0, 0.3, 0],
                      [0, 0, 0, 0, 0.5],
                      [0, 0, 0, 0, 0  ]])
        lganm = sempler.LGANM(W, (1,2), (1,2))
        noise_distributions = [sempler.noise.normal(m,v) for (m,v) in zip(lganm.means, lganm.variances)]
        assignments = [None, None, lambda x: .4 * x, lambda x: .2*x[:,0] + .3 * x[:,1], lambda x: .5*x]
        anm = sempler.ANM(W, assignments, noise_distributions)
        interventions = sempler.generators.intervention_targets(lganm.p, K, (0,3))
        for targets in interventions:
            print(targets)
            means, variances = np.random.uniform(0,5,len(targets)), np.random.uniform(2,3,len(targets))
            interventions_lganm = dict((t, (m, v)) for (t,m,v) in zip(targets, means, variances))
            interventions_anm = dict((t, sempler.noise.normal(m,v)) for (t,m,v) in zip(targets, means, variances))
            # Sample each SCMs
            # TODO: Combine different interventions in one
            n = round(1e6)
            if len(targets) <= 1:
                samples_anm = anm.sample(n, do_interventions = interventions_anm)
                samples_lganm = lganm.sample(n, do_interventions = interventions_lganm)
            elif len(targets) == 2:
                samples_anm = anm.sample(n, shift_interventions = interventions_anm)
                samples_lganm = lganm.sample(n, shift_interventions = interventions_lganm)
            elif len(targets) == 3:
                samples_anm = anm.sample(n, noise_interventions = interventions_anm)
                samples_lganm = lganm.sample(n, noise_interventions = interventions_lganm)
            # Check that the distribution is the same
            self.assertTrue(sempler.utils.same_normal(samples_anm, samples_lganm, debug=False))
