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



        
