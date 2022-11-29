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

import sempler.generators
import sempler.semi

# Tests for the DAG generation


def covs(samples):
    return np.array([np.cov(sample, rowvar=False) for sample in samples])


def means(samples):
    return np.array([np.mean(sample, axis=0) for sample in samples])


class GaussianDistributionTests(unittest.TestCase):
    def test_distances(self):
        p = 5
        i = 0
        W = sempler.generators.dag_avg_deg(p, 2.7, 0.5, 1, random_state=i)
        true_A = (W != 0).astype(int)
        scm = sempler.LGANM(W, (1, 2), (1, 2))
        n = 1000
        samples = [scm.sample(n)]
        samples += [scm.sample(n, noise_interventions={0: (10, 4)})]
        sample_covs = covs(samples)
        sample_means = means(samples)
        # Fit DRF SCM with true graph
        drf_scm = sempler.semi.DRFNet(true_A, samples, verbose=True)
        semi_sample = drf_scm.sample()
        cov_diffs = sample_covs - covs(semi_sample)
        mean_diffs = sample_means - means(semi_sample)
        print(cov_diffs / sample_covs)
        print(mean_diffs / sample_means)
