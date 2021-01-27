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
Main module containing the SCM classes (sempler.ANM and
sempler.LGANM), the sempler.NormalDistribution class and additional
functions to generate random graphs.
"""

import numpy as np
from copy import deepcopy
import sempler.utils as utils
import sempler.functions as functions

#---------------------------------------------------------------------
# ANM class
class ANM:
    """
    Class to represent a general (acyclic) additive noise model.
    
    Parameters
    ----------
    A : np.array
        The p x p adjacency matrix specifying the functional
        dependencies-
    p : int
        the number of variables.
    assignments : list of function
        the assignment functions of the variables.
    noise_distributions : list of function
        a list of functions representing the noise term distribution
        of each variable.

    """
    
    def __init__(self, A, assignments, noise_distributions):
        """Creates an instance of the ANM class, representing an SCM over p
        observed variables.
        
        Parameters
        ----------
        A : np.array
            The p x p adjacency matrix specifying the functional
            dependencies, where A[i,j] != 0 if i appears in the
            assignment of j (i.e. i -> j).
        functions : list of function or NoneType
            a list of p functions representing the functional
            assignments of each variable. Each function must take as
            many arguments as specified by the adjacency matrix A, or
            be None if the variable has no parents.
        noise_distributions : list of function
            a list of p functions that generate samples of each
            variable's noise distribution (see sempler.noise for
            details).

        """
        try:
            self.ordering = utils.topological_ordering(A)
        except 
        self.p = len(A)
        self.A = deepcopy(A)
        self.assignments = [functions.null if fun is None else deepcopy(fun) for fun in assignments]
        self.noise_distributions = deepcopy(noise_distributions)
            
    def sample(self, n, do_interventions = {}, shift_interventions = {}, noise_interventions = {}, random_state = None):
        """
        Generates n observations from the ANM, under the given do, shift or
        noise interventions. If none are given, sample from the observational
        distribution.
        
        Parameters
        ----------
        n : int
            the size of the sample (i.e. number of observations).
        do_interventions : dict, optional
            a dictionary where keys correspond to the intervened
            variables, and the values are the distribution functions
            (see sempler.noise) to generate samples for each
            intervened variable.
        shift_interventions : dict, optional
            a dictionary where keys correspond to the intervened
            variables, and the values are the distribution functions
            (see sempler.noise) to generate noise samples which are
            added to the intervened variables.
        noise_interventions : dict, optional
            a dictionary where keys correspond to the intervened
            variables, and the values are the distribution functions
            (see sempler.noise) of the new noise.
        random_state: int, optional
            set the random state, for reproducibility.

        Returns
        -------
        X : np.array
            an array containing the sample, where each column
            corresponds to a variable.
        
        """
        # Set random state (if requested)
        np.random.seed(random_state) if random_state is not None else None
        # Sample according to a topological ordering of the connectivity matrix
        X = np.zeros((n, self.p))
        for i in self.ordering:
            # If i is do intervened, sample from the corresponding
            # interventional distribution
            if i in do_interventions:
                X[:,i] = do_interventions[i](n)
            # Otherwise maintain dependence on parents
            else:
                assignment = np.transpose(self.assignments[i](X[:, self.A[:,i] == 1]))
                # Shift-intervention: add noise from given distribution
                if i in shift_interventions:
                    noise = self.noise_distributions[i](n) + shift_interventions[i](n)
                # Noise-intervention: sample noise from given distribution
                elif i in noise_interventions:
                    noise = noise_interventions[i](n)
                # No intervention: sample noise from original distribution
                else:
                    noise = self.noise_distributions[i](n)
                X[:,i] = assignment + noise
        return X
