# Copyright 2020 Juan Luis Gamella Martin

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

import numpy as np
from copy import deepcopy
from sempler import utils, functions
from sempler.utils import matrix_block

#---------------------------------------------------------------------
# ANM class
class ANM:
    """
    Represents an general (acyclic) additive noise model.
    """
    def __init__(self, A, assignments, noise_distributions):
        """Parameters:
          - A: pxp onnectivity matrix (A_ij = 1 if i appears in the assignment of j)
          - functions: list of p functions representing
            the assignments of each variable
          - noise_distributions: list of p functions that generate
            samples of each variable's noise distribution
        """
        self.ordering = utils.topological_ordering(A)
        self.p = len(A)
        self.A = deepcopy(A)
        self.assignments = [functions.null if fun is None else deepcopy(fun) for fun in assignments]
        self.noise_distributions = deepcopy(noise_distributions)
            
    def sample(self, n, do_interventions = {}, shift_interventions = {}, random_state = None):
        """Generates n samples from the ANM, under the given do or shift interventions.
           Parameters:
             - n: the number of samples
             - do_interventions: a dictionary containing the
               distribution functions (see sempler.functions) from
               which to generate samples for each intervened variable
             - shift_interventions: a dictionary containing the
               distribution functions (see sempler.functions) from
               which to generate the noise which is added to each
               intervened variable
             - random_state: (int) seed for the random state generator
        """
        # Set random state (if requested)
        np.random.seed(random_state) if random_state is not None else None
        # Sample according to a topological ordering of the connectivity matrix
        X = np.zeros((n, self.p))
        for i in self.ordering:
            if i in do_interventions:
                X[:,i] = do_interventions[i](n)
            else:
                assignment = np.transpose(self.assignments[i](X[:, self.A[:,i] == 1]))
                noise = self.noise_distributions[i](n)
                shift = shift_interventions[i](n) if i in shift_interventions else 0
                X[:,i] = assignment + noise + shift
        return X


#---------------------------------------------------------------------
# LGANM class
class LGANM:
    """Represents a linear model with Gaussian additive noise
    (i.e. Gaussian Bayesian Network).
    """
    
    def __init__(self, W, variances, means = None):
        """
        Parameters
        - W (np.array): weighted connectivity matrix representing a DAG
        - variances (np.array or tuple): either a vector of variances or a tuple
          indicating range for uniform sampling
        - means (np.array, tuple or None): either a vector of means, a tuple
          indicating the range for uniform sampling or None (zero
          means)
        """
        self.W = W.copy()
        self.p = len(W)

        # Set variances
        if isinstance(variances, tuple):
            self.variances = np.random.uniform(variances[0], variances[1], size=self.p)
        else:
            self.variances = variances.copy()
            
        # Set means
        if means is None:
            self.means = np.zeros(self.p)
        elif isinstance(means, tuple):
            self.means = np.random.uniform(means[0], means[1], size=self.p)
        else:
            self.means = means.copy()
    
    def sample(self, n=100, population=False, do_interventions=None, shift_interventions=None):
        """
        If population is set to False:
          - Generate n samples from a given Linear Gaussian SCM, under the given
            interventions (by default samples observational data)
        if set to True:
          - Return the "symbolic" joint distribution under the given
            interventions (see class NormalDistribution)
        """
        # Must copy as they can be changed by intervention, but we
        # still want to keep the observational SEM
        W = self.W.copy()
        variances = self.variances.copy()
        means = self.means.copy()

        # Perform shift interventions
        if shift_interventions is not None:
            shift_interventions = parse_interventions(shift_interventions)
            targets = shift_interventions[:,0].astype(int)
            means[targets] += shift_interventions[:,1]
            variances[targets] += shift_interventions[:,2]
        
        # Perform do interventions. Note that they take preference
        # i.e. "override" shift interventions
        if do_interventions is not None:
            do_interventions = parse_interventions(do_interventions)
            targets = do_interventions[:,0].astype(int)
            means[targets] = do_interventions[:,1]
            variances[targets] = do_interventions[:,2]
            W[:,targets] = 0
            
        # Sampling by building the joint distribution
        A = np.linalg.inv(np.eye(self.p) - W.T)
        mean = A @ means
        covariance = A @ np.diag(variances) @ A.T
        distribution = NormalDistribution(mean, covariance)
        if not population:
            return distribution.sample(n)
        else:
            return distribution

def parse_interventions(interventions_dict):
    """Used internally by LGANM.sample. Transform the interventions from a dictionary to an array"""
    interventions = []
    for (target, params) in interventions_dict.items():
        # Mean and variance provided
        if type(params) == tuple and len(params) == 2:
            interventions.append([target, params[0], params[1]])
        # Only mean provided, assume we're setting the variable to a deterministic value
        elif type(params) in [float, int]:
            interventions.append([target, params, 0])
        else:
            raise ValueError("Wrongly specified intervention")
    return np.array(interventions)

#---------------------------------------------------------------------
# DAG Generating Functions

def dag_avg_deg(p, k, w_min, w_max, debug=False, random_state=None, return_ordering=False):
    """
    Generate a random graph with p nodes and average degree k
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

def dag_full(p, w_min=1, w_max=1, debug=False):
    """Creates a fully connected DAG (ie. upper triangular adj. matrix
    with all ones)"""
    A = np.triu(np.ones((p,p)), k=1)
    weights = np.random.uniform(w_min, w_max, size=A.shape)
    W = A * weights
    return W

#---------------------------------------------------------------------
# NormalDistribution class
class NormalDistribution():
    """Symbolic representation of a normal distribution that allows for
    marginalization, conditioning and sampling
    
    Attributes:
      - mean: mean vector
      - covariance: covariance matrix
      - p: number of variables

    """
    def __init__(self, mean, covariance):
        self.p = len(mean)
        self.mean = mean.copy()
        self.covariance = covariance.copy()

    def sample(self, n):
        """Sample from the distribution"""
        return np.random.multivariate_normal(self.mean, self.covariance, size=n)

    def marginal(self, X):
        """Return the marginal distribution of the variables with indices X"""
        # Parse params
        X = np.atleast_1d(X)
        # Compute marginal mean/variance
        mean = self.mean[X].copy()
        covariance = matrix_block(self.covariance, X, X).copy()
        return NormalDistribution(mean, covariance)

    def conditional(self, Y, X, x):
        """Return the conditional distribution of the variables with indices Y
        given observations x of the variables with indices X

        """
        # Parse params
        Y = np.atleast_1d(Y)
        X = np.atleast_1d(X)
        x = np.atleast_1d(x)
        if len(X) == 0:
            return self.marginal(Y)
        # See https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
        cov_y = matrix_block(self.covariance, Y, Y)
        cov_x = matrix_block(self.covariance, X, X)
        cov_yx = matrix_block(self.covariance, Y, X)
        cov_xy = matrix_block(self.covariance, X, Y)
        mean_y = self.mean[Y]
        mean_x = self.mean[X]
        mean = mean_y + cov_yx @ np.linalg.inv(cov_x) @ (x - mean_x)
        covariance = cov_y - cov_yx @ np.linalg.inv(cov_x) @ cov_xy
        return NormalDistribution(mean,covariance)

    def regress(self, y, Xs):
        """Compute the coefficients and intercept of regressing y on
        predictors Xs, where the joint distribution is a multivariate
        Gaussian
        """
        coefs = np.zeros(self.p)
        # If predictors are given, perform regression, otherwise just fit
        # intercept
        if Xs:
            cov_y_xs = matrix_block(self.covariance, [y], Xs)
            cov_xs = matrix_block(self.covariance, Xs, Xs)
            coefs[Xs] = cov_y_xs @ np.linalg.inv(cov_xs)
        intercept = self.mean[y] - coefs @ self.mean
        return (coefs, intercept)

    def mse(self, y, Xs):
        """Compute the population MSE of regressing y on predictors Xs, where
        the joint distribution is a multivariate Gaussian
        """
        var_y = self.covariance[y,y]
        # Compute regression coefficients when regressing on Xs
        (coefs_xs, _) = self.regress(y, Xs)
        # Covariance matrix
        cov = self.covariance
        # Computing the MSE
        mse = var_y + coefs_xs @ cov @ coefs_xs.T - 2 * cov[y,:] @ coefs_xs.T
        return mse

    def equal(self, dist, tol=1e-7):
        return np.allclose(self.mean, dist.mean) and np.allclose(self.covariance, dist.covariance)
