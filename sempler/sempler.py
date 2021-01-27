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
    
    Attributes
    ----------
    A : np.array
        The p x p adjacency matrix specifying the functional
        dependencies-
    p : int
        the number of variables.
    assignments : list of functions
        the assignment functions of the variables.
    noise_distributions : list of functions
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
        functions : list of functions or NoneType
            a list of p functions representing the functional
            assignments of each variable. Each function must take as
            many arguments as specified by the adjacency matrix A, or
            be None if the variable has no parents.
        noise_distributions : list of functions
            a list of p functions that generate samples of each
            variable's noise distribution (see sempler.noise for
            details).

        """
        self.ordering = utils.topological_ordering(A)
        self.p = len(A)
        self.A = deepcopy(A)
        self.assignments = [functions.null if fun is None else deepcopy(fun) for fun in assignments]
        self.noise_distributions = deepcopy(noise_distributions)
            
    def sample(self, n, do_interventions = {}, shift_interventions = {}, noise_interventions = {}, random_state = None):
        """Generates n observations from the ANM, under the given do, shift or
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

#---------------------------------------------------------------------
# LGANM class
class LGANM:
    """Represents a linear model with Gaussian additive noise
    (i.e. Gaussian Bayesian Network).

    Attributes
    ----------
    W : np.array:
        connectivity (weights) matrix representing a DAG.
    variances : np.array
        the variances of the noise terms.
    means : np.array
        the means of the noise terms.

    """
    
    def __init__(self, W, variances, means):
        """
        Create a linear Gaussian SCM.
        
        Parameters
        ----------
        W : np.array:
            connectivity (weights) matrix representing a DAG.
        variances : np.array or tuple
            the variances of the noise terms, or a tuple representing
            the lower/upper bounds to sample them from a uniform
            distribution.
        means : np.array or tuple
            the means of the noise terms, or a tuple representing
            the lower/upper bounds to sample them from a uniform
            distribution.

        Returns
        -------

        """
        self.W = W.copy()
        self.p = len(W)

        # Set variances
        if isinstance(variances, tuple) and len(variances) == 2:
            self.variances = np.random.uniform(variances[0], variances[1], size=self.p)
        elif type(variances) == np.ndarray and len(variances) == self.p:
            self.variances = variances.copy()
        else:
            raise ValueError("Wrong value for variances")
            
        # Set means
        if isinstance(means, tuple) and len(means) == 2:
            self.means = np.random.uniform(means[0], means[1], size=self.p)
        elif type(means)==np.ndarray and len(means) == self.p:
            self.means = means.copy()
        else:
            raise ValueError("Wrong value for means")
    
    def sample(self, n=100, population=False, do_interventions=None, shift_interventions=None, noise_interventions=None):
        """Generates n observations from the linear Gaussian SCM, under the
        given do, shift or noise interventions. If none are given,
        sample from the observational distribution.
        
        Parameters
        ----------
        n : int,optional
            the size of the sample (i.e. number of
            observations). Defaults to 100.
        population : bool, optional
            if True, the function returns a symbolic normal
            distribution instead of samples (see
            sempler.NormalDistribution). Defaults to False.
        do_interventions : dict, optional
            a dictionary where keys correspond to the intervened
            variables, and the values are tuples representing the new
            mean/variance of the intervened variable, e.g. {1: (1,2)}.
        shift_interventions : dict, optional
            a dictionary where keys correspond to the intervened
            variables, and the values are tuples representing the
            mean/variance of the noise which is added to the
            intervened variables, e.g. {1: (1,2)}.
        noise_interventions : dict, optional
            a dictionary where keys correspond to the intervened
            variables, and the values are tuples representing the
            mean/variance of the new noise, e.g. {1: (1,2)}.
        random_state: int, optional
            set the random state, for reproducibility.

        Returns
        -------
        X : np.array or sempler.NormalDistribution
            an array containing the sample, where each column
            corresponds to a variable; or, if population=True, a
            symbolic normal distribution (see
            sempler.NormalDistribution).

        """
        # Must copy as they can be changed by interventions, but we
        # still want to keep the observational SEM
        W = self.W.copy()
        variances = self.variances.copy()
        means = self.means.copy()

        # Perform shift interventions
        if shift_interventions:
            shift_interventions = _parse_interventions(shift_interventions)
            targets = shift_interventions[:,0].astype(int)
            means[targets] += shift_interventions[:,1]
            variances[targets] += shift_interventions[:,2]

        # Perform noise interventions. Note that they take preference
        # i.e. "override" shift interventions
        if noise_interventions:
            noise_interventions = _parse_interventions(noise_interventions)
            targets = noise_interventions[:,0].astype(int)
            means[targets] = noise_interventions[:,1]
            variances[targets] = noise_interventions[:,2]
        
        # Perform do interventions. Note that they take preference
        # i.e. "override" shift and noise interventions
        if do_interventions:
            do_interventions = _parse_interventions(do_interventions)
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

def _parse_interventions(interventions_dict):
    """Used internally by LGANM.sample. Transforms the interventions from
    a dictionary to an array"""
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
    """Generate an Erdos-Renyi graph with p nodes and average degree k,
    and orient edges according to a random ordering. Sample the edge
    weights from a uniform distribution.

    Parameters
    ----------
    p : int
        the number of nodes in the graph.
    k : float
        the desired average degree.
    w_min : float
        the lower bound on the sampled weights.
    w_max : float
        the upper bound on the sampled weights.
    debug : bool, optional
        if debug traces should be printed
    random_state : int,optional
        to set the random state for reproducibility.
    return_ordering: bool, optional
        if the topological ordering used to orient the edge should be
        returned.

    Returns
    -------
    W : np.array
       the connectivity (weights) matrix of the generated DAG.
    ordering : np.array, optional
       if return_ordering = True, a topological ordering of the graph.

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

    """Create a fully connected DAG, sampling the weights from a uniform
    distribution.

    Parameters
    ----------
    p : int
        the number of nodes in the graph.
    w_min : float, optional
        the lower bound on the sampled weights. Defaults to 1.
    w_max : float, optional
        the upper bound on the sampled weights. Defaults to 1.

    Returns
    -------
    W : np.array
       the connectivity (weights) matrix of the generated DAG.

    """
    A = np.triu(np.ones((p,p)), k=1)
    weights = np.random.uniform(w_min, w_max, size=A.shape)
    W = A * weights
    return W

#---------------------------------------------------------------------
# NormalDistribution class
class NormalDistribution():
    """Symbolic representation of a normal distribution that allows for
    marginalization, conditioning and sampling
    
    Attributes
    ----------
    mean : np.array
        the marginal means of the variables.
    covariance : np.array
        the covariance matrix of the distribution.
    p : int
        the number of variables.

    """
    def __init__(self, mean, covariance):
        """Create a representation of a normal distribution.

        Parameters
        ----------
        mean : np.array
            the marginal means of the variables.
        covariance : np.array
            the covariance matrix of the distribution.

        Returns
        -------

        """
        self.p = len(mean)
        self.mean = mean.copy()
        self.covariance = covariance.copy()

    def sample(self, n, random_state = None):
        """Generate a sample from the distribution.

        Parameters
        ----------
        n : int
            the size of the sample (i.e. number of observations).
        random_state : int,optional
            to set the random state for reproducibility.

        Returns
        -------
        X : np.array
            an array containing the sample, where each column
            corresponds to a variable.

        """
        np.random.seed(random_state) if random_state is not None else None
        return np.random.multivariate_normal(self.mean, self.covariance, size=n)

    def marginal(self, X):
        """Return the marginal distribution of the variables with indices X.
        
        Parameters
        ----------
        X : int, list of ints or np.array
            the indices of the variables.
        
        Returns
        -------
        distribution : sempler.NormalDistribution
            the marginal distribution.

        """
        # Parse params
        X = np.atleast_1d(X)
        # Compute marginal mean/variance
        mean = self.mean[X].copy()
        covariance = utils.matrix_block(self.covariance, X, X).copy()
        return NormalDistribution(mean, covariance)

    def conditional(self, Y, X, x):
        """Return the conditional distribution of some variables given some
        others' values.
        
        Parameters
        ----------
        Y : int, list of ints or np.array
            the indices of the conditioned variables.
        X : int, list of ints or np.array
            the indices of the variables to condition on.
        x : np.array
            the values of the conditioning variables.
        
        Returns
        -------
        distribution : sempler.NormalDistribution
            the conditional distribution.

        """
        # Parse params
        Y = np.atleast_1d(Y)
        X = np.atleast_1d(X)
        x = np.atleast_1d(x)
        if len(X) == 0:
            return self.marginal(Y)
        # See https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
        cov_y = utils.matrix_block(self.covariance, Y, Y)
        cov_x = utils.matrix_block(self.covariance, X, X)
        cov_yx = utils.matrix_block(self.covariance, Y, X)
        cov_xy = utils.matrix_block(self.covariance, X, Y)
        mean_y = self.mean[Y]
        mean_x = self.mean[X]
        mean = mean_y + cov_yx @ np.linalg.inv(cov_x) @ (x - mean_x)
        covariance = cov_y - cov_yx @ np.linalg.inv(cov_x) @ cov_xy
        return NormalDistribution(mean,covariance)

    def regress(self, y, Xs):
        """Compute the population MLE of the regression coefficients and
        intercept from regressing a variable on a subset of others.

        Parameters
        ----------
        y : int
            the index of the response/predicted/explanatory variable.
        Xs : int, list of ints or np.array
            the indices of the predictor/explanatory variables.
        
        Returns
        -------
        coefs : np.array
            the estimated regression coefficients.
        intercept : float
            the estimated intercept.

        """
        coefs = np.zeros(self.p)
        # If predictors are given, perform regression, otherwise just fit
        # intercept
        Xs = np.atleast_1d(Xs)
        if len(Xs) > 0:
            cov_y_xs = utils.matrix_block(self.covariance, [y], Xs)
            cov_xs = utils.matrix_block(self.covariance, Xs, Xs)
            coefs[Xs] = cov_y_xs @ np.linalg.inv(cov_xs)
        intercept = self.mean[y] - coefs @ self.mean
        return (coefs, intercept)

    def mse(self, y, Xs):
        """Compute the population (i.e. expected) mean squared error resulting
        from regressing a variable on a subset of others.

        The regression coefficients/intercept are the MLE computed in
        NormalDistribution.regress.

        Parameters
        ----------
        y : int
            the index of the response/predicted/explanatory variable.
        Xs : int, list of ints or np.array
            the indices of the predictor/explanatory variables.

        Returns
        -------
        mse : float
           the expected mean squared error.

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
        """Check if this distribution is equal to another
        sempler.NormalDistribution, up to a tolerance.

        Parameters
        ----------
        dist : sempler.NormalDistribution
            the distribution to compare with.
        tol : float, optional
            the allowed (absolute) tolerance in mean and
            covariance. Default is 1e-7.

        Returns
        -------
        equal : bool
            if the two distributions have the same mean/covariance up
            to tolerance tol.

        """        
        return np.allclose(self.mean, dist.mean) and np.allclose(self.covariance, dist.covariance)
