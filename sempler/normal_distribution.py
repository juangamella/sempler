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

"""Module containing the sempler.NormalDistribution class, which
represents a normal distribution (mean + covariance) and allows for
marginalization, conditioning and regression.
"""

import numpy as np
import sempler.utils as utils

#---------------------------------------------------------------------
# NormalDistribution class
class NormalDistribution():
    """Symbolic representation of a normal distribution that allows for
    marginalization, conditioning and sampling
    
    Parameters
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
