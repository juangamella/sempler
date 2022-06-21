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
marginalization, conditioning and regression in the population setting.
"""

import numpy as np
import sempler.utils as utils
import warnings

# ---------------------------------------------------------------------
# NormalDistribution class


class NormalDistribution():
    """Symbolic representation of a normal distribution.

    Parameters
    ----------
    mean : array_like
        The marginal means of the variables.
    covariance : array_like
        The covariance matrix of the distribution.
    check_valid : {'ignore', 'warn', 'raise'}, optional
        Behaviour when the provided covariance matrix is not positive
        definite. Note that semi-definiteness is enough for sampling,
        but not for the conditioning and regression operations.

    Raises
    ------
    ValueError
        If `check_valid='raise'` and the provided matrix is not
        positive semidefinite; *or* if the sizes of the mean vector
        and covariance matrices do not match.

    Examples
    --------

    >>> import sempler
    >>> import numpy as np

    Defining a univariate standard normal distribution:

    >>> distribution = sempler.NormalDistribution(0, 1)

    Defining an isotropic normal with zero mean vector:

    >>> covariance = np.array([[1,0],[0,1]])
    >>> means = np.array([0,0])
    >>> distribution = sempler.NormalDistribution(means, covariance)

    An error is raised if the covariance matrix is not positive
    definite and we set `check_valid='raise'`:

    >>> sempler.NormalDistribution([0,0], [[1,0],[1,1]], check_valid='raise')
    Traceback (most recent call last):
      ...
    ValueError: Covariance matrix is not positive definite.

    >>> sempler.NormalDistribution([0,0], [[1,1],[1,1]], check_valid='raise')
    Traceback (most recent call last):
      ...
    ValueError: Covariance matrix is not positive definite.

    Or if the size of the mean vector and covariance matrix do not
    match:

    >>> sempler.NormalDistribution([0], [[1,0],[0,1]])
    Traceback (most recent call last):
      ...
    ValueError: Mismatch in the size of mean vector and covariance matrix.


    Attributes
    ----------
    mean : numpy.ndarray
        The marginal means of the variables.
    covariance : numpy.ndarray
        The covariance matrix of the distribution.
    p : int
        The number of variables.

    """

    def __init__(self, mean, covariance, check_valid='ignore'):
        mean = np.atleast_1d(mean)
        covariance = np.atleast_2d(covariance)
        # Check positive semidefiniteness
        if check_valid != 'ignore':
            try:
                np.linalg.cholesky(covariance)
            except np.linalg.LinAlgError:
                msg = "Covariance matrix is not positive definite."
                raise ValueError(msg) if check_valid == 'raise' else warnings.warn(msg)
        # Define other parameters
        if len(mean) != len(covariance):
            raise ValueError("Mismatch in the size of mean vector and covariance matrix.")
        self.p = len(mean)
        self.mean = mean.copy()
        self.covariance = covariance.copy()

    def __str__(self):
        return "mean:\n" + str(self.mean) + "\ncovariance:\n" + str(self.covariance)

    def sample(self, n, random_state=None):
        """Generate a sample from the distribution.

        Parameters
        ----------
        n : int
            The size of the sample (i.e. number of observations).
        random_state : int,optional
            To set the random state for reproducibility.

        Returns
        -------
        numpy.ndarray
            An array containing the sample, where each column
            corresponds to a variable.

        Example
        -------

        >>> distribution.sample(5, random_state = 42)
        array([[0.754143  , 1.67465613, 2.45882157, 2.4718102 , 3.8108411 ],
               [2.04520905, 1.21383225, 3.29393672, 4.46230358, 5.79975029],
               [0.99209988, 2.50969225, 4.41981256, 5.3086489 , 5.73704626],
               [0.76152114, 2.94130318, 4.45257529, 5.02206465, 6.25225282],
               [0.71256027, 1.22445591, 1.33646622, 2.19050731, 0.07261   ]])

        """
        np.random.seed(random_state) if random_state is not None else None
        return np.random.multivariate_normal(self.mean, self.covariance, size=n)

    def marginal(self, X):
        """Return the marginal distribution of some variables.

        Parameters
        ----------
        X : array_like
            The indices of the variables. Note that the indices in the
            new distribution are dependent on the order given in X,
            e.g. `marginal([0,1])` and `marginal([1,0])` yield
            different (permuted) distributions.

        Returns
        -------
        marginal : sempler.NormalDistribution
            The marginal distribution.

        Examples
        --------

        >>> marginal = distribution.marginal(0)

        >>> marginal = distribution.marginal([0,1])

        """
        # Parse params
        X = np.atleast_1d(X)
        # Compute marginal mean/variance
        mean = self.mean[X]
        covariance = utils.matrix_block(self.covariance, X, X)
        return NormalDistribution(mean, covariance)

    def conditional(self, Y, X, x):
        """Return the conditional distribution of some variables given some
        others' values.

        Parameters
        ----------
        Y : array_like, list of ints or np.array
            The indices of the conditioned variables. Note that the indices in the
            new distribution are dependent on the order given in Y,
            e.g. `marginal([0,1])` and `marginal([1,0])` yield
            different (permuted) distributions.
        X : array_like, list of ints or np.array
            The indices of the variables to condition on.
        x : array_like
            The values of the conditioning variables.

        Raises
        ------
        ValueError
            If the size of `X` and `x` do not match, or if `X` and `Y` are not disjoint.

        Returns
        -------
        distribution : sempler.NormalDistribution
            the conditional distribution.

        Examples
        --------

        Conditioning a single variable:

        >>> conditional = distribution.conditional(0, 1, .1)

        Conditioning several variables:

        >>> conditional = distribution.conditional([0,1], [2,3], [.2, .3])

        An exception is raised if the size of `X` and `x` do not match:

        >>> distribution.conditional(0, [1], [.1, .2])
        Traceback (most recent call last):
        ...
        ValueError: Mismatch in the size of X and x.

        or if `Y` and `X` are not disjoint:

        >>> distribution.conditional([0], [0,1], [0, .1])
        Traceback (most recent call last):
        ...
        ValueError: X and Y are not disjoint.

        """
        # Parse params
        Y = np.atleast_1d(Y)
        X = np.atleast_1d(X)
        x = np.atleast_1d(x)
        # Check exceptions
        if len(X) != len(x):
            raise ValueError("Mismatch in the size of X and x.")
        if len(set(Y) & set(X)) > 0:
            raise ValueError("X and Y are not disjoint.")
        # Conditioning on nothing = marginalizing
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
        return NormalDistribution(mean, covariance)

    def regress(self, y, Xs):
        """Compute the population MLE of the regression coefficients and
        intercept from regressing a variable on a subset of others.

        Parameters
        ----------
        y : int
            The index of the response/predicted variable.
        Xs : array_like
            The indices of the predictor/explanatory variables.

        Returns
        -------
        coefs : numpy.ndarray
            A p-sized array containing the estimated coefficients in
            the indices in Xs and 0s elsewhere.
        intercept : float
            The estimated intercept.

        Example
        -------
        >>> distribution.regress(0,[1,2])
        (array([ 0.        , -0.33333333,  0.33333333,  0.        ,  0.        ]), 0.6666666666666666)

        Regressing a variable on itself:

        >>> distribution.regress(0,0)
        (array([1., 0., 0., 0., 0.]), 0.0)

        """
        coefs = np.zeros(self.p)
        # If predictors are given, perform regression, otherwise just fit
        # intercept
        Xs = np.atleast_1d(Xs)
        if len(Xs) > 0:
            cov_y_xs = self.covariance[y, Xs]  # utils.matrix_block(self.covariance, y, Xs)
            cov_xs = self.covariance[:, Xs][Xs, :]  # utils.matrix_block(self.covariance, Xs, Xs)
            coefs[Xs] = np.linalg.solve(cov_xs, cov_y_xs)
        intercept = self.mean[y] - coefs @ self.mean
        return (coefs, intercept)

    def mse(self, y, Xs):
        """Compute the population (i.e. expected) mean squared error resulting
        from regressing a variable on a subset of others.

        The regression coefficients/intercept are the MLE computed in
        :func:`~sempler.NormalDistribution.regress`.

        Parameters
        ----------
        y : int
            The index of the response/predicted variable.
        Xs : int, list of ints or np.array
            The indices of the predictor/explanatory variables.

        Returns
        -------
        mse : float
           the expected mean squared error.

        Example
        -------

        >>> distribution.mse(0, [1,2])
        0.24666666666666665

        Regressing a variable on itself yields a zero error:

        >>> distribution.mse(0,0)
        0.0

        """
        var_y = self.covariance[y, y]
        # Compute regression coefficients when regressing on Xs
        (coefs_xs, _) = self.regress(y, Xs)
        # Covariance matrix
        cov = self.covariance
        # Computing the MSE
        mse = var_y + coefs_xs @ cov @ coefs_xs.T - 2 * cov[y, :] @ coefs_xs.T
        return mse

    def equal(self, dist, rtol=1e-5, atol=1e-8):
        """Check if this distribution is equal to another
        sempler.NormalDistribution, up to a tolerance.

        Parameters
        ----------
        dist : sempler.NormalDistribution
            The distribution to compare with.
        rtol : float, optional
            The relative tolerance in the elements of the mean
            and covariance. Default is 1e-5.
        atol : float, optional
            The absolute tolerance in the elements of the mean
            and covariance. Default is 1e-8.

        Returns
        -------
        equal : bool
            If the two distributions have the same mean/covariance up
            to tolerance tol.

        Raises
        ------
        TypeError
           If `dist` is not a sempler.NormalDistribution.

        Example
        -------

        >>> import sempler
        >>> dist1 = sempler.NormalDistribution(0, 1)
        >>> sempler.NormalDistribution(0.5, 1).equal(dist1, atol=0.5)
        True
        >>> sempler.NormalDistribution(0.5, 1).equal(dist1)
        False

        An exception is raised if we attempt to compare with anything else than a sempler.NormalDistribution:

        >>> sempler.NormalDistribution(0,1).equal(1)
        Traceback (most recent call last):
        ...
        TypeError: Unexpected type for "dist".

        """
        # Check type
        if type(dist) != NormalDistribution:
            raise TypeError('Unexpected type for "dist".')
        equal = np.allclose(self.mean, dist.mean, atol=atol, rtol=rtol) and np.allclose(
            self.covariance, dist.covariance, atol=atol, rtol=rtol)
        return equal


# To run the method's doctests
if __name__ == '__main__':
    import doctest
    # Build LGANM
    covariance = np.array([[0.37, 0., 0.37, 0.37, 0.75],
                           [0., 0.95, 0.95, 0.95, 1.9],
                           [0.37, 0.95, 2.06, 2.06, 4.11],
                           [0.37, 0.95, 2.06, 2.66, 4.71],
                           [0.75, 1.9, 4.11, 4.71, 9.48]])
    distribution = NormalDistribution([1, 2, 3, 4, 5], covariance, check_valid='raise')
    doctest.testmod(extraglobs={'distribution': distribution}, verbose=True)
