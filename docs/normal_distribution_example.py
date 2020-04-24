import numpy as np
import sempler

# Define by mean and covariance
mean = np.array([1,2,3])
covariance = np.array([[1, 2, 4], [2, 6, 5], [4, 5, 1]])
distribution = sempler.NormalDistribution(mean, covariance)

# Marginal distribution of X0 and X1 (also a NormalDistribution object)
marginal = distribution.marginal([0, 1])

# Conditional distribution of X2 on X1=1 (also a NormalDistribution object)
conditional = distribution.conditional(2,1,1)

# Regress X0 on X1 and X2 in the population setting (no estimation errors)
(coefs, intercept) = distribution.regress(0, [1,2])
