import sempler
import sempler.noise as noise
import numpy as np

# Connectivity matrix
A = np.array([[0, 0, 0, 1, 0],
  [0, 0, 1, 0, 0],
  [0, 0, 0, 1, 0],
  [0, 0, 0, 0, 1],
  [0, 0, 0, 0, 0]])

# Noise distributions (see sempler.noise)
noise_distributions = [noise.normal(0,1)] * 5

# Variable assignments
functions = [None, None, np.sin, lambda x: np.exp(x[:,0]) + 2*x[:,1], lambda x: 2*x]

# All together
anm = sempler.ANM(A, functions, noise_distributions)

# Sampling from the observational setting
samples = anm.sample(100)

# Sampling under a shift intervention on variable 1
samples = anm.sample(100, shift_interventions = {1: noise.normal(0,1)})
