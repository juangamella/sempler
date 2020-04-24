import sempler
import numpy as np

# Connectivity matrix
W = np.array([[0, 0, 0, 0.1, 0],
              [0, 0, 2.1, 0, 0],
              [0, 0, 0, 3.2, 0],
              [0, 0, 0, 0, 5.0],
              [0, 0, 0, 0, 0]])

# All together
lganm = sempler.LGANM(W, (0,1), (0,1))

# Sampling from the observational setting
samples = lganm.sample(100)

# Sampling under a shift intervention on variable 1 with standard gaussian noise
samples = lganm.sample(100, shift_interventions = {1: (0,1)})

# Sampling the observational environment in the "population setting"
distribution = lganm.sample(population = True)
