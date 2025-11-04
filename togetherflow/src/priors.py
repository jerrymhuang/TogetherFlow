import numpy as np
from numba import njit


@njit
def complete_pooling_prior():
    weight = np.random.beta(2., 2.)
    radius = np.random.lognormal(0, 0.5)
    # radius = np.random.lognormal(0., 0.25)
    v = np.random.beta(2., 2.) * 2.
    focus = np.random.beta(2, 5)
    return np.array([weight, radius, v, focus], dtype=np.float32)
