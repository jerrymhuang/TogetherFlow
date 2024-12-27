import numpy as np

from numba import njit


@njit
def complete_pooling_prior_fun():
    weight = np.random.beta(2, 5)
    radius = np.random.gamma(2, 2)
    v = np.random.beta(2, 2)
    external_noise = np.random.beta(2, 2)
    internal_noise = np.random.beta(2, 2)
    return np.array([weight, radius, v, external_noise, internal_noise])

@njit
def meta_prior_fun():
    raise NotImplementedError

