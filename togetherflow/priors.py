import numpy as np

from numba import njit


@njit
def complete_pooling_prior():
    weight = np.random.beta(2, 5)
    radius = np.random.beta(2, 2) * 5.
    v = np.random.beta(2, 2) * 2.
    return np.array([weight, radius, v], dtype=np.float32)


@njit
def partial_pooling_prior():
    alpha_w = np.random.uniform(1, 5)
    beta_w = np.random.uniform(1, 5)

    return np.array([alpha_w, beta_w], dtype=np.float32)


@njit
def complete_pooling_prior_with_noise():
    external_noise = np.random.beta(2, 2)
    internal_noise = np.random.beta(2, 2)
    weight = np.random.beta(2, 5)
    radius = np.random.beta(2, 2) * 5.
    v = np.random.beta(2, 2) * 2.
    return np.array([weight, radius, v, external_noise, internal_noise])
