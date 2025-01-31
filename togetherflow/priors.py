import numpy as np

from numba import njit


@njit
def complete_pooling_prior():
    weight = np.random.beta(2, 5)
    radius = np.random.beta(2, 2) * 5.
    v = np.random.beta(2, 2) * 2.
    return np.array([weight, radius, v], dtype=np.float32)


@njit
def hyperprior():

    hyperprior = np.zeros(2, dtype=np.float32)
    alpha_w = np.random.uniform(1, 5)
    beta_w = np.random.uniform(1, 5)

    hyperprior[0] = alpha_w
    hyperprior[1] = beta_w

    return hyperprior


@njit
def partial_pooling_prior(hyperprior, num_agents=12):
    alpha_w, beta_w = hyperprior[0], hyperprior[1]

    weights = np.random.beta(alpha_w, beta_w, size=(num_agents, 1))

    global_prior = np.zeros((2, 1), dtype=np.float32)
    radius = np.random.beta(2, 2) * 5.
    v = np.random.beta(2, 2) * 2.

    global_prior[0] = radius
    global_prior[1] = v

    return np.concatenate((weights, global_prior), axis=-1)


@njit
def complete_pooling_prior_with_noise():
    external_noise = np.random.beta(2, 2)
    internal_noise = np.random.beta(2, 2)
    weight = np.random.beta(2, 5)
    radius = np.random.beta(2, 2) * 5.
    v = np.random.beta(2, 2) * 2.
    return np.array([weight, radius, v, external_noise, internal_noise])
