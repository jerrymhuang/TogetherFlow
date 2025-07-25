import numpy as np

from numba import njit


@njit
def fancy_complete_pooling_prior():
    """
    Function that samples the free parameters under a complete pooling scheme
    based on their respective prior distributions.

    Returns
    -------
    np.ndarray of the following sampled priors:
    -   weight: the influence weight of the agent that emphasizes their tendency to move
                individually or collectively.
    -   radius: the sensing radius of the agent for calculating external influences.
    -   v:      global movement velocity (drift rate) of the agent.
    """
    weight = np.random.beta(2, 5)
    radius = np.random.beta(2, 2) * 5.
    v = np.random.beta(2, 2) * 2.
    return np.array([weight, radius, v], dtype=np.float32)

@njit
def complete_pooling_prior():
    weight = np.random.beta(2, 2)
    radius = np.random.gamma(5, 1)
    v = np.random.beta(2, 2) * 2.
    #focus = np.random.beta(2, 5)
    return np.array([weight, radius, v], dtype=np.float32)


@njit
def partial_pooling_hyper_prior():
    """
    Function that samples the hyperpriors of the free parameters under a partial pooling scheme.

    Returns
    -------
    np.ndarray of the following sampled priors:
    -   alpha_w : alpha parameter of the Beta distribution for the sampled weights.
    -   beta_w  : beta parameter of the Beta distribution for the sampled weights.
    """

    hyperprior = np.zeros((4, ), dtype=np.float32)

    # This is redundant, but doing so for clarity.
    alpha_w = np.random.random() * 4 + 1
    beta_w = np.random.random() * 4 + 1
    alpha_r = np.random.random() * 4 + 1
    beta_r = np.random.random() * 4 + 1

    hyperprior[0] = alpha_w
    hyperprior[1] = beta_w
    hyperprior[2] = alpha_r
    hyperprior[3] = beta_r

    return hyperprior


@njit
def partial_pooling_local_prior(hyperprior, num_agents=12):
    """
    Function that samples the free parameters under a partial pooling scheme.

    Parameters
    ----------
    hyperprior : np.ndarray of size (2)
        The sampled hyperpriors
    num_agents : int, default=12
        The number of agents to sample

    Returns
    -------
    np.ndarray of the following sampled priors:
    -   weights :   the influence weights of the individual agents that
                    emphasizes their tendency to move individually or
                    collectively.
    -   radius  :   the sensing radius of the agent for calculating
                    external influences. This radius needs to be sampled
                    globally for the simulation to function properly.
    -   v       :   global movement velocity (drift rate) of the agent.
    """

    alpha_w, beta_w = hyperprior[0].item(), hyperprior[1].item()
    weights = np.random.beta(alpha_w, beta_w, size=(num_agents, 1)).astype(np.float32)

    alpha_v, beta_v = hyperprior[2].item(), hyperprior[3].item()
    drifts = np.random.beta(alpha_v, beta_v, size=(num_agents, 1)).astype(np.float32)

    return np.concatenate((weights, drifts), axis=-1)


@njit
def partial_pooling_shared_prior():

    shared_prior = np.zeros((1, ), dtype=np.float32)

    # weight = np.random.beta(2, 5)
    radius = np.random.beta(2, 2) * 5.

    # shared_prior[0] = weight
    shared_prior[0] = radius

    return shared_prior


@njit
def complete_pooling_prior_with_noise():
    external_noise = np.random.beta(2, 2)
    internal_noise = np.random.beta(2, 2)
    weight = np.random.beta(2, 5)
    radius = np.random.beta(2, 2) * 5.
    v = np.random.beta(2, 2) * 2.
    return np.array([weight, radius, v, external_noise, internal_noise])
