
import numpy as np

class Vicsek:

    def __init__(self,
                 num_agents: int = 12,
                 rng: np.random.Generator = None
    ):
        """
        Vicsek model for particle motion.

        Parameters
        ----------
        num_agents : int, default: 12
            Number of agents.
        rng : np.random.Generator, optional
            Random number generator.
            If it is None, a default random number generator will be used.
        """
        self.num_agents = num_agents
        self.rng = rng
        if self.rng is None:
            self.rng = np.random.default_rng()


    def prior(self):
        """
        Generates a random draw from the custom prior over the
        three key parameters of interest for agent movement:
        1) sensing radius (r);
        2) agent speed (v);
        3) variance of rotational noise (eta).


        Returns
        -------
        params  : np.ndarray of shape (3, )
            A single draw from the prior.
        """
        r = self.rng.beta(2., 5.)
        v = self.rng.beta(2., 2.)
        eta = self.rng.uniform(-np.pi * 0.25, np.pi * 0.25)
        return np.append(r, v, eta)


    def observation_model(self):
        raise NotImplementedError