
import numpy as np


class Model:

    def prior(self):
        raise NotImplementedError

    def observation_model(self):
        raise NotImplementedError

    def __call__(self):
        prior_draws = self.prior()
        observation = self.observation_model()


