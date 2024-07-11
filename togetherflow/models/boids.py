
import numpy as np
from .model import Model


class Boids(Model):
    def __init__(self):
        raise NotImplementedError

    def prior(self):
        raise NotImplementedError

    def observation_model(self):
        raise NotImplementedError
