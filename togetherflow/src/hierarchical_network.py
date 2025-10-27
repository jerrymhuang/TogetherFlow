import keras
from bayesflow.networks import SummaryNetwork
from bayesflow.utils.serialization import serializable

@serializable("bayesflow.networks")
class HierarchicalNetwork(SummaryNetwork):
    def __init__(self, networks, **kwargs):
        super().__init__(**kwargs)
        self.networks = list(networks)
        self._final_output_shape = None


    def call(self, x, return_all=False, **kwargs):
        if return_all:
            outputs = []
            for network in self.networks:
                x = network(x, **kwargs)
                outputs.append(x)
            return outputs
        else:
            for network in self.networks:
                x = network(x, **kwargs)
            return x
