"""Latent weight export must preserve simulations and match agent identity."""
import unittest
import numpy as np
from numba import njit
from togetherflow import TogetherFlowSimulator
from togetherflow.priors import partial_pooling_prior


@njit
def zero_spread_prior():
    return np.array([.3, 2.5, 1., .2, 0.], dtype=np.float32)


class AgentWeightTests(unittest.TestCase):
    def simulator(self, export, prior=partial_pooling_prior):
        return TogetherFlowSimulator(num_agents=4, time_horizon=1, seed=123,
            prior=prior, param_names=('mu_w','r','v','noise','sigma_w'),
            include_agent_weights=export)

    def test_export_preserves_observations_and_rng(self):
        base = self.simulator(False).sample(8)
        exported = self.simulator(True).sample(8)
        self.assertNotIn('agent_weights', base)
        for key in base:
            np.testing.assert_array_equal(base[key], exported[key])
        weights = exported['agent_weights']
        self.assertEqual(weights.shape, (8,4))
        self.assertTrue(np.all((weights>0)&(weights<1)))
        self.assertTrue(np.any(np.std(weights,axis=1)>0))
        np.testing.assert_array_equal(weights,self.simulator(True).sample(8)['agent_weights'])

    def test_zero_spread_collapses_to_population_weight(self):
        data = self.simulator(True,zero_spread_prior).sample(4)
        np.testing.assert_allclose(data['agent_weights'],.3,atol=1e-7)

    def test_rejects_nonhierarchical_export(self):
        with self.assertRaises(ValueError):
            TogetherFlowSimulator(include_agent_weights=True)


if __name__ == '__main__':
    unittest.main()
