"""Verify radius inference and exclusion of true-radius observation features."""
import os
os.environ.setdefault('KERAS_BACKEND', 'jax')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
import unittest
from dataclasses import replace
import numpy as np
from variable_radius_variants import ARMS, BASE, PARTIAL, CONTROL
from run_variant import build_simulator, build_adapter

class VariableRadiusTests(unittest.TestCase):
    def test_radius_is_random_and_inferred(self):
        for arm in ARMS:
            self.assertIn('r', arm.infer)
            draws = np.stack([arm.prior() for _ in range(32)])
            self.assertGreater(np.std(draws[:, 1]), 0)
            self.assertTrue(np.all(draws[:, 1] > 0))
            self.assertFalse(set(arm.channels) & {'neighbors', 'distances', 'neighbor_fluctuations'})

    def test_adapter_excludes_oracle_features(self):
        for arm in ARMS:
            small = replace(arm, time_horizon=1, num_agents=4)
            raw = build_simulator(small, 123).sample(3)
            adapter = build_adapter(small)
            expected = adapter(raw)['summary_variables']
            changed = {k: v.copy() for k, v in raw.items()}
            for key in ['r', 'neighbors', 'distances', 'neighbor_fluctuations']:
                changed[key] = changed[key] * 7 + 1
            np.testing.assert_array_equal(expected, adapter(changed)['summary_variables'])
            self.assertEqual(adapter(raw)['inference_variables'].shape[-1], len(arm.infer))
            self.assertTrue(np.isfinite(expected).all())

    def test_hierarchical_control_matches(self):
        for key in ['time_horizon', 'summary_net', 'inference_net', 'constrain_parameters',
                    'epochs', 'batch_size', 'num_batches_per_epoch', 'channels', 'reference_radii']:
            self.assertEqual(getattr(PARTIAL, key), getattr(CONTROL, key))

if __name__ == '__main__':
    unittest.main()
