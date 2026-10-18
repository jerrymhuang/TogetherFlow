"""Scientific configuration invariants; run with unittest and the main environment."""
import os
os.environ.setdefault('KERAS_BACKEND', 'jax')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
import unittest
import numpy as np
from revision_variants import ARMS, BASE, SWITCH, PARTIAL, CONTROL
from run_variant import build_simulator
from dataclasses import replace


class RevisionDesignTests(unittest.TestCase):
    def test_fixed_radius_never_inferred(self):
        for arm in ARMS:
            self.assertNotIn('r', arm.infer)
            draws = np.stack([arm.prior() for _ in range(8)])
            self.assertTrue(np.all(draws[:, 1] == draws[0, 1]), arm.slug)
            self.assertTrue(np.isfinite(draws).all(), arm.slug)

    def test_hierarchical_control_matches(self):
        for name in ['time_horizon','summary_net','inference_net','constrain_parameters',
                     'epochs','batch_size','num_batches_per_epoch','channels']:
            self.assertEqual(getattr(PARTIAL,name),getattr(CONTROL,name),name)

    def test_switching_observations_match(self):
        self.assertEqual(BASE.channels, SWITCH.channels)
        self.assertNotIn('salience', SWITCH.channels)
        self.assertIsNone(BASE.salience_sigma)
        self.assertIsNotNone(SWITCH.salience_sigma)

    def test_reproducible_heldout_simulations(self):
        arm = replace(SWITCH, time_horizon=1, num_agents=4)
        a = build_simulator(arm, 123).sample(2)
        b = build_simulator(arm, 123).sample(2)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        np.testing.assert_array_equal(a['r'], np.full_like(a['r'], 2.5))


if __name__ == '__main__':
    unittest.main()
