"""The offline correction reduces the LR without stopping or restoring weights."""
import os
os.environ.setdefault('KERAS_BACKEND', 'jax')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
import keras
from run_main_paper_offline import regularized_components, OFFLINE_REGULARIZATION


class OfflineRegularizationTests(unittest.TestCase):
    def test_plateau_reduces_lr_without_stopping_or_changing_weights(self):
        args = SimpleNamespace(regularized_offline=True, online=False, patience=0)
        summary, subnet, options, callbacks = regularized_components(args)
        model = keras.Sequential([keras.Input((1,)), keras.layers.Dense(1)])
        model.compile(optimizer=options['optimizer'], loss='mse')
        original = [v.copy() for v in model.get_weights()]
        model.stop_training = False
        for callback in callbacks:
            callback.set_model(model)
            callback.on_train_begin()
        for epoch in range(1 + OFFLINE_REGULARIZATION['lr_patience']):
            logs = {'val_loss': 1.0}
            for callback in callbacks:
                callback.on_epoch_end(epoch, logs)
        self.assertAlmostEqual(logs['next_learning_rate'], 1e-4, places=8)
        self.assertFalse(model.stop_training)
        import numpy as np
        for before, after in zip(original, model.get_weights()):
            np.testing.assert_array_equal(before, after)
        self.assertEqual(summary['dropout'], .25)
        self.assertEqual(subnet['dropout'], .15)
        self.assertFalse(any(isinstance(c, keras.callbacks.EarlyStopping) for c in callbacks))

    def test_explicit_online_profile_matches_offline_settings(self):
        summary, subnet, options, callbacks = regularized_components(
            SimpleNamespace(regularized_online=True, online=True, patience=0))
        self.assertEqual(summary['dropout'], .25)
        self.assertEqual(subnet['dropout'], .15)
        self.assertAlmostEqual(float(keras.ops.convert_to_numpy(options['optimizer'].learning_rate)), 2e-4)
        self.assertEqual(options['optimizer'].weight_decay, .05)
        self.assertTrue(any(isinstance(c, keras.callbacks.ReduceLROnPlateau) for c in callbacks))
        with self.assertRaisesRegex(ValueError, 'online training'):
            regularized_components(SimpleNamespace(regularized_online=True, online=False, patience=0))
        with self.assertRaisesRegex(ValueError, 'without early stopping'):
            regularized_components(SimpleNamespace(regularized_online=True, online=True, patience=20))

    def test_online_cannot_accidentally_use_offline_profile(self):
        with self.assertRaisesRegex(ValueError, 'offline training'):
            regularized_components(SimpleNamespace(regularized_offline=True, online=True, patience=0))
        self.assertEqual(regularized_components(SimpleNamespace()), ({}, {}, {}, []))


if __name__ == '__main__':
    unittest.main()
