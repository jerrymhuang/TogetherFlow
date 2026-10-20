"""Guard against mixing simulation banks after an interrupted generation."""
import os
os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from run_main_paper_offline import bank
from variable_radius_variants import BASE


class BankIntegrityTests(unittest.TestCase):
    def test_resume_matches_uninterrupted_generation(self):
        variant = replace(BASE, num_agents=4, time_horizon=1.0)
        calls = 0
        interrupted = False

        def simulator(config, seed):
            class FakeSimulator:
                def sample(self, n):
                    nonlocal calls
                    calls += 1
                    if interrupted and calls == 2:
                        raise RuntimeError("simulated interruption")
                    rng = np.random.default_rng(seed)
                    return {key: rng.normal(size=(n, 2, 1))
                            for key in (*config.infer, *config.channels, "neighbors", "distances")}
            return FakeSimulator()

        with tempfile.TemporaryDirectory() as tmp, patch("run_main_paper_offline.build_simulator", simulator):
            uninterrupted = bank(Path(tmp) / "control", variant, 10, 123, chunk_size=4)
            interrupted = True
            calls = 0
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                bank(Path(tmp) / "resume", variant, 10, 123, chunk_size=4)
            interrupted = False
            resumed = bank(Path(tmp) / "resume", variant, 10, 123, chunk_size=4)
            self.assertFalse(set(resumed) & {"neighbors", "distances"})
            for key in uninterrupted:
                self.assertIsInstance(resumed[key], np.memmap)
                np.testing.assert_array_equal(resumed[key], uninterrupted[key])
                np.testing.assert_array_equal(resumed[key][:3], uninterrupted[key][:3])
            with self.assertRaisesRegex(ValueError, "specification changed"):
                bank(Path(tmp) / "resume", variant, 10, 124, chunk_size=4)


if __name__ == "__main__":
    unittest.main()
