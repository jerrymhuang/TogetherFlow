"""Dependency and simulator checks for graphical weight/radius pooling."""

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import numpy as np
from scipy.special import logit
from graphical_partial_pooling import (
    AGENTS,
    RADIUS_MAX,
    POPULATION_SCALES,
    build_simulator,
    sample_agents,
    simulate_trajectory,
)
from run_graphical_partial_pooling import physical_parameters


class GraphicalPoolingTests(unittest.TestCase):
    def population(self):
        values = np.array([0.4, 0.3, 2.5, 0.2, 1.0, 0.2])
        return logit(values / POPULATION_SCALES).astype("float32")

    def test_graph_preserves_joint_agent_conditioning(self):
        graph = build_simulator().graph.invert()
        self.assertEqual(
            graph.network_composition(), {0: ["population"], 1: ["agents"]}
        )
        self.assertEqual(
            graph.network_conditions(),
            {
                0: ["observations"],
                1: ["population", "observations"],
            },
        )
        self.assertEqual(len(graph.summary_network_input_shapes()), 2)

    def test_graph_samples_are_reproducible_and_have_physical_support(self):
        simulator = build_simulator()
        np.random.seed(91)
        first = simulator.sample(2)
        np.random.seed(91)
        second = simulator.sample(2)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        self.assertEqual(first["features"].shape, (2, 1, 1224))
        self.assertEqual(first["agents"].shape, (2, 24))
        physical = physical_parameters(first)
        self.assertEqual(physical.shape, (2, 30))
        self.assertTrue(np.isfinite(first["features"]).all())
        self.assertTrue(np.all((physical[:, 6:18] > 0) & (physical[:, 6:18] < 1)))
        self.assertTrue(
            np.all((physical[:, 18:] > 0) & (physical[:, 18:] < RADIUS_MAX))
        )

    def test_supplied_radii_determine_receiver_neighborhoods(self):
        population = self.population()
        radii = np.linspace(0.5, 9.0, AGENTS)
        agents = np.concatenate([np.zeros(AGENTS), logit(radii / RADIUS_MAX)])
        positions, _, neighbors, *_ = simulate_trajectory(population, agents, 42)
        distances = np.linalg.norm(
            positions[0, :, None] - positions[0, None, :], axis=-1
        )
        expected = ((distances > 0) & (distances <= radii[:, None])).sum(axis=-1)
        np.testing.assert_array_equal(neighbors[1], expected)

    def test_observations_use_local_draws_without_resampling_from_population(self):
        population = self.population()
        agents = sample_agents(population)["agents"]
        changed = population.copy()
        changed[:4] += 1.0
        first = simulate_trajectory(population, agents, 73)
        second = simulate_trajectory(changed, agents, 73)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
        # Changing the supplied weights must affect the motion.
        different_agents = agents.copy()
        different_agents[:AGENTS] += 3.0
        different = simulate_trajectory(population, different_agents, 73)
        self.assertGreater(np.max(np.abs(first[0] - different[0])), 1e-5)

    def test_zero_spreads_collapse_agent_parameters_to_population_centers(self):
        population = self.population()
        population[[1, 3]] = -np.inf
        agents = sample_agents(population)["agents"]
        physical = physical_parameters({"population": population, "agents": agents})
        np.testing.assert_allclose(physical[6:18], physical[0], rtol=1e-6)
        np.testing.assert_allclose(physical[18:], physical[2], rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
