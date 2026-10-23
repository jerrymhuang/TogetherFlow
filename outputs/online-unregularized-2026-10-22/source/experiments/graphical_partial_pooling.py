"""Graph nodes for the 12-agent weight/radius hierarchy.

Agent parameters form one vector node: their shared trajectory couples their
posteriors. All graph sampling functions are module-level for model reloads.
"""

import keras
import numpy as np
from numba import njit
from scipy.special import expit, logit
from bayesflow.experimental.graphical_simulator import GraphicalSimulator
from bayesflow.experimental.graphical_approximator import GraphicalApproximator
import bayesflow as bf

from togetherflow.simulator import simulator_fun
from run_agent_radius_pooling import AGENTS, RADIUS_MAX, observed_features
from run_agent_partial_pooling import BEACONS, DT

DURATION = 20.0
POPULATION_SCALES = np.array([1.0, 1.5, RADIUS_MAX, 0.75, 2.0, 1.0])


def sample_population():
    """Return transformed (mu_w, sigma_w, mu_r, sigma_r, v, eta)."""
    mu_r = np.random.lognormal(np.log(2.5), 0.35)
    while mu_r >= RADIUS_MAX:
        mu_r = np.random.lognormal(np.log(2.5), 0.35)
    physical = np.array(
        [
            np.random.beta(2, 2),
            1.5 * np.random.beta(1, 3),
            mu_r,
            0.75 * np.random.beta(2, 2),
            2.0 * np.random.beta(2, 2),
            np.random.beta(2, 2),
        ]
    )
    return {
        "population": logit(
            np.clip(physical / POPULATION_SCALES, 1e-6, 1 - 1e-6)
        ).astype("float32")
    }


def sample_agents(population):
    """Draw all agent weights/radii conditional on the population parameters."""
    mu_w, sigma_w, mu_r, sigma_r, _, _ = expit(population) * POPULATION_SCALES
    weights = np.random.normal(logit(mu_w), sigma_w, AGENTS)
    radii = np.random.normal(logit(mu_r / RADIUS_MAX), sigma_r, AGENTS)
    return {"agents": np.concatenate([weights, radii]).astype("float32")}


@njit
def simulate_trajectory(population, agents, seed):
    """Run the existing motion kernel with the graph's exact latent values."""
    np.random.seed(seed)
    physical = POPULATION_SCALES / (1 + np.exp(-population))
    weights = 1 / (1 + np.exp(-agents[:AGENTS]))
    radii = RADIUS_MAX / (1 + np.exp(-agents[AGENTS:]))
    theta = np.empty((int(DURATION / DT), 4))
    theta[:, 0] = physical[0]
    theta[:, 1] = physical[2]
    theta[:, 2] = physical[4]
    theta[:, 3] = physical[5]
    return simulator_fun(
        theta,
        num_agents=AGENTS,
        num_beacons=len(BEACONS),
        dt=DT,
        time_horizon=DURATION,
        beacon_strengths=np.ones(len(BEACONS), dtype=np.float32),
        beacon_strength_path=np.zeros((0, len(BEACONS))),
        reference_radii=np.zeros(0),
        relative_heading=True,
        diffusive_heading=True,
        obstacles=np.zeros((0, 3)),
        init_positions=np.zeros((0, 2)),
        init_rotations=np.zeros(0),
        fixed_beacons=BEACONS,
        beacon_assignment=np.zeros(0, dtype=np.int64),
        fixed_agent_weights=weights,
        fixed_agent_radii=radii,
    )


def sample_observations(population, agents):
    seed = int(np.random.randint(0, 2**31 - 1))
    positions, rotations, *_ = simulate_trajectory(population, agents, seed)
    features = observed_features(positions[None], rotations[None]).reshape(-1)
    # The graphical API expects an observation axis, even for one group.
    return {"features": features[None, :]}


def build_simulator():
    simulator = GraphicalSimulator()
    simulator.add_node("population", sample_fn=sample_population)
    simulator.add_node("agents", sample_fn=sample_agents)
    simulator.add_node("observations", sample_fn=sample_observations)
    simulator.add_edge("population", "agents")
    simulator.add_edge("population", "observations")
    simulator.add_edge("agents", "observations")
    return simulator


def build_approximator(simulator, widths=(256, 256, 256)):
    graph = simulator.graph.invert()
    # Flatten the singleton observation axis without learning a new summary.
    summary_shapes = graph.summary_network_input_shapes()
    summaries = [keras.Sequential([keras.layers.Flatten()]) for _ in summary_shapes]
    networks = [
        bf.networks.FlowMatching(
            subnet_kwargs={"widths": widths, "time_embedding_dim": 32}
        )
        for _ in graph.network_composition()
    ]
    # FlowMatching.compute_metrics requires its base distribution to be built.
    # GraphicalApproximator does not build these networks automatically.
    shapes = simulator.graph.output_shapes()
    variables = simulator.graph.variable_names()
    for index, network in enumerate(networks):
        target_size = int(graph.inference_variable_shapes()[index][-1])
        condition_size = sum(
            int(np.prod(shapes[name][1:]))
            for node in graph.network_conditions()[index]
            for name in variables[node]
        )
        condition_size += len(shapes["features"]) - 2
        network.build((None, target_size), conditions_shape=(None, condition_size))
    return GraphicalApproximator(
        graph,
        inference_networks=networks,
        summary_networks=summaries,
        standardize="all",
    )
