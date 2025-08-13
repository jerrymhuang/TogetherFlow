import os

if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "jax"

import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import bayesflow as bf

from ..src import TogetherFlowSimulator, GRU


if __name__ == "__main__":

    # Define simulator
    simulator = TogetherFlowSimulator(
        num_beacons=49,
        dt=0.1,
        time_horizon=100.
    )

    # Define adapter
    adapter = (
        bf.adapters.Adapter()
        .convert_dtype("float64", "float32")
        .as_time_series(["positions", "rotations", "neighbors"])
        .expand_dims(['w', 'r', 'v'], axis=-1)
        .concatenate(['w', 'r', 'v'], into="inference_variables")
        .concatenate(["positions", "rotations", "neighbors"], into="summary_variables", axis=-1)
    )

    # Define networks
    summary_net = GRU()
    inference_net = bf.networks.DiffusionModel()

    # Set up workflow
    workflow = bf.workflows.BasicWorkflow(
        simulator=simulator,
        adapter=adapter,
        summary_net=summary_net,
        inference_net=inference_net
    )

    # Either generate or retrieve dataset
    training_set = workflow.simulate(batch_shape=(10000,))
    validation_set = workflow.simulate(batch_shape=(500,))

    # Start training
    history = workflow.fit_offline(
        data=training_set,
        validation_data=validation_set,
        batch_size=64,
        epochs=300
    )

    # Diagnostics
    fig_size = (12, 4)

    figures = workflow.plot_default_diagnostics(
        test_data=500,
        loss_kwargs={"figsize": fig_size, "label_fontsize": 12},
        recovery_kwargs={"figsize": fig_size, "label_fontsize": 12},
        calibration_ecdf_kwargs={"figsize": fig_size, "legend_fontsize": 8, "difference": True, "label_fontsize": 12},
        z_score_contraction_kwargs={"figsize": fig_size, "label_fontsize": 12}
    )
