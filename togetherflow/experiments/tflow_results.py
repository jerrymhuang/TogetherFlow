import os
os.environ["KERAS_BACKEND"] = "jax"

import keras
import pathlib
import logging
import numpy as np
import pandas as pd
import bayesflow as bf
import matplotlib.pyplot as plt

from src.simulator import TogetherFlowSimulator


def load_npz_dict(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}

if __name__ == "__main__":

    epochs = 100
    fig_dir = pathlib.Path("figures")
    fig_size = (16, 4)
    color = "#4e2a84"
    variable_names = [r"$w$", r"$r$", r"$v$", r"$\eta$"]

    # Define simulator
    simulator = TogetherFlowSimulator(
        num_agents=49,
        num_beacons=4,
        dt=0.1,
        time_horizon=60.,
        downsample_factor=1.,
        gather=True,
        return_summary=False,
    )

    estimator = keras.saving.load_model("../checkpoints/tflow_complete_pooling_bdlstm_fm_3e4_100_summary/model.keras")
    print("success")

    adapter = (
        bf.adapters.Adapter()
        .convert_dtype("float64", "float32")
        .concatenate(['w', 'r', 'v', "noise"], into="inference_variables")
        .concatenate([
            "positions",
            "rotations",
            "neighbors",
            "distances",
            #"max_dists",
            "angular_velocities",
            "neighbor_fluctuations"
        ], into="summary_variables", axis=-1)
    )

    summary_net = bf.networks.DeepSet()
    inference_net = bf.networks.FlowMatching(subnet_kwargs={"widths": (128,)*3})

    workflow = bf.workflows.BasicWorkflow(
        simulator=simulator,
        adapter=adapter,
        summary_network=summary_net,
        inference_network=inference_net,
    )

    metrics = workflow.compute_default_diagnostics(test_data=300)
    print(metrics)
    metrics.to_csv(f"./results/tflow_complete_pooling_bdlstm_fm{epochs}_3e4_summary.csv", index=False)

    val_sims = simulator.sample(batch_size=200)
    post_draws = estimator.sample(conditions=val_sims, num_samples=300)

    f = bf.diagnostics.plots.recovery(
        estimates=post_draws,
        targets=val_sims,
        variable_names=variable_names
    )

    fig_path = fig_dir / f"tflow_complete_pooling_bdlstm_fm{epochs}_3e4_recovery.png"
    f.savefig(fig_path)
    plt.close(f)
    logging.info(f"Saved plot at {fig_path}")

    # figures = workflow.plot_default_diagnostics(
    #     test_data=300,
    #     variable_names=[r"$w$", r"$r$", r"$v$", r"$\eta$"],
    #     loss_kwargs={"figsize": fig_size, "label_fontsize": 16, "train_color": color},
    #     recovery_kwargs={
    #         "figsize": fig_size,
    #         "label_fontsize": 16,
    #         "title_fontsize": 20,
    #         "color": color
    #     },
    #     coverage_kwargs={
    #         "figsize": fig_size,
    #         "color": color,
    #         "label_fontsize": 16,
    #         "legend_fontsize": 12,
    #         "title_fontsize": 20,
    #         "difference": False
    #     },
    #     calibration_ecdf_kwargs={
    #         "figsize": fig_size,
    #         "legend_fontsize": 16,
    #         "difference": False,
    #         "label_fontsize": 11,
    #         "rank_ecdf_color": color
    #     },
    #     z_score_contraction_kwargs={"figsize": fig_size, "label_fontsize": 12, "color": color},
    # )
    #
    # for plot_name, fig in figures.items():
    #     fig_path = figure_dir / f"tflow_complete_pooling_{plot_name}_bdfm{epochs}_3e4_summary.png"
    #     fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    #     plt.close(fig)
    #     logging.info(f"Saved diagnostic plot to {fig_path}")