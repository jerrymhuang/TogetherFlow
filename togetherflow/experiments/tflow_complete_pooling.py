import os

os.environ["KERAS_BACKEND"] = "jax"

import pathlib
import logging

import numpy as np
import matplotlib.pyplot as plt
import bayesflow as bf

from src import TogetherFlowSimulator, SummaryNet


def save_npz_dict(d, path):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # flatten first level of dict into a single compressed NPZ
    np.savez_compressed(p, **{k: np.asarray(v) for k, v in d.items()})

def load_npz_dict(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


if __name__ == "__main__":

    debug = True
    gather = True
    epochs = 100
    return_summary = True

    # Define simulator
    simulator = TogetherFlowSimulator(
        num_agents=49,
        num_beacons=4,
        dt=0.1,
        time_horizon=60.,
        downsample_factor=1.,
        gather=gather,
        return_summary=False,
    )

    sim = simulator.sample(batch_size=2)

    print(sim["w"].shape)
    print(sim["positions"].shape)
    print(sim["rotations"].shape)
    print(np.concatenate((sim["positions"], sim["rotations"]), axis=2).shape)

    # Define adapter
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

    if debug:
        sample = adapter(simulator.sample(2))
        print(sample["summary_variables"].shape)

    # Define networks
    # summary_net = SummaryNet(keras.Sequential([
    #     keras.layers.Conv1D(filters=32, kernel_size=2, strides=2, activation="swish"),
    #     keras.layers.LayerNormalization(),
    #     keras.layers.SpatialDropout1D(0.1),
    #
    #     keras.layers.Conv1D(filters=32, kernel_size=2, strides=2, activation="swish"),
    #     keras.layers.LayerNormalization(),
    #     keras.layers.SpatialDropout1D(0.1),
    #
    #     keras.layers.LSTM(512, return_sequences=True, recurrent_dropout=0.1),
    #     keras.layers.GlobalAveragePooling1D(),
    #     keras.layers.Dropout(0.1),
    #     keras.layers.Dense(64, activation="swish"),
    # ]))

    summary_net = SummaryNet()

    # summary_net = bf.networks.TimeSeriesNetwork(dropout=0.2)
    # summary_net = HierarchicalNetwork([
    #     keras.layers.TimeDistributed(keras.layers.LSTM(512)),
    #     keras.layers.Lambda(lambda x: keras.ops.mean(x, axis=1))
    # ])

    # summary_net = SummaryNet(keras.Sequential([
    #     keras.layers.Conv1D(filters=32, kernel_size=3, strides=2, activation="swish"),
    #     keras.layers.Conv1D(filters=32, kernel_size=3, strides=2, activation="swish"),
    #     keras.layers.LayerNormalization(),
    #     keras.layers.LSTM(512),
    #     keras.layers.LayerNormalization(),
    #     keras.layers.Dropout(0.2),
    #     keras.layers.Dense(64, activation="swish"),
    # ]))

    # summary_net = GRU()
    # inference_net = bf.networks.CouplingFlow(depth=2, transform="spline")
    inference_net = bf.networks.FlowMatching(subnet_kwargs={"widths": (128,)*3})
    # inference_net = bf.networks.DiffusionModel()

    # Set up workflow
    workflow = bf.workflows.BasicWorkflow(
        simulator=simulator,
        adapter=adapter,
        summary_network=summary_net,
        inference_network=inference_net,
        checkpoint_filepath=f"../checkpoints/tflow_complete_pooling_bdlstm_fm_3e4_{epochs}"
    )

    outdir = pathlib.Path("dataset")
    figure_dir = pathlib.Path("figures")
    train_path = outdir / ("train_3e4s.npz" if gather else "train_0.npz")
    val_path   = outdir / ("val_3e4s.npz" if gather else "val_0.npz")
    meta_path  = outdir / "meta.json"

    if train_path.exists() and val_path.exists():
        # training_set   = load_npz_dict(train_path)
        validation_set = load_npz_dict(val_path)
    else:
        # logging.info("Generating training set...")
        # training_set   = workflow.simulate((30000,))
        logging.info("Generating validation set...")
        validation_set = workflow.simulate(300)
        # save_npz_dict(training_set, train_path)
        save_npz_dict(validation_set, val_path)

    # # Start training
    # history = workflow.fit_offline(
    #     data=training_set,
    #     validation_data=validation_set,
    #     batch_size=64,
    #     epochs=epochs
    # )

    history = workflow.fit_online(
        epochs=100,
        batch_size=32,
        num_batches_per_epoch=100
    )

    # Diagnostics
    fig_size = (16, 4)

    metrics = workflow.compute_default_diagnostics(test_data=300)
    print(metrics)
    metrics.to_csv(f"./results/tflow_complete_pooling_bdfm{epochs}_3e4.csv", index=False)

    color = "#4e2a84"

    figures = workflow.plot_default_diagnostics(
        test_data=300,
        variable_names=[r"$w$", r"$r$", r"$v$", r"$\eta$"],
        loss_kwargs={"figsize": fig_size, "label_fontsize": 12, "train_color": color},
        recovery_kwargs={"figsize": fig_size, "label_fontsize": 12, "color": color},
        coverage_kwargs={
            "figsize": fig_size,
            "color": color,
            "label_fontsize": 16,
            "legend_fontsize": 11,
            "difference": False
        },
        calibration_ecdf_kwargs={
            "figsize": fig_size,
            "legend_fontsize": 16,
            "difference": False,
            "label_fontsize": 11,
            "rank_ecdf_color": color
        },
        z_score_contraction_kwargs={"figsize": fig_size, "label_fontsize": 12, "color": color},
    )

    for plot_name, fig in figures.items():
        fig_path = figure_dir / f"tflow_complete_pooling_{plot_name}_bdfm{epochs}_3e4.png"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logging.info(f"Saved diagnostic plot to {fig_path}")
