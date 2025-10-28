import os

os.environ["KERAS_BACKEND"] = "jax"

import pathlib
import logging

import keras
import numpy as np
import matplotlib.pyplot as plt
import bayesflow as bf

from src import TogetherFlowSimulator, GRU, SummaryNet, TogetherNet, HierarchicalNetwork


def save_npz_dict(d, path):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # flatten first level of dict into a single compressed NPZ
    np.savez_compressed(p, **{k: np.asarray(v) for k, v in d.items()})

def load_npz_dict(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


if __name__ == "__main__":

    debug = False
    gather = True

    # Define simulator
    simulator = TogetherFlowSimulator(
        num_agents=49,
        num_beacons=4,
        dt=0.1,
        time_horizon=60.,
        downsample_factor=1.,
        gather=gather
    )

    # Define adapter
    adapter = (
        bf.adapters.Adapter()
        .convert_dtype("float64", "float32")
        .concatenate(['w', 'r', 'v'], into="inference_variables")
        .concatenate([
            "positions",
            "rotations",
            "neighbors",
            "distances",
            "angular_velocities",
            "neighbor_fluctuations"
        ], into="summary_variables", axis=-1)
    )

    if debug:
        sample = adapter(simulator.sample((1,)))
        print(sample["summary_variables"].shape)

    # Define networks
    summary_net = SummaryNet(keras.Sequential([
        keras.layers.Conv1D(filters=32, kernel_size=2, strides=2, activation="swish"),
        keras.layers.Conv1D(filters=32, kernel_size=2, strides=2, activation="swish"),
        keras.layers.LSTM(512),
        keras.layers.Dense(64)
    ]))

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

    inference_net = bf.networks.FlowMatching()

    # Set up workflow
    workflow = bf.workflows.BasicWorkflow(
        simulator=simulator,
        adapter=adapter,
        summary_network=summary_net,
        inference_network=inference_net
    )

    outdir = pathlib.Path("dataset")
    figure_dir = pathlib.Path("figures")
    train_path = outdir / ("train.npz" if gather else "train_0.npz")
    val_path   = outdir / ("val.npz" if gather else "val_0.npz")
    meta_path  = outdir / "meta.json"

    if train_path.exists() and val_path.exists():
        training_set   = load_npz_dict(train_path)
        validation_set = load_npz_dict(val_path)
    else:
        logging.info("Generating training set...")
        training_set   = workflow.simulate((5000,))
        logging.info("Generating validation set...")
        validation_set = workflow.simulate((300,))
        save_npz_dict(training_set, train_path)
        save_npz_dict(validation_set, val_path)
        # meta = dict(
        #     version="v1",
        #     created=time.strftime("%Y-%m-%d %H:%M:%S"),
        #     simulator="TogetherFlowSimulator",
        #     train_batch_shape="(10000,)",
        #     val_batch_shape="(200,)",
        # )
        # meta_path.parent.mkdir(parents=True, exist_ok=True)
        # meta_path.write_text(json.dumps(meta, indent=2))

    # Start training
    history = workflow.fit_offline(
        data=training_set,
        validation_data=validation_set,
        batch_size=32,
        epochs=10
    )

    # Diagnostics
    fig_size = (12, 4)

    figures = workflow.plot_default_diagnostics(
        test_data=validation_set,
        variable_names=["w", "r", "v"],
        loss_kwargs={"figsize": fig_size, "label_fontsize": 12},
        recovery_kwargs={"figsize": fig_size, "label_fontsize": 12},
        calibration_ecdf_kwargs={"figsize": fig_size, "legend_fontsize": 8, "difference": True, "label_fontsize": 12},
        z_score_contraction_kwargs={"figsize": fig_size, "label_fontsize": 12}
    )

    for plot_name, fig in figures.items():
        fig_path = figure_dir / f"tflow_complete_pooling_{plot_name}.png"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logging.info(f"Saved diagnostic plot to {fig_path}")
