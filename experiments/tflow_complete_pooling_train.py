import os

os.environ["KERAS_BACKEND"] = "jax"

import logging
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import bayesflow as bf

from togetherflow import TogetherFlowSimulator, SummaryNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")

ROOT     = pathlib.Path(__file__).parent.parent
CKPT_DIR = ROOT / "outputs" / "checkpoints"
FIG_DIR  = ROOT / "outputs" / "figures"
RES_DIR  = ROOT / "outputs" / "results"
DATA_DIR = ROOT / "data"

# ── Run configuration ─────────────────────────────────────────────────────────
EXP_NAME = "tflow_complete_pooling"
NET_TAG  = "bdlstm_fm"
EPOCHS   = 100
ONLINE   = True

BATCH_SIZE            = 64
NUM_BATCHES_PER_EPOCH = 50    # online only
TRAIN_SIZE            = 3_000 # offline only
VAL_SIZE              = 300   # offline only

# ── Simulator ─────────────────────────────────────────────────────────────────
NUM_AGENTS   = 49
NUM_BEACONS  = 4
DT           = 0.1
TIME_HORIZON = 60.

# ── Diagnostics ───────────────────────────────────────────────────────────────
COLOR     = "#6969ff"
VAR_NAMES = [r"$w$", r"$r$", r"$v$", r"$\eta$"]
FIG_SIZE  = (16, 4)

RUN_NAME = f"{EXP_NAME}_{NET_TAG}_{EPOCHS}"


def save_npz(d, path):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in d.items()})


def load_npz(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


if __name__ == "__main__":

    # ── Simulator ─────────────────────────────────────────────────────────────
    simulator = TogetherFlowSimulator(
        num_agents=NUM_AGENTS,
        num_beacons=NUM_BEACONS,
        dt=DT,
        time_horizon=TIME_HORIZON,
        output_mode="flat",
    )

    # ── Adapter ───────────────────────────────────────────────────────────────
    adapter = (
        bf.adapters.Adapter()
        .convert_dtype("float64", "float32")
        .concatenate(["w", "r", "v", "noise"], into="inference_variables")
        .concatenate([
            "positions",
            "rotations",
            "neighbors",
            "distances",
            "angular_velocities",
            "neighbor_fluctuations",
        ], into="summary_variables", axis=-1)
    )

    # ── Networks ──────────────────────────────────────────────────────────────
    summary_net   = SummaryNet()
    inference_net = bf.networks.FlowMatching(subnet_kwargs={"widths": (128,) * 3})

    # ── Workflow ──────────────────────────────────────────────────────────────
    workflow = bf.workflows.BasicWorkflow(
        simulator=simulator,
        adapter=adapter,
        summary_network=summary_net,
        inference_network=inference_net,
        checkpoint_filepath=str(CKPT_DIR / RUN_NAME),
    )

    # ── Training ──────────────────────────────────────────────────────────────
    if ONLINE:
        logging.info("Online training — %d epochs × %d batches × %d", EPOCHS, NUM_BATCHES_PER_EPOCH, BATCH_SIZE)
        workflow.fit_online(
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            num_batches_per_epoch=NUM_BATCHES_PER_EPOCH,
        )
        test_data = VAL_SIZE

    else:
        train_path = DATA_DIR / f"train_{EXP_NAME}.npz"
        val_path   = DATA_DIR / f"val_{EXP_NAME}.npz"

        if train_path.exists() and val_path.exists():
            logging.info("Loading cached dataset from %s", DATA_DIR)
            training_set   = load_npz(train_path)
            validation_set = load_npz(val_path)
        else:
            logging.info("Generating training set (%d samples)...", TRAIN_SIZE)
            training_set = workflow.simulate(TRAIN_SIZE)
            logging.info("Generating validation set (%d samples)...", VAL_SIZE)
            validation_set = workflow.simulate(VAL_SIZE)
            save_npz(training_set, train_path)
            save_npz(validation_set, val_path)
            logging.info("Dataset saved to %s", DATA_DIR)

        logging.info("Offline training — %d epochs", EPOCHS)
        workflow.fit_offline(
            data=training_set,
            validation_data=validation_set,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
        )
        test_data = validation_set

    # ── Diagnostics ───────────────────────────────────────────────────────────
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("Computing diagnostics...")
    metrics = workflow.compute_default_diagnostics(test_data=test_data)
    logging.info("\n%s", metrics.to_string())
    metrics.to_csv(RES_DIR / f"{RUN_NAME}_metrics.csv", index=False)

    figures = workflow.plot_default_diagnostics(
        test_data=test_data,
        variable_names=VAR_NAMES,
        loss_kwargs={
            "figsize": FIG_SIZE, "label_fontsize": 16, "train_color": COLOR,
        },
        recovery_kwargs={
            "figsize": FIG_SIZE, "label_fontsize": 16, "title_fontsize": 20, "color": COLOR,
        },
        coverage_kwargs={
            "figsize": FIG_SIZE, "color": COLOR, "label_fontsize": 16,
            "legend_fontsize": 12, "title_fontsize": 20, "difference": False,
        },
        calibration_ecdf_kwargs={
            "figsize": FIG_SIZE, "legend_fontsize": 16, "difference": False,
            "label_fontsize": 12, "rank_ecdf_color": COLOR,
        },
        z_score_contraction_kwargs={
            "figsize": FIG_SIZE, "label_fontsize": 12, "color": COLOR,
        },
    )

    for plot_name, fig in figures.items():
        path = FIG_DIR / f"{RUN_NAME}_{plot_name}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logging.info("Saved %s", path.name)
