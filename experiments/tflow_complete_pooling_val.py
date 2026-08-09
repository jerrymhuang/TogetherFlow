import os

os.environ["KERAS_BACKEND"] = "jax"

import keras
import logging
import pathlib

import matplotlib.pyplot as plt
import bayesflow as bf

from togetherflow import TogetherFlowSimulator

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")

ROOT     = pathlib.Path(__file__).parent.parent
CKPT_DIR = ROOT / "outputs" / "checkpoints"
FIG_DIR  = ROOT / "outputs" / "figures"
RES_DIR  = ROOT / "outputs" / "results"

# ── Must match the training run ───────────────────────────────────────────────
EXP_NAME = "tflow_complete_pooling"
NET_TAG  = "bdlstm_fm"
EPOCHS   = 100

NUM_AGENTS       = 49
NUM_BEACONS      = 4
DT               = 0.1
TIME_HORIZON     = 60.
TEST_SIZE        = 300
NUM_POST_SAMPLES = 1_000

COLOR     = "#6969ff"
VAR_NAMES = [r"$w$", r"$r$", r"$v$", r"$\eta$"]
FIG_SIZE  = (16, 4)

RUN_NAME = f"{EXP_NAME}_{NET_TAG}_{EPOCHS}"


if __name__ == "__main__":

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt_path = CKPT_DIR / RUN_NAME / "model.keras"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}.\n"
            f"Run tflow_complete_pooling_train.py first with matching EXP_NAME/NET_TAG/EPOCHS."
        )
    logging.info("Loading estimator from %s", ckpt_path)
    estimator = keras.saving.load_model(str(ckpt_path))

    # ── Test data ─────────────────────────────────────────────────────────────
    simulator = TogetherFlowSimulator(
        num_agents=NUM_AGENTS,
        num_beacons=NUM_BEACONS,
        dt=DT,
        time_horizon=TIME_HORIZON,
        output_mode="flat",
    )

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

    logging.info("Generating %d test samples...", TEST_SIZE)
    test_sims = simulator.sample(batch_size=TEST_SIZE)

    # ── Posterior samples ─────────────────────────────────────────────────────
    logging.info("Drawing %d posterior samples per observation...", NUM_POST_SAMPLES)
    post_draws = estimator.sample(conditions=test_sims, num_samples=NUM_POST_SAMPLES)

    # ── Save diagnostics ──────────────────────────────────────────────────────
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    plots = {
        "recovery": bf.diagnostics.plots.recovery(
            estimates=post_draws,
            targets=test_sims,
            variable_names=VAR_NAMES,
            figsize=FIG_SIZE,
            color=COLOR,
            label_fontsize=16,
            title_fontsize=20,
        ),
        "calibration_ecdf": bf.diagnostics.plots.calibration_ecdf(
            estimates=post_draws,
            targets=test_sims,
            variable_names=VAR_NAMES,
            figsize=FIG_SIZE,
            difference=False,
            legend_fontsize=16,
            label_fontsize=12,
            rank_ecdf_color=COLOR,
        ),
        "coverage": bf.diagnostics.plots.coverage(
            estimates=post_draws,
            targets=test_sims,
            variable_names=VAR_NAMES,
            figsize=FIG_SIZE,
            color=COLOR,
            label_fontsize=16,
            legend_fontsize=12,
            title_fontsize=20,
            difference=False,
        ),
        "z_score_contraction": bf.diagnostics.plots.z_score_contraction(
            estimates=post_draws,
            targets=test_sims,
            variable_names=VAR_NAMES,
            figsize=FIG_SIZE,
            label_fontsize=12,
            color=COLOR,
        ),
    }

    for plot_name, fig in plots.items():
        path = FIG_DIR / f"{RUN_NAME}_{plot_name}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logging.info("Saved %s", path.name)
