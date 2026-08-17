"""Run one variant end to end and emit an amortized-workflow results folder.

    uv run python experiments/run_variant.py v0-baseline
    uv run python experiments/run_variant.py --list
    uv run python experiments/run_variant.py --all          # the full night

Output per variant, under outputs/variants/<slug>/:
    checkpoints/model.keras, loss.png, recovery.png, calibration_ecdf.png,
    coverage.png, z_score_contraction.png, metrics.csv, history.json, report.md
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import argparse
import json
import logging
import pathlib
import sys
import time
import traceback

import matplotlib
matplotlib.use("Agg")           # unattended overnight runs have no display
import matplotlib.pyplot as plt
import numpy as np
import keras
import bayesflow as bf

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / ".claude" / "skills" / "amortized-workflow"))

from togetherflow import TogetherFlowSimulator
from togetherflow.simulator import (
    make_logit_random_walk_expander,
    make_ou_salience_process,
)
from togetherflow.networks import SummaryNet, TransformerSummaryNet
from variants import ALL_VARIANTS, BY_SLUG, PARAM_BOUNDS
from scripts.inspect_training import inspect_history
from scripts.check_diagnostics import check_diagnostics, suggest_next_steps
from report import write_report

ROOT = pathlib.Path(__file__).parent.parent
OUT_ROOT = ROOT / "outputs" / "variants"

FIGURE_NAMES = {
    "losses": "loss.png",
    "recovery": "recovery.png",
    "calibration_ecdf": "calibration_ecdf.png",
    "coverage": "coverage.png",
    "z_score_contraction": "z_score_contraction.png",
}

LATEX_NAMES = {
    "w": r"$w$", "r": r"$r$", "v": r"$v$",
    "noise": r"$\eta$", "alpha": r"$\alpha$", "kappa": r"$\kappa$",
    "w0": r"$w_0$", "tau": r"$\tau$",
    "mu_w": r"$\mu_w$", "sigma_w": r"$\sigma_w$",
}

# Posterior sampling for diagnostics defaults to one batch of
# (num_test_datasets * num_samples) rows — 300 x 1000 = 300,000 here. FlowMatching
# absorbs that; DiffusionModel does not, because adaptive stochastic integration
# keeps far more intermediate state per row, and v0-diffusion died requesting
# 4.47 GiB on a 12 GB card *after* training had completed. Chunking costs nothing
# numerically: each batch is an independent draw from the same posterior.
DIAGNOSTIC_BATCH_SIZE = 25
DIAGNOSTIC_KWARGS = {"approximator_kwargs": {"batch_size": DIAGNOSTIC_BATCH_SIZE}}

# BayesFlow 2.0.12 labels diagnostic rows differently from the names the skill's
# check_diagnostics() looks up. Without this mapping the lookups miss silently
# and every report loses its calibration and contraction ratings.
METRIC_ALIASES = {
    "Calibration Error": "ECE",
    "Posterior Contraction": "Post. Contraction",
    "Log Gamma": "Log-gamma",
}


def normalize_metric_index(metrics):
    renamed = metrics.rename(index=METRIC_ALIASES)
    expected = {"NRMSE", "ECE", "Post. Contraction"}
    missing = expected - set(renamed.index)
    if missing:
        logging.warning(
            "diagnostic rows %s not found (present: %s) — report ratings will be incomplete",
            sorted(missing), list(renamed.index),
        )
    return renamed


def build_simulator(variant, seed):
    if variant.expander == "static":
        expander = None                      # the simulator's default broadcast
    elif variant.expander == "random_walk_w":
        # tau lives in the last slot and nothing in the kernel reads it; it acts
        # only here, shaping the w path. That is why it must not be called
        # "alpha" or "kappa", which the kernel does read.
        expander = make_logit_random_walk_expander(
            w_col=variant.param_names.index("w0"),
            tau_col=variant.param_names.index("tau"),
            dt=variant.dt,
        )
    else:
        raise ValueError(f"unknown expander '{variant.expander}'")

    # Time-varying beacon salience. The paths are emitted whenever the process
    # is on, but only reach the network if the variant lists "salience" among
    # its channels — that difference is exactly the v8 observed/hidden pair.
    if variant.salience_sigma is None:
        salience_process = None
    else:
        salience_process = make_ou_salience_process(
            num_beacons=variant.num_beacons,
            dt=variant.dt,
            sigma_s=variant.salience_sigma,
            tau_s=variant.salience_tau,
        )

    return TogetherFlowSimulator(
        expander=expander,
        salience_process=salience_process,
        include_salience_paths=salience_process is not None,
        salience_sensitivity=variant.salience_sensitivity,
        switch_margin=variant.switch_margin,
        num_agents=variant.num_agents,
        num_beacons=variant.num_beacons,
        dt=variant.dt,
        time_horizon=variant.time_horizon,
        output_mode="flat",
        prior=variant.prior,
        param_names=variant.param_names,
        reference_radii=variant.reference_radii,
        beacon_strengths=variant.beacon_strengths,
        beacon_spread=variant.beacon_spread,
        relative_heading=variant.relative_heading,
        diffusive_heading=variant.diffusive_heading,
        repulsion_radius=variant.repulsion_radius,
        repulsion_gain=variant.repulsion_gain,
        seed=seed,
    )


def build_adapter(variant):
    """Explicit adapter — constraints first, then routing.

    .constrain() maps bounded parameters to an unconstrained space so the flow
    never has to place mass outside the support. This matters most for w and
    noise, which live on [0, 1].
    """
    adapter = bf.adapters.Adapter()
    # An arm whose prior leaves the default support declares its own bounds;
    # constraining to [0, 1] while the prior draws to 1.5 would put real prior
    # mass outside the region the flow is allowed to occupy.
    bounds_table = variant.param_bounds or PARAM_BOUNDS
    if variant.constrain_parameters:
        for name in variant.infer:
            bounds = bounds_table.get(name)
            if bounds:
                adapter = adapter.constrain(name, **bounds)
    return (
        adapter
        .convert_dtype("float64", "float32")
        .concatenate(list(variant.infer), into="inference_variables")
        .concatenate(list(variant.channels), into="summary_variables", axis=-1)
    )


def _train(workflow, variant, train_data, val_data, results_dir):
    """Fit the workflow and persist the history. Returns the Keras History."""
    callbacks = []
    if variant.early_stopping_patience:
        # restore_best_weights is the point: without it the diagnostics run on
        # the final, over-trained weights rather than the best ones.
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=variant.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ))

    if variant.online:
        logging.info("[%s] ONLINE training %d epochs x %d batches...",
                     variant.slug, variant.epochs, variant.num_batches_per_epoch)
        history = workflow.fit_online(
            epochs=variant.epochs,
            batch_size=variant.batch_size,
            num_batches_per_epoch=variant.num_batches_per_epoch,
            validation_data=val_data,
            callbacks=callbacks,
            verbose=2,
        )
    else:
        logging.info("[%s] training %d epochs on %d sims...",
                     variant.slug, variant.epochs, variant.n_train)
        history = workflow.fit_offline(
            data=train_data,
            epochs=variant.epochs,
            batch_size=variant.batch_size,
            validation_data=val_data,
            callbacks=callbacks,
            verbose=2,
        )

    with open(results_dir / "history.json", "w") as f:
        json.dump(history.history, f)
    return history


def load_trained(workflow, results_dir):
    """Restore a completed training run so diagnostics can be redone alone.

    Training writes checkpoints/model.keras before diagnostics run, so an arm
    that trains successfully and then fails in diagnostics — the v0-diffusion
    OOM — does not need its 35 minutes of training repeated.
    """
    path = results_dir / "checkpoints" / "model.keras"
    if not path.exists():
        raise FileNotFoundError(f"no trained checkpoint at {path}")
    workflow.approximator = keras.saving.load_model(str(path))
    logging.info("[%s] restored trained approximator from %s", results_dir.name, path)
    return workflow


def run(variant, force=False, diagnostics_only=False):
    results_dir = OUT_ROOT / variant.slug
    if (results_dir / "report.md").exists() and not force and not diagnostics_only:
        logging.info("[%s] report.md exists — skipping (use --force to rerun)", variant.slug)
        return "skipped"
    results_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    logging.info("[%s] %s", variant.slug, variant.title)

    # ── Simulate the offline bank ────────────────────────────────────────────
    # One bank, split into train/val/test. Offline rather than online: the
    # simulator is fast enough that regenerating every epoch costs ~8x more
    # wall-clock than training on a fixed bank.
    sim = build_simulator(variant, seed=variant.seed)
    # Online training generates its own training batches, so only the validation
    # and test splits need to come from a bank.
    total = (variant.n_val + variant.n_test) if variant.online else \
            (variant.n_train + variant.n_val + variant.n_test)
    logging.info("[%s] simulating %d datasets...", variant.slug, total)
    t0 = time.time()
    data = sim.sample(batch_size=total)
    # The kernel emits float64, but the adapter casts to float32 before the
    # networks ever see it. Casting the bank up front halves resident memory,
    # which is what decides whether the night survives unattended.
    data = {k: np.asarray(v, dtype=np.float32) for k, v in data.items()}
    gb = sum(v.nbytes for v in data.values()) / 1e9
    logging.info("[%s] simulated in %.1fs — bank %.2f GB", variant.slug, time.time() - t0, gb)

    if variant.online:
        train_data = None
        val_data  = {k: v[:variant.n_val] for k, v in data.items()}
        test_data = {k: v[variant.n_val:] for k, v in data.items()}
    else:
        i, j = variant.n_train, variant.n_train + variant.n_val
        train_data = {k: v[:i] for k, v in data.items()}
        val_data   = {k: v[i:j] for k, v in data.items()}
        test_data  = {k: v[j:] for k, v in data.items()}
    del data

    # ── Networks ─────────────────────────────────────────────────────────────
    # summary_dim follows the 3x-parameters heuristic; the inference net uses the
    # workflow's Base sizing rather than the smaller net used in earlier runs.
    n_params = len(variant.infer)
    summary_dim = 3 * n_params
    if variant.summary_net == "bdlstm":
        summary_net = SummaryNet(summary_dim=summary_dim)
    elif variant.summary_net == "transformer":
        summary_net = TransformerSummaryNet(summary_dim=summary_dim)
    else:
        raise ValueError(f"unknown summary_net '{variant.summary_net}'")
    # Matched subnet across both estimators, so v0-diffusion differs from
    # v0-reference in the generative process and nothing else that we control.
    subnet_kwargs = {"widths": (256,) * 4, "time_embedding_dim": 32}
    if variant.inference_net == "flow_matching":
        inference_net = bf.networks.FlowMatching(subnet_kwargs=subnet_kwargs)
    elif variant.inference_net == "diffusion":
        # Required by CompositionalWorkflow, which the partial-pooling work needs.
        inference_net = bf.networks.DiffusionModel(subnet_kwargs=subnet_kwargs)
    else:
        raise ValueError(f"unknown inference_net '{variant.inference_net}'")

    workflow = bf.workflows.BasicWorkflow(
        simulator=sim,
        adapter=build_adapter(variant),
        summary_network=summary_net,
        inference_network=inference_net,
        standardize="all",
        checkpoint_filepath=str(results_dir / "checkpoints"),
        # Deliberately False: diagnostics are computed on the FINAL epoch.
        #
        # Best-epoch checkpointing would give better numbers — several arms
        # degrade after their minimum, v0-diffusion worst at 0.864 (epoch 229)
        # against 1.482 at epoch 300 — but it would put these arms on a
        # different protocol from the thirteen arms of the 2026-08-09 night that
        # are not being re-run, and every one of those is read against
        # v0-reference. A baseline measured differently from the arms compared
        # to it is a worse problem than a suboptimal but uniform protocol.
        # Revisit only if the whole set is re-run together.
        save_best_only=False,
    )

    # ── Train ────────────────────────────────────────────────────────────────
    if diagnostics_only:
        workflow = load_trained(workflow, results_dir)
        with open(results_dir / "history.json") as f:
            history_dict = json.load(f)
        training_report = inspect_history(history_dict)
        history = None
    else:
        history = _train(workflow, variant, train_data, val_data, results_dir)
        training_report = inspect_history(history.history)

    # ── Diagnostics ──────────────────────────────────────────────────────────
    logging.info("[%s] computing diagnostics...", variant.slug)
    metrics = normalize_metric_index(
        workflow.compute_default_diagnostics(
            test_data=test_data, as_data_frame=True, **DIAGNOSTIC_KWARGS
        )
    )
    # index=True: the row labels ARE the metric names and check_diagnostics
    # looks them up by name.
    metrics.to_csv(results_dir / "metrics.csv")

    var_names = [LATEX_NAMES.get(p, p) for p in variant.infer]
    figures = workflow.plot_default_diagnostics(
        test_data=test_data,
        variable_names=var_names,
        **DIAGNOSTIC_KWARGS,
    )
    for key, fig in figures.items():
        fig.savefig(results_dir / FIGURE_NAMES[key], dpi=150, bbox_inches="tight")
        plt.close(fig)

    diag_report = check_diagnostics(metrics)
    next_steps = suggest_next_steps(training_report, diag_report)

    elapsed = time.time() - t_start
    write_report(
        results_dir=results_dir,
        variant=variant,
        metrics=metrics,
        training_report=training_report,
        diag_report=diag_report,
        next_steps=next_steps,
        summary_dim=summary_dim,
        elapsed_s=elapsed,
    )
    logging.info("[%s] done in %.1f min -> %s", variant.slug, elapsed / 60, results_dir)
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slug", nargs="?", help="variant slug to run")
    ap.add_argument("--all", action="store_true", help="run every variant in order")
    ap.add_argument("--list", action="store_true", help="list variants and exit")
    ap.add_argument("--force", action="store_true", help="rerun even if report.md exists")
    ap.add_argument("--diagnostics-only", action="store_true",
                    help="skip training, restore checkpoints/model.keras and redo diagnostics")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.list:
        for v in ALL_VARIANTS:
            print(f"  {v.slug:<28} {v.addresses:<28} {v.title}")
        return

    if args.all:
        results = {}
        for v in ALL_VARIANTS:
            try:
                results[v.slug] = run(v, force=args.force, diagnostics_only=args.diagnostics_only)
            except Exception:
                # One bad variant must not take down the night.
                logging.error("[%s] FAILED:\n%s", v.slug, traceback.format_exc())
                results[v.slug] = "failed"
        print("\n=== NIGHT SUMMARY ===")
        for slug, status in results.items():
            print(f"  {status:<8} {slug}")
        return

    if not args.slug:
        ap.error("pass a slug, --all, or --list")
    if args.slug not in BY_SLUG:
        ap.error(f"unknown variant '{args.slug}'. Use --list.")
    run(BY_SLUG[args.slug], force=args.force, diagnostics_only=args.diagnostics_only)


if __name__ == "__main__":
    main()
