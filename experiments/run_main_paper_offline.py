"""Regenerate the main-paper offline budgets using the revised radius-free inputs.

Run from the repository root with the reference environment and a CUDA device:
    python experiments/run_main_paper_offline.py

Training banks are memory mapped: 30K full trajectories exceed laptop RAM.
The 3K bank is a nested subset of the 30K bank; validation/test banks are shared.
Checkpoints and posterior draws are retained, and diagnostics reuse saved draws.
"""
import os

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("NUMBA_NUM_THREADS", "12")

import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bayesflow as bf
import jax
import keras
import matplotlib.pyplot as plt
import numpy as np

from run_variant import build_adapter, build_simulator, LATEX_NAMES, FIGURE_NAMES, normalize_metric_index
from togetherflow.networks import TransformerSummaryNet
from variable_radius_variants import BASE


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=str) + "\n")
    temporary.replace(path)


def bank(directory, variant, size, seed, chunk_size=64):
    """Generate reproducible chunks, flushing before advancing the resume cursor."""
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "bank.json"
    spec = dict(size=size, seed=seed, chunk_size=chunk_size,
                channels=list(variant.channels), infer=list(variant.infer),
                agents=variant.num_agents, horizon=variant.time_horizon, dt=variant.dt,
                prior=variant.prior.__name__, reference_radii=variant.reference_radii,
                relative_heading=variant.relative_heading, diffusive_heading=variant.diffusive_heading,
                source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                               for p in ("togetherflow/simulator.py", "togetherflow/priors.py")})
    completed = 0
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["spec"] != spec:
            raise ValueError(f"Bank specification changed: {directory}")
        completed = state["completed"]
    keys = list(variant.infer) + list(variant.channels)
    arrays = {k: np.load(directory / f"{k}.npy", mmap_mode="r+") for k in keys} if completed else {}
    for start in range(completed, size, chunk_size):
        stop = min(start + chunk_size, size)
        raw = build_simulator(variant, seed + start).sample(stop - start)
        for key in keys:
            values = np.asarray(raw[key], dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite simulated values: {key}, {start}")
            if key not in arrays:
                arrays[key] = np.lib.format.open_memmap(
                    directory / f"{key}.npy", mode="w+", dtype="float32",
                    shape=(size, *values.shape[1:]))
            arrays[key][start:stop] = values
            arrays[key].flush()
        del raw
        write_json(state_path, dict(spec=spec, completed=stop))
        logging.info("Bank %s: %d/%d", directory.name, stop, size)
    return {k: np.load(directory / f"{k}.npy", mmap_mode="r") for k in keys}


class EpochRecord(keras.callbacks.Callback):
    def __init__(self, directory):
        super().__init__()
        self.directory = directory
        self.rows = []
        self.started = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.rows.append({"epoch": epoch + 1, **{k: float(v) for k, v in (logs or {}).items()}})
        write_json(self.directory / "epochs.json", self.rows)
        write_json(self.directory / "status.json", dict(
            stage="training", epoch=epoch + 1, elapsed_seconds=time.time() - self.started,
            metrics=self.rows[-1]))


def manifest(variant, budget, args):
    config = asdict(replace(variant, online=False, n_train=budget, epochs=args.epochs,
                           batch_size=args.batch_size, early_stopping_patience=args.patience,
                           n_val=args.n_eval, n_test=args.n_eval))
    config["prior"] = variant.prior.__name__
    files = ["experiments/run_main_paper_offline.py", "experiments/run_variant.py",
             "experiments/variants.py", "experiments/variable_radius_variants.py",
             "togetherflow/simulator.py", "togetherflow/priors.py",
             "togetherflow/networks/transformer.py"]
    return dict(
        purpose="smoke validation only" if args.smoke else "main-paper revised offline recovery",
        variant=config, training_budget=budget, maximum_epochs=args.epochs,
        batch_size=args.batch_size, early_stopping_patience=args.patience,
        checkpoint_selection="minimum validation loss; restore best weights",
        validation_size=args.n_eval, test_size=args.n_eval, posterior_draws=args.draws,
        seeds={"training_bank": BASE.seed, "validation_bank": BASE.seed + 100000,
               "test_bank": BASE.seed + 200000, "network": BASE.seed,
               "posterior": BASE.seed + 300000, "diagnostics": BASE.seed + 400000},
        training_design="nested fixed training banks; shared independent validation and test banks",
        optimizer="BayesFlow offline default AdamW, weight decay 0.005, cosine schedule, 5% warmup",
        nrmse_normalization="prior", diagnostics_aggregation="BayesFlow default median",
        packages={p: importlib.metadata.version(p) for p in
                  ("bayesflow", "keras", "jax", "jaxlib", "numpy", "numba", "matplotlib", "pandas")},
        hardware=str(jax.devices()),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files},
        comparison_note="Online reference uses 480000 fresh simulations and Adam; the original paper also differs in dynamics, inputs, architecture and NRMSE definition.")


def run_budget(variant, budget, train, validation, test, directory, args):
    directory.mkdir(parents=True, exist_ok=True)
    spec = json.loads(json.dumps(manifest(variant, budget, args)))
    manifest_path = directory / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != spec:
        raise ValueError(f"Existing run specification differs: {directory}; use a new output directory")
    write_json(manifest_path, spec)
    if (directory / "complete.json").exists():
        logging.info("Already complete: %s", directory)
        return
    keras.utils.set_random_seed(BASE.seed)
    checkpoints = directory / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    workflow = bf.workflows.BasicWorkflow(
        simulator=build_simulator(variant, BASE.seed), adapter=build_adapter(variant),
        summary_network=TransformerSummaryNet(summary_dim=3 * len(variant.infer)),
        inference_network=bf.networks.FlowMatching(
            subnet_kwargs={"widths": (256,) * 4, "time_embedding_dim": 32}),
        standardize="all", checkpoint_filepath=str(checkpoints),
        checkpoint_name="best_model", save_best_only=True)
    final_model = checkpoints / "model.keras"
    if (directory / "training_complete.json").exists():
        workflow.approximator = keras.saving.load_model(final_model)
        workflow.history = keras.callbacks.History()
        workflow.history.history = json.loads((directory / "history.json").read_text())
    else:
        if (directory / "epochs.json").exists() and not args.restart_training:
            raise RuntimeError(f"Interrupted training at {directory}; use --restart-training to restart explicitly")
        logging.info("Training fixed bank N=%d, maximum %d epochs", budget, args.epochs)
        history = workflow.fit_offline(
            data={k: v[:budget] for k, v in train.items()}, epochs=args.epochs,
            batch_size=args.batch_size, validation_data=validation, verbose=2,
            callbacks=[EpochRecord(directory), keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=args.patience, restore_best_weights=True, verbose=1)])
        write_json(directory / "history.json", history.history)
        # EarlyStopping restores the validation-selected weights after the last epoch.
        workflow.approximator.save(final_model)
        write_json(directory / "training_complete.json", dict(
            epochs=len(history.history["loss"]),
            best_epoch=int(np.argmin(history.history["val_loss"])) + 1,
            best_val_loss=float(np.min(history.history["val_loss"]))))
    write_json(directory / "status.json", {"stage": "posterior sampling"})
    posterior_path = directory / "posterior.npz"
    if posterior_path.exists():
        with np.load(posterior_path) as saved:
            samples = {k: saved[f"samples_{k}"] for k in variant.infer}
            for k in variant.infer:
                np.testing.assert_array_equal(saved[f"truth_{k}"], test[k])
    else:
        # Small condition batches bound the memory needed for ODE integration.
        samples = workflow.sample(num_samples=args.draws, conditions=test, batch_size=25,
                                  seed=BASE.seed + 300000)
        samples = {k: np.asarray(samples[k]) for k in variant.infer}
        for k, value in samples.items():
            if not np.isfinite(value).all() or value.shape[:2] != (args.n_eval, args.draws):
                raise ValueError(f"Invalid posterior draws for {k}: {value.shape}")
        np.savez_compressed(posterior_path,
            **{f"samples_{k}": v for k, v in samples.items()},
            **{f"truth_{k}": test[k] for k in variant.infer})
    write_json(directory / "status.json", {"stage": "diagnostics"})
    # Use identical empirical-prior bootstrap draws for both training budgets.
    np.random.seed(BASE.seed + 400000)
    metrics = normalize_metric_index(workflow.compute_default_diagnostics(
        test_data=test, samples=samples, variable_keys=list(variant.infer),
        root_mean_squared_error_kwargs={"normalize": "prior"}))
    metrics.to_csv(directory / "metrics.csv")
    # Retain range normalization too, clearly labeled, for auditing. This keeps
    # BayesFlow's median aggregation; it is not the old paper's global-mean formula.
    metrics_range = workflow.compute_default_diagnostics(
        test_data=test, samples=samples, variable_keys=list(variant.infer),
        root_mean_squared_error_kwargs={"normalize": "range"})
    metrics_range.to_csv(directory / "metrics_range_normalized.csv")
    figures = workflow.plot_default_diagnostics(
        test_data=test, samples=samples, variable_keys=list(variant.infer),
        variable_names=[LATEX_NAMES[k] for k in variant.infer])
    for name, fig in figures.items():
        path = directory / FIGURE_NAMES.get(name, name + ".png")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
    (directory / "report.md").write_text(
        f"# Revised offline reference: {budget:,} simulations\n\n"
        "Fixed training bank; independent shared validation and test banks. "
        "Checkpoint selected by validation loss. Metrics and figures use the same saved posterior draws.\n\n"
        + metrics.to_string() + "\n\n"
        "Primary NRMSE is prior-normalized; the separate range-normalized CSV is for auditing. "
        "These runs change the original simulator dynamics, observable inputs, and summary architecture, "
        "so comparison with the original paper cannot isolate the effect of training budget.\n")
    write_json(directory / "complete.json", {"stage": "complete", "completed_at_unix": time.time()})
    write_json(directory / "status.json", {"stage": "complete"})
    keras.backend.clear_session()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/main-paper-revision")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--n-eval", type=int, default=300)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-full-shape", action="store_true",
                        help="Validate the full trajectory shape and research batch size on tiny banks")
    parser.add_argument("--restart-training", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()
    args.smoke = args.smoke or args.smoke_full_shape
    if not args.smoke and jax.default_backend() != "gpu":
        raise RuntimeError("Research training requires a GPU; refusing accidental CPU run")
    if args.smoke:
        args.output = args.output / ("smoke-full-shape" if args.smoke_full_shape else "smoke")
        args.epochs, args.patience, args.n_eval, args.draws = 1, 2, 4, 16
        args.batch_size = 32 if args.smoke_full_shape else 4
        variant = BASE if args.smoke_full_shape else replace(BASE, num_agents=4, time_horizon=2.0)
        budgets = (32, 64) if args.smoke_full_shape else (8, 16)
    else:
        variant, budgets = BASE, (3000, 30000)
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(args.output / "run.log"), logging.StreamHandler()])
    logging.info("Devices: %s", jax.devices())
    (args.output / "environment.txt").write_text(subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True) if importlib.util.find_spec("pip")
        else "\n".join(sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())))
    data_root = args.output / "banks"
    validation = bank(data_root / "validation", variant, args.n_eval, BASE.seed + 100000)
    test = bank(data_root / "test", variant, args.n_eval, BASE.seed + 200000)
    # The smaller budget is an exact prefix of this common fixed training bank.
    train = bank(data_root / "train", variant, max(budgets), BASE.seed)
    if not args.generate_only:
        for budget in budgets:
            run_budget(variant, budget, train, validation, test,
                       args.output / f"offline-{budget}", args)


if __name__ == "__main__":
    main()
