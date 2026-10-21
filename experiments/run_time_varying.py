"""Train Superstats to infer a weight trajectory and constant model parameters.

TogetherFlow generates complete trajectories because positions and headings
persist between steps. Superstats trains offline on these simulation banks,
using recurrent embeddings and separate coupling flows for w_t and scalars.

Run a small check:
    .venv-superstats/bin/python experiments/run_time_varying.py --smoke

Run the matched fixed-radius experiment:
    .venv-superstats/bin/python experiments/run_time_varying.py --fixed-radius --match-baseline
"""

import argparse
import csv
from dataclasses import dataclass, field
from collections.abc import Callable
import importlib.metadata
import json
import os
import pathlib
import sys
import hashlib

os.environ.setdefault("KERAS_BACKEND", "jax")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import keras
import bayesflow as bf

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from superstats.workflow import Workflow
from plot_path_posteriors import plot_posteriors
import superstats.diagnostics as ssd

from togetherflow.simulator import (
    TogetherFlowSimulator,
    make_logit_random_walk_expander,
)
from togetherflow.priors import prior_nonstationary

ROOT = pathlib.Path(__file__).parent.parent
DT = 0.1
NUM_BEACONS = 4
BEACON_SPREAD = 50.0


@dataclass
class RunConfig:
    output_dir: pathlib.Path = ROOT / "outputs/revision/superstats-path"
    time_horizon: float = 30.0
    num_agents: int = 49
    seed: int = 20260809
    reference_radii: list | None = None
    constrain_parameters: bool = False
    data_keys: list[str] = field(
        default_factory=lambda: [
            "positions",
            "rotations",
            "neighbors",
            "distances",
            "angular_velocities",
            "neighbor_fluctuations",
        ]
    )
    target_keys: list[str] = field(
        default_factory=lambda: [
            "w_t",
            "r_t",
            "v_t",
            "eta_t",
            "tau_t",
        ]
    )
    target_names: list[str] = field(
        default_factory=lambda: [
            r"$w_t$",
            r"$r$",
            r"$v$",
            r"$\eta$",
            r"$\tau$",
        ]
    )
    prior: Callable = prior_nonstationary


def simulate(config, n, seed):
    sim = TogetherFlowSimulator(
        num_agents=config.num_agents,
        num_beacons=NUM_BEACONS,
        dt=DT,
        time_horizon=config.time_horizon,
        output_mode="flat",
        prior=config.prior,
        param_names=("w0", "r", "v", "noise", "tau"),
        relative_heading=True,
        diffusive_heading=True,
        beacon_spread=BEACON_SPREAD,
        expander=make_logit_random_walk_expander(0, 4, DT),
        include_parameter_paths=True,
        seed=seed,
        reference_radii=config.reference_radii,
    )
    raw = sim.sample(n)
    num_steps = int(config.time_horizon / DT)

    data = {k: raw[k].astype(np.float32) for k in config.data_keys}
    # Normalised clock, so the network can locate itself in the trial.
    data["time_steps"] = np.tile(
        np.linspace(0.0, 1.0, num_steps, dtype=np.float32)[None, :, None], (n, 1, 1)
    )
    data["w_t"] = raw["w0_path"].astype(np.float32)
    data["r_t"] = raw["r_path"].astype(np.float32)
    data["v_t"] = raw["v_path"].astype(np.float32)
    data["eta_t"] = raw["noise_path"].astype(np.float32)
    data["tau_t"] = raw["tau_path"].astype(np.float32)
    for key in config.target_keys:
        if key != "w_t":
            data[key] = data[key][:, 0, :]
    return data


def simulation_bank(config, directory, n, seed):
    """Bound RAM and simulation work with restartable, disk-backed chunks."""
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "bank.json"
    spec = dict(
        n=n,
        seed=seed,
        horizon=config.time_horizon,
        agents=config.num_agents,
        channels=config.data_keys,
        targets=config.target_keys,
        radii=config.reference_radii,
        prior=config.prior.__name__,
        chunk_size=32,
        source_sha256={
            str(p): hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
            for p in (
                "togetherflow/simulator.py",
                "togetherflow/priors.py",
                "experiments/run_time_varying.py",
            )
        },
    )
    start = 0
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["spec"] != spec:
            raise ValueError(f"Bank specification changed: {directory}")
        start = state["completed"]
    arrays = (
        {p.stem: np.load(p, mmap_mode="r+") for p in directory.glob("*.npy")}
        if start
        else {}
    )
    for offset in range(start, n, 32):
        count = min(32, n - offset)
        chunk = simulate(config, count, seed + offset)
        for key, values in chunk.items():
            if key not in arrays:
                arrays[key] = np.lib.format.open_memmap(
                    directory / f"{key}.npy",
                    mode="w+",
                    dtype="float32",
                    shape=(n, *values.shape[1:]),
                )
            arrays[key][offset : offset + count] = values
            arrays[key].flush()
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(spec=spec, completed=offset + count)))
        tmp.replace(state_path)
        print(f"bank {directory.name}: {offset + count}/{n}", flush=True)
    return {key: np.load(directory / f"{key}.npy", mmap_mode="r") for key in arrays}


def build_adapter(config):
    """Mirrors superstats' default adapter, with our channels as the data keys."""
    summary_keys = ["time_steps", *config.data_keys]
    adapter = bf.Adapter()
    if config.constrain_parameters:
        from variants import PARAM_BOUNDS

        names = {"w_t": "w", "r_t": "r", "v_t": "v", "eta_t": "noise", "tau_t": "tau"}
        for key in config.target_keys:
            adapter = adapter.constrain(key, **PARAM_BOUNDS[names[key]])
    return (
        adapter.convert_dtype("float64", "float32")
        .as_time_series(summary_keys)
        .concatenate(["w_t"], into="inference_variables")
        .concatenate(
            [k for k in config.target_keys if k != "w_t"], into="invariant_variables"
        )
        .concatenate(summary_keys, into="summary_variables")
    )


def stack_targets(config, data):
    """(num_sim, num_steps, num_params), matching the diagnostics contract."""
    trajectories = []
    num_steps = data["w_t"].shape[1]
    for key in config.target_keys:
        values = data[key]
        if values.ndim == 2:
            # Repeat scalar truths only for the per-step diagnostic arrays.
            shape = (len(values), num_steps, values.shape[-1])
            values = np.broadcast_to(values[:, None, :], shape)
        trajectories.append(values)
    return np.concatenate(trajectories, axis=-1)


def as_estimate_array(config, samples, num_steps):
    """Arrange draws as (datasets, draws, time steps, parameters)."""
    if isinstance(samples, dict):
        if "inference_variables" in samples:
            estimates = samples["inference_variables"]
        else:
            trajectories = []
            for key in config.target_keys:
                values = samples[key]
                if values.ndim != 4:
                    shape = (*values.shape[:2], num_steps, values.shape[-1])
                    values = np.broadcast_to(values[:, :, None, :], shape)
                trajectories.append(values)
            estimates = np.concatenate(trajectories, axis=-1)
    else:
        estimates = samples
    estimates = np.asarray(estimates)
    if estimates.ndim != 4:
        raise ValueError(f"expected 4-D posterior samples, got {estimates.shape}")
    return estimates


def parse_config():
    """Read command-line options and apply experiment defaults."""
    config = RunConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke", action="store_true", help="tiny run to validate the pipeline"
    )
    parser.add_argument(
        "--fixed-radius",
        action="store_true",
        help="Fix radius at 2.5 and exclude it from inference",
    )
    parser.add_argument(
        "--variable-radius",
        action="store_true",
        help="Infer radius using inputs measured at fixed reference radii",
    )
    parser.add_argument(
        "--output-dir", type=pathlib.Path, help="Separate destination for a new run"
    )
    parser.add_argument(
        "--match-baseline",
        action="store_true",
        help="60s, radius-free inputs, reference seed, 30K bank, 300 epochs, 300 validation/test cases and 1000 draws",
    )
    parser.add_argument(
        "--smoke-full-shape",
        action="store_true",
        help="One epoch at full trajectory shape and batch size",
    )
    args = parser.parse_args()
    args.smoke = args.smoke or args.smoke_full_shape
    if args.fixed_radius and args.variable_radius:
        parser.error("Choose fixed or variable radius, not both")
    config.output_dir = (
        ROOT
        / "outputs"
        / "revision"
        / ("superstats-fixed-path" if args.fixed_radius else "superstats-path")
    )
    if args.variable_radius:
        from variants import R_FREE_CHANNELS, REFERENCE_RADII as radii

        config.data_keys = list(R_FREE_CHANNELS)
        config.reference_radii = list(radii)
        config.output_dir = (
            ROOT / "outputs" / "variable-radius" / "superstats-variable-path"
        )
    if args.fixed_radius:
        from revision_variants import fix_radius

        config.prior = fix_radius(prior_nonstationary)
        config.target_keys = ["w_t", "v_t", "eta_t", "tau_t"]
        config.target_names = [r"$w_t$", r"$v$", r"$\eta$", r"$\tau$"]
    if args.match_baseline:
        from variable_radius_variants import BASE

        config.time_horizon, config.seed = BASE.time_horizon, BASE.seed
        config.constrain_parameters = True
        config.data_keys, config.reference_radii = (
            list(BASE.channels),
            list(BASE.reference_radii),
        )
    if args.smoke:
        config.output_dir = config.output_dir.with_name(
            config.output_dir.name + "-smoke"
        )
        if not args.smoke_full_shape:
            config.time_horizon, config.num_agents = 2.0, 4
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    return config, args


def training_budget(args):
    if args.smoke:
        n_train, n_val, n_test, epochs, num_samples = (
            (32 if args.smoke_full_shape else 16),
            4,
            4,
            1,
            20,
        )
    else:
        n_train, n_val, n_test, epochs, num_samples = 4000, 400, 300, 150, 250

    if args.match_baseline and not args.smoke:
        n_train, n_val, n_test, epochs, num_samples = 30000, 300, 300, 300, 1000
    return n_train, n_val, n_test, epochs, num_samples


def write_manifest(config, args, n_train, n_val, n_test, epochs, num_samples):
    """Record settings and reject accidental reuse of a different run."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "superstats": importlib.metadata.version("superstats"),
        "superstats_source": json.loads(
            importlib.metadata.distribution("superstats").read_text("direct_url.json")
            or "null"
        ),
        "bayesflow": importlib.metadata.version("bayesflow"),
        "fixed_radius": 2.5 if args.fixed_radius else None,
        "smoke": args.smoke,
        "seed": config.seed,
        "matched_baseline": args.match_baseline,
        "constrain_parameters": config.constrain_parameters,
        "checkpoint_selection": "minimum validation loss; early stopping patience 20",
        "comparison_note": "Dynamic-weight extension; fixed-radius arm conditions on r=2.5. Path diagnostics are not scalar baseline metrics.",
        "time_horizon": config.time_horizon,
        "num_agents": config.num_agents,
        "data_keys": config.data_keys,
        "reference_radii": config.reference_radii,
        "targets": config.target_keys,
        "n_train": n_train,
        "n_test": n_test,
        "varying_targets": ["w_t"],
        "invariant_targets": [k for k in config.target_keys if k != "w_t"],
        "approximator": "marginal",
        "mode": "smoothing",
        "n_val": n_val,
        "epochs": epochs,
        "num_samples": num_samples,
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "experiments/run_time_varying.py",
                "togetherflow/simulator.py",
                "togetherflow/priors.py",
            )
        },
    }
    manifest_path = config.output_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError(
            f"Run specification changed at {config.output_dir}; use a new output directory"
        )
    manifest_path.write_text(json.dumps(manifest, indent=2))


def load_banks(config, n_train, n_val, n_test):
    """Generate missing simulations or reopen the saved banks."""
    num_steps = int(config.time_horizon / DT)
    print(
        f"simulating {n_train + n_val + n_test} datasets at num_steps={num_steps} ..."
    )
    train = simulation_bank(
        config, config.output_dir / "banks/train", n_train, config.seed
    )
    val = simulation_bank(
        config, config.output_dir / "banks/validation", n_val, config.seed + 100_000
    )
    test = simulation_bank(
        config, config.output_dir / "banks/test", n_test, config.seed + 200_000
    )
    gb = sum(v.nbytes for v in train.values()) / 1e9
    print(f"  train bank {gb:.2f} GB")

    return train, val, test


def train_model(config, train, val, epochs, resume_diagnostics):
    """Train with validation-based stopping, or load a completed run."""
    keras.utils.set_random_seed(config.seed)
    workflow = Workflow(
        model=None,  # data comes from our own kernel
        adapter=build_adapter(config),
        embedding_network="recurrent",
        varying_inference_network="coupling",
        invariant_inference_network="coupling",
        checkpoint_filepath=str(config.output_dir / "checkpoints"),
    )

    # Select the weights with the lowest validation loss.
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=20, restore_best_weights=True, verbose=1
        )
    ]
    if resume_diagnostics:
        history = json.loads((config.output_dir / "history.json").read_text())
    else:
        history = workflow.fit_offline(
            data=train,
            validation_data=val,
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=2,
        )
        # Persist selected weights before recording training completion.
        workflow.approximator.save(config.output_dir / "checkpoints/model.keras")
        history = {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        }
        (config.output_dir / "history.json").write_text(json.dumps(history))

    return workflow, history


def sample_posteriors(config, workflow, test, num_samples):
    """Save posterior draws, or reuse draws after checking their test data."""
    print("sampling posteriors ...")
    posterior_path = config.output_dir / "posterior.npz"
    if posterior_path.exists():
        with np.load(posterior_path) as saved:
            samples = {k: saved[f"samples_{k}"] for k in config.target_keys}
            for k in config.target_keys:
                np.testing.assert_array_equal(saved[f"truth_{k}"], test[k])
    else:
        samples = workflow.sample(
            data=test, num_samples=num_samples, batch_size=1, seed=config.seed + 300_000
        )
        temporary = posterior_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            **{f"samples_{k}": v for k, v in samples.items()},
            **{f"truth_{k}": test[k] for k in config.target_keys},
        )
        temporary.replace(posterior_path)
    return samples


def save_diagnostics(config, workflow, samples, test):
    """Export scalar and per-step recovery metrics and posterior plots."""
    num_steps = int(config.time_horizon / DT)
    # Scalar diagnostics use the same BayesFlow definitions as the reference.
    from run_variant import normalize_metric_index

    invariant_keys = [k for k in config.target_keys if k != "w_t"]
    np.random.seed(config.seed + 400_000)
    scalar_metrics = workflow.workflow.compute_default_diagnostics(
        samples={k: samples[k] for k in invariant_keys},
        test_data={k: test[k] for k in invariant_keys},
        variable_keys=invariant_keys,
        root_mean_squared_error_kwargs={"normalize": "prior"},
    )
    normalize_metric_index(scalar_metrics).to_csv(
        config.output_dir / "invariant_metrics.csv"
    )
    estimates = as_estimate_array(config, samples, num_steps)
    targets = stack_targets(config, test)
    print(f"  estimates {estimates.shape}  targets {targets.shape}")

    nrmse = ssd.nrmse_per_step(estimates, targets)
    contraction = ssd.posterior_contraction_per_step(estimates, targets)
    ece = ssd.calibration_error_per_step(estimates, targets)

    np.savez(
        config.output_dir / "per_step_metrics.npz",
        nrmse=nrmse,
        contraction=contraction,
        ece=ece,
    )
    with (config.output_dir / "per_step_metrics.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "parameter", "NRMSE", "contraction", "ECE"])
        for i in range(num_steps):
            for j, name in enumerate(config.target_keys):
                writer.writerow(
                    [i * DT, name, nrmse[i, j], contraction[i, j], ece[i, j]]
                )
    np.savez_compressed(
        config.output_dir / "posterior_examples.npz",
        estimates=estimates[:3],
        targets=targets[:3],
    )

    print(
        f"\n{'param':<8}{'NRMSE (early/mid/late)':>30}{'contraction (early/mid/late)':>32}"
    )
    thirds = [
        slice(0, num_steps // 3),
        slice(num_steps // 3, 2 * num_steps // 3),
        slice(2 * num_steps // 3, num_steps),
    ]
    for j, name in enumerate(config.target_keys):
        mean_nrmse = [nrmse[s, j].mean() for s in thirds]
        mean_contraction = [contraction[s, j].mean() for s in thirds]
        print(
            f"{name:<8}{mean_nrmse[0]:>10.3f}{mean_nrmse[1]:>10.3f}{mean_nrmse[2]:>10.3f}"
            f"{mean_contraction[0]:>12.3f}{mean_contraction[1]:>10.3f}{mean_contraction[2]:>10.3f}"
        )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    t = np.arange(num_steps) * DT
    for arr, ax, label in (
        (nrmse, axes[0], "NRMSE"),
        (contraction, axes[1], "posterior contraction"),
        (ece, axes[2], "calibration error"),
    ):
        for j, nm in enumerate(config.target_names):
            ax.plot(t, arr[:, j], label=nm, lw=1.4)
        ax.set_xlabel("time (s)")
        ax.set_ylabel(label)
    axes[0].legend(fontsize=8)
    fig.suptitle("Per-step recovery of the time-varying influence weight")
    fig.tight_layout()
    fig.savefig(
        config.output_dir / "per_step_diagnostics.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    plot_posteriors(estimates[:3], targets[:3], config.target_keys, config.output_dir)


def check_checkpoint(config):
    """Reject incomplete training before generating any simulation banks."""
    resume_diagnostics = (config.output_dir / "history.json").exists()
    if (
        not resume_diagnostics
        and (config.output_dir / "checkpoints/model.keras").exists()
    ):
        raise RuntimeError(
            "Interrupted training checkpoint exists; use a new output directory"
        )
    if (
        resume_diagnostics
        and not (config.output_dir / "checkpoints/model.keras").exists()
    ):
        raise FileNotFoundError("Training history exists without a saved model")

    return resume_diagnostics


def main():
    config, args = parse_config()
    resume_diagnostics = check_checkpoint(config)
    n_train, n_val, n_test, epochs, num_samples = training_budget(args)
    write_manifest(config, args, n_train, n_val, n_test, epochs, num_samples)
    train, val, test = load_banks(config, n_train, n_val, n_test)
    workflow, history = train_model(config, train, val, epochs, resume_diagnostics)
    samples = sample_posteriors(config, workflow, test, num_samples)
    save_diagnostics(config, workflow, samples, test)
    completion = {"stage": "complete", "epochs": len(history["loss"])}
    (config.output_dir / "complete.json").write_text(json.dumps(completion))
    print(f"\nwrote {config.output_dir}")


if __name__ == "__main__":
    main()
