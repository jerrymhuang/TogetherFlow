"""Time-varying w as a parameter PATH, estimated with superstats.

    uv run python experiments/run_v5b_path.py --smoke     # tiny end-to-end check
    uv run python experiments/run_v5b_path.py             # full run

`v5-timevarying-w` infers the two scalars (w0, tau) that generate the path and
treats the path itself as a latent nuisance. That answered a narrower question
than reviewer point 6 asks: the reviewer wants to know whether the model can
capture a shift between individual exploration and social following, which is a
claim about the trajectory of w, not about the volatility of the process behind
it. tau turned out non-identifiable (contraction 0.076) — but that does not show
the path is unrecoverable, and the two must not be conflated.

This run estimates w_t at every timestep and reports superstats' per-step
diagnostics.

Why the data is simulated here rather than by superstats
--------------------------------------------------------
`superstats.simulation.GenerativeModel` flattens parameters to
(batch_size * num_steps,) and calls the model once, so each step's observation
must be independent of the previous state given that step's parameters. Our
dynamics carry position and heading across steps, so it cannot drive this
simulator. `Workflow` accepts `model=None` and trains from a pre-simulated
bank, which is the integration point that does work — superstats supplies the
sequence-producing network, the training loop and the per-step diagnostics, and
we supply the data.

That does force offline training, so unlike the online arms this one genuinely
can overfit and uses early stopping with restore_best_weights.
"""

import argparse
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

from togetherflow.simulator import TogetherFlowSimulator, make_logit_random_walk_expander
from togetherflow.priors import prior_nonstationary

ROOT = pathlib.Path(__file__).parent.parent
OUT = ROOT / "outputs" / "variants" / "v5b-path"

# 30 s rather than the usual 60. The path needs time to drift, but observables
# stop carrying parameter information once agents reach the walls at ~15 s, and
# a bidirectional recurrent net over 600 steps x 343 channels is expensive for
# information that is not there. 30 s keeps the informative window plus enough
# beyond it for drift to accumulate.
TIME_HORIZON = 30.0
DT = 0.1
NUM_AGENTS = 49
NUM_BEACONS = 4
BEACON_SPREAD = 50.0
SEED = 20260809

REFERENCE_RADII = None
CONSTRAIN_PARAMETERS = False

DATA_KEYS = ["positions", "rotations", "neighbors", "distances",
             "angular_velocities", "neighbor_fluctuations"]

# Only w_t is a trajectory. Development Superstats infers the other targets
# as scalars through its separate invariant inference network.
TARGET_KEYS = ["w_t", "r_t", "v_t", "eta_t", "tau_t"]
TARGET_NAMES = [r"$w_t$", r"$r$", r"$v$", r"$\eta$", r"$\tau$"]


def simulate(n, seed):
    sim = TogetherFlowSimulator(
        num_agents=NUM_AGENTS, num_beacons=NUM_BEACONS, dt=DT,
        time_horizon=TIME_HORIZON, output_mode="flat", prior=prior_nonstationary,
        param_names=("w0", "r", "v", "noise", "tau"),
        relative_heading=True, diffusive_heading=True, beacon_spread=BEACON_SPREAD,
        expander=make_logit_random_walk_expander(0, 4, DT),
        include_parameter_paths=True, seed=seed, reference_radii=REFERENCE_RADII,
    )
    raw = sim.sample(n)
    T = int(TIME_HORIZON / DT)

    data = {k: raw[k].astype(np.float32) for k in DATA_KEYS}
    # Normalised clock, so the network can locate itself in the trial.
    data["time_steps"] = np.tile(
        np.linspace(0.0, 1.0, T, dtype=np.float32)[None, :, None], (n, 1, 1)
    )
    data["w_t"] = raw["w0_path"].astype(np.float32)
    data["r_t"] = raw["r_path"].astype(np.float32)
    data["v_t"] = raw["v_path"].astype(np.float32)
    data["eta_t"] = raw["noise_path"].astype(np.float32)
    data["tau_t"] = raw["tau_path"].astype(np.float32)
    for key in TARGET_KEYS:
        if key != "w_t":
            data[key] = data[key][:, 0, :]
    return data


def simulation_bank(directory, n, seed):
    """Bound RAM and simulation work with restartable, disk-backed chunks."""
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "bank.json"
    spec = dict(n=n, seed=seed, horizon=TIME_HORIZON, agents=NUM_AGENTS,
                channels=DATA_KEYS, targets=TARGET_KEYS, radii=REFERENCE_RADII,
                prior=prior_nonstationary.__name__, chunk_size=32,
                source_sha256={str(p): hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                               for p in ("togetherflow/simulator.py", "togetherflow/priors.py", "experiments/run_v5b_path.py")})
    start = 0
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["spec"] != spec:
            raise ValueError(f"Bank specification changed: {directory}")
        start = state["completed"]
    arrays = {p.stem: np.load(p, mmap_mode="r+") for p in directory.glob("*.npy")} if start else {}
    for offset in range(start, n, 32):
        count = min(32, n-offset)
        chunk = simulate(count, seed+offset)
        for key, values in chunk.items():
            if key not in arrays:
                arrays[key] = np.lib.format.open_memmap(directory / f"{key}.npy", mode="w+",
                    dtype="float32", shape=(n, *values.shape[1:]))
            arrays[key][offset:offset+count] = values
            arrays[key].flush()
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(spec=spec, completed=offset+count)))
        tmp.replace(state_path)
        print(f"bank {directory.name}: {offset+count}/{n}", flush=True)
    return {key: np.load(directory / f"{key}.npy", mmap_mode="r") for key in arrays}


def build_adapter():
    """Mirrors superstats' default adapter, with our channels as the data keys."""
    summary_keys = ["time_steps", *DATA_KEYS]
    adapter = bf.Adapter()
    if CONSTRAIN_PARAMETERS:
        from variants import PARAM_BOUNDS
        names = {"w_t": "w", "r_t": "r", "v_t": "v", "eta_t": "noise", "tau_t": "tau"}
        for key in TARGET_KEYS:
            adapter = adapter.constrain(key, **PARAM_BOUNDS[names[key]])
    return (
        adapter
        .convert_dtype("float64", "float32")
        .as_time_series(summary_keys)
        .concatenate(["w_t"], into="inference_variables")
        .concatenate([k for k in TARGET_KEYS if k != "w_t"], into="invariant_variables")
        .concatenate(summary_keys, into="summary_variables")
    )


def stack_targets(data):
    """(num_sim, num_steps, num_params), matching the diagnostics contract."""
    return np.concatenate([
        data[k] if data[k].ndim == 3 else np.broadcast_to(
            data[k][:, None, :], (len(data[k]), data["w_t"].shape[1], data[k].shape[-1])
        ) for k in TARGET_KEYS
    ], axis=-1)


def as_estimate_array(samples, n_sim, T):
    """Coerce Workflow.sample output to (num_sim, num_samples, num_steps, num_params)."""
    if isinstance(samples, dict):
        if "inference_variables" in samples:
            arr = samples["inference_variables"]
        else:
            arr = np.concatenate([
                samples[k] if samples[k].ndim == 4 else np.broadcast_to(
                    samples[k][:, :, None, :], (*samples[k].shape[:2], T, samples[k].shape[-1])
                ) for k in TARGET_KEYS
            ], axis=-1)
    else:
        arr = samples
    arr = np.asarray(arr)
    if arr.ndim != 4:
        raise ValueError(f"expected 4-D posterior samples, got {arr.shape}")
    return arr


def main():
    global OUT, TIME_HORIZON, NUM_AGENTS, TARGET_KEYS, TARGET_NAMES, DATA_KEYS, REFERENCE_RADII, SEED, CONSTRAIN_PARAMETERS
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny run to validate the pipeline")
    ap.add_argument("--fixed-radius", action="store_true", help="Revision specification: radius fixed at 2.5, excluded from targets")
    ap.add_argument("--variable-radius", action="store_true", help="Infer radius with radius-independent inputs in a separate companion suite")
    ap.add_argument("--output-dir", type=pathlib.Path, help="Separate destination for a new run")
    ap.add_argument("--match-baseline", action="store_true", help="60s, radius-free inputs, reference seed, 30K bank, 300 epochs, 300 validation/test cases and 1000 draws")
    ap.add_argument("--smoke-full-shape", action="store_true", help="One epoch at full trajectory shape and batch size")
    args = ap.parse_args()
    args.smoke = args.smoke or args.smoke_full_shape
    if args.fixed_radius and args.variable_radius:
        ap.error("Choose fixed or variable radius, not both")
    OUT = ROOT / "outputs" / "revision" / ("superstats-fixed-path" if args.fixed_radius else "superstats-path")
    if args.variable_radius:
        from variants import R_FREE_CHANNELS, REFERENCE_RADII as radii
        DATA_KEYS = list(R_FREE_CHANNELS)
        REFERENCE_RADII = list(radii)
        OUT = ROOT / "outputs" / "variable-radius" / "superstats-variable-path"
    if args.fixed_radius:
        global prior_nonstationary
        from revision_variants import fix_radius
        prior_nonstationary = fix_radius(prior_nonstationary)
        TARGET_KEYS = ["w_t", "v_t", "eta_t", "tau_t"]
        TARGET_NAMES = [r"$w_t$", r"$v$", r"$\eta$", r"$\tau$"]
    if args.match_baseline:
        from variable_radius_variants import BASE
        TIME_HORIZON, SEED = BASE.time_horizon, BASE.seed
        CONSTRAIN_PARAMETERS = True
        DATA_KEYS, REFERENCE_RADII = list(BASE.channels), list(BASE.reference_radii)
    if args.smoke:
        OUT = OUT.with_name(OUT.name + "-smoke")
        if not args.smoke_full_shape:
            TIME_HORIZON, NUM_AGENTS = 2.0, 4
    if args.output_dir is not None:
        OUT = args.output_dir
    resume_diagnostics = (OUT / "history.json").exists()
    if not resume_diagnostics and (OUT / "checkpoints/model.keras").exists():
        raise RuntimeError("Interrupted training checkpoint exists; use a new output directory")
    if resume_diagnostics and not (OUT / "checkpoints/model.keras").exists():
        raise FileNotFoundError("Training history exists without a saved model")

    if args.smoke:
        n_train, n_val, n_test, epochs, num_samples = (32 if args.smoke_full_shape else 16), 4, 4, 1, 20
    else:
        n_train, n_val, n_test, epochs, num_samples = 4000, 400, 300, 150, 250

    if args.match_baseline and not args.smoke:
        n_train, n_val, n_test, epochs, num_samples = 30000, 300, 300, 300, 1000
    OUT.mkdir(parents=True, exist_ok=True)
    T = int(TIME_HORIZON / DT)
    import importlib.metadata
    manifest = {
        "superstats": importlib.metadata.version("superstats"),
        "superstats_source": json.loads(importlib.metadata.distribution("superstats").read_text("direct_url.json") or "null"),
        "bayesflow": importlib.metadata.version("bayesflow"),
        "fixed_radius": 2.5 if args.fixed_radius else None,
        "smoke": args.smoke, "seed": SEED,
        "matched_baseline": args.match_baseline,
        "constrain_parameters": CONSTRAIN_PARAMETERS,
        "checkpoint_selection": "minimum validation loss; early stopping patience 20",
        "comparison_note": "Dynamic-weight extension; fixed-radius arm conditions on r=2.5. Path diagnostics are not scalar baseline metrics.",
        "time_horizon": TIME_HORIZON, "num_agents": NUM_AGENTS,
        "data_keys": DATA_KEYS, "reference_radii": REFERENCE_RADII,
        "targets": TARGET_KEYS, "n_train": n_train, "n_test": n_test,
        "varying_targets": ["w_t"],
        "invariant_targets": [k for k in TARGET_KEYS if k != "w_t"],
        "approximator": "marginal", "mode": "smoothing",
        "n_val": n_val, "epochs": epochs, "num_samples": num_samples,
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("experiments/run_v5b_path.py", "togetherflow/simulator.py", "togetherflow/priors.py")},
    }
    manifest_path = OUT / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError(f"Run specification changed at {OUT}; use a new output directory")
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"simulating {n_train + n_val + n_test} datasets at T={T} ...")
    train = simulation_bank(OUT / "banks/train", n_train, SEED)
    val = simulation_bank(OUT / "banks/validation", n_val, SEED + 100_000)
    test = simulation_bank(OUT / "banks/test", n_test, SEED + 200_000)
    gb = sum(v.nbytes for v in train.values()) / 1e9
    print(f"  train bank {gb:.2f} GB")

    keras.utils.set_random_seed(SEED)
    workflow = Workflow(
        model=None,                          # data comes from our own kernel
        adapter=build_adapter(),
        embedding_network="recurrent",
        varying_inference_network="coupling",
        invariant_inference_network="coupling",
        checkpoint_filepath=str(OUT / "checkpoints"),
    )

    # Offline training really can overfit — unlike the online arms, where fresh
    # batches make it impossible — so best weights are restored here.
    callbacks = [keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True, verbose=1
    )]
    if resume_diagnostics:
        from types import SimpleNamespace
        history = SimpleNamespace(history=json.loads((OUT / "history.json").read_text()))
    else:
        history = workflow.fit_offline(
            data=train, validation_data=val, epochs=epochs, batch_size=32,
            callbacks=callbacks, verbose=2,
        )
        # Persist selected weights before recording training completion.
        workflow.approximator.save(OUT / "checkpoints/model.keras")
        with open(OUT / "history.json", "w") as f:
            json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f)

    print("sampling posteriors ...")
    posterior_path = OUT / "posterior.npz"
    if posterior_path.exists():
        with np.load(posterior_path) as saved:
            samples = {k: saved[f"samples_{k}"] for k in TARGET_KEYS}
            for k in TARGET_KEYS:
                np.testing.assert_array_equal(saved[f"truth_{k}"], test[k])
    else:
        samples = workflow.sample(data=test, num_samples=num_samples, batch_size=1, seed=SEED + 300_000)
        temporary = posterior_path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **{f"samples_{k}": v for k, v in samples.items()},
                            **{f"truth_{k}": test[k] for k in TARGET_KEYS})
        temporary.replace(posterior_path)
    # Scalar diagnostics use the same BayesFlow definitions as the reference.
    from run_variant import normalize_metric_index
    invariant_keys = [k for k in TARGET_KEYS if k != "w_t"]
    np.random.seed(SEED + 400_000)
    scalar_metrics = workflow.workflow.compute_default_diagnostics(
        samples={k: samples[k] for k in invariant_keys},
        test_data={k: test[k] for k in invariant_keys},
        variable_keys=invariant_keys,
        root_mean_squared_error_kwargs={"normalize": "prior"})
    normalize_metric_index(scalar_metrics).to_csv(OUT / "invariant_metrics.csv")
    est = as_estimate_array(samples, n_test, T)
    tgt = stack_targets(test)
    print(f"  estimates {est.shape}  targets {tgt.shape}")

    nrmse = ssd.nrmse_per_step(est, tgt)
    contr = ssd.posterior_contraction_per_step(est, tgt)
    ece = ssd.calibration_error_per_step(est, tgt)

    np.savez(OUT / "per_step_metrics.npz", nrmse=nrmse, contraction=contr, ece=ece)
    import csv
    with (OUT / "per_step_metrics.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "parameter", "NRMSE", "contraction", "ECE"])
        for i in range(T):
            for j, name in enumerate(TARGET_KEYS):
                writer.writerow([i * DT, name, nrmse[i,j], contr[i,j], ece[i,j]])
    np.savez_compressed(OUT / "posterior_examples.npz", estimates=est[:3], targets=tgt[:3])

    print(f"\n{'param':<8}{'NRMSE (early/mid/late)':>30}{'contraction (early/mid/late)':>32}")
    thirds = [slice(0, T // 3), slice(T // 3, 2 * T // 3), slice(2 * T // 3, T)]
    for j, name in enumerate(TARGET_KEYS):
        n3 = [nrmse[s, j].mean() for s in thirds]
        c3 = [contr[s, j].mean() for s in thirds]
        print(f"{name:<8}{n3[0]:>10.3f}{n3[1]:>10.3f}{n3[2]:>10.3f}"
              f"{c3[0]:>12.3f}{c3[1]:>10.3f}{c3[2]:>10.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    t = np.arange(T) * DT
    for arr, ax, label in ((nrmse, axes[0], "NRMSE"),
                           (contr, axes[1], "posterior contraction"),
                           (ece, axes[2], "calibration error")):
        for j, nm in enumerate(TARGET_NAMES):
            ax.plot(t, arr[:, j], label=nm, lw=1.4)
        ax.set_xlabel("time (s)"); ax.set_ylabel(label)
    axes[0].legend(fontsize=8)
    fig.suptitle("Per-step recovery of the time-varying influence weight")
    fig.tight_layout()
    fig.savefig(OUT / "per_step_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    plot_posteriors(est[:3], tgt[:3], TARGET_KEYS, OUT)

    (OUT / "complete.json").write_text(json.dumps({"stage": "complete", "epochs": len(history.history["loss"])}))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
