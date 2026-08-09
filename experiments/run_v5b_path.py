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
simulator. `Workflow` accepts `simulator=None` and trains from a pre-simulated
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

os.environ.setdefault("KERAS_BACKEND", "jax")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import keras
import bayesflow as bf

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from superstats.workflow import Workflow
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

DATA_KEYS = ["positions", "rotations", "neighbors", "distances",
             "angular_velocities", "neighbor_fluctuations"]

# w_t is the local (per-step) target; the rest are time-invariant but are still
# predicted at every step, which is superstats' convention — `verify_time_invariant`
# exists precisely to check that a constant is recovered as a constant.
TARGET_KEYS = ["w_t", "r_t", "v_t", "eta_t", "tau_t"]
TARGET_NAMES = [r"$w_t$", r"$r$", r"$v$", r"$\eta$", r"$\tau$"]


def simulate(n, seed):
    sim = TogetherFlowSimulator(
        num_agents=NUM_AGENTS, num_beacons=NUM_BEACONS, dt=DT,
        time_horizon=TIME_HORIZON, output_mode="flat", prior=prior_nonstationary,
        param_names=("w0", "r", "v", "noise", "tau"),
        relative_heading=True, diffusive_heading=True, beacon_spread=BEACON_SPREAD,
        expander=make_logit_random_walk_expander(0, 4, DT),
        include_parameter_paths=True, seed=seed,
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
    return data


def build_adapter():
    """Mirrors superstats' default adapter, with our channels as the data keys."""
    summary_keys = ["time_steps", *DATA_KEYS]
    return (
        bf.Adapter()
        .convert_dtype("float64", "float32")
        .as_time_series(summary_keys)
        .concatenate(TARGET_KEYS, into="inference_variables")
        .concatenate(summary_keys, into="summary_variables")
    )


def stack_targets(data):
    """(num_sim, num_steps, num_params), matching the diagnostics contract."""
    return np.concatenate([data[k] for k in TARGET_KEYS], axis=-1)


def as_estimate_array(samples, n_sim, T):
    """Coerce Workflow.sample output to (num_sim, num_samples, num_steps, num_params)."""
    if isinstance(samples, dict):
        if "inference_variables" in samples:
            arr = samples["inference_variables"]
        else:
            arr = np.concatenate([samples[k] for k in TARGET_KEYS], axis=-1)
    else:
        arr = samples
    arr = np.asarray(arr)
    if arr.ndim != 4:
        raise ValueError(f"expected 4-D posterior samples, got {arr.shape}")
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny run to validate the pipeline")
    args = ap.parse_args()

    if args.smoke:
        n_train, n_val, n_test, epochs, num_samples = 120, 40, 24, 2, 40
    else:
        n_train, n_val, n_test, epochs, num_samples = 4000, 400, 300, 150, 250

    OUT.mkdir(parents=True, exist_ok=True)
    T = int(TIME_HORIZON / DT)

    print(f"simulating {n_train + n_val + n_test} datasets at T={T} ...")
    train = simulate(n_train, SEED)
    val = simulate(n_val, SEED + 100_000)
    test = simulate(n_test, SEED + 200_000)
    gb = sum(v.nbytes for v in train.values()) / 1e9
    print(f"  train bank {gb:.2f} GB")

    workflow = Workflow(
        simulator=None,                      # data comes from our own kernel
        adapter=build_adapter(),
        summary_network="recurrent",         # sequence-producing, per-step outputs
        inference_network="coupling",
        checkpoint_filepath=str(OUT / "checkpoints"),
    )

    # Offline training really can overfit — unlike the online arms, where fresh
    # batches make it impossible — so best weights are restored here.
    callbacks = [keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True, verbose=1
    )]
    history = workflow.fit_offline(
        data=train, validation_data=val, epochs=epochs, batch_size=32,
        callbacks=callbacks, verbose=2,
    )
    with open(OUT / "history.json", "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f)

    print("sampling posteriors ...")
    samples = workflow.sample(data=test, num_samples=num_samples, batch_size=4)
    est = as_estimate_array(samples, n_test, T)
    tgt = stack_targets(test)
    print(f"  estimates {est.shape}  targets {tgt.shape}")

    nrmse = ssd.nrmse_per_step(est, tgt)
    contr = ssd.posterior_contraction_per_step(est, tgt)
    ece = ssd.calibration_error_per_step(est, tgt)

    np.savez(OUT / "per_step_metrics.npz", nrmse=nrmse, contraction=contr, ece=ece)

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

    try:
        # variable_keys must be explicit: with simulator=None the helper would
        # otherwise introspect self.simulator.local_keys and fail.
        f = workflow.plot_time_varying_posterior(
            estimates=est, targets=tgt,
            variable_keys=TARGET_KEYS, variable_names=TARGET_NAMES,
        )
        figure = f if isinstance(f, plt.Figure) else getattr(f, "figure", None)
        if figure is not None:
            figure.savefig(OUT / "time_varying_posterior.png", dpi=150, bbox_inches="tight")
            plt.close(figure)
    except Exception as exc:                                    # noqa: BLE001
        print(f"  plot_time_varying_posterior failed: {type(exc).__name__}: {exc}")

    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
