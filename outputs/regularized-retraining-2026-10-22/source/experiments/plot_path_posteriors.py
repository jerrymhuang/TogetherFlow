"""Replot saved path runs with Superstats, without retraining.

Usage: .venv-superstats/bin/python experiments/plot_path_posteriors.py RUN_DIR [...]
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import superstats.diagnostics as ssd

NAMES = {"w_t": r"$w_t$", "r_t": r"$r$", "v_t": r"$v$",
         "eta_t": r"$\eta$", "tau_t": r"$\tau$"}


def plot_posteriors(estimates, targets, keys, output):
    """Use explicit mapping selection; array variable_keys do not select columns.

    Legacy runs predict constants at each step. Superstats' marginal helper
    pools their sample/step axes within each dataset and subsamples to the
    original draw count. This is a mixture of stepwise marginals, not a new
    jointly inferred scalar posterior. New runs tile scalar draws only for
    compatibility with the saved-example and per-step diagnostic formats.
    """
    output = Path(output)
    if estimates.ndim != 4 or targets.shape != (estimates.shape[0], *estimates.shape[2:]):
        raise ValueError("Incompatible posterior and target shapes")
    if len(keys) != estimates.shape[-1] or "w_t" not in keys:
        raise ValueError("Target keys must match posterior columns and include w_t")
    invariant = [k for k in keys if k != "w_t"]
    est = {k: estimates[..., j:j+1] for j, k in enumerate(keys)}
    tgt = {k: targets[..., j:j+1] for j, k in enumerate(keys)}
    for k in invariant:
        if not np.allclose(tgt[k], tgt[k][:, :1, :]):
            raise ValueError(f"Expected constant ground truth for {k}")
    fig = ssd.plot_time_varying_posterior(
        est, tgt, variable_keys=["w_t"], variable_names=[NAMES["w_t"]],
        marginal=False, uncertainty_fun="ci",
    )
    fig.savefig(output / "time_varying_posterior.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    # New scalar draws were only tiled for the legacy export schema. Avoid
    # resampling duplicates when their posterior is exactly constant in time.
    scalar_est = {k: est[k][:, :, :1, :] if np.array_equal(
        est[k], np.broadcast_to(est[k][:, :, :1, :], est[k].shape)) else est[k]
        for k in invariant}
    for i in range(len(estimates)):
        fig = ssd.plot_marginals(
            scalar_est, {k: tgt[k] for k in invariant}, variable_keys=invariant,
            variable_names=[NAMES[k] for k in invariant], data_idx=i,
        )
        fig.savefig(output / f"time_invariant_posterior_{i+1}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    source = importlib.metadata.distribution("superstats").read_text("direct_url.json")
    (output / "posterior_plot_manifest.json").write_text(json.dumps({
        "superstats": importlib.metadata.version("superstats"),
        "superstats_source": json.loads(source or "null"),
        "varying_targets": ["w_t"], "invariant_targets": invariant,
        "invariant_plot": "Superstats plot_marginals, per dataset; sample/step pooling and subsampling",
        "trajectory_intervals": "pointwise 95% and 65% equal-tailed intervals",
        "training_results_unchanged": True,
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    for run in args.runs:
        manifest = json.loads((run / "manifest.json").read_text())
        with np.load(run / "posterior_examples.npz") as archive:
            plot_posteriors(archive["estimates"], archive["targets"], manifest["targets"], run)
        print(f"Updated posterior plots: {run}")


if __name__ == "__main__":
    main()
