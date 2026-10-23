"""Plot calibration for saved joint weight/radius posteriors without retraining.

Usage:
    PYTHONPATH=. .venv/bin/python experiments/plot_partial_pooling_calibration.py
    PYTHONPATH=. .venv/bin/python experiments/plot_partial_pooling_calibration.py RUN_DIR ...
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import bayesflow as bf

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = [
    ROOT / "outputs/graphical-partial-pooling/offline",
    ROOT / "outputs/agent-radius-pooling/bounded-radius-online",
]
PARAMETER_NAMES = [
    r"$\mu_w$",
    r"$\sigma_w$",
    r"$\mu_r$",
    r"$\sigma_r$",
    r"$v$",
    r"$\eta$",
]
PARAMETER_NAMES += [rf"$w_{{{i}}}$" for i in range(1, 13)]
PARAMETER_NAMES += [rf"$r_{{{i}}}$" for i in range(1, 13)]
COLORS = ["#132a70"] * 6 + ["#6969ff"] * 12 + ["#4e2a84"] * 12


def plot_calibration(directory, seed=831):
    posterior_path = directory / "posterior.npz"
    with np.load(posterior_path) as saved:
        samples, truth = saved["samples"], saved["truth"]
    if samples.ndim != 3 or samples.shape[-1] != 30:
        raise ValueError(f"Expected (groups, draws, 30) samples in {posterior_path}")
    if truth.shape != (samples.shape[0], samples.shape[2]):
        raise ValueError(f"Truth and posterior dimensions disagree in {posterior_path}")
    if not np.isfinite(samples).all() or not np.isfinite(truth).all():
        raise ValueError(f"Nonfinite posterior draws or truths in {posterior_path}")

    layout = dict(
        estimates=samples,
        targets=truth,
        variable_names=PARAMETER_NAMES,
        num_row=5,
        num_col=6,
        figsize=(21, 17.5),
        label_fontsize=12,
        title_fontsize=20,
        tick_fontsize=10,
        legend_fontsize=9,
    )
    for name, plot, options, curve_label in [
        (
            "coverage",
            bf.diagnostics.plots.coverage,
            {"difference": False},
            "Empirical Coverage",
        ),
        (
            "calibration_ecdf",
            bf.diagnostics.plots.calibration_ecdf,
            {
                "difference": True,
                "stacked": False,
                "ecdf_bands_kwargs": {"confidence": 0.95, "num_simulations": 1000},
            },
            "Rank ECDF",
        ),
    ]:
        # Use the same simulated ECDF bands for runs with the same test size.
        np.random.seed(seed)
        figure = plot(**layout, **options)
        for axis, color in zip(figure.axes, COLORS):
            for line in axis.lines:
                if line.get_label() == curve_label:
                    line.set_color(color)
            if name == "calibration_ecdf":
                axis.axhline(0, color="black", linestyle="dashed", linewidth=0.8)
        figure.tight_layout()
        for extension in ("pdf", "png"):
            figure.savefig(
                directory / f"{name}.{extension}", dpi=180, bbox_inches="tight"
            )
        plt.close(figure)
        print(f"Saved {directory / name} (.pdf and .png)", flush=True)

    metadata = {
        "bayesflow": bf.__version__,
        "groups": samples.shape[0],
        "draws": samples.shape[1],
        "parameters": 30,
        "seed": seed,
        "posterior_sha256": hashlib.sha256(posterior_path.read_bytes()).hexdigest(),
        "coverage": "Central credible-interval coverage; 95% beta-binomial uncertainty ribbons",
        "calibration_ecdf": "Marginal fractional-rank ECDF minus uniform CDF; 95% simultaneous bands within each parameter",
        "layout": "Six population/shared parameters, twelve weights, twelve radii",
    }
    (directory / "calibration_plots.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--seed", type=int, default=831)
    args = parser.parse_args()
    for directory in args.runs:
        plot_calibration(directory, args.seed)


if __name__ == "__main__":
    main()
