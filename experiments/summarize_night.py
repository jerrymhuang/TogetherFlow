"""Collate the per-variant reports into one cross-arm summary.

    uv run python experiments/summarize_night.py

Reads every outputs/variants/<slug>/metrics.csv produced by run_variant.py and
writes outputs/variants/SUMMARY.md: one table per diagnostic with variants as
rows, plus a section per reviewer point pulling the relevant arms together.

Deliberately reads only the on-disk artifacts, so it can be rerun any time
without retraining and works on a partial night.
"""

import os

# Importing `variants` reaches togetherflow/__init__, which pulls in bayesflow
# and therefore keras — which defaults to the TensorFlow backend if unset.
os.environ.setdefault("KERAS_BACKEND", "jax")
# This script only reads CSVs, but the import chain above initialises JAX, which
# would otherwise claim GPU memory out from under a training run. Pin it to CPU
# so the summary is safe to run while the night is still going.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from variants import ALL_VARIANTS

ROOT = pathlib.Path(__file__).parent.parent
OUT_ROOT = ROOT / "outputs" / "variants"

METRICS = ["NRMSE", "ECE", "Post. Contraction"]
METRIC_LABEL = {
    "NRMSE": "Recovery (NRMSE — lower is better)",
    "ECE": "Calibration error (lower is better)",
    "Post. Contraction": "Posterior contraction (higher is better)",
}
ALL_PARAMS = ["w", "r", "v", "noise", "alpha"]


def load():
    rows = {}
    for v in ALL_VARIANTS:
        path = OUT_ROOT / v.slug / "metrics.csv"
        if not path.exists():
            continue
        rows[v.slug] = pd.read_csv(path, index_col=0)
    return rows


def metric_table(loaded, metric):
    lines = [f"### {METRIC_LABEL[metric]}", ""]
    present = [p for p in ALL_PARAMS if any(p in df.columns for df in loaded.values())]
    lines.append("| Variant | " + " | ".join(present) + " |")
    lines.append("|---------|" + "|".join(["---"] * len(present)) + "|")
    for slug, df in loaded.items():
        if metric not in df.index:
            continue
        cells = []
        for p in present:
            cells.append(f"{float(df.loc[metric, p]):.3f}" if p in df.columns else "—")
        lines.append(f"| `{slug}` | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def main():
    loaded = load()
    if not loaded:
        print(f"No metrics.csv found under {OUT_ROOT}. Has the night run?")
        return

    by_slug = {v.slug: v for v in ALL_VARIANTS}
    done = list(loaded)
    missing = [v.slug for v in ALL_VARIANTS if v.slug not in loaded]

    L = ["# Reviewer-response experiment night — cross-variant summary", ""]
    L.append(f"{len(done)} of {len(ALL_VARIANTS)} variants completed.")
    if missing:
        L.append("")
        L.append("**Missing / not yet run:** " + ", ".join(f"`{m}`" for m in missing))
    L.append("")
    L.append("Each variant has its own full report at "
             "`outputs/variants/<slug>/report.md`. This file compares them.")
    L.append("")

    L.append("## Variants")
    L.append("")
    L.append("| Variant | Reviewer point | Training | Summary net | Description |")
    L.append("|---------|----------------|----------|-------------|-------------|")
    for slug in done:
        v = by_slug[slug]
        mode = "online" if v.online else "offline"
        L.append(f"| `{slug}` | {v.addresses} | {mode} | {v.summary_net} | {v.title} |")
    L.append("")
    L.append(
        "> **Read the offline arms with care.** Every offline arm overfits: "
        "validation loss bottoms near epoch 11 and then degrades while training "
        "loss keeps falling, leaving posteriors that are narrow but "
        "miscalibrated (high contraction, high calibration error). Their absolute "
        "numbers are not trustworthy; only comparisons *between* offline arms, "
        "which share the flaw, carry information. Online arms do not have this "
        "problem."
    )
    L.append("")

    L.append("## Diagnostics across arms")
    L.append("")
    for m in METRICS:
        L += metric_table(loaded, m)

    # ── Per-reviewer-point readings ──────────────────────────────────────────
    L.append("## Readings by reviewer point")
    L.append("")

    L.append("### Point 2 — sensing radius identifiability")
    L.append("")
    L.append(
        "Compare `v0-baseline` (r inferred from observables that are themselves "
        "computed at the true r), `v3c-rfree-observables` (r inferred from "
        "observables that do not depend on r), `v3a-fixed-radius` (r not inferred "
        "at all), and `v7-prior-bounded-radius` (r restricted to the room). The "
        "contraction column for `r` is the quantity to read: if it stays low in "
        "`v3c` while the baseline looks acceptable, the baseline's apparent "
        "identifiability came from the leaked observables, and the honest "
        "conclusion is non-identifiability. If `v3a` improves w/v/noise, that is "
        "the argument for fixing r as a known constant."
    )
    L.append("")

    L.append("### Point 3 — velocity calibration")
    L.append("")
    L.append(
        "Compare the `v` column of `v0-baseline` against "
        "`v7-prior-slow-velocity`. A large improvement implicates the prior's "
        "upper tail — where an agent crosses the room inside the observation "
        "window and the trajectory stops carrying speed information — rather than "
        "the architecture."
    )
    L.append("")

    L.append("### Point 7 — prior sensitivity")
    L.append("")
    L.append(
        "The four `v7-*` arms against `v0-baseline`. Diagnostics that hold across "
        "all five are robust to prior choice; any that move substantially should "
        "be reported as prior-dependent in the paper."
    )
    L.append("")

    L.append("### Point 8 — beacon switching")
    L.append("")
    L.append(
        "The `alpha` column in `v2-salience-spread50` and `v2-salience-spread10`. "
        "Recoverable alpha means the training simulator can represent the Unity "
        "twin's beacon-switching rather than approximating it away. The two arms "
        "differ only in beacon placement, so the gap between them measures how "
        "much the published geometry (beacons mostly outside the room) suppressed "
        "the effect."
    )
    L.append("")

    L.append("### Summary-network choice")
    L.append("")
    L.append(
        "`v0-baseline-online` (Conv1D + BiLSTM) against `v0-transformer-online` "
        "(TimeSeriesTransformer). The two arms differ only in the sequence model — "
        "same prior, same observables, same seed, same online budget — so the gap "
        "is attributable to the architecture. Calibration error is the column to "
        "read first; posterior contraction says which net extracts more "
        "information from the trajectories. Note the transformer is the smaller "
        "model here (~229k parameters against ~633k), so a tie favours the "
        "transformer on parameter efficiency."
    )
    L.append("")

    L.append("### Training mode (offline vs online)")
    L.append("")
    L.append(
        "`v0-baseline` (offline, 5,000 simulations) against `v0-baseline-online` "
        "(online, ~480,000). This pair is the evidence that the offline "
        "simulation budget, not the architecture or the model, drove the poor "
        "calibration reported across the offline arms."
    )
    L.append("")

    L.append("## Not addressed by this night")
    L.append("")
    L.append("| Reviewer point | Status |")
    L.append("|----------------|--------|")
    L.append("| Part 1 — collision avoidance | Addressed: `v1-collision`, `v1-collision-kappa`, plus `outputs/scenarios/` |")
    L.append("| Part 1 — navigation scenarios | Addressed outside this harness: `outputs/scenarios/report.md` |")
    L.append("| 4 — partial pooling / hierarchical | Needs CompositionalWorkflow + a DiffusionModel estimator (V6) — deferred |")
    L.append("| 5 — digital twin technical detail | Writing task, not an experiment |")
    L.append("| 6 — time-varying w | Simulator supports parameter paths; needs a prior on (w0, tau) and an expander (V5) — deferred |")
    L.append("| 9 — 2D constraint and path to 3D | Writing task, not an experiment |")
    L.append("")

    out = OUT_ROOT / "SUMMARY.md"
    out.write_text("\n".join(L))
    print(f"Wrote {out}  ({len(done)}/{len(ALL_VARIANTS)} variants)")


if __name__ == "__main__":
    main()
