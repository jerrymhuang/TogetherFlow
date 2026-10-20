"""Build extension-specific figures from saved results and the salience rule.

Run from the repository root with .venv/bin/python. No inference is retrained.
The salience example is a deterministic illustrative layout, not a test bank.
"""
from pathlib import Path
import argparse
import csv
import json
import os
import re
import shutil
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "togetherflow-figure-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
FIGURES = OUT / "figures"
DATA = OUT / "figure_data"
FIGURES.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})


def salience_selection():
    beacons = np.array([[-12., -12.], [-12., 12.], [12., -12.], [12., 12.]])
    strengths = np.array([1., 1., 1., 8.])
    x, y = np.meshgrid(np.linspace(-4, 4, 121), np.linspace(-5, 5, 151))
    points = np.stack([x, y], axis=-1)
    distance = np.linalg.norm(points[..., None, :] - beacons, axis=-1)
    colors = ["#60a0c0", "#e5b65b", "#77b58b", "#7755a5"]
    alphas = np.linspace(0., 3., 301)
    fraction = np.array([(np.argmax(strengths**a / distance, axis=-1) == 3).mean()
                         for a in alphas])
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.0), layout="constrained")
    for ax, alpha, letter in zip(axes.flat[:3], (0., .25, 1.), "abc"):
        selected = np.argmax(strengths**alpha / distance, axis=-1)
        ax.pcolormesh(x, y, selected, cmap=ListedColormap(colors), vmin=-.5,
                      vmax=3.5, shading="nearest", rasterized=True)
        ax.set(xlim=(-4, 4), ylim=(-5, 5), xlabel="Room x", ylabel="Room y",
               title=rf"({letter}) Selected beacon, $\alpha={alpha:g}$", aspect="equal")
    ax = axes.flat[3]
    ax.plot(alphas, fraction, color=colors[3], lw=2)
    for a in (0., .25, 1.):
        f = (np.argmax(strengths**a / distance, axis=-1) == 3).mean()
        ax.plot(a, f, "o", color=colors[3])
    ax.set(xlabel=r"Salience exponent $\alpha$", ylabel="Grid fraction selecting beacon 4",
           xlim=(0, 3), ylim=(0, 1.04), title="(d) Selection response")
    ax.grid(alpha=.2)
    fig.legend(handles=[Patch(color=c, label=f"Beacon {i + 1}" +
                            (" (salience 8)" if i == 3 else " (salience 1)"))
                        for i, c in enumerate(colors)], loc="outside lower center", ncol=2)
    fig.savefig(FIGURES / "ext-salience-selection.pdf", bbox_inches="tight")
    plt.close(fig)
    with (DATA / "salience_selection.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha", "salient_beacon_grid_fraction"])
        writer.writerows(zip(alphas, fraction))
    (DATA / "salience_design.json").write_text(json.dumps({
        "purpose": "deterministic mechanism illustration, not an inference experiment",
        "beacons": beacons.tolist(), "strengths": strengths.tolist(),
        "room": [8, 10], "grid_shape": list(x.shape),
        "rule": "argmax(s_b ** alpha / distance)",
    }, indent=2) + "\n")


def individual_posteriors(style="auto", posterior_file=None):
    if style not in ("auto", "violin", "interval"):
        raise ValueError("Individual plot style must be auto, violin, or interval.")
    source = ROOT / "outputs/agent-radius-pooling/bounded-radius-online/agent_posteriors.csv"
    with source.open() as f:
        rows = [row for row in csv.DictReader(f) if int(row["trial"]) in (0, 1)]
    for trial in (0, 1):
        if sorted(int(row["agent"]) for row in rows if int(row["trial"]) == trial) != list(range(1, 13)):
            raise ValueError("Expected all 12 agent slots in each of the first two held-out groups.")
    posterior_file = Path(posterior_file) if posterior_file is not None else source.parent / "posterior.npz"
    if style == "auto":
        style = "violin" if posterior_file.is_file() else "interval"
    samples = None
    if style == "violin":
        if not posterior_file.is_file():
            raise FileNotFoundError(
                f"Violin plots require the actual posterior draws: {posterior_file}. "
                "Restore posterior.npz from the bounded-radius-online run, containing "
                "physical-scale 'samples' (groups, draws, 30) and 'truth' (groups, 30). "
                "The CSV's medians and interval endpoints cannot determine a posterior density. "
                "The existing figure has not been changed.")
        with np.load(posterior_file, allow_pickle=False) as stored:
            samples = stored["samples"][:2].copy()
            truths = stored["truth"][:2].copy()
        if (samples.ndim != 3 or samples.shape[0] != 2 or samples.shape[1] < 2
                or samples.shape[2] != 30 or truths.shape != (2, 30)):
            raise ValueError("Expected posterior samples for at least two groups, two draws, and 30 targets.")
        if not np.isfinite(samples).all() or not np.isfinite(truths).all():
            raise ValueError("Posterior samples and truths must be finite.")
        if (np.any(samples[:, :, 6:18] < 0) or np.any(samples[:, :, 6:18] > 1)
                or np.any(samples[:, :, 18:30] < 0) or np.any(samples[:, :, 18:30] > 10)):
            raise ValueError("Expected physical-scale weights in [0,1] and radii in [0,10].")
        for row in rows:
            trial, agent = int(row["trial"]), int(row["agent"]) - 1
            for name, slot in (("weight", 6 + agent), ("radius", 18 + agent)):
                if not np.isclose(truths[trial, slot], float(row[name + "_truth"]), rtol=1e-6, atol=1e-7):
                    raise ValueError("Posterior truths do not match the original held-out groups and agent ordering.")
    fig, axes = plt.subplots(2, 2, figsize=(9.1, 5.6), sharex=True, sharey="row",
                             layout="constrained")
    for col, trial in enumerate((0, 1)):
        group = sorted((row for row in rows if int(row["trial"]) == trial),
                       key=lambda row: int(row["agent"]))
        agents = np.array([int(row["agent"]) for row in group])
        for r, name, color, ylabel in [(0, "weight", "#6969dd", r"Influence weight $w_a$"),
                                        (1, "radius", "#593b86", r"Sensing radius $r_a$")]:
            ax = axes[r, col]
            if style == "violin":
                start = 6 if name == "weight" else 18
                draws = samples[trial, :, start:start + 12]
                lo, med, hi = np.quantile(draws, [.025, .5, .975], axis=0)
                parts = ax.violinplot(draws, positions=agents, widths=.75,
                                      showmeans=False, showmedians=False,
                                      showextrema=False, points=150, bw_method="scott")
                for body in parts["bodies"]:
                    body.set_facecolor(color)
                    body.set_edgecolor(color)
                    body.set_alpha(.45)
                ax.vlines(agents, lo, hi, color=color, linewidth=1.6, zorder=3)
                ax.scatter(agents, med, facecolor="white", edgecolor=color, s=17, zorder=4)
            else:
                med = np.array([float(row[name + "_median"]) for row in group])
                lo = np.array([float(row[name + "_q025"]) for row in group])
                hi = np.array([float(row[name + "_q975"]) for row in group])
                ax.errorbar(agents - .1, med, yerr=np.array([med - lo, hi - med]),
                            fmt="o", color=color, ms=4, capsize=2,
                            label="Posterior median and 95% interval")
            truth = np.array([float(row[name + "_truth"]) for row in group])
            assert np.all(lo <= med) and np.all(med <= hi)
            ax.scatter(agents + .1, truth, marker="x", color="black", s=24,
                       zorder=3, label="Ground truth")
            ax.set(xlim=(.5, 12.5), xticks=np.arange(1, 13), ylabel=ylabel)
            ax.grid(axis="y", alpha=.18)
            if r == 0:
                ax.set(title=f"Held-out group {trial + 1}", ylim=(0, 1))
            else:
                ax.set(xlabel="Agent slot")
    radius_max = max(float(row[key]) for row in rows
                     for key in ("radius_q975", "radius_truth"))
    if style == "violin":
        radius_max = max(radius_max, float(samples[:, :, 18:30].max()))
    axes[1, 0].set_ylim(0, radius_max * 1.05)
    if style == "violin":
        handles = [Patch(facecolor="#6969dd", alpha=.45, label="Posterior density"),
                   Line2D([], [], color="#6969dd", lw=1.6, label="95% credible interval"),
                   Line2D([], [], color="#6969dd", marker="o", markerfacecolor="white",
                          linestyle="none", label="Posterior median"),
                   Line2D([], [], color="black", marker="x", linestyle="none", label="Ground truth")]
        fig.legend(handles=handles, loc="outside lower center", ncol=2)
    else:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=2)
    fig.savefig(FIGURES / "ext-individual-posteriors.pdf", bbox_inches="tight")
    plt.close(fig)
    with (DATA / "individual_examples.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    caption = (
        r"Individual posterior distributions for the first two held-out groups in the 12-agent hierarchy, selected by index. The upper row shows influence weights and the lower row sensing radii. Violin widths show kernel-density estimates from posterior draws, white points mark posterior medians, vertical lines show marginal $95\%$ credible intervals, and black crosses show simulated truths. Agent slots identify individuals within each group, not persistent individuals across groups."
        if style == "violin" else
        r"Individual posterior estimates for the first two held-out groups in the 12-agent hierarchy, selected by index. The upper row shows influence weights and the lower row sensing radii. Colored points and bars show posterior medians and marginal $95\%$ credible intervals; black crosses show simulated truths. Agent slots identify individuals within each group, not persistent individuals across groups."
    )
    (DATA / "individual_caption.tex").write_text(caption + "\n")
    info = {"source": str(posterior_file) if style == "violin" else str(source),
            "selection": "trial indices 0 and 1; all 12 agents per trial",
            "style": style, "density": "actual posterior draws" if style == "violin" else None}
    if style == "violin":
        np.savez_compressed(DATA / "individual_posterior_draws.npz", samples=samples, truth=truths)
        info["draws_per_group"] = samples.shape[1]
    (DATA / "individual_plot.json").write_text(json.dumps(info, indent=2) + "\n")
    manuscript = OUT / "model_extensions.tex"
    if manuscript.is_file():
        text = manuscript.read_text()
        for block in re.findall(r"\\begin\{figure\}.*?\\end\{figure\}", text, re.S):
            if r"\label{fig:individual-posteriors}" in block:
                revised = re.sub(r"\\caption\{[^\n]*\}", lambda _: r"\caption{" + caption + "}", block, count=1)
                manuscript.write_text(text.replace(block, revised, 1))
                break
    sources_file = OUT / "figure_sources.csv"
    if sources_file.is_file():
        with sources_file.open() as f:
            sources = list(csv.DictReader(f))
        for row in sources:
            if row["asset"] == "figures/ext-individual-posteriors.pdf":
                row["source"] = info["source"] + "; trial indices 0 and 1"
        with sources_file.open("w") as f:
            writer = csv.DictWriter(f, fieldnames=["asset", "source"])
            writer.writeheader()
            writer.writerows(sources)
    provenance_file = DATA / "provenance.json"
    if provenance_file.is_file():
        provenance = json.loads(provenance_file.read_text())
        provenance["hierarchy"] = info
        provenance_file.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Individual posterior plot: {style}. Caption: {DATA / 'individual_caption.tex'}")
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--individual-style", choices=("auto", "violin", "interval"), default="auto")
    parser.add_argument("--posterior-file", type=Path, help="Physical-scale posterior.npz from the selected hierarchy.")
    parser.add_argument("--individual-only", action="store_true", help="Build only the individual posterior figure.")
    args = parser.parse_args()
    try:
        # Validate requested violin data before replacing any figure assets.
        hierarchy = individual_posteriors(args.individual_style, args.posterior_file)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    if args.individual_only:
        return
    shutil.copy2(ROOT / "outputs/scenarios/figures/crossing-groups.png",
                 FIGURES / "ext-collision-crossing.png")
    shutil.copy2(ROOT / "outputs/variable-radius/superstats-variable-path/time_varying_posterior.png",
                 FIGURES / "ext-timevarying-paths.png")
    salience_selection()
    report = (ROOT / "outputs/scenarios/report.md").read_text()
    crossing = report.split("## Two groups crossing with opposing goals", 1)[1].split("\n## ", 1)[0]
    (DATA / "crossing_scenario.md").write_text("# Two groups crossing with opposing goals\n" + crossing)
    (DATA / "provenance.json").write_text(json.dumps({
        "collision": {"source": "outputs/scenarios/figures/crossing-groups.png",
                      "metrics": "outputs/scenarios/report.md",
                      "purpose": "saved fixed-parameter behavioral example; not prior-averaged recovery"},
        "salience": {"source": "analytic score rule; see salience_design.json",
                     "purpose": "illustration, not a new inference result"},
        "time_varying": {"source": "outputs/variable-radius/superstats-variable-path/time_varying_posterior.png",
                         "selection": "first three test datasets, as selected by run_v5b_path.py"},
        "hierarchy": hierarchy,
    }, indent=2) + "\n")
    print("Built four extension-specific figures; no training performed.")


if __name__ == "__main__":
    main()
