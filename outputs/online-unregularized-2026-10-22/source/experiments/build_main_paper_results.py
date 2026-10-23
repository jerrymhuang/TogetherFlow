"""Assemble a reviewable manuscript fragment only after both research runs finish."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PARAMS = ("w", "r", "v", "noise")
LABELS = (r"$w$", r"$r$", r"$v$", r"$\eta$")


def build(source, destination):
    manifests, metrics = {}, {}
    for n in (3000, 30000):
        run = source / f"offline-{n}"
        if not (run / "complete.json").exists():
            raise RuntimeError(f"Research run has not finished: {run}")
        manifests[n] = json.loads((run / "manifest.json").read_text())
        spec = manifests[n]
        if spec["training_budget"] != n or spec["purpose"] != "main-paper revised offline recovery":
            raise ValueError("Refusing to publish smoke or mislabeled results")
        metrics[n] = pd.read_csv(run / "metrics.csv", index_col=0)
        if not np.isfinite(metrics[n].loc[["NRMSE", "ECE", "Post. Contraction"], list(PARAMS)]).all().all():
            raise ValueError("Metrics contain missing or nonfinite values")
        with np.load(run / "posterior.npz") as posterior:
            for key in PARAMS:
                if posterior[f"samples_{key}"].shape != (spec["test_size"], spec["posterior_draws"], 1):
                    raise ValueError(f"Posterior dimensions differ from the manifest: {key}")
                if n == 30000:
                    with np.load(source / "offline-3000/posterior.npz") as previous:
                        np.testing.assert_array_equal(posterior[f"truth_{key}"], previous[f"truth_{key}"])
    for key in ("seeds", "test_size", "validation_size", "posterior_draws", "packages", "source_sha256"):
        if manifests[3000][key] != manifests[30000][key]:
            raise ValueError(f"Runs do not match on {key}")
    destination.mkdir(parents=True, exist_ok=True)
    figures = destination / "figures"
    figures.mkdir(exist_ok=True)
    for n in (3000, 30000):
        for name in ("recovery", "coverage", "calibration_ecdf", "z_score_contraction", "loss"):
            for suffix in ("pdf", "png"):
                shutil.copy2(source / f"offline-{n}/{name}.{suffix}", figures / f"main-{name}-{n}.{suffix}")
        for name in ("metrics.csv", "metrics_range_normalized.csv", "manifest.json", "history.json", "training_complete.json"):
            shutil.copy2(source / f"offline-{n}" / name, destination / f"offline-{n}-{name}")
        recovery = source / f"offline-{n}/sampling_recovery.json"
        if recovery.exists():
            shutil.copy2(recovery, destination / f"offline-{n}-sampling_recovery.json")

    def score(n, metric, key):
        return f"{metrics[n].loc[metric, key]:.3f}"

    def comparison(metric):
        return ", ".join(f"{label}: ${score(3000, metric, key)}$ to ${score(30000, metric, key)}$"
                         for key, label in zip(PARAMS, LABELS))

    table_rows = []
    for label, key in (("Calibration error (ECE)", "ECE"), ("NRMSE", "NRMSE"),
                       ("Posterior contraction", "Post. Contraction")):
        table_rows.append(label + " & " + " & ".join(score(n, key, p)
                          for n in (3000, 30000) for p in PARAMS) + r" \\")
    table = "\n".join([
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Recovery and calibration for the revised model under two fixed offline training budgets, using a shared independent test bank. NRMSE uses prior-only prediction error as its normalizer; NRMSE and contraction are medians across test simulations.}",
        r"\label{tab:metrics}", r"\small", r"\begin{tabular}{lcccccccc}", r"\toprule",
        r" & \multicolumn{4}{c}{$N=3{,}000$} & \multicolumn{4}{c}{$N=30{,}000$} \\",
        r"Metric & $w$ & $r$ & $v$ & $\eta$ & $w$ & $r$ & $v$ & $\eta$ \\",
        r"\midrule", *table_rows, r"\bottomrule", r"\end{tabular}", r"\end{table}"])

    def figure(name, label, caption):
        return "\n".join([r"\begin{figure}[htbp]", r"\centering",
            rf"\includegraphics[width=\textwidth]{{main-{name}-3000.pdf}}\\[4pt]",
            rf"\includegraphics[width=\textwidth]{{main-{name}-30000.pdf}}",
            rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{figure}"])

    a = manifests[3000]
    selected = {n: json.loads((source / f"offline-{n}/training_complete.json").read_text()) for n in (3000, 30000)}
    method = (
        r"\noindent We assess calibration and parameter recovery for the revised Python model using two fixed offline training budgets, $3{,}000$ and $30{,}000$ simulations. Both infer $(w,r,v,\eta)$ with the priors and observation channels of the variable-radius reference in \autoref{sec:model-extensions}: inputs contain trajectories and neighbor counts at fixed reference distances, excluding statistics evaluated at the unknown true radius. The revised simulator uses relative-heading steering and additive diffusive heading noise. Each trial contains $49$ agents observed for $60$ seconds at $\Delta t=0.1$ seconds."
    )
    training = (
        rf"Both estimators use the same transformer summary network and flow-matching posterior architecture as the extension reference. The smaller training bank is a subset of the larger bank; the runs share ${a['validation_size']}$ independent validation simulations and ${a['test_size']}$ independent test simulations, with ${a['posterior_draws']:,}$ posterior draws per test case. Training uses one initialization seed, batch size ${a['batch_size']}$, and at most ${a['maximum_epochs']}$ epochs, with validation-loss early stopping after ${a['early_stopping_patience']}$ epochs without improvement. The selected epochs are ${selected[3000]['best_epoch']}$ and ${selected[30000]['best_epoch']}$ for the smaller and larger budgets, respectively. Each estimator reuses its fixed bank across epochs; the extension reference instead consumes $480{{,}}000$ fresh online simulations. These runs do not provide an equal-budget comparison of offline and online training."
    )
    calibration = (
        r"\noindent We evaluate computational faithfulness using empirical coverage of marginal credible intervals. Expected calibration error (ECE) averages the absolute difference between empirical and nominal coverage over the diagnostic's interval levels; lower values indicate closer agreement. \autoref{fig:sbc} shows the coverage curves and \autoref{tab:metrics} reports ECE. From the smaller to the larger budget, ECE changes as follows: "
        + comparison("ECE") + "."
    )
    if (metrics[30000].loc["ECE", list(PARAMS)] > metrics[3000].loc["ECE", list(PARAMS)]).all():
        calibration += (
            " For this training seed, calibration error increases for all four parameters. "
            "The larger-budget coverage curves generally lie below nominal coverage, indicating overly confident marginal posteriors. "
            "The gains in recovery therefore coexist with weaker calibration."
        )
    recovery = (
        r"\noindent \autoref{fig:recovery} compares simulated parameters with posterior estimates. We report the median, across test simulations, of posterior-draw RMSE normalized by the median RMSE of prior-only predictions bootstrapped from the same test truths. Posterior contraction is the median clipped proportional reduction in variance relative to the empirical prior variance; both diagnostics use the same saved posterior draws as the plots. From $3{,}000$ to $30{,}000$ training simulations, NRMSE changes as follows: "
        + comparison("NRMSE") + "."
    )
    interpretation = (
        "Posterior contraction changes as follows: " + comparison("Post. Contraction")
        + r". Calibration and contraction describe different properties: small ECE can coexist with broad, weakly informative posteriors. These results assess recovery under the specified simulated model and observation design; they do not establish structural non-identifiability of sensing radius. Differences from the original main-paper estimates also reflect revised dynamics, observables, architecture, and metric definitions."
    )
    fragment = "\n\n".join([
        "% Generated from completed research runs; see README.md for provenance.",
        r"\section{Simulation Study}" + "\n" + r"\label{sec:sim-study}", method, training,
        r"\subsection{Computational Faithfulness: Testing Calibration}",
        figure("coverage", "fig:sbc", r"Empirical coverage for the revised offline estimators trained on $3{,}000$ (top) and $30{,}000$ (bottom) simulations."),
        calibration, r"\subsection{Model Sensitivity: Testing Parameter Recovery}",
        figure("recovery", "fig:recovery", r"Parameter recovery for $3{,}000$ (top) and $30{,}000$ (bottom) offline training simulations. Points show posterior medians and bars marginal $95\%$ credible intervals."),
        table, recovery, interpretation]) + "\n"
    (destination / "sim_study.tex").write_text(fragment)
    (destination / "results_table.tex").write_text(table + "\n")
    (destination / "preview.tex").write_text("\n".join([
        r"\documentclass[11pt]{article}", r"\usepackage[margin=22mm]{geometry}",
        r"\usepackage{amsmath,amssymb,graphicx,booktabs}", r"\usepackage[hidelinks]{hyperref}",
        r"\graphicspath{{figures/}}", r"\begin{document}", r"\input{sim_study.tex}", r"\end{document}", ""]))
    (destination / "README.md").write_text(
        "# Revised main-paper offline results\n\n"
        f"Sources: `{source.relative_to(ROOT) if source.is_relative_to(ROOT) else source}`. Both runs are complete; the original paper fragment remains unchanged.\n\n"
        "The 3K bank is nested in the 30K bank. Validation and test banks are shared and use distinct seed ranges. "
        "Manifests, epoch histories and metrics are included here; raw banks, posterior samples and checkpoints remain in the source folders. "
        "All figures and primary metrics use the same posterior samples. Primary NRMSE uses the empirical-prior bootstrap normalization with a shared bootstrap seed; range-normalized scores are supplied separately and retain the current median aggregation.\n\n"
        "The fragment updates the simulation study. The main model and inference-method descriptions must also describe the corrected steering, diffusive noise, reference-distance observations, transformer network and revised metrics. "
        "A reference to the extension appendix remains unresolved in the standalone preview.\n\n"
        + ("The 30K held-out sampling pass was restarted from the saved checkpoint after it stopped advancing. "
           "The recovery retained the sampler, batch size and shared random seed sequence and saved completed batches; "
           "the recovery metadata is included. No training was repeated.\n"
           if (source / "offline-30000/sampling_recovery.json").exists() else ""))
    if shutil.which("pdflatex"):
        for _ in range(2):
            result = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "preview.tex"],
                                    cwd=destination, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(result.stdout[-3000:])
    with zipfile.ZipFile(destination.with_suffix(".zip"), "w", zipfile.ZIP_DEFLATED) as archive:
        for path in destination.rglob("*"):
            if path.is_file() and path.suffix not in (".aux", ".log", ".out"):
                archive.write(path, path.relative_to(destination.parent))
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/main-paper-revision")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/docs/main_paper_revision")
    parser.add_argument("--wait", action="store_true", help="Assemble automatically when both research runs finish")
    args = parser.parse_args()
    if args.wait:
        while not all((args.source / f"offline-{n}/complete.json").exists() for n in (3000, 30000)):
            time.sleep(30)
    print(build(args.source, args.output), flush=True)


if __name__ == "__main__":
    main()
