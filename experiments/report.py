"""Generate report.md for a variant run, following the amortized-workflow template.

The descriptive paragraphs are fixed text from the template; everything else is
filled from the actual run. Two blocks are specific to this project: a header
naming the reviewer point the variant answers, and a configuration table
recording the generative settings that distinguish arms from one another.
"""

DESC_CONVERGENCE = (
    "The training loss curve shows the optimization objective over epochs. A healthy curve "
    "decreases smoothly and plateaus. Key warning signs: (i) a growing gap between training "
    "and validation loss indicates overfitting; (ii) loss still visibly decreasing at the "
    "final epoch means the network could benefit from more epochs; (iii) NaN spikes indicate "
    "numerical instability, often caused by extreme simulator outputs or missing standardization."
)

DESC_RECOVERY = (
    "Each panel plots the posterior median (point estimate) against the true parameter value "
    "across held-out simulations. Points falling on the diagonal indicate perfect recovery. "
    "Vertical bars represent 95% posterior credible intervals — their width reflects estimation "
    "uncertainty. Systematic deviations from the diagonal reveal bias; wide intervals indicate "
    "the data are only weakly informative for that parameter."
)

DESC_CALIBRATION = (
    "**Calibration ECDF** — Simulation-based calibration (SBC) plots show the empirical CDF of "
    "posterior ranks. Well-calibrated posteriors produce ECDFs close to the uniform diagonal. "
    "Lines consistently above the diagonal indicate overconfident (too narrow) posteriors; lines "
    "below indicate underconfident (too wide) posteriors."
)

DESC_COVERAGE = (
    "**Coverage** — Shows the fraction of held-out true values falling within nominal credible "
    "intervals (e.g., 50%, 80%, 95%). Well-calibrated models yield empirical coverage matching "
    "the nominal level. Under-coverage means the credible intervals are too narrow; over-coverage "
    "means they are too wide."
)

DESC_CONTRACTION = (
    "The z-score–contraction plot summarizes posterior quality in two dimensions. The x-axis shows "
    "**posterior contraction** — the fraction by which the posterior variance has shrunk relative "
    "to the prior variance. Values near 0 indicate no information gain (the data are uninformative "
    "for that parameter); values near 1 indicate near-complete information gain. The y-axis shows "
    "the **posterior z-score** — the average standardized deviation between the posterior mean and "
    "the true value. Symmetric values around 0 indicate an unbiased (Gaussian-like) posterior. The "
    "ideal region is the middle-right corner (z-scores distributed around 0, high contraction)."
)


def _assess_convergence(tr):
    ok = tr.get("overall", {}).get("ok", False)
    issues = tr.get("overall", {}).get("issues", [])
    if ok and not issues:
        return ("Training converged without detected issues: the loss decreased and plateaued, "
                "and no divergence between training and validation loss was flagged.")
    return ("Training completed with the following flags: " + "; ".join(issues) + ". "
            "Treat the downstream diagnostics as provisional until these are addressed.")


def _assess_group(diag, key, good, bad_word):
    params = diag.get("parameters", {})
    good_p = [p for p, d in params.items() if d.get(key) in good]
    bad_p = [p for p, d in params.items() if d.get(key) not in good and key in d]
    bits = []
    if good_p:
        bits.append(f"{', '.join(good_p)} show {bad_word} in the acceptable range")
    if bad_p:
        bits.append(f"{', '.join(bad_p)} fall short")
    return ("; ".join(bits) + ".") if bits else "No per-parameter ratings were available."


def write_report(results_dir, variant, metrics, training_report, diag_report,
                 next_steps, summary_dim, elapsed_s):
    params = list(metrics.columns)
    rows = list(metrics.index)

    # Config table — the fields that actually differ between arms.
    cfg = [
        ("Reviewer point addressed", variant.addresses),
        ("Prior", variant.prior.__name__ if hasattr(variant.prior, "__name__") else str(variant.prior)),
        ("Inferred parameters", ", ".join(variant.infer)),
        ("Observable channels", ", ".join(variant.channels)),
        ("Reference radii (r-free counts)", str(variant.reference_radii) if variant.reference_radii else "—"),
        ("Beacon strengths", str(variant.beacon_strengths) if variant.beacon_strengths else "uniform"),
        ("Beacon spread", f"{variant.beacon_spread:g}"),
        ("Heading update",
         "relative bearing (rotationally invariant)"
         if getattr(variant, "relative_heading", False)
         else "absolute bearing (legacy; NOT rotationally invariant)"),
        ("Agents / beacons", f"{variant.num_agents} / {variant.num_beacons}"),
        ("Time horizon / dt", f"{variant.time_horizon:g}s / {variant.dt:g}"),
        ("Seed", str(variant.seed)),
    ]

    train_cfg = [
        ("Inference network", "FlowMatching (Base, widths 256x4, time_embedding_dim 32)"),
        ("Summary network",
         {"bdlstm": f"SummaryNet — Conv1D + BiLSTM (summary_dim={summary_dim})",
          "transformer": f"TransformerSummaryNet — TimeSeriesTransformer, Base "
                         f"(summary_dim={summary_dim})"}.get(
             getattr(variant, "summary_net", "bdlstm"), variant.summary_net)),
        ("Epochs", str(variant.epochs)),
        ("Batch size", str(variant.batch_size)),
        ("Validation data", f"{variant.n_val} simulations"),
        ("Test data", f"{variant.n_test} simulations"),
        ("Training mode", "online" if getattr(variant, "online", False) else "offline"),
        ("Early stopping", f"patience {variant.early_stopping_patience}, best weights restored"
                           if getattr(variant, "early_stopping_patience", 0) else "off"),
        ("Simulation budget",
         f"{variant.epochs} x {variant.num_batches_per_epoch} x {variant.batch_size} "
         f"(online, fresh each batch)" if getattr(variant, "online", False)
         else f"{variant.n_train} training simulations"),
        ("Wall-clock", f"{elapsed_s / 60:.1f} min"),
    ]

    L = []
    L.append(f"# {variant.title}")
    L.append("")
    L.append(f"**Variant:** `{variant.slug}`  ")
    L.append(f"**Addresses:** {variant.addresses}")
    L.append("")
    L.append("## Why this arm exists")
    L.append("")
    L.append(variant.rationale)
    L.append("")

    L.append("## Generative Configuration")
    L.append("")
    L.append("| Setting | Value |")
    L.append("|---------|-------|")
    for k, v in cfg:
        L.append(f"| {k} | {v} |")
    L.append("")

    L.append("## Training and Network Configuration")
    L.append("")
    L.append("| Setting | Value |")
    L.append("|---------|-------|")
    for k, v in train_cfg:
        L.append(f"| {k} | {v} |")
    L.append("")

    L.append("## Convergence")
    L.append("")
    L.append("![Training loss](loss.png)")
    L.append("")
    L.append(DESC_CONVERGENCE)
    L.append("")
    L.append(f"**Assessment:** {_assess_convergence(training_report)}")
    L.append("")

    L.append("## Parameter Recovery")
    L.append("")
    L.append("![Parameter recovery](recovery.png)")
    L.append("")
    L.append(DESC_RECOVERY)
    L.append("")
    L.append("**Assessment:** " + _assess_group(
        diag_report, "recovery", ("excellent", "good"), "recovery"))
    L.append("")

    L.append("## Calibration and Coverage")
    L.append("")
    L.append("![Calibration ECDF](calibration_ecdf.png)")
    L.append("")
    L.append(DESC_CALIBRATION)
    L.append("")
    L.append("![Coverage](coverage.png)")
    L.append("")
    L.append(DESC_COVERAGE)
    L.append("")
    L.append("**Assessment:** " + _assess_group(
        diag_report, "calibration", ("excellent", "fair"), "calibration"))
    L.append("")

    L.append("## Posterior Z-Score and Contraction")
    L.append("")
    L.append("![Z-score and contraction](z_score_contraction.png)")
    L.append("")
    L.append(DESC_CONTRACTION)
    L.append("")
    L.append("**Assessment:** " + _assess_group(
        diag_report, "contraction", ("high", "medium"), "contraction"))
    L.append("")

    L.append("## Numerical Diagnostic Summary")
    L.append("")
    L.append("| Metric | " + " | ".join(params) + " |")
    L.append("|--------|" + "|".join(["-----"] * len(params)) + "|")
    for row in rows:
        vals = [f"{float(metrics.loc[row, p]):.3f}" for p in params]
        L.append(f"| {row} | " + " | ".join(vals) + " |")
    L.append("")
    for p in params:
        L.append(f"**{p}** — {diag_report['summary'].get(p, 'no metrics available')}")
        L.append("")

    L.append("## Suggested Next Steps")
    L.append("")
    for n, step in enumerate(next_steps, 1):
        L.append(f"{n}. {step}")
    L.append("")

    (results_dir / "report.md").write_text("\n".join(L))
