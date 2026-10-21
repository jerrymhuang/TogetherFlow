"""Build the switching appendix table and figure from saved transfer metrics."""

from pathlib import Path
import csv
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "outputs/variable-radius/switching-transfer"
PARAMETERS = ["w", "r", "v", "noise"]
CONDITIONS = [
    ("reference", "reference"),
    ("reference", "switching"),
    ("switching", "reference"),
    ("switching", "switching"),
]
LABELS = {"reference": "Static", "switching": "Switching"}
(HERE / "figures").mkdir(exist_ok=True)
(HERE / "source_metrics").mkdir(exist_ok=True)
rows, metrics = [], {}
for trained, tested in CONDITIONS:
    name = f"rev-variable-{trained}__on__rev-variable-{tested}.csv"
    source = SOURCE / name
    shutil.copy2(source, HERE / "source_metrics" / name)
    with source.open() as file:
        metrics[trained, tested] = {row[""]: row for row in csv.DictReader(file)}
    for metric in ["NRMSE", "Post. Contraction", "ECE"]:
        for parameter in PARAMETERS:
            rows.append(
                {
                    "trained": LABELS[trained],
                    "tested": LABELS[tested],
                    "metric": metric,
                    "parameter": parameter,
                    "value": metrics[trained, tested][metric][parameter],
                    "source": str(source.relative_to(ROOT)),
                }
            )
with (HERE / "transfer_table_values.csv").open("w") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

body = []
for index, (trained, tested) in enumerate(CONDITIONS):
    if index:
        body.append(r"\midrule")
    body.append(
        "% Source: "
        + str(
            (
                SOURCE / f"rev-variable-{trained}__on__rev-variable-{tested}.csv"
            ).relative_to(ROOT)
        )
    )
    for number, metric in enumerate(["NRMSE", "Post. Contraction", "ECE"]):
        label = "Posterior contraction" if metric == "Post. Contraction" else metric
        values = " & ".join(
            f"{float(metrics[trained, tested][metric][p]):.3f}" for p in PARAMETERS
        )
        prefix = (
            rf"\multirow{{3}}{{*}}{{{LABELS[trained]}}} & \multirow{{3}}{{*}}{{{LABELS[tested]}}}"
            if number == 0
            else " & "
        )
        body.append(f"{prefix} & {label} & {values} " + r"\\")
(HERE / "transfer_table_rows.tex").write_text("\n".join(body) + "\n")

plt.rcParams.update(
    {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}
)
figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
colors = ["#7895b2", "#d67a3b", "#6e5699", "#2d8077"]
x = np.arange(4)
for axis, metric in zip(axes, ["NRMSE", "ECE"]):
    for index, (trained, tested) in enumerate(CONDITIONS):
        values = [float(metrics[trained, tested][metric][p]) for p in PARAMETERS]
        axis.bar(
            x + (index - 1.5) * 0.19,
            values,
            width=0.18,
            color=colors[index],
            label=f"{LABELS[trained]} → {LABELS[tested]}",
        )
    axis.set_xticks(x, [r"$w$", r"$r$", r"$v$", r"$\eta$"])
    axis.set_ylabel(metric)
    axis.set_axisbelow(True)
    axis.grid(axis="y", alpha=0.2)
handles, labels = axes[0].get_legend_handles_labels()
figure.legend(
    handles,
    labels,
    loc="upper center",
    ncol=4,
    frameon=False,
    title="Training generator → test generator",
    fontsize=10,
)
figure.tight_layout(rect=(0, 0, 1, 0.82))
for extension in ["pdf", "png"]:
    figure.savefig(
        HERE / "figures" / f"ext-switching-transfer.{extension}",
        dpi=200,
        bbox_inches="tight",
    )
plt.close(figure)
shutil.copy2(SOURCE / "design.json", HERE / "source_metrics" / "design.json")
print("Built table rows, unrounded values, and transfer-summary figures.")

# Keep the manuscript table synchronized with the exported source values.
fragment = HERE / "beacon_switching_transfer.tex"
if fragment.exists():
    text = fragment.read_text()
    start = text.index("% BEGIN TRANSFER ROWS") + len("% BEGIN TRANSFER ROWS")
    end = text.index("% END TRANSFER ROWS")
    fragment.write_text(text[:start] + "\n" + "\n".join(body) + "\n" + text[end:])
fixed = HERE / "source_metrics" / "fixed_radius"
fixed.mkdir(exist_ok=True)
for trained, tested in [
    ("reference", "reference"),
    ("reference", "switching"),
    ("switching", "switching"),
]:
    name = f"rev-fixed-{trained}__on__rev-fixed-{tested}.csv"
    shutil.copy2(ROOT / "outputs/revision/switching-transfer" / name, fixed / name)
