"""Build the switching recovery/calibration table and copy matching figures."""

import csv
import hashlib
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARAMETERS = ["w", "r", "v", "noise"]
METRICS = ["NRMSE", "Post. Contraction", "ECE"]
ARMS = [("reference", "Reference"), ("switching", "Switching")]


def main():
    (HERE / "figures").mkdir(exist_ok=True)
    (HERE / "source_metrics").mkdir(exist_ok=True)
    table_rows, values, figures = [], [], []
    for index, (arm, label) in enumerate(ARMS):
        source = ROOT / "outputs/variable-radius" / f"rev-variable-{arm}"
        metrics_path = source / "metrics.csv"
        with metrics_path.open() as file:
            metrics = {row[""]: row for row in csv.DictReader(file)}
        shutil.copy2(metrics_path, HERE / "source_metrics" / f"{arm}-metrics.csv")
        shutil.copy2(
            source / "manifest.json", HERE / "source_metrics" / f"{arm}-manifest.json"
        )
        if index:
            table_rows.append(r"\midrule")
        table_rows.append("% Metrics: " + str(metrics_path.relative_to(ROOT)))
        for number, metric in enumerate(METRICS):
            display = (
                "Posterior contraction" if metric == "Post. Contraction" else metric
            )
            cells = " & ".join(
                f"{float(metrics[metric][parameter]):.3f}" for parameter in PARAMETERS
            )
            prefix = rf"\multirow{{3}}{{*}}{{{label}}}" if number == 0 else ""
            table_rows.append(f"{prefix} & {display} & {cells} " + r"\\")
            for parameter in PARAMETERS:
                values.append(
                    dict(
                        model=label,
                        metric=metric,
                        parameter=parameter,
                        value=metrics[metric][parameter],
                        source=str(metrics_path.relative_to(ROOT)),
                    )
                )
        for kind in ["recovery", "coverage", "calibration_ecdf"]:
            original = source / f"{kind}.png"
            destination = HERE / "figures" / f"ext-{kind}-{arm}.png"
            shutil.copy2(original, destination)
            figures.append(
                dict(
                    figure=str(destination.relative_to(HERE)),
                    source=str(original.relative_to(ROOT)),
                    sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                )
            )
    for name, rows in [("table_values.csv", values), ("figure_sources.csv", figures)]:
        with (HERE / name).open("w") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    body = "\n".join(table_rows) + "\n"
    (HERE / "table_rows.tex").write_text(body)
    fragment = HERE / "beacon_switching.tex"
    text = fragment.read_text()
    start = text.index("% BEGIN RECOVERY ROWS") + len("% BEGIN RECOVERY ROWS")
    end = text.index("% END RECOVERY ROWS")
    fragment.write_text(text[:start] + "\n" + body + text[end:])
    print("Updated recovery/calibration table and six matching figures.")


if __name__ == "__main__":
    main()
