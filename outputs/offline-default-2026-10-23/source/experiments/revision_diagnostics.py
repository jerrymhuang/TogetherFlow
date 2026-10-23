"""Conservative report helpers; numerical metrics, not heuristic grades, are evidence."""
import math


def inspect_history(history):
    issues = []
    for key, values in history.items():
        if any(not math.isfinite(float(v)) for v in values):
            issues.append(f"Non-finite values in {key}")
    return {"overall": {"ok": not issues, "issues": issues}}


def check_diagnostics(metrics):
    return {"parameters": {}, "summary": {
        p: "; ".join(f"{m}={float(metrics.loc[m, p]):.3f}" for m in metrics.index)
        for p in metrics.columns}}


def suggest_next_steps(training_report, diag_report):
    return training_report['overall']['issues'] + [
        "Inspect recovery, coverage, and rank plots jointly; no automatic threshold establishes identifiability or convergence."]
