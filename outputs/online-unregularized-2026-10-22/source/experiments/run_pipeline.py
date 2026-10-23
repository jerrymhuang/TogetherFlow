"""Unattended driver for the full variant rerun.

    uv run python experiments/run_pipeline.py              # run everything not yet done
    uv run python experiments/run_pipeline.py --force      # rerun everything
    uv run python experiments/run_pipeline.py --dry-run    # show the plan and exit
    uv run python experiments/run_pipeline.py --only v0-reference v2-salience-spread10

Design notes for a multi-hour unattended run:

* **Each arm runs in its own subprocess.** Training eleven arms in one process
  accumulates JAX/GPU allocations across arms; a fresh process per arm returns
  all of it to the OS on exit. This is the main reason this driver exists rather
  than looping in Python.
* **Resumable.** An arm whose `report.md` already exists is skipped, so an
  interrupted run continues where it stopped instead of starting over.
* **Isolated failures.** A crashing arm is recorded and the pipeline moves on.
* **Per-arm timeout**, so one hung arm cannot consume the whole night.
* **Live status** in `outputs/variants/STATUS.md`, rewritten after every arm, so
  progress is visible without reading logs.
"""

import argparse
import datetime as dt
import os
import pathlib
import subprocess
import sys
import time

# CRITICAL: this driver imports `variants`, which reaches togetherflow -> bayesflow
# -> keras -> jax. On import JAX preallocates ~75% of the GPU (9 GB of 12 GB here)
# *in this process*, even though the driver never computes anything. Every child
# arm is then left with too little memory to create CUDA library handles, and
# fails during initialisation with errors that look like unrelated kernel bugs:
#   "Autotuning failed ... No configs could be compiled"
#   "INTERNAL: BlasLt is unavailable"
#   "gpusolverDnCreate(&handle) failed: cuSolver internal error"
# Pinning the driver to CPU keeps the GPU entirely for the arms. Must be set
# before any import that reaches jax. run_arm() removes it for the children.
os.environ["JAX_PLATFORMS"] = "cpu"

sys.path.insert(0, str(pathlib.Path(__file__).parent))

ROOT = pathlib.Path(__file__).parent.parent
OUT_ROOT = ROOT / "outputs" / "variants"
LOG_DIR = OUT_ROOT / "logs"

# Generous: the slowest arm so far took ~13 min. This is a hang guard, not a budget.
PER_ARM_TIMEOUT_S = 90 * 60


def _now():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_status(results, planned, started_at):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    done = sum(1 for v in results.values() if v["status"] in ("ok", "skipped"))
    lines = [
        "# Pipeline status",
        "",
        f"Started {started_at}  ·  updated {_now()}",
        f"**{done} / {len(planned)} arms complete**",
        "",
        "| Arm | Status | Wall-clock | Log |",
        "|-----|--------|-----------|-----|",
    ]
    for slug in planned:
        r = results.get(slug)
        if r is None:
            lines.append(f"| `{slug}` | pending | — | — |")
        else:
            mins = f"{r['seconds'] / 60:.1f} min" if r["seconds"] else "—"
            lines.append(
                f"| `{slug}` | {r['status']} | {mins} | [log](logs/{slug}.log) |"
            )
    lines += ["", "Per-arm reports: `outputs/variants/<slug>/report.md`",
              "Cross-arm comparison: `outputs/variants/SUMMARY.md`", ""]
    (OUT_ROOT / "STATUS.md").write_text("\n".join(lines))


def gpu_used_mib():
    """Currently allocated GPU memory, or None if it cannot be determined."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode == 0:
            return int(out.stdout.strip().splitlines()[0])
    except Exception:                                       # noqa: BLE001
        pass
    return None


def wait_for_free_gpu(threshold_mib=1000, timeout_s=180):
    """Block until the GPU is essentially idle before starting the next arm.

    A finished arm's memory is not always released the instant its process exits,
    and starting the next arm against a still-occupied GPU makes cuBLAS/autotuner
    initialisation fail in ways that look like unrelated kernel bugs ("No configs
    could be compiled", "BlasLt is unavailable"). Waiting here removes a whole
    class of spurious mid-pipeline failures.
    """
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        used = gpu_used_mib()
        if used is None or used < threshold_mib:
            return used
        time.sleep(5)
    return gpu_used_mib()


def run_arm(slug, force):
    """Run one arm in a fresh subprocess. Returns (status, seconds)."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{slug}.log"

    cmd = [sys.executable, str(ROOT / "experiments" / "run_variant.py"), slug]
    if force:
        cmd.append("--force")

    env = dict(os.environ)
    env.setdefault("KERAS_BACKEND", "jax")
    # The driver pins itself to CPU (see top of file); the arms must NOT inherit
    # that or they would train on CPU and take days.
    env.pop("JAX_PLATFORMS", None)
    # Do NOT set --xla_gpu_enable_triton_gemm=false here: it forces a cuBLASLt
    # fallback that is unavailable in this build and fails every arm with
    # "INTERNAL: BlasLt is unavailable". The autotuner failures that motivated it
    # were GPU memory pressure from a previous arm, which wait_for_free_gpu below
    # handles at the source.

    t0 = time.time()
    with open(log_path, "w") as log:
        log.write(f"# {slug} — started {_now()}\n")
        log.flush()
        try:
            proc = subprocess.run(
                cmd, stdout=log, stderr=subprocess.STDOUT,
                env=env, cwd=str(ROOT), timeout=PER_ARM_TIMEOUT_S,
            )
            status = "ok" if proc.returncode == 0 else f"failed (rc={proc.returncode})"
        except subprocess.TimeoutExpired:
            status = f"timeout (>{PER_ARM_TIMEOUT_S // 60}min)"
    return status, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rerun arms that already have a report")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    ap.add_argument("--only", nargs="+", metavar="SLUG", help="run only these arms")
    args = ap.parse_args()

    os.environ.setdefault("KERAS_BACKEND", "jax")
    from variants import ALL_VARIANTS, BY_SLUG

    planned = [v.slug for v in ALL_VARIANTS]
    if args.only:
        unknown = [s for s in args.only if s not in BY_SLUG]
        if unknown:
            ap.error(f"unknown arm(s): {', '.join(unknown)}")
        planned = list(args.only)

    print(f"=== Pipeline: {len(planned)} arm(s) ===")
    for slug in planned:
        v = BY_SLUG[slug]
        already = (OUT_ROOT / slug / "report.md").exists()
        mark = "RERUN" if (already and args.force) else ("skip" if already else "run")
        print(f"  [{mark:>5}] {slug:<26} net={v.summary_net:<11} "
              f"rel_heading={str(v.relative_heading):<5} infer={len(v.infer)}")
    if args.dry_run:
        print("\n--dry-run: nothing executed.")
        return

    started_at = _now()
    results, t_start = {}, time.time()
    write_status(results, planned, started_at)

    for i, slug in enumerate(planned, 1):
        if (OUT_ROOT / slug / "report.md").exists() and not args.force:
            print(f"[{i}/{len(planned)}] {slug}: already complete, skipping")
            results[slug] = {"status": "skipped", "seconds": 0.0}
            write_status(results, planned, started_at)
            continue

        used = wait_for_free_gpu()
        if used is not None and used >= 1000:
            print(f"[{i}/{len(planned)}] warning: GPU still holds {used} MiB "
                  f"after waiting; starting anyway", flush=True)
        print(f"[{i}/{len(planned)}] {slug}: running ... (GPU {used} MiB free-check)",
              flush=True)
        status, secs = run_arm(slug, args.force)

        # One retry: the failures seen so far were transient GPU-initialisation
        # problems, which a fresh process after a settle usually clears. A
        # genuine bug fails identically twice and costs only the extra attempt.
        if status != "ok":
            print(f"[{i}/{len(planned)}] {slug}: {status} — retrying once", flush=True)
            wait_for_free_gpu()
            time.sleep(15)
            status2, secs2 = run_arm(slug, force=True)
            status = status2 if status2 == "ok" else f"{status} (retry: {status2})"
            secs += secs2

        results[slug] = {"status": status, "seconds": secs}
        print(f"[{i}/{len(planned)}] {slug}: {status} in {secs / 60:.1f} min", flush=True)
        write_status(results, planned, started_at)

    # Refresh the cross-arm comparison from whatever completed.
    try:
        subprocess.run(
            [sys.executable, str(ROOT / "experiments" / "summarize_night.py")],
            cwd=str(ROOT), timeout=600,
            env=dict(os.environ, KERAS_BACKEND="jax", JAX_PLATFORMS="cpu"),
        )
    except Exception as e:                                  # noqa: BLE001
        print(f"summary step failed: {e}")

    total = time.time() - t_start
    print(f"\n=== PIPELINE DONE in {total / 3600:.2f} h ===")
    for slug in planned:
        r = results[slug]
        print(f"  {r['status']:<20} {slug:<26} {r['seconds'] / 60:6.1f} min")
    failed = [s for s, r in results.items() if r["status"] not in ("ok", "skipped")]
    print(f"\n{len(planned) - len(failed)}/{len(planned)} succeeded.")
    if failed:
        print("Failed arms (see outputs/variants/logs/<slug>.log):")
        for s in failed:
            print(f"  - {s}")


if __name__ == "__main__":
    main()
