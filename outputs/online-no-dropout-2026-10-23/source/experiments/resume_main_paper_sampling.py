"""Resume held-out inference with persistent batches, retaining the trained models.

Runs the original offline runner, replacing only its sampling entry point with
an equivalent sequence of condition batches and a shared random seed generator.
Completed physical-scale posterior batches and RNG states are saved atomically.
"""
import os
os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from pathlib import Path
import hashlib
import json
import logging
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import keras
import bayesflow as bf
import run_main_paper_offline as runner

ORIGINAL_SAMPLE = bf.workflows.BasicWorkflow.sample
RUN = ROOT / "outputs/main-paper-revision/offline-30000"
KEYS = ("w", "r", "v", "noise")


def sample_in_saved_batches(self, *, num_samples, conditions, batch_size=25, seed=None, **kwargs):
    if not isinstance(seed, int):
        raise ValueError("An explicit integer sampling seed is required")
    count = len(conditions[KEYS[0]])
    directory = RUN / "posterior_chunks"
    directory.mkdir(exist_ok=True)
    spec = {
        "checkpoint_sha256": hashlib.sha256((RUN / "checkpoints/model.keras").read_bytes()).hexdigest(),
        "draws": num_samples, "test_size": count, "batch_size": batch_size, "seed": seed,
        "truth_sha256": hashlib.sha256(b"".join(np.asarray(conditions[k]).tobytes() for k in KEYS)).hexdigest(),
    }
    path = directory / "spec.json"
    if path.exists() and json.loads(path.read_text()) != spec:
        raise ValueError("Cached posterior batches belong to a different checkpoint or test bank")
    runner.write_json(path, spec)
    generator = keras.random.SeedGenerator(seed)
    results = {k: [] for k in KEYS}
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        output = directory / f"batch-{start:03d}-{stop:03d}.npz"
        before = np.array(generator.state)
        if output.exists():
            with np.load(output) as cached:
                np.testing.assert_array_equal(cached["seed_before"], before)
                for key in KEYS:
                    np.testing.assert_array_equal(cached[f"truth_{key}"], conditions[key][start:stop])
                samples = {k: cached[f"samples_{k}"] for k in KEYS}
                generator.state.assign(cached["seed_after"])
            logging.info("Restored posterior batch %d:%d", start, stop)
        else:
            logging.info("Sampling posterior batch %d:%d", start, stop)
            runner.write_json(RUN / "status.json", {
                "stage": "posterior sampling", "completed_test_cases": start,
                "total_test_cases": count, "batch_started_at_unix": time.time(),
            })
            samples = ORIGINAL_SAMPLE(self, num_samples=num_samples,
                conditions={k: np.asarray(v[start:stop]) for k, v in conditions.items()},
                batch_size=batch_size, seed=generator, **kwargs)
            samples = {k: np.asarray(samples[k]) for k in KEYS}
            for key, values in samples.items():
                if values.shape != (stop - start, num_samples, 1) or not np.isfinite(values).all():
                    raise ValueError(f"Invalid posterior batch for {key}")
            temporary = output.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, seed_before=before, seed_after=np.array(generator.state),
                **{f"samples_{k}": samples[k] for k in KEYS},
                **{f"truth_{k}": conditions[k][start:stop] for k in KEYS})
            temporary.replace(output)
            logging.info("Saved posterior batch %d:%d", start, stop)
        for key in KEYS:
            results[key].append(samples[key])
    return {k: np.concatenate(v, axis=0) for k, v in results.items()}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True,
                        format="%(asctime)s %(levelname)s %(message)s")
    runner.write_json(RUN / "sampling_recovery.json", {
        "started_at_unix": time.time(),
        "reason": "Original sampling pass stopped advancing after 150/300 test cases; training was complete.",
        "method": "Reload saved validation-selected checkpoint; retain original sampler, batch size and shared seed sequence; save completed batches.",
        "entrypoint_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })
    bf.workflows.BasicWorkflow.sample = sample_in_saved_batches
    runner.main()
