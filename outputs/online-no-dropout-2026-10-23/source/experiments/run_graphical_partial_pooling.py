"""Train graphical inference for population and individual weights and radii.

Usage:
    PYTHONPATH=. .venv/bin/python experiments/run_graphical_partial_pooling.py --smoke

This uses fixed simulation banks and two flow-matching networks. The population
network conditions on the entire group's features; the agent network conditions
on those features and the population parameters. All 24 agent parameters are
inferred jointly.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("NUMBA_NUM_THREADS", "4")

import bayesflow as bf
import keras
import numpy as np

from graphical_partial_pooling import build_simulator, build_approximator, DURATION
from run_agent_radius_pooling import (
    AGENTS,
    RADIUS_MAX,
    PARAMETER_NAMES,
    physical_targets,
    diagnostics,
    plot_recovery,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "outputs/graphical-partial-pooling/offline"
    )
    parser.add_argument("--train", type=int, default=5000)
    parser.add_argument("--validation", type=int, default=300)
    parser.add_argument("--test", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=831)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Small banks and networks; two training epochs",
    )
    args = parser.parse_args()
    if args.smoke:
        args.train, args.validation, args.test = 16, 4, 4
        args.epochs, args.samples, args.batch_size = 2, 20, 8
        if args.out == parser.get_default("out"):
            args.out = args.out.with_name("smoke")
    if (
        min(
            args.train,
            args.validation,
            args.test,
            args.epochs,
            args.samples,
            args.batch_size,
        )
        < 1
    ):
        parser.error("All sizes and epoch counts must be positive")
    if args.train < 2 or args.test < 2 or args.samples < 2:
        parser.error(
            "Training/test banks and posterior sample counts must be at least two"
        )
    return args


def make_bank(simulator, size, seed, path):
    """Save the graph's transformed parameters and observed group features."""
    np.random.seed(seed)
    chunks = []
    for start in range(0, size, 32):
        chunks.append(simulator.sample(min(32, size - start)))
        print(f"{path.stem}: {min(start + 32, size)}/{size}", flush=True)
    bank = {
        key: np.concatenate([chunk[key] for chunk in chunks]).astype("float32")
        for key in chunks[0]
    }
    np.savez_compressed(path, **bank)
    return bank


def physical_parameters(data):
    transformed = np.concatenate([data["population"], data["agents"]], axis=-1)
    return physical_targets(transformed, RADIUS_MAX)


def sample_posteriors(approximator, features, num_samples):
    """Bound memory by sampling one held-out group at a time."""
    draws = []
    for index in range(len(features)):
        samples = approximator.sample(
            num_samples=num_samples,
            conditions={"features": features[index : index + 1]},
        )
        draws.append(physical_parameters(samples))
    samples = np.concatenate(draws)
    if not np.isfinite(samples).all():
        raise ValueError("Posterior draws contain nonfinite values")
    return samples


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    keras.utils.set_random_seed(args.seed)
    simulator = build_simulator()
    widths = (32, 32) if args.smoke else (256, 256, 256)
    approximator = build_approximator(simulator, widths=widths)
    composition = approximator.graph.network_composition()
    conditions = approximator.graph.network_conditions()
    expected_composition = {0: ["population"], 1: ["agents"]}
    expected_conditions = {0: ["observations"], 1: ["population", "observations"]}
    if composition != expected_composition or conditions != expected_conditions:
        raise ValueError(
            f"Unexpected posterior factorization: {composition}, {conditions}"
        )

    sources = [
        "experiments/run_graphical_partial_pooling.py",
        "experiments/graphical_partial_pooling.py",
        "togetherflow/simulator.py",
        "experiments/run_agent_radius_pooling.py",
        "experiments/run_agent_partial_pooling.py",
    ]
    manifest = vars(args) | {
        "out": str(args.out),
        "bayesflow": bf.__version__,
        "backend": keras.backend.backend(),
        "simulator": "GraphicalSimulator",
        "approximator": "GraphicalApproximator",
        "training": "offline",
        "agents": AGENTS,
        "duration": DURATION,
        "radius_max": RADIUS_MAX,
        "parameters": PARAMETER_NAMES,
        "network": "FlowMatching",
        "widths": widths,
        "time_embedding_dim": 32,
        "network_composition": composition,
        "network_conditions": conditions,
        "checkpoint_selection": "minimum validation loss; patience 25; restore best weights",
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in sources
        },
        "status": "running",
    }
    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    (args.out / "graph.json").write_text(
        json.dumps(
            {
                "generative_edges": list(simulator.graph.edges),
                "network_composition": composition,
                "network_conditions": conditions,
            },
            indent=2,
        )
    )
    train = make_bank(simulator, args.train, args.seed, args.out / "train.npz")
    validation = make_bank(
        simulator, args.validation, args.seed + 100_000, args.out / "validation.npz"
    )
    test = make_bank(simulator, args.test, args.seed + 200_000, args.out / "test.npz")

    keras.utils.set_random_seed(args.seed + 300_000)
    approximator.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))
    validation_data = bf.datasets.OfflineDataset(
        validation,
        batch_size=args.batch_size,
        adapter=approximator.adapter,
        shuffle=False,
    )
    history = approximator.fit(
        dataset=train,
        validation_data=validation_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=25,
                restore_best_weights=True,
            )
        ],
    )
    history = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }
    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    approximator.save(args.out / "model.keras")
    manifest.update(
        epochs_completed=len(history["loss"]),
        selected_epoch=int(np.argmin(history["val_loss"])) + 1,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Reload before sampling so every completed run also verifies its checkpoint.
    restored = keras.models.load_model(args.out / "model.keras", compile=False)
    keras.utils.set_random_seed(args.seed + 400_000)
    samples = sample_posteriors(restored, test["features"], args.samples)
    truth, prior_truth = physical_parameters(test), physical_parameters(train)
    np.savez_compressed(args.out / "posterior.npz", samples=samples, truth=truth)
    np.savez_compressed(
        args.out / "evaluation_bank.npz",
        features=test["features"],
        truth=truth,
        prior_truth=prior_truth,
    )
    diagnostics(args.out, truth, samples, prior_truth)
    plot_recovery(args.out)
    manifest["status"] = "complete"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Saved graphical partial-pooling results to {args.out}", flush=True)


if __name__ == "__main__":
    main()
