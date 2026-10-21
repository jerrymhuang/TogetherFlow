"""Matched 2x2 architectures and offline budgets against the four-parameter reference.

Run one arm per process. The queue manages GPU serialization.
"""
import os
os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import argparse
from dataclasses import replace
import logging
from pathlib import Path
import jax
from run_main_paper_offline import bank, run_budget
from variable_radius_variants import BASE

NETWORKS = {f"{summary}-{inference}": (summary, inference)
            for summary in ("transformer", "bdlstm")
            for inference in ("flow_matching", "diffusion")}
ARMS = [*NETWORKS, "offline-3000", "offline-30000"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=ARMS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--regularized-offline", action="store_true")
    parser.add_argument("--banks", type=Path, help="Reuse existing banks without changing simulated data")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--training-only", action="store_true", help="Validation pilot; do not evaluate test posteriors")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    if args.regularized_offline and args.arm in NETWORKS:
        parser.error("regularized-offline applies only to the offline budgets")
    args.matched = True
    args.online = args.arm in NETWORKS
    args.patience, args.batch_size = 0, 32
    args.n_eval, args.draws, args.restart_training = 300, 1000, False
    variant = BASE
    if args.online:
        summary, inference = NETWORKS[args.arm]
        variant = replace(BASE, summary_net=summary, inference_net=inference)
        budget = args.epochs * variant.num_batches_per_epoch * variant.batch_size
    else:
        budget = int(args.arm.split('-')[-1])
    if args.smoke:
        args.epochs, args.n_eval, args.draws, args.batch_size = 1, 4, 8, 4
        variant = replace(variant, num_agents=4, time_horizon=2., epochs=1, num_batches_per_epoch=1, batch_size=4)
        budget = 4 if args.online else (8 if budget == 3000 else 16)
    elif jax.default_backend() != "gpu":
        raise RuntimeError("Research retraining requires GPU")
    variant = replace(variant, online=args.online, epochs=args.epochs, batch_size=args.batch_size,
                      early_stopping_patience=0, n_val=args.n_eval, n_test=args.n_eval)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data_root = args.banks or args.output / "banks"
    validation = bank(data_root / "validation", variant, args.n_eval, BASE.seed + 100000)
    test = None if args.training_only else bank(data_root / "test", variant, args.n_eval, BASE.seed + 200000)
    # Generate the common maximum bank even when the 3K arm is first.
    train = None if args.online else bank(data_root / "train", variant,
        16 if args.smoke else 30000, BASE.seed)
    run_budget(variant, budget, train, validation, test, args.output / args.arm, args)
    if not args.patience:
        import json
        trained = json.loads((args.output / args.arm / "training_complete.json").read_text())
        if trained["epochs"] != args.epochs or trained["selected_epoch"] != args.epochs:
            raise RuntimeError("Full-epoch/final-weight contract violated")


if __name__ == "__main__":
    main()
