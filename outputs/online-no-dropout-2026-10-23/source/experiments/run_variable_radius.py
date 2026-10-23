"""Run isolated revision arms and the 2x2 switching mismatch evaluation."""
import os
os.environ.setdefault('KERAS_BACKEND', 'jax')
import argparse
from dataclasses import replace
import json
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np
from variable_radius_variants import ARMS, BY_SLUG, BASE, SWITCH
import run_variant as runner


def cross_evaluate(smoke=False):
    import keras
    import bayesflow as bf
    root = runner.OUT_ROOT
    suffix = '-smoke' if smoke else ''
    arms = [prepare(BASE, smoke), prepare(SWITCH, smoke)]
    out = root / ('switching-transfer' + suffix)
    out.mkdir(parents=True, exist_ok=True)
    # Cache each test bank so both estimators see exactly the same test datasets.
    banks = {v.slug: runner.build_simulator(v, v.seed + 200_000).sample(v.n_test) for v in arms}
    for trained in arms:
        path = root / trained.slug / 'checkpoints' / 'model.keras'
        if not path.exists():
            raise FileNotFoundError(f'Train {trained.slug} first: {path}')
        model = keras.saving.load_model(path)
        workflow = bf.workflows.BasicWorkflow(adapter=runner.build_adapter(trained))
        workflow.approximator = model
        for tested in arms:
            metrics = runner.normalize_metric_index(workflow.compute_default_diagnostics(
                test_data=banks[tested.slug], as_data_frame=True,
                **runner.DIAGNOSTIC_KWARGS))
            metrics.to_csv(out / f'{trained.slug}__on__{tested.slug}.csv')
    (out / 'design.json').write_text(json.dumps({'test_seed': BASE.seed + 200_000,
        'scope': 'Static versus time-varying salience sensitivity with simulated inferred radius',
        'smoke': smoke}, indent=2))


def prepare(v, smoke):
    if not smoke:
        return v
    return replace(v, slug=v.slug+'-smoke', epochs=1, num_batches_per_epoch=1,
                   batch_size=4, n_train=8, n_val=4, n_test=4, time_horizon=2,
                   num_agents=4, early_stopping_patience=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('slug', nargs='?')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--cross-evaluate', action='store_true')
    args = ap.parse_args()
    runner.OUT_ROOT = runner.ROOT / 'outputs' / 'variable-radius'
    if args.list:
        for v in ARMS:
            print(v.slug, v.addresses)
    elif args.cross_evaluate:
        cross_evaluate(args.smoke)
    elif args.slug in BY_SLUG:
        variant = prepare(BY_SLUG[args.slug], args.smoke)
        folder = runner.OUT_ROOT / variant.slug
        resume_diagnostics = (folder / 'history.json').exists() and (folder / 'checkpoints/model.keras').exists() and not (folder / 'report.md').exists()
        runner.run(variant, diagnostics_only=resume_diagnostics)
    else:
        ap.error('Choose a slug from --list or --cross-evaluate')


if __name__ == '__main__':
    main()
