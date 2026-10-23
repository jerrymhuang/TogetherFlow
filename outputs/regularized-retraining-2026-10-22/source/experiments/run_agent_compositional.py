"""CPU pilot: joint population/agent inference using one complete group block.

No agent-wise likelihood factorization or multi-trial composition is assumed.
Run: PYTHONPATH=. .venv/bin/python experiments/run_agent_compositional.py
"""
import os
os.environ.setdefault('KERAS_BACKEND', 'jax')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('NUMBA_NUM_THREADS', '4')
import argparse
import json
from pathlib import Path
import numpy as np
import keras
import bayesflow as bf
from scipy.special import expit, logit
from togetherflow import TogetherFlowSimulator
from revision_variants import PARTIAL


@bf.utils.serialization.serializable('custom')
class AgentSummary(bf.networks.SummaryNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = keras.Sequential([
            keras.layers.Conv1D(32, 3, activation='swish'),
            keras.layers.GRU(48), keras.layers.Dense(32)])

    def build(self, input_shape):
        self.net.build(input_shape)
        super().build(input_shape)

    def call(self, x, training=False):
        return self.net(x, training=training)


def make_bank(n, seed, agents):
    sim = TogetherFlowSimulator(num_agents=agents, num_beacons=PARTIAL.num_beacons,
        time_horizon=20, dt=.1, downsample=True, downsample_factor=5,
        prior=PARTIAL.prior, param_names=PARTIAL.param_names,
        relative_heading=PARTIAL.relative_heading,
        diffusive_heading=PARTIAL.diffusive_heading,
        include_agent_weights=True, seed=seed)
    raw = sim.sample(n)
    # Preserve agent ordering. All features are observable, never latent weights
    # or neighbor counts at an unknown radius. The full group is one input.
    x = np.concatenate([raw['positions'], np.sin(raw['rotations']),
        np.cos(raw['rotations']), raw['angular_velocities']], axis=-1).astype('float32')
    physical = np.concatenate([raw[k] for k in ['mu_w', 'v', 'noise', 'sigma_w']]
        + [raw['agent_weights']], axis=-1)
    scales = np.array([1, 2, 1, 1.5] + [1]*agents)
    # Transform before the adapter; no nonlinear adapter Jacobian in composition.
    targets = logit(np.clip(physical/scales, 1e-6, 1-1e-6)).astype('float32')
    return {'targets': targets, 'observations': x}, physical, scales


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=int, default=4096)
    parser.add_argument('--validation', type=int, default=512)
    parser.add_argument('--test', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--samples', type=int, default=256)
    parser.add_argument('--agents', type=int, default=4)
    parser.add_argument('--seed', type=int, default=731)
    parser.add_argument('--out', type=Path, default=Path('outputs/agent-compositional/pilot'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    keras.utils.set_random_seed(args.seed)
    train, truth_train, scales = make_bank(args.train, args.seed, args.agents)
    val, _, _ = make_bank(args.validation, args.seed+100000, args.agents)
    test, truth, _ = make_bank(args.test, args.seed+200000, args.agents)
    adapter = (bf.adapters.Adapter().concatenate(['targets'], into='inference_variables')
        .concatenate(['observations'], into='summary_variables'))
    workflow = bf.workflows.CompositionalWorkflow(adapter=adapter,
        inference_network=bf.networks.DiffusionModel(
            subnet_kwargs={'widths': (128,128), 'time_embedding_dim': 16}),
        summary_network=AgentSummary(), standardize='all')
    manifest = vars(args) | {'out': str(args.out), 'bayesflow': bf.__version__,
        'workflow': type(workflow).__name__, 'composition_blocks': 1,
        'sampling_method': 'sample',
        'radius': 2.5, 'time_horizon': 20, 'backend': 'jax-cpu',
        'purpose': 'exploratory reduced-agent pilot; not comparable to full suite'}
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    history = workflow.fit_offline(train, validation_data=val, epochs=args.epochs,
        batch_size=64, verbose=2)
    workflow.approximator.save(args.out/'model.keras')
    (args.out/'history.json').write_text(json.dumps({k: list(map(float,v)) for k,v in history.history.items()}))
    # BayesFlow rejects compositional_sample with only one evidence block.
    # Use this workflow's ordinary sampler for a joint within-group posterior.
    # Multiple interacting agents must NOT be passed as independent blocks.
    draws = workflow.sample(num_samples=args.samples,
        conditions={'observations': test['observations']},
        batch_size=8, sample_shape=())['targets']
    physical_draws = expit(draws)*scales
    mean = physical_draws.mean(axis=1)
    low, high = np.quantile(physical_draws, [.05,.95], axis=1)
    prior_mean = truth_train.mean(axis=0)
    prior_var = truth_train.var(axis=0)
    names = ['mu_w','v','noise','sigma_w']+[f'w_{i}' for i in range(args.agents)]
    import pandas as pd
    metrics = pd.DataFrame({'parameter': names,
        'rmse': np.sqrt(np.mean((mean-truth)**2, axis=0)),
        'prior_mean_rmse': np.sqrt(np.mean((prior_mean-truth)**2,axis=0)),
        'coverage90': np.mean((truth>=low)&(truth<=high),axis=0),
        'contraction': 1-physical_draws.var(axis=1).mean(axis=0)/prior_var})
    metrics.to_csv(args.out/'metrics.csv', index=False)
    rows = []
    for trial in range(args.test):
        for agent in range(args.agents):
            j = 4+agent
            rows.append({'trial':trial, 'agent':agent, 'truth':truth[trial,j],
                'mean':mean[trial,j], 'q05':low[trial,j], 'q95':high[trial,j]})
    pd.DataFrame(rows).to_csv(args.out/'agent_posteriors.csv',index=False)
    centered_truth = truth[:,4:] - truth[:,4:].mean(axis=1,keepdims=True)
    centered_mean = mean[:,4:] - mean[:,4:].mean(axis=1,keepdims=True)
    summary = {'within_group_rmse':float(np.sqrt(np.mean((centered_mean-centered_truth)**2))),
        'no_individual_differences_rmse':float(np.sqrt(np.mean(centered_truth**2))),
        'within_group_correlation':float(np.corrcoef(centered_mean.ravel(),centered_truth.ravel())[0,1])}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2))
    np.savez_compressed(args.out/'posterior.npz', truth=truth, samples=physical_draws)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7,4))
    for trial in range(min(4,args.test)):
        indices = np.arange(args.agents)+trial*(args.agents+1)
        ax.vlines(indices, low[trial,4:], high[trial,4:], color='C0')
        ax.scatter(indices, mean[trial,4:], color='C0', label='Posterior mean / 90% interval' if trial==0 else None)
        ax.scatter(indices, truth[trial,4:], color='C1', marker='x', label='Simulated truth' if trial==0 else None)
    ax.set(xlabel='Agents, separated by held-out group',ylabel='Influence weight',ylim=(0,1))
    ax.legend(); fig.tight_layout(); fig.savefig(args.out/'agent_posteriors.png'); plt.close(fig)
    print(metrics.to_string(index=False), flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
