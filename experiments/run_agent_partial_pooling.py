"""Matched complete/partial pooling example with observable per-agent features.

PYTHONPATH=. .venv/bin/python experiments/run_agent_partial_pooling.py
The controlled example fixes sensing radius and beacon layout; it is not a
re-analysis of the previous random-beacon experiments or human observations.
"""
import os
os.environ.setdefault('KERAS_BACKEND', 'jax')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('NUMBA_NUM_THREADS', '4')
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import keras
import bayesflow as bf
from numba import njit
from scipy.special import expit, logit
from togetherflow import TogetherFlowSimulator
from revision_variants import PARTIAL

BEACONS = np.array([[-3.,-4.],[-3.,4.],[3.,-4.],[3.,4.]],dtype=np.float32)
DT = .1
RADIUS = 2.5


@njit
def pooled_prior():
    theta = PARTIAL_PRIOR()
    theta[4] = 0.
    return theta


PARTIAL_PRIOR = PARTIAL.prior


def wrap(x):
    return (x+np.pi) % (2*np.pi)-np.pi


def observed_features(positions, rotations):
    """Use observed motion and known layout only; retain ordered agent features.

    Mirror the simulator's arithmetic neighbor-angle average, then wrap it.
    Its alignment rule is NOT the circular mean. Only use transitions starting
    >0.21 from walls: speed <=2 and dt=.1 ensure no reflection in that step.
    These are summaries, not a claimed exact or sufficient likelihood.
    """
    p = np.asarray(positions)[:,:-1]
    r = np.asarray(rotations)[:,:-1]
    distances = np.linalg.norm(p[:,:,:,None,:]-p[:,:,None,:,:],axis=-1)
    mask = (distances>0)&(distances<=RADIUS)
    counts = mask.sum(axis=-1)
    average = (mask*r[:,:,None,:]).sum(axis=-1)/np.maximum(counts,1)
    internal = np.where(counts>0,wrap(average-r),0.)
    delta = BEACONS[None,None,None,:,:]-p[:,:,:,None,:]
    nearest = np.argmin(np.sum(delta**2,axis=-1),axis=-1)
    offset = np.take_along_axis(delta,nearest[...,None,None],axis=3)[...,0,:]
    external = wrap(np.arctan2(offset[...,1],offset[...,0])-r)
    x = DT*(external-internal)
    y = wrap(np.asarray(rotations)[:,1:]-r)-DT*internal
    valid = (np.abs(p[...,0])<3.79)&(np.abs(p[...,1])<4.79)
    n = valid.sum(axis=1)
    def avg(a):
        return (a*valid).sum(axis=1)/np.maximum(n,1)
    xx, xy, yy = avg(x*x), avg(x*y), avg(y*y)
    slope = xy/np.maximum(xx,1e-8)
    residual = np.maximum(yy-xy*xy/np.maximum(xx,1e-8),0.)
    speed = np.linalg.norm(np.asarray(positions)[:,1:]-p,axis=-1)/DT
    # No clipping to [0,1]: noisy raw slopes carry uncertainty information.
    features = np.stack([np.clip(slope,-10,10),xx,xy,yy,
        np.sqrt(residual/DT),n/positions.shape[1],avg(speed),avg(counts),
        avg(external**2),avg(internal**2)],axis=-1)
    return features.astype(np.float32)


def make_bank(n, seed, agents, pooling):
    sim = TogetherFlowSimulator(num_agents=agents,num_beacons=len(BEACONS),
        time_horizon=20,dt=DT,prior=PARTIAL_PRIOR if pooling=='partial' else pooled_prior,
        param_names=PARTIAL.param_names,relative_heading=True,diffusive_heading=True,
        fixed_beacons=BEACONS,include_agent_weights=True,seed=seed)
    banks, truths = [], []
    for start in range(0,n,256):
        raw = sim.sample(min(256,n-start))
        positions = raw['positions'].reshape(len(raw['mu_w']),-1,agents,2)
        features = observed_features(positions,raw['rotations'])
        truth = np.concatenate([raw[k] for k in ['mu_w','v','noise','sigma_w']]
            +[raw['agent_weights']],axis=-1)
        banks.append(features.reshape(len(features),-1))
        truths.append(truth)
    truth = np.concatenate(truths)
    indices = list(range(4+agents)) if pooling=='partial' else [0,1,2]
    scales = np.array([1.,2.,1.,1.5]+[1.]*agents)[indices]
    target = logit(np.clip(truth[:,indices]/scales,1e-6,1-1e-6)).astype(np.float32)
    return {'targets':target,'features':np.concatenate(banks)}, truth, scales


def evaluate(draws, truth, prior_truth, pooling, seed, generator, folder):
    mean = draws.mean(axis=1)
    lo, hi = np.quantile(draws,[.05,.95],axis=1)
    wt = truth[:,4:]
    predicted = mean[:,4:]
    centered_truth = wt-wt.mean(axis=1,keepdims=True)
    centered_pred = predicted-predicted.mean(axis=1,keepdims=True)
    def rmse(x): return float(np.sqrt(np.mean(x*x)))
    # Pairwise uncertainty is computed from joint draws, retaining dependencies.
    contrasts = draws[:,:,4:]-draws[:,:,4:].mean(axis=-1,keepdims=True)
    cl, ch = np.quantile(contrasts,[.05,.95],axis=1)
    ranks = (draws[:,:,4:] < wt[:,None,:]).mean(axis=1)
    result = dict(seed=seed,estimator=pooling,generator=generator,
        agent_rmse=rmse(predicted-wt),
        centered_rmse=rmse(centered_pred-centered_truth),
        zero_difference_rmse=rmse(centered_truth),
        agent_coverage90=float(np.mean((wt>=lo[:,4:])&(wt<=hi[:,4:]))),
        centered_coverage90=float(np.mean((centered_truth>=cl)&(centered_truth<=ch))),
        mean_interval_width=float(np.mean(hi[:,4:]-lo[:,4:])),
        prior_mean_rmse=rmse(prior_truth[:,4:].mean(axis=0)-wt),
        sigma_rmse=rmse(mean[:,3]-truth[:,3]))
    result['centered_correlation'] = (float(np.corrcoef(centered_truth.ravel(),centered_pred.ravel())[0,1])
        if np.std(centered_truth)>1e-7 and np.std(centered_pred)>1e-7 else None)
    names = ['mu_w','v','noise','sigma_w']+[f'w_{i}' for i in range(wt.shape[1])]
    pd.DataFrame(dict(parameter=names,rmse=np.sqrt(np.mean((mean-truth)**2,axis=0)),
        coverage90=np.mean((truth>=lo)&(truth<=hi),axis=0))).to_csv(folder/'metrics.csv',index=False)
    rows=[]
    for trial in range(len(truth)):
        for agent in range(wt.shape[1]):
            rows.append(dict(trial=trial,agent=agent,truth=wt[trial,agent],
                posterior_mean=predicted[trial,agent],q05=lo[trial,4+agent],
                q95=hi[trial,4+agent],posterior_rank=ranks[trial,agent]))
    pd.DataFrame(rows).to_csv(folder/'agent_posteriors.csv',index=False)
    np.savez_compressed(folder/'posterior.npz',truth=truth,samples=draws)
    (folder/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    return result


def plot_comparison(root, seed):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharey=True)
    for row,pooling in enumerate(['partial','complete']):
        data=np.load(root/f'seed-{seed}'/pooling/'on-partial'/'posterior.npz')
        draws,truth=data['samples'][:,:,4:],data['truth'][:,4:]
        low,high=np.quantile(draws,[.05,.95],axis=1)
        means=draws.mean(axis=1)
        for trial in range(2):
            ax=axes[row,trial]; x=np.arange(truth.shape[1])
            ax.vlines(x,low[trial],high[trial],color='C0')
            ax.scatter(x,means[trial],label='Posterior mean / 90% interval')
            ax.scatter(x,truth[trial],marker='x',color='C1',label='Simulated truth')
            ax.set(title=f'{pooling.capitalize()} pooling, test group {trial}',
                xlabel='Agent',ylabel='Influence weight',ylim=(0,1))
    axes[0,0].legend(fontsize=8)
    fig.tight_layout();fig.savefig(root/'pooling_comparison.png');plt.close(fig)


def plot_agent_recovery(root, seed):
    """BayesFlow recovery diagnostic for all inferred parameters, one seed."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    source = root/f'seed-{seed}'/'partial'/'on-partial'/'posterior.npz'
    with np.load(source) as data:
        truth, samples = data['truth'], data['samples']
    agents = truth.shape[-1] - 4
    names = [r'$\mu_w$', r'$v$', r'$\eta$', r'$\sigma_w$']
    names += [rf'$w_{{{i + 1}}}$' for i in range(agents)]
    cols = 4
    rows = (len(names) + cols - 1) // cols
    fig = bf.diagnostics.plots.recovery(
        estimates=samples, targets=truth, variable_names=names,
        num_col=cols, num_row=rows, figsize=(14, 3.5*rows),
        xlabel='Ground truth', ylabel='Posterior median',
        uncertainty_agg_kwargs={'prob': .95},
        label_fontsize=12, title_fontsize=16, metric_fontsize=11,
        tick_fontsize=10, markersize=3)
    # Recolor only data artists, preserving BayesFlow's reference diagonals.
    from matplotlib.container import ErrorbarContainer
    for ax in fig.axes[4:len(names)]:
        for container in ax.containers:
            if isinstance(container, ErrorbarContainer):
                points, caps, bars = container.lines
                if points is not None:
                    points.set_color('#6969ff')
                for artist in (*caps, *bars):
                    artist.set_color('#6969ff')
    fig.tight_layout()
    fig.savefig(root/'parameter_recovery.png', dpi=180)
    fig.savefig(root/'parameter_recovery.pdf')
    plt.close(fig)
    (root/'parameter_recovery.json').write_text(json.dumps({
        'seed': seed, 'source': str(source), 'num_test_groups': len(truth),
        'diagnostic': 'bayesflow.diagnostics.plots.recovery',
        'point_estimate': 'median', 'credible_interval': .95,
        'global_color': '#132a70', 'agent_color': '#6969ff', 'figure_title': None,
        'parameters': ['mu_w', 'v', 'eta', 'sigma_w'] +
                      [f'w_{i+1}' for i in range(agents)]}, indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',type=int,default=4096)
    p.add_argument('--validation',type=int,default=512)
    p.add_argument('--test',type=int,default=128)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--samples',type=int,default=256)
    p.add_argument('--agents',type=int,default=8)
    p.add_argument('--seeds',type=int,nargs='+',default=[831])
    p.add_argument('--out',type=Path,default=Path('outputs/agent-partial-pooling/example'))
    a=p.parse_args()
    if min(a.train,a.validation,a.test,a.epochs,a.samples,a.agents)<1 or a.test<2:
        p.error('positive sizes and at least two test groups required')
    a.out.mkdir(parents=True,exist_ok=False)
    manifest=vars(a)|{'out':str(a.out),'bayesflow':bf.__version__,
        'workflow':'BasicWorkflow','inference_network':'FlowMatching',
        'beacons':BEACONS.tolist(),'radius':RADIUS,'dt':DT,'time_horizon':20,
        'features':'observable per-agent turning moments; full-group conditioning',
        'target':'joint population and agent weights; complete control shares one weight',
        'seed_design':'separate 1,000,000-wide blocks per training seed',
        'purpose':'controlled synthetic example; not a matched comparison to earlier pilot'}
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    results=[]
    for seed in a.seeds:
        base=seed*1000000
        tests={kind:make_bank(a.test,base+200000,a.agents,kind) for kind in ['partial','complete']}
        for pooling in ['partial','complete']:
            folder=a.out/f'seed-{seed}'/pooling;folder.mkdir(parents=True)
            keras.utils.set_random_seed(seed)
            train,prior_truth,scales=make_bank(a.train,base,a.agents,pooling)
            val,_,_=make_bank(a.validation,base+100000,a.agents,pooling)
            adapter=(bf.adapters.Adapter().concatenate(['targets'],into='inference_variables')
                .concatenate(['features'],into='inference_conditions'))
            workflow=bf.workflows.BasicWorkflow(adapter=adapter,
                inference_network=bf.networks.FlowMatching(subnet_kwargs={
                    'widths':(128,128),'time_embedding_dim':16}),standardize='all')
            print(f'Training {pooling}, seed {seed}',flush=True)
            history=workflow.fit_offline(train,validation_data=val,epochs=a.epochs,batch_size=64,verbose=2)
            workflow.approximator.save(folder/'model.keras')
            (folder/'history.json').write_text(json.dumps({k:list(map(float,v)) for k,v in history.history.items()}))
            for generator,(test,truth,_) in tests.items():
                evaluation=folder/f'on-{generator}';evaluation.mkdir()
                draws=workflow.sample(num_samples=a.samples,conditions={'features':test['features']},
                    batch_size=16,seed=seed+17)['targets']
                physical=expit(draws)*scales
                if pooling=='complete':
                    physical=np.concatenate([physical,np.zeros_like(physical[...,:1]),
                        np.repeat(physical[...,:1],a.agents,axis=-1)],axis=-1)
                results.append(evaluate(physical,truth,prior_truth,pooling,seed,generator,evaluation))
                pd.DataFrame(results).to_csv(a.out/'comparison.csv',index=False)
    plot_comparison(a.out,a.seeds[0])
    plot_agent_recovery(a.out,a.seeds[0])
    print(pd.DataFrame(results).to_string(index=False),flush=True)


if __name__=='__main__':
    main()
