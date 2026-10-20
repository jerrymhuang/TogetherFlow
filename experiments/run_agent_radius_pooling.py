"""12-agent joint weight/radius partial pooling; 300 held-out groups by default."""
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
from run_agent_partial_pooling import BEACONS, DT, wrap

AGENTS = 12
REFERENCE_RADII = np.array([.5,.75,1.,1.25,1.5,2.,2.5,3.,3.5,4.,5.,6.,8.,12.])
GLOBAL_NAMES = ['mu_w', 'sigma_w', 'mu_r', 'sigma_r', 'v', 'eta']
PARAMETER_NAMES = GLOBAL_NAMES + [f'w_{i+1}' for i in range(AGENTS)] + [f'r_{i+1}' for i in range(AGENTS)]
# Upper bound on individual radii and their median; inf restores the lognormal.
RADIUS_MAX = 10.
RADIUS_SLOTS = [2]+list(range(18,18+AGENTS))
WEIGHT_COLOR = '#6969ff'
RADIUS_COLOR = '#4e2a84'  # RGB(78, 42, 132)
ONLINE_SEED_SPAN = 700000


def scales(radius_max=RADIUS_MAX):
    """Physical support scales: NaN denotes a positive parameter mapped with log."""
    out = np.array([1.,1.5,np.nan,.75,2.,1.]+[1.]*AGENTS+[np.nan]*AGENTS)
    out[RADIUS_SLOTS] = radius_max if np.isfinite(radius_max) else np.nan
    return out


def make_prior(radius_max=RADIUS_MAX):
    @njit
    def prior():
        mu_w = np.random.beta(2.,2.)
        # Truncated to the radius support; at log(2.5) +- 0.35 this almost never rejects.
        mu_r = np.random.lognormal(np.log(2.5),.35)
        while mu_r >= radius_max:
            mu_r = np.random.lognormal(np.log(2.5),.35)
        return hyperparameters(mu_w,mu_r)
    return prior


@njit
def hyperparameters(mu_w, mu_r):
    v = 2.*np.random.beta(2.,2.)
    noise = np.random.beta(2.,2.)
    sigma_w = 1.5*np.random.beta(1.,3.)
    sigma_r = .75*np.random.beta(2.,2.)
    # Simulator order: weight center, radius center, speed, noise, then spreads.
    return np.array([mu_w,mu_r,v,noise,sigma_w,sigma_r],dtype=np.float32)


radius_hierarchy_prior = make_prior()


def transform_targets(physical, radius_max=RADIUS_MAX):
    scale = scales(radius_max)
    bounded = np.isfinite(scale)
    out = np.empty_like(physical,dtype=np.float32)
    out[...,bounded] = logit(np.clip(physical[...,bounded]/scale[bounded],1e-6,1-1e-6))
    out[...,~bounded] = np.log(physical[...,~bounded])
    return out


def physical_targets(transformed, radius_max=RADIUS_MAX):
    scale = scales(radius_max)
    bounded = np.isfinite(scale)
    out = np.empty_like(transformed)
    out[...,bounded] = expit(transformed[...,bounded])*scale[bounded]
    out[...,~bounded] = np.exp(transformed[...,~bounded])
    return out


def observed_features(positions, rotations):
    """Fixed-reference-radius summaries of observations; no latent radii inputs.

    Each candidate radius is a predetermined observable feature definition, not
    the unknown radius used by the simulator. The whole group supplies context.
    """
    p, r = positions[:,:-1], rotations[:,:-1]
    distances = np.linalg.norm(p[:,:,:,None,:]-p[:,:,None,:,:],axis=-1)
    offset = BEACONS[None,None,None,:,:]-p[:,:,:,None,:]
    nearest = np.argmin(np.sum(offset**2,axis=-1),axis=-1)
    offset = np.take_along_axis(offset,nearest[...,None,None],axis=3)[...,0,:]
    external = wrap(np.arctan2(offset[...,1],offset[...,0])-r)
    turn = wrap(rotations[:,1:]-r)
    valid = (np.abs(p[...,0])<3.79)&(np.abs(p[...,1])<4.79)
    n = valid.sum(axis=1)
    def avg(x): return (x*valid).sum(axis=1)/np.maximum(n,1)
    speed = np.linalg.norm(positions[:,1:]-p,axis=-1)/DT
    features = [n/positions.shape[1], avg(speed), avg(turn**2), avg(external**2)]
    for radius in REFERENCE_RADII:
        mask = (distances>0)&(distances<=radius)
        counts = mask.sum(axis=-1)
        average = (mask*r[:,:,None,:]).sum(axis=-1)/np.maximum(counts,1)
        internal = np.where(counts>0,wrap(average-r),0.)
        x, y = DT*(external-internal), turn-DT*internal
        xx, xy, yy = avg(x*x), avg(x*y), avg(y*y)
        slope = xy/np.maximum(xx,1e-8)
        residual = np.maximum(yy-xy*xy/np.maximum(xx,1e-8),0.)
        features += [np.clip(slope,-10,10),xx,xy,yy,np.sqrt(residual/DT),
                     avg(counts),avg(internal**2)]
    return np.stack(features,axis=-1).astype(np.float32)


def build_simulator(seed, radius_max=RADIUS_MAX):
    return TogetherFlowSimulator(num_agents=AGENTS,num_beacons=len(BEACONS),
        time_horizon=20,dt=DT,prior=make_prior(radius_max),radius_upper=radius_max,
        param_names=('mu_w','mu_r','v','noise','sigma_w','sigma_r'),
        relative_heading=True,diffusive_heading=True,fixed_beacons=BEACONS,
        include_agent_weights=True,include_agent_radii=True,seed=seed)


def simulate(sim, n):
    """Draw n groups; return observed features and physical-scale truths."""
    raw = sim.sample(n)
    positions = raw['positions'].reshape(n,-1,AGENTS,2)
    features = observed_features(positions,raw['rotations']).reshape(n,-1)
    truth = np.concatenate([raw[k] for k in
        ['mu_w','sigma_w','mu_r','sigma_r','v','noise']]+
        [raw['agent_weights'],raw['agent_radii']],axis=-1)
    return features, truth


def make_bank(n, seed, radius_max=RADIUS_MAX):
    sim = build_simulator(seed,radius_max)
    features, truths = zip(*[simulate(sim,min(64,n-start)) for start in range(0,n,64)])
    truth = np.concatenate(truths)
    return {'targets':transform_targets(truth,radius_max),'features':np.concatenate(features)},truth


class OnlineSimulator(bf.simulators.Simulator):
    """Fresh feature/target batches for fit_online; seeds advance every call."""
    def __init__(self, seed, radius_max=RADIUS_MAX):
        self.sim = build_simulator(seed,radius_max)
        self.radius_max = radius_max

    def sample(self, batch_shape, **kwargs):
        n = batch_shape if isinstance(batch_shape,int) else int(np.prod(batch_shape))
        features, truth = simulate(self.sim,n)
        return {'targets':transform_targets(truth,self.radius_max),'features':features}


def plot_recovery(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.container import ErrorbarContainer
    with np.load(root/'posterior.npz') as data:
        samples,truth=data['samples'],data['truth']
    names=[r'$\mu_w$',r'$\sigma_w$',r'$\mu_r$',r'$\sigma_r$',r'$v$',r'$\eta$']
    names += [rf'$w_{{{i+1}}}$' for i in range(AGENTS)]
    names += [rf'$r_{{{i+1}}}$' for i in range(AGENTS)]
    fig=bf.diagnostics.plots.recovery(estimates=samples,targets=truth,
        variable_names=names,num_row=5,num_col=6,figsize=(21,17.5),
        uncertainty_agg_kwargs={'prob':.95},xlabel='Ground truth',ylabel='Estimate',
        label_fontsize=16,title_fontsize=22,metric_fontsize=10,tick_fontsize=9,markersize=2)
    # Global panels keep the default; weights and radii get their own colors.
    for ax,color in zip(fig.axes[6:],[WEIGHT_COLOR]*AGENTS+[RADIUS_COLOR]*AGENTS):
        for container in ax.containers:
            if isinstance(container,ErrorbarContainer):
                points,caps,bars=container.lines
                if points is not None: points.set_color(color)
                for artist in (*caps,*bars): artist.set_color(color)
    fig.tight_layout()
    fig.savefig(root/'parameter_recovery.png',dpi=180)
    fig.savefig(root/'parameter_recovery.pdf')
    plt.close(fig)


def diagnostics(root, truth, samples, prior_truth):
    median=np.median(samples,axis=1)
    low,high=np.quantile(samples,[.025,.975],axis=1)
    pd.DataFrame({'parameter':PARAMETER_NAMES,
        'rmse':np.sqrt(np.mean((median-truth)**2,axis=0)),
        'prior_median_rmse':np.sqrt(np.mean((np.median(prior_truth,axis=0)-truth)**2,axis=0)),
        'coverage95':np.mean((truth>=low)&(truth<=high),axis=0),
        'mean_interval_width':np.mean(high-low,axis=0),
        'contraction':1-samples.var(axis=1).mean(axis=0)/prior_truth.var(axis=0)
        }).to_csv(root/'metrics.csv',index=False)
    summaries={}
    for name,section in [('weight',slice(6,18)),('radius',slice(18,30))]:
        actual=truth[:,section].astype(float)
        # Center each joint draw before aggregating; use means for centered RMSE.
        draws=samples[:,:,section]
        centered_draws=draws-draws.mean(axis=-1,keepdims=True)
        prediction=centered_draws.mean(axis=1)
        centered=actual-actual.mean(axis=1,keepdims=True)
        cl,ch=np.quantile(centered_draws,[.025,.975],axis=1)
        summaries[name]={'centered_rmse':float(np.sqrt(np.mean((prediction-centered)**2))),
            'zero_difference_rmse':float(np.sqrt(np.mean(centered**2))),
            'centered_correlation':float(np.corrcoef(centered.ravel(),prediction.ravel())[0,1]),
            'centered_coverage95':float(np.mean((centered>=cl)&(centered<=ch)))}
    (root/'summary.json').write_text(json.dumps(summaries,indent=2,allow_nan=False))
    rows=[]
    for trial in range(len(truth)):
        for i in range(AGENTS):
            row={'trial':trial,'agent':i+1}
            for name,j in [('weight',6+i),('radius',18+i)]:
                row.update({f'{name}_truth':float(truth[trial,j]),
                    f'{name}_median':float(median[trial,j]),
                    f'{name}_q025':float(low[trial,j]),f'{name}_q975':float(high[trial,j])})
            row['posterior_weight_radius_correlation']=float(np.corrcoef(samples[trial,:,6+i],samples[trial,:,18+i])[0,1])
            rows.append(row)
    pd.DataFrame(rows).to_csv(root/'agent_posteriors.csv',index=False)
    print(json.dumps(summaries,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training',choices=['online','offline'],default='online',
                   help='online: fresh simulations every batch; offline: fixed --train bank')
    p.add_argument('--train',type=int,default=5000,
                   help='offline training bank; online runs simulate it only as the prior reference')
    p.add_argument('--batches-per-epoch',type=int,default=50,help='online only')
    p.add_argument('--validation',type=int,default=300)
    p.add_argument('--test',type=int,default=300)
    p.add_argument('--epochs',type=int,default=300)
    p.add_argument('--samples',type=int,default=1000)
    p.add_argument('--seed',type=int,default=831)
    p.add_argument('--radius-max',type=float,default=RADIUS_MAX,
                   help='upper bound on radii (inf: unbounded lognormal, as in validation-selected/)')
    p.add_argument('--out',type=Path,default=Path('outputs/agent-radius-pooling/bounded-radius-online'))
    a=p.parse_args()
    if min(a.train,a.validation,a.test,a.epochs,a.samples)<2:
        p.error('sizes and epochs must be at least 2')
    if not a.radius_max>2.5:
        p.error('--radius-max must exceed the prior radius center 2.5')
    bounded=np.isfinite(a.radius_max)
    online=a.training=='online'
    # Online seeds start at base+300000, after the validation/test ranges.
    if online and a.epochs*a.batches_per_epoch*32>ONLINE_SEED_SPAN:
        p.error('online budget would overlap the next seed block')
    a.out.mkdir(parents=True,exist_ok=False)
    manifest=vars(a)|{'out':str(a.out),'agents':AGENTS,'bayesflow':bf.__version__,
        'workflow':'BasicWorkflow','network':'FlowMatching','backend':'jax-cpu',
        'parameters':PARAMETER_NAMES,'reference_radii':REFERENCE_RADII.tolist(),
        'beacons':BEACONS.tolist(),'dt':DT,'duration':20,
        'mu_r_prior':'LogNormal(log(2.5), 0.35)'+(f' truncated below {a.radius_max:g}' if bounded else ''),
        'sigma_r_prior':'0.75 * Beta(2,2)',
        'mu_w_prior':'Beta(2,2)','sigma_w_prior':'1.5 * Beta(1,3)',
        'v_prior':'2 * Beta(2,2)','eta_prior':'Beta(2,2)',
        'local_prior':'logit(w_i) ~ Normal(logit(mu_w),sigma_w); '+
            (f'logit(r_i/{a.radius_max:g}) ~ Normal(logit(mu_r/{a.radius_max:g}),sigma_r)' if bounded
             else 'log(r_i) ~ Normal(log(mu_r),sigma_r)')+'; independent conditional on globals',
        'radius_target_transform':f'logit(r/{a.radius_max:g})' if bounded else 'log(r)',
        'recovery':'BayesFlow medians / 95% intervals; 5 rows x 6 columns; no title',
        'global_color':'#132a70','weight_color':WEIGHT_COLOR,'radius_color':RADIUS_COLOR,
        'evaluation_alignment':'variants.py n_test=300 and BasicWorkflow default num_samples=1000',
        'training':a.training,
        'simulations_seen':a.epochs*a.batches_per_epoch*32 if online else a.train,
        'checkpoint_selection':'final weights; no early stopping (fresh data cannot be overfit)' if online
            else 'minimum validation loss; patience 25; restore best weights',
        'status':'running'}
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    keras.utils.set_random_seed(a.seed)
    base=a.seed*1000000
    # Online runs never train on this bank; it only supplies the prior reference
    # for the contraction and prior-median metrics, matching the offline runs.
    print('Simulating training/prior-reference bank',flush=True)
    train,prior_truth=make_bank(a.train,base,a.radius_max)
    print('Simulating validation and held-out banks',flush=True)
    val,_=make_bank(a.validation,base+100000,a.radius_max)
    test,truth=make_bank(a.test,base+200000,a.radius_max)
    adapter=(bf.adapters.Adapter().concatenate(['targets'],into='inference_variables')
             .concatenate(['features'],into='inference_conditions'))
    workflow=bf.workflows.BasicWorkflow(adapter=adapter,
        simulator=OnlineSimulator(base+300000,a.radius_max) if online else None,
        inference_network=bf.networks.FlowMatching(subnet_kwargs={
            'widths':(256,256,256),'time_embedding_dim':32}),standardize='all')
    if online:
        history=workflow.fit_online(epochs=a.epochs,batch_size=32,
            num_batches_per_epoch=a.batches_per_epoch,validation_data=val,verbose=2)
    else:
        history=workflow.fit_offline(train,validation_data=val,epochs=a.epochs,batch_size=32,verbose=2,
            callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',patience=25,restore_best_weights=True)])
    manifest['epochs_completed']=len(history.history['loss'])
    manifest['selected_epoch']=int(np.argmin(history.history['val_loss']))+1
    workflow.approximator.save(a.out/'model.keras')
    (a.out/'history.json').write_text(json.dumps({k:list(map(float,v)) for k,v in history.history.items()}))
    # Save banks for evaluation recovery without retraining if sampling fails.
    np.savez_compressed(a.out/'evaluation_bank.npz',features=test['features'],truth=truth,prior_truth=prior_truth)
    draws=workflow.sample(num_samples=a.samples,conditions={'features':test['features']},
                          batch_size=5,seed=a.seed+17)['targets']
    samples=physical_targets(draws,a.radius_max)
    if not np.isfinite(samples).all():
        raise ValueError('Nonfinite posterior draws; inspect training before plotting')
    np.savez_compressed(a.out/'posterior.npz',truth=truth,samples=samples)
    diagnostics(a.out,truth,samples,prior_truth)
    plot_recovery(a.out)
    manifest['status']='complete'
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()
