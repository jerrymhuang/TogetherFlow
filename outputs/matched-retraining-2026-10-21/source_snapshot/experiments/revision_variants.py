"""Fixed-radius revision arms. Historical variants and outputs are preserved."""
from dataclasses import replace
from numba import njit
from variants import (V3A_FIXED, V6A_PARTIAL_POOLING, V5_TIMEVARYING_W,
                      V8_SALIENCE_HIDDEN, V7_WIDE, V7_TIGHT,
                      V7_SLOW_V, V7_ETA_SMOOTH, V7_ETA_DIFFUSE)


def fix_radius(prior, radius=2.5):
    @njit
    def draw():
        theta = prior()
        theta[1] = radius
        return theta
    draw.__name__ = f"{prior.__name__}_fixed_r_{radius:g}"
    return draw


def fixed_variant(source, slug, **kwargs):
    return replace(source, slug=slug, prior=fix_radius(source.prior),
                   infer=tuple(p for p in source.infer if p != 'r'), **kwargs)


BASE = replace(V3A_FIXED, slug='rev-fixed-reference', addresses='R2 point 2',
               rationale='Revision reference: radius fixed by model design at 2.5 room units. This is not an empirical estimate of human sensing radius.')
PARTIAL = fixed_variant(V6A_PARTIAL_POOLING, 'rev-fixed-partial', addresses='R2 point 4')
CONTROL = replace(BASE, slug='rev-fixed-partial-control', summary_net='bdlstm',
                  inference_net='diffusion', constrain_parameters=False, time_horizon=20,
                  addresses='R2 point 4: matched complete-pooling control',
                  rationale='Match hierarchical arm estimator, architecture family, duration, constraints, and training budget. Output dimension differs by necessity.')
SWITCH = fixed_variant(V8_SALIENCE_HIDDEN, 'rev-fixed-switching', addresses='R2 point 7',
                       rationale='Salience-driven switching with the same observed features as the static reference. Cross-evaluate both trained estimators on both generators; this is a Python mismatch proxy, not verified Unity parity.')
ARMS = [BASE, CONTROL, PARTIAL,
        fixed_variant(V5_TIMEVARYING_W, 'rev-fixed-timevarying', addresses='R2 point 5'), SWITCH]
for radius in (1.25, 4.0):
    ARMS.append(replace(BASE, slug=f'rev-fixed-radius-{radius:g}',
                        prior=fix_radius(V3A_FIXED.prior, radius),
                        rationale=f'Design sensitivity: fix radius at {radius:g}; retrain and evaluate under that same assumption.'))
for source in (V7_WIDE, V7_TIGHT, V7_SLOW_V, V7_ETA_SMOOTH, V7_ETA_DIFFUSE):
    ARMS.append(fixed_variant(source, 'rev-fixed-' + source.slug.removeprefix('v7-'), addresses='R2 point 6'))
# Matched ablations of training regime and simulator mechanisms, not the old paper's exact replication.
ARMS += [replace(BASE, slug='rev-fixed-offline-3k', online=False, n_train=3000,
                 early_stopping_patience=20, addresses='R2 point 3',
                 rationale='Offline 3K with validation-selected weights versus fresh online batches. Other reference settings matched.'),
         replace(BASE, slug='rev-fixed-legacy-heading', relative_heading=False, addresses='R2 point 3'),
         replace(BASE, slug='rev-fixed-legacy-noise', diffusive_heading=False, addresses='R2 point 3')]
BY_SLUG = {v.slug: v for v in ARMS}
