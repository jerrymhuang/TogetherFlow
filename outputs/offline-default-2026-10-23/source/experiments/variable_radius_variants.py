"""Variable-radius companion suite; true-radius features never enter inference."""
from dataclasses import replace
from variants import (V0, V1_COLLISION, V1_COLLISION_KAPPA, V2_SPREAD50,
    V5_TIMEVARYING_W, V6A_PARTIAL_POOLING, V8_SALIENCE_HIDDEN,
    V8_SALIENCE_OBSERVED, R_FREE_CHANNELS, REFERENCE_RADII)


def variable(source, name, observed=False):
    return replace(source, slug='rev-variable-' + name,
        channels=list(R_FREE_CHANNELS) + (['salience'] if observed else []),
        reference_radii=list(REFERENCE_RADII),
        rationale='Radius is drawn per trial from LogNormal(0, 0.5) and inferred. '
        'Inputs use observed trajectories and counts at fixed reference distances; '
        'no neighbor statistics computed at the true radius enter inference. '
        'This companion analysis preserves all fixed-radius results.')

BASE = variable(V0, 'reference')
SWITCH = variable(V8_SALIENCE_HIDDEN, 'switching')
PARTIAL = variable(V6A_PARTIAL_POOLING, 'partial')
CONTROL = replace(BASE, slug='rev-variable-partial-control',
    summary_net=PARTIAL.summary_net, inference_net=PARTIAL.inference_net,
    constrain_parameters=PARTIAL.constrain_parameters, time_horizon=PARTIAL.time_horizon)
ARMS = [BASE, variable(V1_COLLISION, 'collision'),
    variable(V1_COLLISION_KAPPA, 'collision-kappa'), variable(V2_SPREAD50, 'salience'),
    SWITCH, variable(V8_SALIENCE_OBSERVED, 'switching-observed', observed=True),
    variable(V5_TIMEVARYING_W, 'timevarying'), CONTROL, PARTIAL]
BY_SLUG = {v.slug: v for v in ARMS}
