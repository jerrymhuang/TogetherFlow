# Round 1 — archived 2026-08-04

Superseded. Kept as evidence, not as results.

**Why superseded**
- Nine arms trained **offline** on a 5,000-simulation bank and overfit badly
  (val/train loss ratio ~9.5x; validation loss bottomed near epoch 11 then
  degraded ~60%). Every parameter rated "poor calibration; poor — overconfident".
- Ran on the pre-migration stack (jax 0.10.1, keras 3.14.1, numba 0.65.1).
- Used the legacy heading update, which is not rotationally invariant.

**Still worth reading**
- `v0-baseline` vs `v0-baseline-online` — the pair that established simulation
  budget, not architecture, caused the calibration failure.
- `v0-baseline-online` vs `v0-transformer-online` — first summary-network
  comparison; the transformer roughly halved velocity NRMSE and calibration error.
- `v3c-rfree-observables` vs `v0-baseline` — recovery of r degraded ~5x once the
  observables no longer contained r. Offline and overfit, so directional only.
