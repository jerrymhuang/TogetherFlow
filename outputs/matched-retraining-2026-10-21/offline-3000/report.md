# Matched reference: 3,000 simulations

nested fixed training banks; shared independent validation and test banks. Checkpoint selection: final epoch; no early stopping. Metrics and figures use the same saved posterior draws.

                          w         r         v     noise
NRMSE              0.316516  0.419702  0.218469  0.361551
Log-gamma              -inf      -inf      -inf      -inf
ECE                0.455000  0.458333  0.433333  0.446667
Post. Contraction  0.995452  0.996241  0.996052  0.993430

Primary NRMSE is prior-normalized; the separate range-normalized CSV is for auditing. These runs change the original simulator dynamics, observable inputs, and summary architecture, so comparison with the original paper cannot isolate the effect of training budget.
