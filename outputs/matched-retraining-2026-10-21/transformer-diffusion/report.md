# Matched reference: 480,000 simulations

fresh online simulations; shared validation and test banks. Checkpoint selection: final epoch; no early stopping. Metrics and figures use the same saved posterior draws.

                          w         r         v     noise
NRMSE              0.370579  0.767741  0.291180  0.483506
Log-gamma          1.186391  4.467488  2.584126  3.631632
ECE                0.017895  0.025702  0.019211  0.007456
Post. Contraction  0.871242  0.434992  0.912649  0.745629

Primary NRMSE is prior-normalized; the separate range-normalized CSV is for auditing. These runs change the original simulator dynamics, observable inputs, and summary architecture, so comparison with the original paper cannot isolate the effect of training budget.
