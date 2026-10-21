# Matched reference: 30,000 simulations

nested fixed training banks; shared independent validation and test banks. Checkpoint selection: final epoch; no early stopping. Metrics and figures use the same saved posterior draws.

                          w          r         v      noise
NRMSE              0.230945   0.490646  0.179551   0.309688
Log-gamma         -6.056293 -11.853892 -5.504904 -12.473277
ECE                0.077895   0.076228  0.070702   0.121491
Post. Contraction  0.958124   0.839078  0.969803   0.915889

Primary NRMSE is prior-normalized; the separate range-normalized CSV is for auditing. These runs change the original simulator dynamics, observable inputs, and summary architecture, so comparison with the original paper cannot isolate the effect of training budget.
