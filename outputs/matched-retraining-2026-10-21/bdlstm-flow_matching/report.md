# Matched reference: 480,000 simulations

fresh online simulations; shared validation and test banks. Checkpoint selection: final epoch; no early stopping. Metrics and figures use the same saved posterior draws.

                          w           r         v       noise
NRMSE              0.706914    0.643828  1.092293    0.837652
Log-gamma              -inf -120.448178      -inf -285.221097
ECE                0.366667    0.229035  0.400000    0.271140
Post. Contraction  0.899101    0.813279  0.862296    0.758350

Primary NRMSE is prior-normalized; the separate range-normalized CSV is for auditing. These runs change the original simulator dynamics, observable inputs, and summary architecture, so comparison with the original paper cannot isolate the effect of training budget.
