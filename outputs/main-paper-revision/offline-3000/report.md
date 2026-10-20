# Revised offline reference: 3,000 simulations

Fixed training bank; independent shared validation and test banks. Checkpoint selected by validation loss. Metrics and figures use the same saved posterior draws.

                          w         r         v     noise
NRMSE              0.605850  0.719848  0.486213  0.645094
Log-gamma         -8.069564 -3.343557  1.350798 -1.919601
ECE                0.008158  0.043070  0.038070  0.036140
Post. Contraction  0.668604  0.610433  0.776822  0.544888

Primary NRMSE is prior-normalized; the separate range-normalized CSV is for auditing. These runs change the original simulator dynamics, observable inputs, and summary architecture, so comparison with the original paper cannot isolate the effect of training budget.
