# Revised offline reference: 30,000 simulations

Fixed training bank; independent shared validation and test banks. Checkpoint selected by validation loss. Metrics and figures use the same saved posterior draws.

                           w          r          v      noise
NRMSE               0.266471   0.530845   0.238911   0.345745
Log-gamma         -12.805746 -12.566701 -14.450763 -13.762409
ECE                 0.083596   0.050439   0.075702   0.052193
Post. Contraction   0.943274   0.783743   0.949342   0.885474

Primary NRMSE is prior-normalized; the separate range-normalized CSV is for auditing. These runs change the original simulator dynamics, observable inputs, and summary architecture, so comparison with the original paper cannot isolate the effect of training budget.
