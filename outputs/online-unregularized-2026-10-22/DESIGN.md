# Online training without regularization

Fresh 300-epoch transformer/flow-matching online run, 50 batches per epoch and batch size 32 (480,000 fresh simulations). Explicit zero summary and inference dropout; default online Adam with no weight decay. Default learning-rate schedule: 5% warmup to 0.0005, then cosine decay. Gradient clipping at norm 1.5 is retained for numerical stability. No validation-plateau schedule or early stopping; evaluate final weights.

Frozen matched-reference simulator, adapter, architecture widths, initialization and evaluation banks are preserved. Evaluation reuses outputs/regularized-retraining-2026-10-22/banks (300 validation datasets, 300 test datasets; 1,000 posterior draws per test dataset). This changes both regularization and the optimizer/schedule relative to the regularized online run; it does not isolate dropout alone.

The source snapshot and hashes are stored here. Status: transformer-flow_matching/status.json; training log: training.log. Completion requires transformer-flow_matching/complete.json, including posterior diagnostics.
