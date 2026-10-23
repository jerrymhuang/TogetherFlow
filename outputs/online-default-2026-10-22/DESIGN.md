# Default BayesFlow online training

Use the existing flat transformer/flow-matching model with no regularization profile or custom training overrides. Network defaults: summary dropout 0.10, inference dropout 0.05. BayesFlow online defaults: Adam without weight decay; initial_learning_rate 0.0005 with default 5% warmup/cosine schedule and clipnorm 1.5.

300 epochs, 50 fresh simulation batches per epoch, batch size 32 (480,000 simulations). Final weights, no early stopping. Frozen matched-reference simulator, model architecture, data adapter and initialization. Reuse shared validation/test banks from outputs/regularized-retraining-2026-10-22/banks. Evaluate 300 test datasets with 1,000 posterior draws each.

Status: transformer-flow_matching/status.json. Logs: training.log. Completion: transformer-flow_matching/complete.json.
