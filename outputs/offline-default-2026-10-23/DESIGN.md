# BayesFlow default offline retraining

Fresh 3K and 30K offline runs of the flat transformer/flow-matching model. No regularization profile is passed. BayesFlow 2.0.14 offline defaults: AdamW with weight decay 0.005, peak learning rate 0.0005, 5% warmup and cosine decay, clipnorm 1.5. Existing network defaults: summary dropout 0.10 and inference dropout 0.05.

Both runs use 300 epochs, batch size 32, final weights and no early stopping, matching the existing experiment design. Nested fixed training banks, shared independent validation/test banks and initialization are preserved from outputs/regularized-retraining-2026-10-22/banks. Evaluation: 300 test datasets and 1,000 posterior draws per dataset. No new simulations are needed for the banks.

Frozen matched-reference source is retained in source/ with hashes in source_sha256.json. Existing results are preserved. Each arm has its own training log, manifest, status.json, checkpoints, posterior samples and diagnostics. An arm is complete only when its complete.json exists.
