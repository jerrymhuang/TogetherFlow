# Saved experiment results

Historical artifacts, not new runs. Reviewer numbering in old reports is stale.

NRMSE normalization and simulator versions must be verified before comparing historical and new runs. Most historical arms use features computed at the true radius. Use fixed-radius or radius-independent analyses for deployable inference claims.

No metrics were found for `v8-salience-hidden`. The superstats path run has a diagnostic PNG but no numeric metrics archive locally.

## NRMSE

| Variant | w | r | v | noise | kappa | alpha | w0 | tau | mu_w | sigma_w |
|---|---|---|---|---|---|---|---|---|---|---|
| v0-bdlstm-diffusion | 0.3466 | 0.0631 | 0.2499 | 0.6787 | — | — | — | — | — | — |
| v0-bdlstm-online | 0.2896 | 0.0772 | 0.2110 | 0.5096 | — | — | — | — | — | — |
| v0-diffusion | 0.7064 | 0.0532 | 0.2761 | 0.8443 | — | — | — | — | — | — |
| v0-legacy-heading | 1.0143 | 0.0293 | 0.1285 | 0.9719 | — | — | — | — | — | — |
| v0-reference | 0.3424 | 0.0591 | 0.1963 | 0.4415 | — | — | — | — | — | — |
| v1-collision | 0.2217 | 0.0492 | 0.1605 | 0.4353 | — | — | — | — | — | — |
| v1-collision-kappa | 0.2617 | 0.0560 | 0.1527 | 0.6119 | 0.6720 | — | — | — | — | — |
| v2-salience-spread10 | 0.1075 | 0.0731 | 0.1785 | 0.5946 | — | 0.9586 | — | — | — | — |
| v2-salience-spread50 | 0.2374 | 0.0576 | 0.1574 | 0.4104 | — | 0.9667 | — | — | — | — |
| v3a-fixed-radius | 0.1795 | — | 0.1714 | 0.3553 | — | — | — | — | — | — |
| v3c-rfree-observables | 0.2692 | 0.8442 | 0.2425 | 0.3542 | — | — | — | — | — | — |
| v4-horizon-40 | 0.3543 | 0.0602 | 0.2077 | 0.5118 | — | — | — | — | — | — |
| v4-horizon-80 | 0.2660 | 0.0535 | 0.1777 | 0.3934 | — | — | — | — | — | — |
| v4-long-horizon | 0.3280 | 0.0577 | 0.2120 | 0.5598 | — | — | — | — | — | — |
| v4-short-horizon | 0.3712 | 0.0634 | 0.1979 | 0.4574 | — | — | — | — | — | — |
| v5-timevarying-w | — | 0.0543 | 0.2212 | 0.5563 | — | — | 0.4734 | 0.9784 | — | — |
| v6a-partial-pooling | — | 0.0610 | 0.3028 | 0.9479 | — | — | — | — | 0.5265 | 0.9506 |
| v7-prior-bounded-radius | 0.2217 | 0.0594 | 0.1726 | 0.3988 | — | — | — | — | — | — |
| v7-prior-eta-diffuse | 0.3393 | 0.0551 | 0.1965 | 0.3634 | — | — | — | — | — | — |
| v7-prior-eta-smooth | 0.2908 | 0.0562 | 0.2045 | 0.5619 | — | — | — | — | — | — |
| v7-prior-slow-velocity | 0.3235 | 0.0635 | 0.2123 | 0.4158 | — | — | — | — | — | — |
| v7-prior-tight | 0.3578 | 0.0884 | 0.2553 | 0.7290 | — | — | — | — | — | — |
| v7-prior-wide | 0.2582 | 0.0451 | 0.1549 | 0.3668 | — | — | — | — | — | — |
| v8-salience-observed | 0.3604 | 0.0556 | 0.2144 | 0.4536 | — | — | — | — | — | — |

## ECE

| Variant | w | r | v | noise | kappa | alpha | w0 | tau | mu_w | sigma_w |
|---|---|---|---|---|---|---|---|---|---|---|
| v0-bdlstm-diffusion | 0.0116 | 0.0126 | 0.0243 | 0.0302 | — | — | — | — | — | — |
| v0-bdlstm-online | 0.0264 | 0.0686 | 0.0462 | 0.0059 | — | — | — | — | — | — |
| v0-diffusion | 0.0386 | 0.0531 | 0.0089 | 0.0096 | — | — | — | — | — | — |
| v0-legacy-heading | 0.0082 | 0.1289 | 0.0111 | 0.0229 | — | — | — | — | — | — |
| v0-reference | 0.0153 | 0.0288 | 0.0128 | 0.0082 | — | — | — | — | — | — |
| v1-collision | 0.0250 | 0.0307 | 0.0125 | 0.0261 | — | — | — | — | — | — |
| v1-collision-kappa | 0.0187 | 0.0560 | 0.0186 | 0.0132 | 0.0202 | — | — | — | — | — |
| v2-salience-spread10 | 0.0231 | 0.0304 | 0.0065 | 0.0339 | — | 0.0154 | — | — | — | — |
| v2-salience-spread50 | 0.0269 | 0.0608 | 0.0139 | 0.0098 | — | 0.0154 | — | — | — | — |
| v3a-fixed-radius | 0.0130 | — | 0.0213 | 0.0151 | — | — | — | — | — | — |
| v3c-rfree-observables | 0.0153 | 0.0096 | 0.0195 | 0.0158 | — | — | — | — | — | — |
| v4-horizon-40 | 0.0084 | 0.0375 | 0.0208 | 0.0098 | — | — | — | — | — | — |
| v4-horizon-80 | 0.0122 | 0.0326 | 0.0216 | 0.0212 | — | — | — | — | — | — |
| v4-long-horizon | 0.0064 | 0.0540 | 0.0204 | 0.0123 | — | — | — | — | — | — |
| v4-short-horizon | 0.0289 | 0.0126 | 0.0236 | 0.0047 | — | — | — | — | — | — |
| v5-timevarying-w | — | 0.0475 | 0.0185 | 0.0144 | — | — | 0.0175 | 0.0321 | — | — |
| v6a-partial-pooling | — | 0.0282 | 0.0231 | 0.0090 | — | — | — | — | 0.0096 | 0.0339 |
| v7-prior-bounded-radius | 0.0040 | 0.0202 | 0.0275 | 0.0072 | — | — | — | — | — | — |
| v7-prior-eta-diffuse | 0.0132 | 0.0507 | 0.0095 | 0.0315 | — | — | — | — | — | — |
| v7-prior-eta-smooth | 0.0096 | 0.0451 | 0.0171 | 0.0132 | — | — | — | — | — | — |
| v7-prior-slow-velocity | 0.0073 | 0.0217 | 0.0089 | 0.0264 | — | — | — | — | — | — |
| v7-prior-tight | 0.0118 | 0.0128 | 0.0085 | 0.0205 | — | — | — | — | — | — |
| v7-prior-wide | 0.0204 | 0.0363 | 0.0241 | 0.0188 | — | — | — | — | — | — |
| v8-salience-observed | 0.0211 | 0.1006 | 0.0113 | 0.0102 | — | — | — | — | — | — |

## Post. Contraction

| Variant | w | r | v | noise | kappa | alpha | w0 | tau | mu_w | sigma_w |
|---|---|---|---|---|---|---|---|---|---|---|
| v0-bdlstm-diffusion | 0.8869 | 0.9966 | 0.9373 | 0.5220 | — | — | — | — | — | — |
| v0-bdlstm-online | 0.9162 | 0.9938 | 0.9544 | 0.7316 | — | — | — | — | — | — |
| v0-diffusion | 0.5217 | 0.9974 | 0.9173 | 0.2685 | — | — | — | — | — | — |
| v0-legacy-heading | 0.0033 | 0.9991 | 0.9822 | 0.1028 | — | — | — | — | — | — |
| v0-reference | 0.8896 | 0.9967 | 0.9616 | 0.7954 | — | — | — | — | — | — |
| v1-collision | 0.9498 | 0.9979 | 0.9756 | 0.8204 | — | — | — | — | — | — |
| v1-collision-kappa | 0.9339 | 0.9973 | 0.9744 | 0.6317 | 0.6169 | — | — | — | — | — |
| v2-salience-spread10 | 0.9877 | 0.9952 | 0.9635 | 0.6895 | — | 0.0222 | — | — | — | — |
| v2-salience-spread50 | 0.9354 | 0.9970 | 0.9758 | 0.8281 | — | 0.0944 | — | — | — | — |
| v3a-fixed-radius | 0.9705 | — | 0.9689 | 0.8813 | — | — | — | — | — | — |
| v3c-rfree-observables | 0.9331 | 0.2928 | 0.9473 | 0.8734 | — | — | — | — | — | — |
| v4-horizon-40 | 0.8844 | 0.9967 | 0.9531 | 0.7280 | — | — | — | — | — | — |
| v4-horizon-80 | 0.9352 | 0.9974 | 0.9659 | 0.8411 | — | — | — | — | — | — |
| v4-long-horizon | 0.8930 | 0.9969 | 0.9536 | 0.6868 | — | — | — | — | — | — |
| v4-short-horizon | 0.8811 | 0.9966 | 0.9605 | 0.7840 | — | — | — | — | — | — |
| v5-timevarying-w | — | 0.9973 | 0.9529 | 0.6920 | — | — | 0.7994 | 0.0764 | — | — |
| v6a-partial-pooling | — | 0.9970 | 0.9069 | 0.0811 | — | — | — | — | 0.7637 | 0.0921 |
| v7-prior-bounded-radius | 0.9533 | 0.9965 | 0.9697 | 0.8485 | — | — | — | — | — | — |
| v7-prior-eta-diffuse | 0.8796 | 0.9971 | 0.9633 | 0.8709 | — | — | — | — | — | — |
| v7-prior-eta-smooth | 0.9185 | 0.9974 | 0.9602 | 0.6922 | — | — | — | — | — | — |
| v7-prior-slow-velocity | 0.9074 | 0.9964 | 0.9566 | 0.8320 | — | — | — | — | — | — |
| v7-prior-tight | 0.8849 | 0.9928 | 0.9285 | 0.4814 | — | — | — | — | — | — |
| v7-prior-wide | 0.9248 | 0.9982 | 0.9725 | 0.8648 | — | — | — | — | — | — |
| v8-salience-observed | 0.8823 | 0.9970 | 0.9553 | 0.7713 | — | — | — | — | — | — |

## Artifact availability

| Variant | Metrics | Checkpoint | Manifest |
|---|---|---|---|
| v0-bdlstm-diffusion | yes | missing | missing |
| v0-bdlstm-online | yes | missing | missing |
| v0-diffusion | yes | missing | missing |
| v0-legacy-heading | yes | missing | missing |
| v0-reference | yes | missing | missing |
| v1-collision | yes | missing | missing |
| v1-collision-kappa | yes | missing | missing |
| v2-salience-spread10 | yes | missing | missing |
| v2-salience-spread50 | yes | missing | missing |
| v3a-fixed-radius | yes | missing | missing |
| v3c-rfree-observables | yes | missing | missing |
| v4-horizon-40 | yes | missing | missing |
| v4-horizon-80 | yes | missing | missing |
| v4-long-horizon | yes | missing | missing |
| v4-short-horizon | yes | missing | missing |
| v5-timevarying-w | yes | missing | missing |
| v6a-partial-pooling | yes | missing | missing |
| v7-prior-bounded-radius | yes | missing | missing |
| v7-prior-eta-diffuse | yes | missing | missing |
| v7-prior-eta-smooth | yes | missing | missing |
| v7-prior-slow-velocity | yes | missing | missing |
| v7-prior-tight | yes | missing | missing |
| v7-prior-wide | yes | missing | missing |
| v8-salience-observed | yes | missing | missing |
