# Further offline regularization pilots

Compare stronger summary/inference dropout (0.40/0.25) and stronger AdamW weight decay (0.15) separately with the completed regularized 3K reference (0.25/0.15 dropout, 0.05 decay). Use the frozen matched-reference source, identical banks and initialization, and 120 epochs. The baseline is the first 120 epochs of the existing 300-epoch run; its validation-plateau schedule has no dependence on the total epoch count. Each candidate starts from scratch.

Primary comparison: mean validation loss over epochs 91–120, rather than a noisy single minimum. Also report 30-epoch window means to assess deterioration. No test bank or posterior evaluation is loaded by these pilots. One seed is exploratory; promising settings need longer validation before adoption. Original results and online settings are preserved.
