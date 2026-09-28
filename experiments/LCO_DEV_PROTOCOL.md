# Leave-combinations-out: the development protocol

This is the procedure for selecting, calibrating and reporting anything evaluated under the leave-combinations-out (LCO) protocol. It follows from one property of the data: every image of a combination carries the same label vector, so any held-out per-species metric compares whole combinations, and its effective sample size is the number of held-out combinations, not the number of images. On the nine test combinations of the seed-1337 partition, 63 to 90% of a decoder's per-species score variance lies between combinations.

Two consequences follow. The validation split of the LCO partition (an image-level split of the trained-on combinations) says nothing about the compositional shift, so it must not be used to choose between models or read-outs. And the nine test combinations are for reporting, once; anything chosen on them is chosen on the test set. Development therefore needs its own held-out combinations, and enough of them.

## The folds

`data/lco_dev_folds.json` names combinations only. It partitions the 26 trained-on combinations of order 2 to 4 of the seed-1337 LCO partition into five folds; each is held out exactly once, each fold spans orders 2 to 4, and the five singletons stay in every fold's training pool. The partition's own held-out singleton, `bt`, already leaves one species without a pure culture in every fold, so every fold matches the test partition in that respect, and the training pool of every fold covers all six species. `tools/build_lco_dev_folds.py` regenerates the file (`--seed`, `--n_folds`). Its `--hold_out_singletons` option also holds out one singleton per fold, which leaves two species per fold without a pure culture and makes the folds measurably harder than the test they stand in for (ChannelGroup 0.41 mean AUROC on those folds against 0.63 on the pooled test partitions).

| fold | held-out (dev) combinations |
|---|---|
| 0 | `bs_fj`, `fj_pf`, `bt_ka_fj`, `bt_mx_pf`, `mx_ka_pf`, `bt_mx_ka_fj` |
| 1 | `bt_ka`, `bt_mx`, `bs_mx_pf`, `bt_mx_ka`, `ka_fj_pf` |
| 2 | `bt_pf`, `mx_fj`, `bs_fj_pf`, `bs_ka_fj`, `bs_ka_fj_pf` |
| 3 | `bs_mx`, `mx_ka`, `bs_bt_mx`, `bt_fj_pf`, `bt_ka_fj_pf` |
| 4 | `bs_ka`, `mx_pf`, `bs_mx_ka`, `mx_fj_pf`, `bs_mx_ka_pf` |

## The procedure

1. **Train** on the fold's `train_combos`, using their images from the partition's train split.
2. **Calibrate** thresholds and any hyper-parameter that needs in-distribution data on the fold's `train_combos`, using their images from the partition's val split. Nothing from `dev_combos` is seen at this step.
3. **Evaluate** on the fold's `dev_combos`: all their images (the partition's train and val parts).
4. **Pool** the per-image predictions and scores of the dev combinations over the five folds. The pooled set contains the 26 trained-on combinations of order 2 to 4, each held out once, and is the development analogue of the test split.
5. **Intervals.** Every pooled metric carries a 95% interval from a bootstrap over combinations (2,000 resamples of combinations with replacement, recomputing the metric from the pooled per-image predictions of the resampled combinations). `src/compositional/intervals.py::heldout_report` does this; it is also the tool for the test partition and for pooling several partitions (resampling unit `seed:combination`).
6. **Select** with the pooled metrics: primary criterion mean per-species AUROC (threshold-free, so the constant all-present predictor cannot score on it); secondary criterion macro-F1 margin over the constant predictor. Per-sample F1 alone is not a selection criterion, because a predictor that marks every species present scores 0.67 on order-3 combinations and 0.80 on order-4 ones without discriminating anything.
7. **Report** the nine test combinations once, after selection, with the same intervals, and alongside the other two partitions (seeds 1338 and 1339 hold out different combinations; pool the three for 27 held-out combinations when a species-level statement has to be resolved). A statement about a specific species is made only when its interval excludes chance.

`experiments/run_comp_nested_cv.py` applies the procedure to the closed-form decoders from the unit cache without re-extracting features; a trained decoder needs one training per fold.
