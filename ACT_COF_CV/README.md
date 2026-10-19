# COF tube classifier

Predicts from a single photo whether a COF synthesis in a Falcon tube was **successful**
(crystalline, label 1) or **failed** (label 0). The goal is a quick **pre-screen before PXRD**:
tubes the model rates as likely crystalline are measured first.

The pipeline is built for a **small dataset** (about 150 photos) and is strict about honest
evaluation:
- photos of the same synthesis condition are never split between training and testing;
- the test labels stay sealed until the final model is frozen;
- the test set is scored exactly once.

> **Data are not part of this repository.** Photos, labels and all results are excluded by
> `.gitignore`. To run the pipeline, provide your own photos (see [Data](#data)).

## Status

Proof of concept on one photo setup (COF-144, 151 photos, 52 successes).

| | Cross-validation (dev) | Sealed test (30 tubes, 10 successes) |
|---|---|---|
| Selected model `hand_RF` | PR-AUC 0.87 | PR-AUC 0.58 [0.31, 0.90], F1 0.52 [0.20, 0.78] |

The drop from cross-validation to the test set is mainly due to the labels: many tubes look like
the opposite label, because the original success/fail labels were assigned by eye. A re-test with
labels from a PXRD crystallinity formula is planned.

## How it works

Every photo goes through the same steps:

1. **Mask:** a fixed tube mask, derived once from the dev photos without using labels.
2. **Colour correction:** gray-world white balance and exposure correction against the white
   holder.
3. **Two kinds of features:**
   - **hand features**: 25 colour and layering numbers measured inside the liquid (L\*a\*b\* in 4
     bands from the tip, yellow/green/brown fractions, layer boundary, ...);
   - **CLIP features**: frozen OpenCLIP ViT-B/32 embeddings of 3 views (full, lower half,
     upper half), reduced by PCA.
4. **Classifier:** logistic regression (LR) or random forest (RF).

Six models compete, {hand, clip, hand+clip} x {LR, RF}, each with and without data augmentation,
under 5 x 5 repeated grouped cross-validation. They are compared with two baselines: "always
fail", and a one-feature model on the brown fraction. Fixed selection rules pick a primary model,
preferring the simpler model when scores are within one standard deviation. A decision threshold
is then chosen on out-of-fold predictions.

## Setup

```powershell
conda create -n cof144_clf python=3.11 -y
conda activate cof144_clf
pip install -r requirements.txt          # exact versions used: requirements_lock.txt
```

A CPU is enough. The CLIP weights (`ViT-B-32` / `laion2b_s34b_b79k`, about 600 MB) are downloaded
once into the Hugging Face cache. `features_clip.py` checks for them and asks for
`--allow-download` if they are missing. The seed is 42 everywhere, and everything printed is also
appended to `run_log.txt`.

## Data

Photos are named by synthesis condition, with an optional replicate tag and `_Y` / `_N` for
success / fail:

```text
p9mmol_100ul_1p5hr_250dc_Y.jpg     # condition 9mmol_100ul_1.5hr_250dc, success ("1p5" = 1.5)
p6mmol_75ul_1hr_150dc_v2_N.jpg     # replicate photo (tag v2), fail
```

`step1.py` reads a folder of such photos into the project:

```powershell
python step1.py <photo folder> .
```

It writes:
- `anon_images/<id>.jpg`: copies under random ids, so no model ever sees a filename;
- `manifest.csv` (id, label, group, replicate tag, original name);
- `unlabeled.csv` and `excluded.csv`.

Ids stay the same when `step1.py` runs again. Photos must be 208 x 340 px crops with the same
framing (the mask is fixed).

## Running the pipeline

Run the stages in order. After every stage the script saves figures to `figures/<stage>/`, prints
a summary and stops, so you can check the result before going on.

| # | Command | What it does |
|---|---|---|
| 00 | `python data_checks.py` | ids vs files, image sizes, labels, duplicates, groups |
| 01 | `python data_split.py` | grouped stratified test split (20 %), 5 x 5 CV folds on dev, seals test labels |
| 02 | `python mask.py`, then `python mask.py --set-showcase random --approve` | fixed tube mask from dev photos; you approve it |
| 03 | `python preprocess.py` | colour correction, background blanked |
| 04 | `python features_hand.py` | meniscus/tip detection, hand features |
| 05 | `python features_clip.py` | CLIP embeddings of 3 views |
| 06 | `python augment.py` | cached augmented views (flip, small rotation/shift, brightness/contrast) |
| 07 | `python train_cv.py`, then `python train_cv.py --confirm-selection` | model comparison and selection |
| 08 | `python evaluate.py` | misclassified photos, shortcut checks, `review_labels.csv` |
| 09 | `python threshold.py` | decision threshold from out-of-fold predictions |
| 10 | `python freeze.py` | final models trained on all dev photos, `frozen_config.json` |
| 11 | `python final_test.py`, then `python final_test.py --answer yes` | locks test predictions, **then** opens the sealed labels; bootstrap CIs |
| 12 | `python predict.py --demo` | example prediction |

To use the frozen model on new photos:

```powershell
python predict.py path\to\photo.jpg --annotate
python predict.py path\to\folder --out predictions.csv
```

## Safeguards against leakage

- **Manifest hash:** `splits_meta.json` stores the SHA-256 of `manifest.csv`, and every later
  stage aborts if it changed. Re-splitting (`data_split.py --confirm-rebuild`) first archives all
  previous outputs to `archive/<timestamp>/`.
- **Sealed test labels:** `labels_test_SEALED.csv` is read by a single function in
  `final_test.py`, and only after the test predictions are written and hashed.
  `common.assert_sealed_isolation()` checks at every stage that no other file refers to it.
- **No test images during development:** test images cannot be loaded before stage 11.
- **Fitting inside the fold:** everything that is fitted (colour range, imputer, scaler, PCA,
  classifier) is fitted on the training part of each fold only. Held-out photos are never
  augmented.
- **Shortcut checks:** stage 08 tests whether nuisance variables (fill height, raw brightness,
  white-balance ratios) predict the label better than chance.

## Changing labels and re-testing

- `python apply_label_changes.py --template` writes `retest_protocol/new_labels.csv`, one row per
  photo. Fill in a crystallinity value (or PASS/FAIL, or NA) for every photo, then apply it with
  `python apply_label_changes.py --formula --cutoff <X> [--apply]`.
- `SPLIT_MODE = "fixed_test"` in `config.py` uses a given list of test ids instead of a random
  split, for example the original test tubes plus new photos (`data_split.py
  --save-fixed-test-ids`).
- `python label_review.py` makes blind review sheets of the ambiguous tubes.

After changing labels, run `data_split.py --confirm-rebuild` and all later stages again.

## Configuration

All settings are in `config.py`, grouped by stage: paths, split, mask, normalization, colour
ranges, CLIP, augmentation, models, threshold. Settings specific to the chemical system (image
size, colour ranges, number of bands, label parser) are collected in `SYSTEM_PROFILES`. Lines
marked `# [auto]` are written by the scripts themselves.

## Files

| File | Role |
|---|---|
| `config.py` | all settings |
| `common.py` | shared helpers: logging, stage start/stop, sealing checks, data loading |
| `visualize.py` | figure helpers |
| `step1.py` | photo ingestion (anonymized ids) |
| `data_checks.py` ... `predict.py` | stages 00-12 (see the table above) |
| `apply_label_changes.py`, `label_review.py` | label curation tools |

## Limitations

- **One setup only.** The fixed mask, the 208 x 340 framing, the white-holder colour correction
  and the colour ranges are specific to one photo rig and one COF system. Other benches need a
  tube detector and a colour reference card.
- **Small data.** Confidence intervals on 30 test tubes are wide; treat all numbers as
  preliminary.
- **It is a pre-screen.** PXRD stays the reference measurement.
