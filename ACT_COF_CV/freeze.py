"""
Stage 10 -- freeze. Train the two final models on the whole DEV set, with identical features
and hyper-parameter selection, and freeze every setting in frozen_config.json.

  PRIMARY  : the model and augmentation setting chosen by the stage-07 rules (and confirmed
             with train_cv.py --confirm-selection)
  SECONDARY: the same model type and hyper-parameters with the OPPOSITE augmentation choice
             (primary uses augmentation -> secondary uses none; primary uses none -> HEAVY);
             reported for information only

Inputs : cv_results/selection.json (confirmed), cv_results/thresholds.json, the approved
         tube/reference masks, splits_meta.json
Outputs: models/primary.joblib and models/secondary.joblib (colour ranges, PCA, imputer,
         scaler, classifier, threshold, review band); frozen_config.json (all settings plus
         file hashes, checked again by final_test.py and predict.py);
         figures/10_freeze/frozen_settings.png
NOTHING may be tuned after this point.

Usage:  python freeze.py                 first freeze
        python freeze.py --refreeze      redo the freeze (only before the final test, stage 11)
        python freeze.py --render-only   redraw the summary figure from frozen_config.json
"""
from __future__ import annotations

import argparse
import datetime
import sys

import joblib
import numpy as np

import common
import config as C
import features_hand as FH
import train_cv as TC
import visualize as V

STAGE = "10_freeze"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refreeze", action="store_true",
                    help="overwrite an existing freeze (only allowed before the final test)")
    ap.add_argument("--render-only", action="store_true",
                    help="only redraw figures/10_freeze/frozen_settings.png from "
                         "frozen_config.json (nothing is retrained or changed)")
    args = ap.parse_args()
    fig_dir = common.start_stage(STAGE)
    if args.render_only:
        common.require_files(C.FROZEN_CONFIG_JSON, hint="run freeze.py first")
        render_summary(common.json_load(C.FROZEN_CONFIG_JSON), fig_dir)
        common.stop_here(STAGE, next_cmd="final_test.py")
        return

    # --- preconditions -------------------------------------------------------------------------
    common.require_mask_approved()
    common.require_files(C.CV_DIR / "selection.json", C.CV_DIR / "thresholds.json",
                         hint="run train_cv.py (confirmed) and threshold.py first")
    if common.final_test_completed():
        print("ABORT: the final test has already been run -- the freeze can never change now.")
        sys.exit(2)
    if C.FROZEN_CONFIG_JSON.exists() and not args.refreeze:
        print("ABORT: frozen_config.json already exists. Use --refreeze only if you really need "
              "to redo the freeze before the final test.")
        sys.exit(2)
    selection = common.json_load(C.CV_DIR / "selection.json")
    thresholds = common.json_load(C.CV_DIR / "thresholds.json")
    if not selection.get("confirmed"):
        print("ABORT: stage-07 selection not confirmed (python train_cv.py --confirm-selection).")
        sys.exit(2)
    meta = common.json_load(C.SPLITS_META_JSON)

    # --- training images of the final models ---------------------------------------------------
    if C.SPLIT_MODE == "fixed_60_20_20" and not C.RETRAIN_ON_TRAIN_VAL:
        splits = common.load_splits()
        train_ids = sorted(splits.loc[splits["split"] == "train", "id"])
        print("fixed_60_20_20 with RETRAIN_ON_TRAIN_VAL=False -> final models use TRAIN only")
    else:
        train_ids = common.dev_ids()
    need_aug = any(selection[r]["setting"] != "NONE" for r in ("primary", "secondary"))
    # the secondary model has the same model type as the primary, so one check covers both
    need_clip = TC.needs_clip_image(selection["primary"]["model"])
    data = TC.DevData(need_aug=need_aug, need_clip=need_clip)
    C.MODELS_DIR.mkdir(exist_ok=True)

    # --- fit and save both final models --------------------------------------------------------
    models = {}
    for role in ("primary", "secondary"):
        info = selection[role]
        params = info["params"]
        bundle = TC.fit_bundle(data, train_ids, info["model"], info["setting"], params)
        bundle.update({"role": role, "threshold": thresholds[role]["threshold"],
                       "threshold_objective": thresholds[role]["objective"],
                       "review_band": list(C.REVIEW_BAND) if C.REVIEW_BAND else None})
        path = C.MODELS_DIR / f"{role}.joblib"
        joblib.dump(bundle, path)
        models[role] = {
            "file": f"models/{role}.joblib", "sha256": common.sha256_file(path),
            "model": info["model"], "augmentation_setting": info["setting"],
            "hyperparameters": params,
            "pca_components": params.get("n_pca") if isinstance(params, dict) else None,
            "threshold": thresholds[role]["threshold"],
            "threshold_objective": thresholds[role]["objective"],
            "color_ranges": bundle["color_ranges"], "n_train_rows": bundle["n_train_rows"],
            "n_train_images": bundle["n_train_images"],
            "n_train_success_images": bundle["n_train_success_images"],
            "cv_estimate": {k: info[k] for k in ("pr_auc_mean", "pr_auc_sd", "f1_mean", "f1_sd")},
        }
        print(f"{role.upper():9s} {info['model']} / {info['setting']} {params} -> trained "
              f"on {bundle['n_train_rows']} rows ({bundle['n_train_images']} DEV images, "
              f"{bundle['n_train_success_images']} successes); "
              f"threshold {bundle['threshold']:.3f}")

    # --- end-to-end consistency check ----------------------------------------------------------
    # The deployment path (raw image -> featurize_raw()) must reproduce the prediction obtained
    # from the cached training features; otherwise final_test.py / predict.py would score
    # images with different features than the model was trained on.
    tube, ref = common.load_masks()
    embedder = None
    if need_clip:
        import features_clip as FC
        embedder = FC.ClipEmbedder()
    primary_bundle = joblib.load(C.MODELS_DIR / "primary.joblib")
    check_ids = list(C.SHOWCASE_IDS)[:3]
    max_diff = 0.0
    for img_id in check_ids:
        feat = TC.featurize_raw(common.load_image(img_id), tube, ref, img_id, embedder)
        p_raw = TC.bundle_predict(
            primary_bundle, [TC.bundle_hand_row(primary_bundle, feat["rec"], feat["px"])],
            [feat["clip"]] if need_clip else None)[0]
        hand = data.hand_matrix(primary_bundle["color_ranges"], [(img_id, 0)]).to_numpy(float)
        p_cached = TC.bundle_predict(primary_bundle, hand,
                                     [data.clip[(img_id, 0)]] if need_clip else None)[0]
        max_diff = max(max_diff, abs(p_raw - p_cached))
    print(f"consistency check (raw image path vs cached training path) on {len(check_ids)} DEV "
          f"images: max |dp| = {max_diff:.2e}")
    if max_diff > 1e-3:
        print("ABORT: deployment preprocessing does not reproduce the training features.")
        sys.exit(2)

    # --- frozen_config.json --------------------------------------------------------------------
    mask_meta = common.json_load(C.MASK_META_JSON)
    frozen = {
        "frozen_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "statement": ("The PRIMARY model was chosen by cross-validation on the DEV set (stage 07, "
                      "confirmed by the user). The SECONDARY model (same model type and "
                      "hyper-parameters, opposite augmentation choice) is reported for "
                      "information only. Nothing may be tuned after this point."),
        "seed": C.SEED, "split_mode": C.SPLIT_MODE, "split_seed": meta["split_seed"],
        "manifest_sha256": meta["manifest_sha256"],
        "retrain_on_train_val": C.RETRAIN_ON_TRAIN_VAL, "n_final_train_images": len(train_ids),
        "mask": {"file": C.TUBE_MASK_NPY.name, "sha256": common.sha256_file(C.TUBE_MASK_NPY),
                 "reference_file": C.REF_MASK_NPY.name,
                 "reference_sha256": common.sha256_file(C.REF_MASK_NPY),
                 "params": C.MASK_PARAMS, "approved": mask_meta.get("approved", False)},
        "normalization": {"wb_reference": C.WB_REFERENCE,
                          "exposure_reference": C.EXPOSURE_REFERENCE,
                          "target_L": C.EXPOSURE_TARGET_L[C.EXPOSURE_REFERENCE],
                          "neutral_gray": C.NEUTRAL_GRAY},
        "features": {"hand_features": FH.model_feature_names(), "n_bands": C.N_BANDS,
                     "color_ranges_config": C.COLOR_RANGES,
                     "height_feature_colors": C.HEIGHT_FEATURE_COLORS,
                     "homogeneity_features": C.INCLUDE_HOMOGENEITY_FEATURES,
                     "surface_detection": {"max_y": C.SURFACE_MAX_Y,
                                           "min_valley": C.SURFACE_MIN_VALLEY,
                                           "min_contrast": C.SURFACE_MIN_CONTRAST,
                                           "strong_step": C.SURFACE_STRONG_STEP,
                                           "min_sharpness": C.SURFACE_MIN_SHARPNESS,
                                           "overrides": C.SURFACE_OVERRIDES}},
        "clip": {"model": C.CLIP_MODEL_NAME, "pretrained": C.CLIP_PRETRAINED,
                 "views": list(C.CLIP_VIEWS), "image_size": C.CLIP_IMAGE_SIZE},
        "augmentation": {"views_per_class": C.AUG_VIEWS_PER_CLASS, "hflip_p": C.AUG_HFLIP_P,
                         "rotation_deg": C.AUG_ROTATION_DEG,
                         "rotation_center": C.AUG_ROTATION_CENTER,
                         "brightness": C.AUG_BRIGHTNESS, "contrast": C.AUG_CONTRAST,
                         "translate_px": C.AUG_TRANSLATE_PX},
        "review_band": list(C.REVIEW_BAND) if C.REVIEW_BAND else None,
        "selection_rule": selection["rules"], "selection_reason": selection["reason"],
        "models": models,
    }
    common.json_dump(frozen, C.FROZEN_CONFIG_JSON)
    print(f"wrote {C.FROZEN_CONFIG_JSON.name} and models/primary.joblib, models/secondary.joblib")
    render_summary(frozen, fig_dir)
    V.update_progression(STAGE)
    common.stop_here(STAGE, next_cmd="final_test.py")


def render_summary(frozen: dict, fig_dir):
    """One-page table of the frozen settings, built from frozen_config.json only."""
    prim, sec = frozen["models"]["primary"], frozen["models"]["secondary"]

    def fmt_params(params):
        short = {"max_depth": "depth", "min_samples_leaf": "min_leaf", "n_estimators": "trees",
                 "n_pca": "pca_comps"}
        if not isinstance(params, dict):
            return str(params)
        return ", ".join(f"{short.get(k, k)}={v}" for k, v in params.items())

    def fmt_brown_range(model_info):
        rng = model_info["color_ranges"]["brown"]
        return (f"{np.round(rng['lo']).astype(int).tolist()} .. "
                f"{np.round(rng['hi']).astype(int).tolist()}")

    def fmt_cv(model_info, metric):
        cv = model_info["cv_estimate"]
        return f"{cv[metric + '_mean']:.3f} +/- {cv[metric + '_sd']:.3f}"

    band = frozen["review_band"] or "none (every tube YES/NO)"
    rows = [["setting", "PRIMARY", "SECONDARY (information only)"],
            ["model", prim["model"], sec["model"]],
            ["augmentation", prim["augmentation_setting"], sec["augmentation_setting"]],
            ["hyper-parameters", fmt_params(prim["hyperparameters"]),
             fmt_params(sec["hyperparameters"])],
            ["PCA components", prim["pca_components"], sec["pca_components"]],
            ["threshold", f"{prim['threshold']:.3f} ({prim['threshold_objective']})",
             f"{sec['threshold']:.3f} ({sec['threshold_objective']})"],
            ["review band", band, band],
            ["CV PR-AUC (DEV)", fmt_cv(prim, "pr_auc"), fmt_cv(sec, "pr_auc")],
            ["CV F1 (DEV)", fmt_cv(prim, "f1"), fmt_cv(sec, "f1")],
            ["training rows / images", f"{prim['n_train_rows']} / {prim['n_train_images']}",
             f"{sec['n_train_rows']} / {sec['n_train_images']}"],
            ["brown range (L*a*b* lo..hi)", fmt_brown_range(prim), fmt_brown_range(sec)],
            ["CLIP", f"{frozen['clip']['model']} / {frozen['clip']['pretrained']}", "same"],
            ["normalization", f"WB {frozen['normalization']['wb_reference']}, exposure "
                              f"{frozen['normalization']['exposure_reference']}", "same"],
            ["mask", f"tube_mask.npy sha256 {frozen['mask']['sha256'][:12]}...", "same"],
            ["split", f"{frozen['split_mode']}, seed {frozen['split_seed']}, manifest "
                      f"{frozen['manifest_sha256'][:12]}...", "same"],
            ["seed", frozen["seed"], frozen["seed"]],
            ["frozen at", frozen["frozen_at"], "same"]]
    V.render_table(rows, fig_dir / "frozen_settings.png",
                   "Frozen settings (stage 10) -- primary chosen by CV on DEV; secondary for "
                   "information only. Nothing may be tuned after this point.",
                   col_widths=[0.22, 0.39, 0.39])


if __name__ == "__main__":
    main()
