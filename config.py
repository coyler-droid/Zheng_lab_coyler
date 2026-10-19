"""
config.py -- every system-specific and run-specific setting of the pipeline.

Every stage script does `import config as C` and reads its settings from here; no other file
holds tunable values.

Adapting the pipeline to a new chemical system / camera set-up
    * Safe to change (then rerun from the stage that uses the setting):
        - ACTIVE_PROFILE and its entry in SYSTEM_PROFILES (image size, colour ranges, bands,
          CLIP model, review band);
        - MASK_PARAMS (stage 02), normalization references (stage 03), hand-feature and
          surface-detection thresholds (stage 04), CLIP settings (stage 05), augmentation
          (stage 06), model grids (stage 07), threshold objective (stage 09), BOOTSTRAP_N.
    * Changing any split setting (stage 01) requires a fresh `python data_split.py` and a
      rerun of every later stage.
    * Do NOT change the file / folder names in "Paths": the scripts, the run archive and the
      sealed-label isolation check (common.assert_sealed_isolation) rely on them.

Auto-managed lines
    Lines ending in "# [auto]" (bottom of this file) are rewritten by the scripts through
    common.set_auto_config(); the comment on each line names the writer. data_split.py resets
    them after a re-split. Keep their one-line `NAME = value  # [auto] ...` format.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent   # project folder; every path below is relative to it

# =============================================================================
# Reproducibility
# =============================================================================
SEED = 42   # single seed for numpy, sklearn, torch, augmentation and bootstrap

# =============================================================================
# Paths -- inputs (produced by step1.py; never modified by the pipeline)
# =============================================================================
IMAGES_DIR = ROOT / "anon_images"          # anonymised photos <id>.jpg, loaded by id only
MANIFEST_CSV = ROOT / "manifest.csv"       # id, label, group, replicate_tag, original_name
EXCLUDED_CSV = ROOT / "excluded.csv"       # conflicting-label condition, deliberately left out
IMAGE_EXT = ".jpg"                         # file extension of the images in IMAGES_DIR

# =============================================================================
# Paths -- outputs written by the stage scripts
# =============================================================================
SPLITS_CSV = ROOT / "splits.csv"                 # id, split, group        (NO labels)
FOLDS_CSV = ROOT / "folds.csv"                   # id, repeat, fold, ...   (NO labels)
LABELS_DEV_CSV = ROOT / "labels_dev.csv"         # id, label for dev ids ONLY
# The sealed test labels. ONLY data_split.py (writes) and final_test.py (reads, through one
# dedicated loader) may refer to this constant -- common.assert_sealed_isolation() enforces it.
SEALED_TEST_LABELS_CSV = ROOT / "labels_test_SEALED.csv"
SPLITS_META_JSON = ROOT / "splits_meta.json"     # split settings + manifest hash (stage 01)

TUBE_MASK_NPY = ROOT / "tube_mask.npy"           # bool HxW, interior of the tube (stage 02)
REF_MASK_NPY = ROOT / "ref_mask.npy"             # bool HxW, neutral holder/background (stage 02)
MASK_META_JSON = ROOT / "mask_meta.json"         # mask parameters + hashes (stage 02)

FEATURES_HAND_CSV = ROOT / "features_hand.csv"           # dev originals until stage 11
FEATURES_HAND_AUG_CSV = ROOT / "features_hand_aug.csv"   # augmented dev views (stage 06)

FIG_DIR = ROOT / "figures"                       # one sub-folder per stage
PROCESSED_DIR = ROOT / "processed"               # masked/, normalized/, augmented/
CACHE_DIR = ROOT / "cache"                       # CLIP embeddings, normalized aug views
CV_DIR = ROOT / "cv_results"                     # out-of-fold predictions, selection, thresholds
MODELS_DIR = ROOT / "models"                     # frozen final models (stage 10)
REPORTS_DIR = ROOT                               # test_report*.csv, review_labels.csv
ARCHIVE_DIR = ROOT / "archive"                   # previous outputs, moved here on a re-split
RUN_LOG = ROOT / "run_log.txt"                   # everything printed by every stage
FROZEN_CONFIG_JSON = ROOT / "frozen_config.json" # full frozen pipeline description (stage 10)

# Stage names, in pipeline order (also the figure sub-folder names).
STAGES = ["01_split", "02_mask", "03_normalization", "04_hand_features", "05_clip",
          "06_augmentation", "07_model_comparison", "08_error_analysis", "09_threshold",
          "10_freeze", "11_final_test", "12_deployment"]
# Script that runs each stage (reference only; not read by the scripts).
STAGE_SCRIPTS = {"00_data_checks": "data_checks.py", "01_split": "data_split.py",
                 "02_mask": "mask.py", "03_normalization": "preprocess.py",
                 "04_hand_features": "features_hand.py", "05_clip": "features_clip.py",
                 "06_augmentation": "augment.py", "07_model_comparison": "train_cv.py",
                 "08_error_analysis": "evaluate.py", "09_threshold": "threshold.py",
                 "10_freeze": "freeze.py", "11_final_test": "final_test.py",
                 "12_deployment": "predict.py --demo"}


# =============================================================================
# Replaceable label parser -- used ONLY when building a manifest for a new data set (as
# step1.py does). The modelling pipeline never calls it: labels come from manifest.csv
# (data_split.py) and afterwards only from labels_dev.csv / the sealed file.
# =============================================================================
def parse_label_yn_suffix(original_name):
    """'..._Y.jpg' -> 1 (success), '..._N.jpg' -> 0 (fail), anything else -> None."""
    stem = Path(original_name).stem
    tail = stem.rsplit("_", 1)[-1].upper()
    return {"Y": 1, "N": 0}.get(tail)


# =============================================================================
# Named colour ranges. The colour-fraction features are generated by LOOPING over this
# dict, so adding an entry adds a feature (frac_<name>) automatically.
#   space "LAB": box on (L*, a*, b*)          L* 0..100, a*/b* about -128..127
#   space "LCH": box on (L*, chroma, hue deg)  hue = atan2(b*, a*) in degrees, 0..360
#   space "HSV": box on (H deg 0..360, S 0..1, V 0..1)
#   "derive": "fold"  -> the box is NOT fixed: inside every training fold it is derived from
#                        LAB percentiles of the lowest band of that fold's training successes
#                        (recorded per fold in cv_results/, final values in frozen_config.json).
# =============================================================================
COLOR_RANGES_COF = {
    # Per-channel 5-95 % box covers ~78 % of the success bottom-band pixels (a 10-90 % box
    # covered ~61 % and dropped dark precipitate).
    "brown": {"space": "LAB", "derive": "fold", "source_band": 1, "source_label": 1,
              "percentiles": (5, 95)},
    # Fixed, label-free hue boxes, set at stage 04 from the hue histogram of all DEV liquid
    # pixels (one yellow peak at ~101 deg, 99 % below 117 deg): "yellow" holds the whole
    # yellow / yellow-green cloud; "green" (true green) is ~0 % here, kept for other systems.
    "yellow": {"space": "LCH", "lo": (35.0, 20.0, 82.0), "hi": (100.0, 200.0, 125.0)},
    "green": {"space": "LCH", "lo": (25.0, 12.0, 125.0), "hi": (100.0, 200.0, 170.0)},
    # --- examples for other systems: uncomment (and check the numbers) to add features -----
    # "orange": {"space": "LCH", "lo": (35.0, 25.0, 45.0), "hi": (85.0, 200.0, 70.0)},
    # "purple": {"space": "LCH", "lo": (10.0, 15.0, 280.0), "hi": (80.0, 200.0, 340.0)},
    # "clear":  {"space": "LCH", "lo": (75.0, 0.0, 0.0),  "hi": (100.0, 8.0, 360.0)},
}
HEIGHT_FEATURE_COLORS = ["brown"]   # colours that get a "<colour> region height / liquid height"
                                    # feature


# =============================================================================
# System profiles -- ACTIVE_PROFILE selects one; its values are unpacked below.
# =============================================================================
SYSTEM_PROFILES = {
    "cof_synthesis": dict(
        image_size=(208, 340),                  # (width, height) px, all images identical
        mask_file="tube_mask.npy",              # informational; scripts use TUBE_MASK_NPY
        label_parser=parse_label_yn_suffix,     # see "Replaceable label parser" above
        liquid_region="surface_to_tip",         # liquid column = detected meniscus -> V-tip
        n_bands=4,                              # vertical bands measured from the tip
        color_ranges=COLOR_RANGES_COF,
        clip_model=("ViT-B-32", "laion2b_s34b_b79k"),   # open_clip (architecture, weights tag)
        # Chosen for this project at stage 09: None = no REVIEW band, every tube gets YES/NO.
        # A (0.3, 0.7) band would have sent 29 % of dev tubes to manual review.
        review_band=None,
    ),
    # Template for another system:
    # "other_system": dict(
    #     image_size=(240, 320),
    #     mask_file="other_mask.npy",
    #     label_parser=parse_label_yn_suffix,     # or your own function
    #     liquid_region="whole_mask",             # e.g. a vial with no meniscus to detect
    #     n_bands=3,
    #     color_ranges={"red": {"space": "LCH", "lo": (20, 20, 0), "hi": (80, 200, 40)}},
    #     clip_model=("ViT-L-14", "openai"),
    #     review_band=(0.25, 0.75),
    # ),
}
ACTIVE_PROFILE = "cof_synthesis"

_P = SYSTEM_PROFILES[ACTIVE_PROFILE]
IMAGE_W, IMAGE_H = _P["image_size"]                  # expected image size, px
LABEL_PARSER = _P["label_parser"]                    # not read by the modelling scripts
LIQUID_REGION = _P["liquid_region"]                  # not read by the modelling scripts
N_BANDS = _P["n_bands"]                              # vertical bands for hand features
COLOR_RANGES = _P["color_ranges"]                    # colour boxes for frac_<name> features
CLIP_MODEL_NAME, CLIP_PRETRAINED = _P["clip_model"]  # open_clip model + pretrained tag
REVIEW_BAND = _P["review_band"]                      # (lo, hi) probability band or None

# --- Multi-class extension (not implemented; binary is the default and only tested mode) ---
# LABEL_MODE = "binary"            # "binary" | "multiclass"
# CLASS_NAMES = ["fail", "success"]  # e.g. ["fail", "partial", "success"] for multiclass
# Required changes when LABEL_MODE == "multiclass":
#   * labels_dev.csv / sealed file hold integer class ids 0..K-1;
#   * StratifiedGroupKFold already stratifies on any integer label;
#   * train_cv.py: LogisticRegression is multinomial (softmax) by default -- keep it; RF
#     unchanged; predict_proba returns K columns; use macro-F1 and per-class PR curves
#     (one-vs-rest average_precision) in evaluate.py; the threshold stage becomes
#     arg-max + a "REVIEW" rule on the top-2 probability margin.

# =============================================================================
# Split (stage 01) -- changing anything here requires a fresh re-split
# =============================================================================
SPLIT_MODE = "cv_plus_test"     # "cv_plus_test" (default) | "fixed_60_20_20" | "fixed_test"
# "fixed_test": the test set is exactly the ids listed in FIXED_TEST_IDS_FILE (e.g. the
# original test tubes + new photos); all other photos are DEV with the usual repeated
# StratifiedGroupKFold. Write the file with `python data_split.py --save-fixed-test-ids`
# BEFORE re-splitting.
FIXED_TEST_IDS_FILE = ROOT / "fixed_test_ids.txt"
TEST_FRACTION = 0.20            # ~fraction in the sealed test set (one of round(1/x) group folds)
N_FOLDS = 5                     # StratifiedGroupKFold folds per repeat (dev set)
N_REPEATS = 5                   # repeats, each with its own shuffle seed
TEST_SUCCESS_RANGE = (9, 12)    # allowed number of successes in the test set, inclusive;
                                # seeds are retried until satisfied
MIN_SUCCESS_PER_FOLD = 6        # every CV held-out fold (or the val split) needs >= this
MAX_SPLIT_ATTEMPTS = 500        # seeds tried (SEED, SEED+1, ...) before giving up
RETRAIN_ON_TRAIN_VAL = True     # fixed_60_20_20 only: final model trained on train+val

# =============================================================================
# Showcase images (fixed set of DEV images followed through every stage)
# =============================================================================
SHOWCASE_N_RANDOM_PER_CLASS = 4  # random showcase images per class
SHOWCASE_N_USER = 4              # hand-picked showcase images (mask.py --set-showcase)
CONTACT_SHEET_N = 24             # random dev images on the stage-02 contact sheet

# =============================================================================
# Tube mask (stage 02). The polygon is derived from DEV images (no labels) by mask.py; these
# parameters are tuned until the mask is approved (`python mask.py --approve`).
# =============================================================================
MASK_PARAMS = {
    # opening of the white holder (rounded rectangle), measured on the dev median image, px
    "window": {"x0": 40, "y0": 17, "x1": 167, "y1": 322, "radius": 48},
    "wall_inset_px": 5,          # move the traced inner walls inward by this much
    "top_y": None,               # None = automatic (just inside the window's top arc)
    "tip_y": None,               # None = automatic (median V-tip of the dev images)
    "tip_round_px": 6,           # rounding of the V-tip corners
    "print_freq_thr": 0.10,      # cut out pixels that are printed ink in > 10 % of dev images
    "print_zone_max_y": 135,     # ... but only above this row (no print lower on the tube)
    "spot_freq_thr": 0.45,       # fixed bright dust spots / glints: in > 45 % of dev images
    "print_dilate_px": 3,        # safety margin around the cut-outs
    "min_row_width_px": 40,      # the mask starts at the first row at least this wide
    "ref_chroma_p90_max": 12.0,  # reference region: neutral in >= 90 % of dev images
    # Extra (x0, y0, x1, y1) rectangles to cut out, added at stage 02 after inspecting DEV
    # images: the printed "2" numeral (left, anywhere in x 45-100 / y 68-125 depending on how
    # the tube is rotated) and the graduation marks (right) leaked into the automatic mask in
    # ~10-15 tubes. Above y ~125 the mask keeps a 41-px central strip (supernatant only).
    "exclude_boxes": [(44, 60, 98, 127), (140, 48, 172, 124)],
}

# =============================================================================
# Normalization (stage 03)
#   WB_REFERENCE: which pixels the gray-world white balance is estimated on
#       "frame"      -> classic gray-world over the whole photo
#       "background" -> neutral white holder / back-light outside the tube (ref_mask.npy)
#       "mask"       -> inside the tube mask (removes the liquid colour itself!)
#   EXPOSURE_REFERENCE: region whose mean luminance is scaled to EXPOSURE_TARGET_L
#       "mask" | "background" | "frame"
# =============================================================================
# Chosen for this project at stage 03 after comparing all options on DEV images:
# "frame" / "mask" left most lighting variation (holder L* SD 7.8 -> 6.7) and lightened dark
# tubes; "background" / "background" reduced it to 1.8 and kept the liquid colour.
WB_REFERENCE = "background"
EXPOSURE_REFERENCE = "background"
EXPOSURE_TARGET_L = {"mask": 70.0, "background": 85.0, "frame": 78.0}   # target mean L*
NEUTRAL_GRAY = (128, 128, 128)   # RGB used for background blanking and square padding

# =============================================================================
# Hand-built features (stage 04)
# =============================================================================
PROFILE_SMOOTH_SIGMA = 2.0       # px, Gaussian smoothing of vertical colour profiles
CENTRAL_STRIP_FRAC = 0.6         # fraction of the mask width used for surface/tip profiles
BOUNDARY_EXCLUDE = (0.06, 0.06)  # ignore this fraction of the column at (tip, surface) ends
SURFACE_MAX_Y = 150              # px row, lowest plausible meniscus (dev images: <= ~130)
SURFACE_MIN_VALLEY = 3.0         # meniscus = thin darker line: L* dip of at least this much
SURFACE_MIN_CONTRAST = 4.0       # ... and a colour step (Delta-E) of at least this much
SURFACE_STRONG_STEP = 12.0       # or: a sharp colour step of at least this Delta-E ...
SURFACE_MIN_SHARPNESS = 0.45     # ... with this fraction of it within +/-2 rows (a gradual
                                 # yellowing ramp inside the liquid scores ~0.3)
SURFACE_GOOD_SCORE = 10.0        # detections scoring below this are flagged "weak"
SURFACE_OVERRIDES = {}           # {id: y_px} manual surface rows for failed detections
INCLUDE_HOMOGENEITY_FEATURES = True    # per-band SD of b* ("spotty vs uniform yellowing");
                                       # chosen a priori at stage 04, before seeing results

# =============================================================================
# CLIP (stage 05) -- frozen embeddings, no fine-tuning
# =============================================================================
CLIP_IMAGE_SIZE = 224                            # px, square CLIP input
CLIP_VIEWS = ("full", "lower", "upper")          # crops of the tube that are embedded
CLIP_OPTIONAL_SECOND = ("ViT-L-14", "openai")    # optional second model (not read by scripts)
PCA_COMPONENTS_GRID = (10, 20, 30)               # PCA sizes tried on the CLIP embeddings

# =============================================================================
# Augmentation (stage 06) -- applied to training folds only
# =============================================================================
AUG_N_VIEWS_GENERATED = 9        # generated & cached once per dev image (seeded by id, k)
AUG_VIEWS_PER_CLASS = {          # how many of the cached views each training image uses,
    "NONE": {1: 0, 0: 0},        # per label (1 = success, 0 = fail)
    "LIGHT": {1: 4, 0: 2},
    "HEAVY": {1: 9, 0: 3},
}
AUG_SETTINGS = ("NONE", "LIGHT", "HEAVY")   # settings compared at stage 07
AUG_HFLIP_P = 0.5                # probability of a horizontal flip
AUG_ROTATION_DEG = 5.0           # uniform in [-5, +5] degrees
AUG_ROTATION_CENTER = "tip"      # "tip" (V-tip of the mask) | "image"
AUG_BRIGHTNESS = 0.15            # multiplicative, uniform in [1-0.15, 1+0.15]
AUG_CONTRAST = 0.15              # multiplicative, uniform in [1-0.15, 1+0.15]
AUG_TRANSLATE_PX = 3             # uniform integer shift in [-3, +3] px, x and y

# =============================================================================
# Models & CV (stage 07)
# =============================================================================
# <feature set>_<classifier>: {hand, clip, hand+clip} x {LR, RF}.
MODELS = ["hand_LR", "hand_RF", "clip_LR", "clip_RF", "hand+clip_LR", "hand+clip_RF"]
# Simplest first: used for "prefer the simpler model when scores are within one std".
MODEL_SIMPLICITY_ORDER = ["hand_LR", "hand_RF", "clip_LR", "clip_RF",
                          "hand+clip_LR", "hand+clip_RF"]
LR_C_GRID = (0.01, 0.1, 1.0, 10.0)   # logistic-regression inverse regularisation strengths
RF_GRID = {"n_estimators": (200, 500), "max_depth": (None, 5), "min_samples_leaf": (1, 5)}
NESTED_CV = False                # optional: inner StratifiedGroupKFold for hyper-parameters
NESTED_INNER_FOLDS = 4           # inner folds when NESTED_CV is True
N_JOBS = -2                      # joblib workers for the CV grid (all cores but one)

# =============================================================================
# Threshold (stage 09)
#   "max_f1" (default) | "recall>=X" (best precision) | "precision>=X" (best recall)
# =============================================================================
THRESHOLD_OBJECTIVE = "max_f1"
THRESHOLD_TARGET = 0.90          # the X for the constrained objectives

# =============================================================================
# Final test (stage 11)
# =============================================================================
BOOTSTRAP_N = 1000               # group-bootstrap resamples for the 95 % CIs of test metrics

# =============================================================================
# AUTO-MANAGED STATE -- written by the scripts named in each comment. Do not reformat.
# =============================================================================
SHOWCASE_IDS = []  # [auto] mask.py --set-showcase
MASK_APPROVED = False  # [auto] mask.py --approve
