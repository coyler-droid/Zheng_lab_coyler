"""
Stage 12 -- deployment: classify new tube photos with the FROZEN model.

  python predict.py <image_or_folder> [--model primary|secondary] [--annotate] [--out preds.csv]
  python predict.py --demo         # stage-12 figure: annotated output for one example image

Loads frozen_config.json (model and tube-mask hashes are checked), the model bundle (primary
by default), the tube mask and reference region, and applies exactly the DEV preprocessing:
normalization, mask, hand features (frozen colour ranges), CLIP (if the model uses it). Prints
the probability of success, YES/NO at the frozen threshold and the REVIEW flag (probability
inside the review band).

Outputs: a folder input (or --out) writes a CSV (default <folder>/predictions.csv) with the
columns file, probability, yes_no, review, call, surface_flag (plus error for skipped images);
--annotate also saves <stem>_annotated.png per image in an "annotated" folder next to the
input. --demo writes figures/12_deployment/annotated_example_output.png.
New photos must have the same framing and size as the training photos (208x340).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common
import config as C
import features_hand as FH
import train_cv as TC
import visualize as V

STAGE = "12_deployment"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_frozen(role: str):
    """Return (frozen_config dict, model bundle of `role`); exit 3 on a hash mismatch."""
    common.require_files(C.FROZEN_CONFIG_JSON, hint="run freeze.py first")
    frozen = common.json_load(C.FROZEN_CONFIG_JSON)
    model_info = frozen["models"][role]
    if common.sha256_file(C.ROOT / model_info["file"]) != model_info["sha256"]:
        print(f"ABORT: {model_info['file']} does not match the hash in frozen_config.json")
        sys.exit(3)
    if common.sha256_file(C.TUBE_MASK_NPY) != frozen["mask"]["sha256"]:
        print("ABORT: tube_mask.npy does not match the hash in frozen_config.json")
        sys.exit(3)
    return frozen, joblib.load(C.ROOT / model_info["file"])


def annotate(raw, feat, bundle, prob, call, tube, title_extra=""):
    """Two-panel figure: input with the tube outline, and the normalized image with the
    feature overlay; the title shows the probability and the call."""
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 5.4))
    axes[0].imshow(V.outline(raw, tube))
    axes[0].set_title("input + tube mask", fontsize=8)
    axes[1].imshow(FH.render_overlay(feat["norm"], tube, feat["rec"],
                                     bundle["color_ranges"]["brown"]))
    axes[1].set_title("normalized: bands (dashed), surface (cyan),\nbrown (red), boundary (green)",
                      fontsize=7)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    color = {"YES": "green", "NO": "black", "REVIEW": "darkorange"}[call]
    fig.suptitle(f"P(success) = {prob:.2f}  ->  {call}\n(threshold {bundle['threshold']:.3f}, "
                 f"REVIEW band {bundle['review_band'] or 'none'}; {bundle['model']} / "
                 f"{bundle['setting']}){title_extra}", fontsize=9, color=color)
    fig.tight_layout()
    return fig


def predict_paths(paths, role="primary", annotate_dir=None):
    """Predict every image in `paths` with the frozen `role` model.

    Images with the wrong size are skipped (a row with an "error" column). Returns the
    per-image DataFrame and the model bundle.
    """
    frozen, bundle = load_frozen(role)
    tube, ref = common.load_masks()
    embedder = None
    if TC.bundle_needs_clip(bundle):
        import features_clip as FC
        embedder = FC.ClipEmbedder(frozen["clip"]["model"], frozen["clip"]["pretrained"])
    rows = []
    for path in paths:
        raw = common.read_rgb(path)
        try:
            feat = TC.featurize_raw(raw, tube, ref, f"new:{path.name}", embedder)
        except ValueError as e:
            print(f"  {path.name}: SKIPPED ({e})")
            rows.append({"file": path.name, "error": str(e)})
            continue
        hand_row = TC.bundle_hand_row(bundle, feat["rec"], feat["px"])
        clip_rows = [feat["clip"]] if TC.bundle_needs_clip(bundle) else None
        prob = float(TC.bundle_predict(bundle, [hand_row], clip_rows)[0])
        call, yesno, review = TC.decide(prob, bundle["threshold"], bundle["review_band"])
        surface_flag = feat["rec"]["surface_flag"]
        rows.append({"file": path.name, "probability": round(prob, 4), "yes_no": yesno,
                     "review": review, "call": call, "surface_flag": surface_flag})
        qc = "" if surface_flag == "ok" else f"   (surface: {surface_flag})"
        print(f"  {path.name:40s} P(success) = {prob:.3f}  {yesno:3s}  -> {call}{qc}")
        if annotate_dir is not None:
            Path(annotate_dir).mkdir(parents=True, exist_ok=True)
            V.save_fig(annotate(raw, feat, bundle, prob, call, tube),
                       Path(annotate_dir) / f"{path.stem}_annotated.png")
    return pd.DataFrame(rows), bundle


def demo():
    """Stage-12 figure: annotated prediction for one (already scored) test image."""
    fig_dir = common.start_stage(STAGE)
    if not common.final_test_completed():
        print("ABORT: run the final test (stage 11) first; the demo uses a test image.")
        sys.exit(2)
    frozen, _ = load_frozen("primary")
    tube, ref = common.load_masks()
    rng = np.random.default_rng(C.SEED)
    img_id = str(rng.choice(common.test_ids()))
    path = common.image_path(img_id)
    df, bundle = predict_paths([path], "primary")
    # featurize again here because predict_paths() does not return the per-image features
    raw = common.read_rgb(path)
    embedder = None
    if TC.bundle_needs_clip(bundle):
        import features_clip as FC
        embedder = FC.ClipEmbedder(frozen["clip"]["model"], frozen["clip"]["pretrained"])
    feat = TC.featurize_raw(raw, tube, ref, f"new:{path.name}", embedder)
    result = df.iloc[0]
    V.save_fig(annotate(raw, feat, bundle, result["probability"], result["call"], tube,
                        title_extra=f"\nexample image {img_id} (a test image, already scored)"),
               fig_dir / "annotated_example_output.png")
    print("\nUsage on new photos:")
    print("  python predict.py path\\to\\photo.jpg --annotate")
    print("  python predict.py path\\to\\folder --out predictions.csv --annotate")
    print("  python predict.py path\\to\\folder --model secondary   (information only)")
    print("\n" + common.resplit_reminder())
    common.stop_here(STAGE)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", help="an image file or a folder of images")
    ap.add_argument("--model", choices=["primary", "secondary"], default="primary")
    ap.add_argument("--annotate", action="store_true", help="save annotated PNGs")
    ap.add_argument("--out", help="CSV path (default for folders: <folder>/predictions.csv)")
    ap.add_argument("--demo", action="store_true", help="stage-12 example figure")
    args = ap.parse_args()
    if args.demo:
        demo()
        return
    if not args.path:
        ap.error("give an image or a folder (or --demo)")
    src = Path(args.path)
    if src.is_dir():
        paths = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT)
    else:
        paths = [src]
    out_dir = src if src.is_dir() else src.parent
    print(f"predicting {len(paths)} image(s) with the {args.model} model")
    df, _ = predict_paths(paths, args.model,
                          (out_dir / "annotated") if args.annotate else None)
    if src.is_dir() or args.out:
        out = Path(args.out) if args.out else out_dir / "predictions.csv"
        df.to_csv(out, index=False)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
