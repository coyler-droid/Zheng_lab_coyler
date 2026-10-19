"""
Stage 06 -- augmentation of DEV images (used in TRAINING folds only).

Allowed (config.AUG_*):
  * horizontal flip -- a mirror about the tube's own axis (fitted to the mask), so the tube
    stays inside the fixed mask
  * rotation uniformly within +/-AUG_ROTATION_DEG, about the V-tip (keeps the narrow tip in
    the mask)
  * brightness and contrast jitter within +/-AUG_BRIGHTNESS / AUG_CONTRAST (same factor for R,
    G and B -> no hue change)
  * translation by up to AUG_TRANSLATE_PX pixels
NOT used, because they would alter the colour / layering signal or move the liquid out of the
fixed mask: hue or saturation jitter, grayscale, vertical flip, 90-degree rotations, blur,
noise, downscaling, any crop or zoom that cuts off the top or bottom of the liquid column.

The RAW image is augmented first; the mask, normalization, hand features and CLIP are then
re-applied. AUG_N_VIEWS_GENERATED views are generated ONCE per dev image (seeded by id and view
number, independent of labels) and cached; train_cv.py selects them per fold by source id
(config.AUG_VIEWS_PER_CLASS, e.g. HEAVY: 9 per success / 3 per failure, LIGHT: 4 / 2).
Augmented views always stay in their source image's fold; held-out folds and the test set are
never augmented.

Augmentation adds robustness to lighting and tube placement -- it does NOT add new chemistry.

Inputs : DEV images, tube/ref masks, processed/normalized/<id>.png (stage 03).
Outputs: processed/augmented/<id>_aug<k>.png (raw views), cache/aug_normalized/<id>_aug<k>.png,
         features_hand_aug.csv, cache/aug_params.csv, CLIP embeddings of the views (cache),
         figures/06_augmentation/*.png.

Usage:  python augment.py
"""
from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common
import config as C
import features_clip as FC
import features_hand as FH
import preprocess
import visualize as V

STAGE = "06_augmentation"
AUG_RAW_DIR = C.PROCESSED_DIR / "augmented"
AUG_NORM_DIR = C.CACHE_DIR / "aug_normalized"
AUG_PARAMS_CSV = C.CACHE_DIR / "aug_params.csv"


def tube_axis(tube: np.ndarray):
    """Least-squares axis x = c0 + c1*y through the row centres of the mask, and the tip.

    Returns (c0, c1, (tip_x, tip_y)).
    """
    rows = np.nonzero(tube.sum(axis=1) >= 4)[0]
    centres = np.array([np.nonzero(tube[y])[0].mean() for y in rows])
    c1, c0 = np.polyfit(rows, centres, 1)
    tip_y = int(rows.max())
    return c0, c1, (c0 + c1 * tip_y, float(tip_y))


def _to3(M):
    """2x3 affine matrix -> 3x3 homogeneous matrix (so transforms can be chained with @)."""
    return np.vstack([M, [0, 0, 1]])


def augment_raw(raw: np.ndarray, img_id: str, k: int, tube: np.ndarray):
    """Augmented RAW view k of an image (deterministic in (SEED, id, k)).

    Returns (uint8 RGB view, dict of the drawn parameters for aug_params.csv).
    """
    # the order of the random draws defines the views -- do not reorder
    rng = common.rng_for("augment", img_id, k)
    flip = bool(rng.random() < C.AUG_HFLIP_P)
    angle = float(rng.uniform(-C.AUG_ROTATION_DEG, C.AUG_ROTATION_DEG))
    bright = float(rng.uniform(1 - C.AUG_BRIGHTNESS, 1 + C.AUG_BRIGHTNESS))
    contrast = float(rng.uniform(1 - C.AUG_CONTRAST, 1 + C.AUG_CONTRAST))
    tx, ty = (int(v) for v in rng.integers(-C.AUG_TRANSLATE_PX, C.AUG_TRANSLATE_PX + 1, 2))

    # one affine warp: flip, then rotation, then translation (composed right-to-left)
    c0, c1, tip = tube_axis(tube)
    M = np.eye(3)
    if flip:                                    # reflection across the tube axis line
        direction = np.array([c1, 1.0]) / np.hypot(c1, 1.0)
        R = 2 * np.outer(direction, direction) - np.eye(2)
        p0 = np.array([c0, 0.0])                # a point on the axis (x at y = 0)
        M = _to3(np.hstack([R, (p0 - R @ p0)[:, None]])) @ M
    centre = tip if C.AUG_ROTATION_CENTER == "tip" else (C.IMAGE_W / 2, C.IMAGE_H / 2)
    M = _to3(cv2.getRotationMatrix2D(centre, angle, 1.0)) @ M
    M = _to3(np.array([[1, 0, tx], [0, 1, ty]], float)) @ M
    out = cv2.warpAffine(raw, M[:2], (C.IMAGE_W, C.IMAGE_H), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101).astype(np.float32)
    # contrast about the image mean, then brightness (same factors for R, G and B)
    m = out.mean()
    out = ((out - m) * contrast + m) * bright
    params = {"id": img_id, "aug": k, "flip": flip, "angle_deg": round(angle, 3),
              "brightness": round(bright, 4), "contrast": round(contrast, 4), "tx": tx, "ty": ty}
    return np.clip(out, 0, 255).astype(np.uint8), params


# --- examples of what is NOT used (sanity figure only) --------------------------------------
def _hue_shift(raw, deg=40):
    """Hue rotation by `deg` degrees (OpenCV 8-bit hue is 0..179, i.e. 2 deg per unit)."""
    hsv = cv2.cvtColor(raw, cv2.COLOR_RGB2HSV)
    hsv[..., 0] = (hsv[..., 0].astype(int) + deg // 2) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def _top_bottom_crop(raw, frac=0.22):
    """Crop `frac` of the height at top and bottom and stretch back to full size."""
    h = raw.shape[0]
    crop = raw[int(frac * h):int((1 - frac) * h)]
    return cv2.resize(crop, (raw.shape[1], h), interpolation=cv2.INTER_LINEAR)


def make_figures(fig_dir, tube, labels, raws, aug_raw, aug_norm, norms):
    show = list(C.SHOWCASE_IDS)[:6]
    ks = list(range(1, 9))
    # (1) original + 8 augmented RAW views (fixed mask outline drawn)
    items, titles = [], []
    for i in show:
        items.append(V.outline(raws[i], tube))
        titles.append(f"{i} | {V.label_str(labels[i])}\noriginal")
        for k in ks:
            items.append(V.outline(aug_raw[(i, k)], tube))
            titles.append(f"{i}\naug{k} (raw)")
    V.save_fig(V.show_grid(items, titles, ncols=9, tile_w=1.45, tile_h=2.35, fontsize=5.5,
                           suptitle="Original + 8 augmented RAW views (green = the fixed mask)"),
               fig_dir / "augmented_raw_views.png")
    # (2) the same views after mask + normalization
    items, titles = [], []
    for i in show:
        items.append(norms[i])
        titles.append(f"{i} | {V.label_str(labels[i])}\noriginal, normalized")
        for k in ks:
            items.append(aug_norm[(i, k)])
            titles.append(f"{i}\naug{k} masked+normalized")
    V.save_fig(V.show_grid(items, titles, ncols=9, tile_w=1.45, tile_h=2.35, fontsize=5.5,
                           suptitle="The same views after re-applying mask and normalization"),
               fig_dir / "augmented_normalized_views.png")
    # (3) what is NOT used
    fig, axes = plt.subplots(3, 4, figsize=(8, 8.8))
    for r, i in enumerate(show[:3]):
        raw = raws[i]
        panels = [(raw, "original", False), (_hue_shift(raw), "hue shift", True),
                  (raw[::-1].copy(), "vertical flip", True),
                  (_top_bottom_crop(raw), "top/bottom crop", True)]
        for c, (im, name, not_used) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(im)
            ax.set_title(f"{i}\n{name}" + (" (NOT USED)" if not_used else ""), fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
            if not_used:
                V.red_x(ax)
    fig.suptitle("Augmentations that are deliberately NOT used", fontsize=10)
    fig.tight_layout()
    V.save_fig(fig, fig_dir / "not_used_augmentations.png")


def main():
    fig_dir = common.start_stage(STAGE)
    common.require_mask_approved()
    common.require_files(C.FEATURES_HAND_CSV, hint="run features_hand.py first")
    tube, ref = common.load_masks()
    labels = common.load_dev_labels()
    ids = common.dev_ids()
    AUG_RAW_DIR.mkdir(parents=True, exist_ok=True)
    AUG_NORM_DIR.mkdir(parents=True, exist_ok=True)

    raws, norms, records, params, clip_items = {}, {}, [], [], []
    aug_raw, aug_norm = {}, {}                  # showcase views only, for the figures
    for i in ids:
        raw = common.load_image(i)
        raws[i] = raw
        norms[i] = common.read_rgb(C.PROCESSED_DIR / "normalized" / f"{i}.png")
        for k in range(1, C.AUG_N_VIEWS_GENERATED + 1):
            view_raw, view_params = augment_raw(raw, i, k, tube)
            _, view_norm, _, raw_stats = preprocess.process_raw(view_raw, tube, ref)
            rec, _ = FH.extract_record(view_norm, tube, view_raw, raw_stats, i, k)
            common.write_rgb(AUG_RAW_DIR / f"{i}_aug{k}.png", view_raw)
            common.write_rgb(AUG_NORM_DIR / f"{i}_aug{k}.png", view_norm)
            records.append(rec)
            params.append(view_params)
            clip_items.append((i, k, view_norm))
            if i in C.SHOWCASE_IDS:
                aug_raw[(i, k)], aug_norm[(i, k)] = view_raw, view_norm
    df = pd.DataFrame(records)
    df.to_csv(C.FEATURES_HAND_AUG_CSV, index=False)
    pd.DataFrame(params).to_csv(AUG_PARAMS_CSV, index=False)
    print(f"generated {len(df)} augmented views ({C.AUG_N_VIEWS_GENERATED} per DEV image) -> "
          "processed/augmented/, cache/aug_normalized/, features_hand_aug.csv")
    print("surface flags on augmented views:", df["surface_flag"].value_counts().to_dict())

    emb = FC.ClipEmbedder()
    n_new = FC.embed_and_cache(emb, clip_items, tube)
    print(f"CLIP: embedded {n_new} new augmented views (cached by id and view number)")

    for s in C.AUG_SETTINGS:
        n_rows = sum(1 + C.AUG_VIEWS_PER_CLASS[s][int(labels[i])] for i in ids)
        n_succ_rows = sum(1 + C.AUG_VIEWS_PER_CLASS[s][1] for i in ids if labels[i] == 1)
        print(f"  {s:6s}: views per success {C.AUG_VIEWS_PER_CLASS[s][1]}, per failure "
              f"{C.AUG_VIEWS_PER_CLASS[s][0]} -> all-DEV training set would be {n_rows} rows "
              f"({n_succ_rows} success rows, {n_rows - n_succ_rows} failure rows)")
    make_figures(fig_dir, tube, labels, raws, aug_raw, aug_norm, norms)
    n_succ = int(sum(labels[i] for i in ids))
    print("\nREMINDER: augmentation adds robustness to lighting and tube placement, but it does "
          f"NOT add new chemistry -- {n_succ} DEV successes stay {n_succ} independent examples.")
    V.update_progression(STAGE)
    common.stop_here(STAGE, next_cmd="train_cv.py")


if __name__ == "__main__":
    main()
