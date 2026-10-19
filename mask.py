"""
Stage 02 -- build ONE fixed tube mask from the DEV images (labels are never used for it).

The camera, holder and tube type are the same in every photo, so a single mask is used for all
images. This keeps the features comparable between images and avoids a per-image segmentation
step that could itself become a (label-correlated) source of variation.

How the mask is derived (all label-free):
  1. Inner tube walls: in every dev image and row, the run of liquid-coloured pixels
     (yellow..brown hue, chroma > 12) that contains the tube axis gives a left and a right
     wall position. Per-row medians over all dev images are fitted with a quadratic in y,
     giving two smooth wall curves; they are moved inward by MASK_PARAMS["wall_inset_px"]
     so that the dark glass edges and their refraction band are excluded.
  2. V-tip: median (over dev images) of the lowest liquid-coloured row on the tube axis.
     Top: MASK_PARAMS["top_y"] (None -> just inside the holder window's top arc).
  3. Holder window: rounded rectangle (MASK_PARAMS["window"]) -- the mask never leaves it.
  4. Printed scale numbers, tick marks, the printed circle label and fixed dust spots:
     thin dark (black-hat) or small bright (top-hat) structures that sit at the SAME pixel in
     many dev images ("print frequency" map) are cut out. The print moves with the tube
     rotation, so the frequency map covers the range of positions seen in the dev images.
  5. Extra rectangles in MASK_PARAMS["exclude_boxes"] are cut out (print that is too variable
     in position to reach the frequency threshold, see config.py).
  6. Reference mask (used by stage 03 for a background-referenced normalization):
     pixels outside the dilated tube region, away from the yellow side bleed, that are
     neutral (low chroma) in >= 90 % of dev images -> white holder and back-light.

Also picks the showcase images: SHOWCASE_N_RANDOM_PER_CLASS random successes and failures from
DEV (seeded) plus a contact sheet of CONTACT_SHEET_N other random DEV images from which the
remaining SHOWCASE_N_USER showcase images are chosen by hand.

Inputs : DEV images and DEV labels (labels only for the showcase selection and figure titles).
Outputs: tube_mask.npy, ref_mask.npy, mask_meta.json, showcase_candidates.json,
         processed/masked/<id>.png, figures/02_mask/*.png.

Usage:
  python mask.py                               # build mask candidate + figures, then STOP
  python mask.py --set-showcase ID ID ID ID    # add the 4 hand-picked showcase images
  python mask.py --set-showcase random         # or: 4 more at random (seed 42)
  python mask.py --approve                     # freeze the mask after reviewing the figures
"""
from __future__ import annotations

import argparse
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

import common
import config as C
import visualize as V

STAGE = "02_mask"
SHOWCASE_JSON = C.ROOT / "showcase_candidates.json"


# =============================================================================
# Colour helpers
# =============================================================================
def rgb_to_lab(rgb_u8: np.ndarray) -> np.ndarray:
    """uint8 RGB -> float32 CIELAB (L 0..100, a/b ~ -128..127)."""
    return cv2.cvtColor(rgb_u8.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def liquid_colour_map(lab: np.ndarray) -> np.ndarray:
    """Pixels whose colour looks like the yellow..brown liquid (hue 40-130 deg, chroma > 12)."""
    a, b = lab[..., 1], lab[..., 2]
    chroma = np.hypot(a, b)
    hue = np.degrees(np.arctan2(b, a)) % 360
    return (chroma > 12) & (hue > 40) & (hue < 130)


# =============================================================================
# Geometry estimation (dev images only, no labels)
# =============================================================================
def window_mask(params=None) -> np.ndarray:
    """Rounded-rectangle opening of the white holder (measured on the dev median image)."""
    p = (params or C.MASK_PARAMS)["window"]
    m = np.zeros((C.IMAGE_H, C.IMAGE_W), np.uint8)
    x0, y0, x1, y1, r = p["x0"], p["y0"], p["x1"], p["y1"], p["radius"]
    # two overlapping rectangles + four corner discs = rounded rectangle
    cv2.rectangle(m, (x0 + r, y0), (x1 - r, y1), 1, -1)
    cv2.rectangle(m, (x0, y0 + r), (x1, y1 - r), 1, -1)
    for cx, cy in [(x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)]:
        cv2.circle(m, (cx, cy), r, 1, -1)
    return m.astype(bool)


def estimate_walls(dev_imgs: list[np.ndarray], axis_x: int = 108):
    """Per-row median wall positions from liquid-coloured runs through the tube axis.

    Returns (rows, median left wall x, median right wall x, median V-tip row or None). Only rows
    measured in at least 10 dev images are kept.
    """
    H = C.IMAGE_H
    lefts = [[] for _ in range(H)]
    rights = [[] for _ in range(H)]
    tips = []
    for img in dev_imgs:
        liquid = liquid_colour_map(rgb_to_lab(img))
        # close tiny gaps (particles, dust) along each row
        liquid = cv2.morphologyEx(liquid.astype(np.uint8), cv2.MORPH_CLOSE,
                                  np.ones((1, 5), np.uint8)).astype(bool)
        # V-tip: lowest row in which a 9-px strip around the axis is mostly liquid-coloured
        axis_is_liquid = liquid[:, axis_x - 4:axis_x + 5].mean(axis=1) > 0.5
        axis_rows = np.where(axis_is_liquid)[0]
        if len(axis_rows):
            tips.append(axis_rows.max())
        for y in range(40, H - 20):
            if not liquid[y, axis_x]:
                continue
            row = liquid[y]
            x_left = axis_x
            while x_left > 0 and row[x_left - 1]:
                x_left -= 1
            x_right = axis_x
            while x_right < C.IMAGE_W - 1 and row[x_right + 1]:
                x_right += 1
            if x_right - x_left >= 25:          # ignore narrow fragments
                lefts[y].append(x_left)
                rights[y].append(x_right)
    ys, xl_med, xr_med = [], [], []
    for y in range(H):
        if len(lefts[y]) >= 10:
            ys.append(y)
            xl_med.append(np.median(lefts[y]))
            xr_med.append(np.median(rights[y]))
    ys, xl_med, xr_med = map(np.asarray, (ys, xl_med, xr_med))
    tip_y = int(np.median(tips)) if tips else None
    return ys, xl_med, xr_med, tip_y


def robust_quadratic(y, x, n_iter=5, k=2.5):
    """Quadratic fit x(y) with iterative outlier rejection.

    Each pass refits on the kept points and keeps points whose residual is below
    k * (MAD-based sigma) + 1 px, so rows disturbed by bubbles or print do not bend the walls.
    """
    keep = np.ones(len(y), bool)
    for _ in range(n_iter):
        coef = np.polyfit(y[keep], x[keep], 2)
        residual = x - np.polyval(coef, y)
        sigma = 1.4826 * np.median(np.abs(residual[keep] - np.median(residual[keep]))) + 1e-6
        keep = np.abs(residual) < k * sigma + 1.0
    return coef


def print_frequency(dev_imgs: list[np.ndarray]):
    """Fraction of dev images in which each pixel is a thin dark stroke (black-hat) or a small
    bright spot (top-hat): printed numbers / ticks / label, tube edges, fixed dust."""
    ink = np.zeros((C.IMAGE_H, C.IMAGE_W), np.float32)
    spot = np.zeros_like(ink)
    kernel_blackhat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    kernel_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    for img in dev_imgs:
        L = rgb_to_lab(img)[..., 0]
        ink += cv2.morphologyEx(L, cv2.MORPH_BLACKHAT, kernel_blackhat) > 8.0
        spot += cv2.morphologyEx(L, cv2.MORPH_TOPHAT, kernel_tophat) > 8.0
    n = float(len(dev_imgs))
    return ink / n, spot / n


def build_masks(dev_imgs: list[np.ndarray], params=None):
    """Return tube_mask, ref_mask and a dict of diagnostics."""
    p = params or C.MASK_PARAMS
    ys, xl, xr, tip_auto = estimate_walls(dev_imgs)
    coef_left = robust_quadratic(ys, xl)
    coef_right = robust_quadratic(ys, xr)
    window = window_mask(p)
    window_inner = cv2.erode(window.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    top_y = p["top_y"] if p["top_y"] is not None else p["window"]["y0"] + 10
    tip_y = p["tip_y"] if p["tip_y"] is not None else tip_auto
    inset = p["wall_inset_px"]

    # cone polygon: left wall top -> tip, then right wall tip -> top
    yy = np.arange(top_y, tip_y + 1)
    x_left = np.polyval(coef_left, yy) + inset
    x_right = np.polyval(coef_right, yy) - inset
    poly = np.concatenate([np.stack([x_left, yy], 1), np.stack([x_right, yy], 1)[::-1]])
    cone = np.zeros((C.IMAGE_H, C.IMAGE_W), np.uint8)
    cv2.fillPoly(cone, [np.round(poly).astype(np.int32)], 1)
    tip_radius = int(p["tip_round_px"])
    if tip_radius > 0:                          # round the V-tip corners
        cone = cv2.morphologyEx(
            cone, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tip_radius + 1,) * 2))
    cone = cone.astype(bool) & window_inner

    ink_f, spot_f = print_frequency(dev_imgs)
    rows = np.arange(C.IMAGE_H)[:, None]
    # printed scale/label only exist in the upper part of the tube; below print_zone_max_y
    # thin dark structures are liquid content (layer edges, clumps) and are kept.
    printed = ((ink_f > p["print_freq_thr"]) & (rows < p["print_zone_max_y"])) | \
              (spot_f > p["spot_freq_thr"])
    printed = cv2.dilate(printed.astype(np.uint8),
                         np.ones((2 * p["print_dilate_px"] + 1,) * 2, np.uint8)).astype(bool)
    boxes = np.zeros_like(cone)
    for (x0, y0, x1, y1) in p["exclude_boxes"]:
        boxes[y0:y1 + 1, x0:x1 + 1] = True
    tube = cone & ~printed & ~boxes
    # keep the largest connected component (drop isolated slivers)
    n_cc, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(tube.astype(np.uint8), 8)
    if n_cc > 1:
        largest = 1 + int(np.argmax(cc_stats[1:, cv2.CC_STAT_AREA]))
        tube = cc_labels == largest
    # the mask starts at the first row that is at least min_row_width_px wide (removes the
    # thin sliver left between the printed numbers and the printed circle label)
    width = tube.sum(axis=1)
    wide = np.where(width >= p["min_row_width_px"])[0]
    if len(wide):
        tube[:wide.min()] = False

    # reference region: neutral holder / back-light outside the tube, away from the image
    # border and from the coloured light that bleeds sideways out of the tube
    chroma_p90 = np.percentile(np.stack([np.hypot(*np.moveaxis(rgb_to_lab(im)[..., 1:], -1, 0))
                                         for im in dev_imgs]), 90, axis=0)
    tube_zone = cv2.dilate(cone.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
    x_grid = np.arange(C.IMAGE_W)[None, :].repeat(C.IMAGE_H, 0)
    y_grid = np.arange(C.IMAGE_H)[:, None].repeat(C.IMAGE_W, 1)
    ref = (~tube_zone & (x_grid > 24) & (x_grid < C.IMAGE_W - 24)
           & (y_grid > 4) & (y_grid < C.IMAGE_H - 4)
           & (chroma_p90 < p["ref_chroma_p90_max"]))
    diag = {"wall_rows": ys, "wall_left": xl, "wall_right": xr, "coef_left": coef_left,
            "coef_right": coef_right, "tip_auto": tip_auto, "top_y": top_y, "tip_y": tip_y,
            "ink_freq": ink_f, "spot_freq": spot_f, "printed": printed, "cone": cone,
            "window": window, "chroma_p90": chroma_p90}
    return tube, ref, diag


def blank_outside(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Copy of img with every pixel outside the mask set to C.NEUTRAL_GRAY."""
    out = img.copy()
    out[~mask] = C.NEUTRAL_GRAY
    return out


# --- OPTIONAL: mask generalization (disabled; the default run uses ONE fixed mask) ----------
# def auto_detect_tube(img, min_wall_len=120):
#     """Find the tube in EACH image instead of using the fixed mask -- for other containers
#     or framing. Option A (conical tubes): Canny edges of the gray image -> cv2.HoughLinesP;
#     keep the longest near-vertical line on each side whose x decreases/increases downward
#     (the two conical walls), close the polygon at their intersection (the tip) and at the
#     highest row of both lines. Option B (any container): threshold the bright, low-chroma
#     holder, take the largest remaining contour (cv2.findContours) and simplify it with
#     cv2.approxPolyDP; for round-bottom vials use cv2.HoughCircles on the bottom.
#     To enable: return the per-image mask here and pass it instead of tube_mask.npy to
#     preprocess.process_raw() and features_hand.extract_record(); inspect the detected masks
#     on DEV images first (a per-image mask becomes part of the features).
#     """
#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 90)
#     lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
#                             minLineLength=min_wall_len, maxLineGap=10)
#     if lines is None:
#         return None
#     segs = [l[0] for l in lines if abs(l[0][2] - l[0][0]) < abs(l[0][3] - l[0][1]) / 2]
#     left = min(segs, key=lambda s: (s[0] + s[2]) / 2)
#     right = max(segs, key=lambda s: (s[0] + s[2]) / 2)
#     poly = np.array([left[:2], left[2:], right[2:], right[:2]], np.int32)
#     m = np.zeros(gray.shape, np.uint8)
#     cv2.fillPoly(m, [cv2.convexHull(poly)], 1)
#     return m.astype(bool)


# =============================================================================
# Showcase selection
# =============================================================================
def showcase_candidates(labels):
    """k random successes + k random failures (seeded) + CONTACT_SHEET_N other random dev ids."""
    rng = np.random.default_rng(C.SEED)
    succ = sorted(labels.index[labels == 1])
    fail = sorted(labels.index[labels == 0])
    k = C.SHOWCASE_N_RANDOM_PER_CLASS
    auto = sorted(rng.choice(succ, k, replace=False)) + sorted(rng.choice(fail, k, replace=False))
    rest = sorted(set(labels.index) - set(auto))
    sheet = sorted(rng.choice(rest, C.CONTACT_SHEET_N, replace=False))
    return [str(a) for a in auto], [str(s) for s in sheet]


def set_showcase(extra: list[str]):
    """Store auto picks + `extra` (or SHOWCASE_N_USER random ids) as config.SHOWCASE_IDS."""
    labels = common.load_dev_labels()
    cand = common.json_load(SHOWCASE_JSON)
    auto = cand["auto"]
    if extra == ["random"]:
        rng = np.random.default_rng(C.SEED)
        rest = sorted(set(labels.index) - set(auto))
        extra = [str(e) for e in sorted(rng.choice(rest, C.SHOWCASE_N_USER, replace=False))]
        print(f"'random' -> picked {C.SHOWCASE_N_USER} more at random (seed {C.SEED}): {extra}")
    bad = [e for e in extra if e not in labels.index]
    if bad:
        print(f"ABORT: not DEV ids: {bad}")
        sys.exit(2)
    if len(extra) != C.SHOWCASE_N_USER or set(extra) & set(auto):
        print(f"ABORT: need {C.SHOWCASE_N_USER} ids that are not already in {auto}")
        sys.exit(2)
    ids = auto + extra
    common.set_auto_config("SHOWCASE_IDS", ids)
    print("config.SHOWCASE_IDS =", ids)
    for i in ids:
        print(f"   {i}  {V.label_str(labels[i])}")
    V.update_progression(STAGE)


# =============================================================================
# Figures
# =============================================================================
def make_figures(fig_dir, ids, imgs, tube, ref, diag, labels, auto, sheet):
    lab_of = {i: labels[i] for i in ids}
    rng = np.random.default_rng(C.SEED + 2)
    pick = [int(k) for k in sorted(rng.choice(len(ids), 12, replace=False))]
    # No features exist yet at this stage, so the close-up examples are random; the overview
    # sheet (a2) shows ALL dev images.

    # (a) 12 dev images with the mask outline
    items = [V.outline(imgs[k], tube) for k in pick]
    titles = [f"{ids[k]} | {V.label_str(lab_of[ids[k]])}\nmask outline" for k in pick]
    V.save_fig(V.show_grid(items, titles, ncols=6,
                           suptitle="(a) Tube mask outline on 12 random DEV images"),
               fig_dir / "a_mask_outline_12.png")

    # (a2) every dev image, small, with the outline (to check the FIXED mask fits all)
    items = [V.outline(im, tube) for im in imgs]
    V.save_fig(V.show_grid(items, [f"{i}" for i in ids], ncols=14, tile_w=1.0, tile_h=1.55,
                           fontsize=5, suptitle="(a2) Mask outline on ALL DEV images"),
               fig_dir / "a2_mask_outline_all_dev.png")

    # (b) before / after blanking
    items, titles = [], []
    for k in pick:
        items += [imgs[k], blank_outside(imgs[k], tube)]
        titles += [f"{ids[k]} | {V.label_str(lab_of[ids[k]])}\nraw",
                   f"{ids[k]}\nbackground blanked"]
    V.save_fig(V.show_grid(items, titles, ncols=8,
                           suptitle="(b) Before / after: holder, scale numbers, ticks, "
                                    "printed label and side bleed blanked"),
               fig_dir / "b_before_after_blanking.png")

    # (c) the mask alone + how it was built
    med = np.median(np.stack(imgs), axis=0).astype(np.uint8)
    fig, axes = plt.subplots(1, 5, figsize=(17, 6.2))
    axes[0].imshow(tube, cmap="gray")
    axes[0].set_title(f"tube mask ({int(tube.sum())} px)")
    axes[1].imshow(med)
    yy = np.arange(diag["top_y"], diag["tip_y"] + 1)
    axes[1].plot(diag["wall_left"], diag["wall_rows"], ".", ms=1.5, color="cyan")
    axes[1].plot(diag["wall_right"], diag["wall_rows"], ".", ms=1.5, color="cyan")
    axes[1].plot(np.polyval(diag["coef_left"], yy), yy, color="lime", lw=1)
    axes[1].plot(np.polyval(diag["coef_right"], yy), yy, color="lime", lw=1)
    axes[1].contour(tube, levels=[0.5], colors="red", linewidths=0.8)
    axes[1].set_title("dev median: wall medians (cyan),\nquadratic fit (green), mask (red)",
                      fontsize=8)
    im2 = axes[2].imshow(np.maximum(diag["ink_freq"], diag["spot_freq"]), cmap="magma",
                         vmin=0, vmax=0.6)
    axes[2].set_title("print / dust frequency\n(fraction of dev images)", fontsize=8)
    fig.colorbar(im2, ax=axes[2], shrink=0.5)
    axes[3].imshow(med)
    axes[3].imshow(np.ma.masked_where(~diag["printed"], diag["printed"]), cmap="autumn",
                   alpha=0.6)
    axes[3].set_title("cut out: frequent print / dust", fontsize=8)
    axes[4].imshow(med)
    axes[4].imshow(np.ma.masked_where(~ref, ref), cmap="winter", alpha=0.5)
    axes[4].set_title("reference region (neutral holder /\nback-light, for stage 03)",
                      fontsize=8)
    for ax in axes:
        ax.set_xticks(range(0, C.IMAGE_W + 1, 40))
        ax.set_yticks(range(0, C.IMAGE_H + 1, 40))
        ax.tick_params(labelsize=6)
    fig.suptitle("(c) The fixed mask and how it was derived (DEV images only, no labels)")
    fig.tight_layout()
    V.save_fig(fig, fig_dir / "c_mask_alone.png")

    # (d) showcase selection
    items = [common.load_image(i) for i in auto]
    titles = [f"{i} | {V.label_str(labels[i])}\nauto pick" for i in auto]
    V.save_fig(V.show_grid(items, titles, ncols=8,
                           suptitle="(d1) Showcase: 4 random successes + 4 random failures "
                                    f"(DEV, seed {C.SEED})"),
               fig_dir / "d1_showcase_auto8.png")
    items = [common.load_image(i) for i in sheet]
    titles = [f"{i} | {V.label_str(labels[i])}" for i in sheet]
    V.save_fig(V.show_grid(items, titles, ncols=8,
                           suptitle="(d2) Contact sheet: 24 other random DEV images -- name 4 "
                                    "(small-clump failures or borderline) for the showcase"),
               fig_dir / "d2_contact_sheet_24.png")


# =============================================================================
# Stage
# =============================================================================
def build(fig_dir):
    """Build the mask candidate, write masks / metadata / masked images / figures."""
    common.set_auto_config("MASK_APPROVED", False)      # any rebuild needs a new approval
    ids = common.dev_ids()
    imgs = [common.load_image(i) for i in ids]
    sizes = {im.shape for im in imgs}
    assert sizes == {(C.IMAGE_H, C.IMAGE_W, 3)}, f"image sizes differ: {sizes}"
    print(f"all {len(ids)} DEV images are {C.IMAGE_W}x{C.IMAGE_H}")

    tube, ref, diag = build_masks(imgs)
    np.save(C.TUBE_MASK_NPY, tube)
    np.save(C.REF_MASK_NPY, ref)
    rows_with_mask = np.where(tube.any(axis=1))[0]
    meta = {"params": C.MASK_PARAMS, "top_y": diag["top_y"], "tip_y": diag["tip_y"],
            "tip_auto": diag["tip_auto"], "coef_left": diag["coef_left"],
            "coef_right": diag["coef_right"], "tube_pixels": int(tube.sum()),
            "ref_pixels": int(ref.sum()), "mask_rows": [int(rows_with_mask.min()),
                                                        int(rows_with_mask.max())],
            "tube_mask_sha256": common.sha256_file(C.TUBE_MASK_NPY),
            "ref_mask_sha256": common.sha256_file(C.REF_MASK_NPY),
            "built_from": f"{len(ids)} DEV images, no labels"}
    common.json_dump(meta, C.MASK_META_JSON)
    print(f"tube mask: {int(tube.sum())} px, rows {meta['mask_rows'][0]}..{meta['mask_rows'][1]}"
          f" (top_y {diag['top_y']}, tip_y {diag['tip_y']}, automatic tip {diag['tip_auto']})")
    print(f"wall fit (x = a*y^2 + b*y + c): left {np.round(diag['coef_left'], 5).tolist()}, "
          f"right {np.round(diag['coef_right'], 5).tolist()}")
    print(f"pixels cut out as frequent print/dust inside the cone: "
          f"{int((diag['cone'] & diag['printed']).sum())}")
    print(f"reference (neutral holder/back-light) region: {int(ref.sum())} px")

    out = C.PROCESSED_DIR / "masked"
    out.mkdir(parents=True, exist_ok=True)
    for i, im in zip(ids, imgs):
        common.write_rgb(out / f"{i}.png", blank_outside(im, tube))
    print(f"wrote processed/masked/<id>.png for {len(ids)} DEV images")

    labels = common.load_dev_labels()
    auto, sheet = showcase_candidates(labels)
    common.json_dump({"auto": auto, "contact_sheet": sheet}, SHOWCASE_JSON)
    make_figures(fig_dir, ids, imgs, tube, ref, diag, labels, auto, sheet)
    print(f"\nShowcase auto picks (4 successes + 4 failures, seed {C.SEED}):")
    for i in auto:
        print(f"   {i}  {V.label_str(labels[i])}")
    print("Contact sheet ids (figures/02_mask/d2_contact_sheet_24.png):")
    print("   " + " ".join(sheet))
    if C.SHOWCASE_IDS:
        V.update_progression(STAGE)
    print("\nPAUSE: check figures/02_mask/; if needed, adjust config.MASK_PARAMS and rerun "
          "until the mask fits, then approve it and name 4 more showcase ids (or 'random').")
    print("   approve  -> python mask.py --approve")
    print("   showcase -> python mask.py --set-showcase ID ID ID ID   |   --set-showcase random")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--approve", action="store_true", help="approve the current mask")
    ap.add_argument("--set-showcase", nargs="+", metavar="ID",
                    help="4 extra DEV showcase ids, or the word 'random'")
    args = ap.parse_args()
    fig_dir = common.start_stage(STAGE)

    if args.set_showcase:
        common.require_files(SHOWCASE_JSON, C.TUBE_MASK_NPY, hint="run `python mask.py` first")
        set_showcase(args.set_showcase)
    if args.approve:
        common.require_files(C.TUBE_MASK_NPY, C.MASK_META_JSON, hint="run `python mask.py` first")
        meta = common.json_load(C.MASK_META_JSON)
        if common.sha256_file(C.TUBE_MASK_NPY) != meta["tube_mask_sha256"]:
            print("ABORT: tube_mask.npy changed since it was built; rerun `python mask.py`.")
            sys.exit(2)
        meta["approved"] = True
        common.json_dump(meta, C.MASK_META_JSON)
        common.set_auto_config("MASK_APPROVED", True)
        print("mask APPROVED and frozen for all later stages "
              f"(sha256 {meta['tube_mask_sha256'][:16]}...)")
    if not (args.approve or args.set_showcase):
        build(fig_dir)
    if C.MASK_APPROVED and C.SHOWCASE_IDS:
        common.stop_here(STAGE, next_cmd="preprocess.py")
    else:
        common.stop_here(STAGE)


if __name__ == "__main__":
    main()
