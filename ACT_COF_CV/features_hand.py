"""
Stage 04 -- hand-built colour / layering features, computed inside the tube mask only.

Per image (normalized, background blanked):
  * CIELAB (LCh / HSV where a colour range needs it)
  * liquid surface (meniscus) and V-tip -> liquid column. The meniscus is a thin darker line
    (L* dip) with a colour step, in the upper part of the tube; if no such line exists inside
    the mask the liquid reaches above the mask top and the column is truncated there. Every
    such case and every weak detection is FLAGGED and listed (fix with config.SURFACE_OVERRIDES).
  * N_BANDS vertical bands measured from the tip: mean L*, a*, b* per band (and the SD of b*
    per band if config.INCLUDE_HOMOGENEITY_FEATURES)
  * colour fractions for every entry of config.COLOR_RANGES (frac_<name>) and the height of the
    brown region as a fraction of the liquid height (height_brown). Fold-derived ranges (brown)
    are recomputed INSIDE every training fold by train_cv.py, so that no held-out image
    influences its own colour range; here they are only shown with a range derived from all
    DEV successes (columns prefixed viz_ are NEVER model inputs)
  * layer boundary: position (fraction of liquid height from the tip) and strength (Delta-E
    step) of the sharpest colour transition of the vertical profile
  * top band minus bottom band colour difference (dL, da, db)
  * shortcut-check-only features (prefix sc_, NEVER model inputs): fill height / frame height,
    raw brightness, raw white-balance ratios, surface position relative to the printed "2" mark
    (diag_ columns are diagnostics only as well)

Inputs : DEV images, tube_mask.npy / ref_mask.npy, normalization (preprocess.process_raw).
Outputs: features_hand.csv keyed by id (DEV ids only until stage 11),
         cache/color_ranges_alldev.json (display-only brown range), figures/04_hand_features/.
The feature functions (extract_record, liquid_pixels, color_features, derive_color_ranges, ...)
are reused by augment.py, train_cv.py, final_test.py and predict.py.

Usage:  python features_hand.py
"""
from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter1d

import common
import config as C
import preprocess
import visualize as V

STAGE = "04_hand_features"
MENISCUS_SKIP_PX = 2          # rows of the meniscus line itself excluded from the column
ALLDEV_RANGES_JSON = C.CACHE_DIR / "color_ranges_alldev.json"


# =============================================================================
# Colour helpers
# =============================================================================
def to_lab(rgb_u8: np.ndarray) -> np.ndarray:
    """uint8 RGB -> float32 CIELAB (L 0..100)."""
    return cv2.cvtColor(rgb_u8.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def lab_to_space(lab_px: np.ndarray, space: str) -> np.ndarray:
    """(N, 3) LAB pixels -> the same pixels in "LAB", "LCH" (hue in degrees) or "HSV"."""
    if space == "LAB":
        return lab_px
    if space == "LCH":
        L, a, b = lab_px[:, 0], lab_px[:, 1], lab_px[:, 2]
        return np.stack([L, np.hypot(a, b), np.degrees(np.arctan2(b, a)) % 360], axis=1)
    if space == "HSV":   # H in degrees, S and V in 0..1
        rgb = cv2.cvtColor(lab_px.reshape(-1, 1, 3).astype(np.float32), cv2.COLOR_LAB2RGB)
        return cv2.cvtColor(np.clip(rgb, 0, 1), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    raise ValueError(space)


def in_range(lab_px: np.ndarray, rng: dict) -> np.ndarray:
    """Boolean per pixel: inside the colour box (hue channels may wrap through 0)."""
    v = lab_to_space(np.asarray(lab_px, np.float32), rng["space"])
    lo, hi = np.asarray(rng["lo"], float), np.asarray(rng["hi"], float)
    hue_ch = {"LCH": 2, "HSV": 0}.get(rng["space"], -1)
    inside = np.ones(len(v), bool)
    for c in range(3):
        if c == hue_ch and lo[c] > hi[c]:          # hue interval wrapping through 0 deg
            inside &= (v[:, c] >= lo[c]) | (v[:, c] <= hi[c])
        else:
            inside &= (v[:, c] >= lo[c]) & (v[:, c] <= hi[c])
    return inside


# =============================================================================
# Geometry: surface (meniscus) and V-tip
# =============================================================================
def mask_row_range(tube: np.ndarray):
    """First and last mask row that is at least 8 px wide."""
    rows = np.where(tube.sum(axis=1) >= 8)[0]
    return int(rows.min()), int(rows.max())


def central_profile(lab: np.ndarray, tube: np.ndarray, frac: float = C.CENTRAL_STRIP_FRAC):
    """Per-row median L*a*b* over the central part of every mask row (gaps interpolated).

    Only the central strip is used, to avoid wall shading and refraction near the mask edges.
    Returns (profile of shape (IMAGE_H, 3), top row, bottom row); rows outside
    top..bottom stay NaN.
    """
    top, bot = mask_row_range(tube)
    prof = np.full((C.IMAGE_H, 3), np.nan, np.float32)
    for y in range(top, bot + 1):
        xs = np.where(tube[y])[0]
        if len(xs) < 4:
            continue
        lo, hi = xs.min(), xs.max()
        centre, half = (lo + hi) / 2, max(2.0, (hi - lo) * frac / 2)
        sel = xs[(xs >= centre - half) & (xs <= centre + half)]
        if len(sel):
            prof[y] = np.median(lab[y, sel], axis=0)
    ys = np.arange(C.IMAGE_H)
    ok = ~np.isnan(prof[:, 0])
    for ch in range(3):
        prof[top:bot + 1, ch] = np.interp(ys[top:bot + 1], ys[ok], prof[ok, ch])
    return prof, top, bot


def detect_surface(lab: np.ndarray, tube: np.ndarray, override_y=None):
    """Return (surface_y, flag, score). flag: ok | weak | above_mask_top | manual_override.

    A row is a meniscus candidate if it is a thin darker line with a colour step (L* dip >=
    SURFACE_MIN_VALLEY and Delta-E step >= SURFACE_MIN_CONTRAST), or a large SHARP colour
    step (Delta-E >= SURFACE_STRONG_STEP, most of it within +/-2 rows) -- e.g. air directly
    over strongly coloured liquid. Gradual colour ramps inside the liquid are not candidates.
    Among the candidates (local maxima of valley + 0.5 * step) the topmost one scoring at least
    35 % of the best is taken, because a layer boundary lower in the liquid can be stronger
    than the meniscus itself.
    """
    prof, top, bot = central_profile(lab, tube)
    if override_y is not None:
        return int(override_y), "manual_override", np.nan
    P = gaussian_filter1d(prof[top:bot + 1], 0.8, axis=0)     # rows relative to `top`
    L = P[:, 0]
    # search only the upper part of the tube (keep >= 40 rows above the mask bottom)
    j_max = min(C.SURFACE_MAX_Y, bot - 40) - top
    if j_max < 4:
        return top, "above_mask_top", 0.0
    js = np.arange(4, j_max + 1)
    step, valley, sharp = np.zeros(len(js)), np.zeros(len(js)), np.zeros(len(js))
    for k, j in enumerate(js):
        # colour step between the 6 rows above and the 6 rows below (1-row gap each side)
        above = P[max(0, j - 7):j - 1].mean(axis=0)
        below = P[j + 2:j + 8].mean(axis=0)
        step[k] = np.linalg.norm(below - above)
        # L* dip of this row relative to rows 3 above / below (the dark meniscus line)
        valley[k] = max(0.0, (L[j - 3] + L[j + 3]) / 2 - L[j])
        # sharpness: share of the +/-7-row colour change that happens within +/-2 rows
        near = np.linalg.norm(P[min(j + 2, len(P) - 1)] - P[max(j - 2, 0)])
        far = np.linalg.norm(P[min(j + 7, len(P) - 1)] - P[max(j - 7, 0)])
        sharp[k] = near / max(far, 1e-6)
    score = valley + 0.5 * step
    candidates = []
    half_win = 3      # the step, the dark line and the sharpness peak a few rows apart
    for k in range(len(js)):
        if not (score[k] >= score[max(0, k - 1)] and score[k] >= score[min(len(js) - 1, k + 1)]):
            continue                                            # not a local maximum
        lo_k, hi_k = max(0, k - half_win), min(len(js), k + half_win + 1)
        k_step = lo_k + int(np.argmax(step[lo_k:hi_k]))
        line_like = valley[lo_k:hi_k].max() >= C.SURFACE_MIN_VALLEY and \
            step[k_step] >= C.SURFACE_MIN_CONTRAST
        sharp_step = step[k_step] >= C.SURFACE_STRONG_STEP and \
            sharp[k_step] >= C.SURFACE_MIN_SHARPNESS
        if line_like or sharp_step:
            candidates.append((int(js[k]), float(score[k])))
    if not candidates:
        return top, "above_mask_top", 0.0
    best_score = max(s for _, s in candidates)
    j, s = next((j, s) for j, s in candidates if s >= 0.35 * best_score)   # topmost strong one
    return top + j, ("ok" if s >= C.SURFACE_GOOD_SCORE else "weak"), float(s)


def detect_tip(lab: np.ndarray, tube: np.ndarray):
    """Lowest mask row that does not look like the empty white holder (L* > 80, chroma < 6).

    Returns (tip_y, flag); flag is "above_mask_bottom" if the tip is > 6 rows above the mask
    bottom and "uncertain" if nothing is found within 45 rows of it.
    """
    prof, _, bot = central_profile(lab, tube)
    for y in range(bot, bot - 45, -1):
        L, a, b = prof[y]
        if not (L > 80 and np.hypot(a, b) < 6):
            return y, ("ok" if bot - y <= 6 else "above_mask_bottom")
    return bot, "uncertain"


def liquid_pixels(lab: np.ndarray, tube: np.ndarray, surface_y: int, tip_y: int):
    """LAB of every mask pixel in the liquid column, its relative height (0 tip .. 1 surface)
    and its row."""
    sel = tube.copy()
    sel[:surface_y + MENISCUS_SKIP_PX] = False
    sel[tip_y + 1:] = False
    yy, xx = np.nonzero(sel)
    relh = (tip_y - yy) / max(tip_y - surface_y, 1)
    return lab[yy, xx].astype(np.float32), relh.astype(np.float32), yy.astype(np.int32)


# =============================================================================
# Feature groups
# =============================================================================
def fixed_feature_names() -> list[str]:
    """Model features that do not depend on fold-derived colour ranges."""
    names = [f"band{k}_{c}" for k in range(1, C.N_BANDS + 1) for c in ("L", "a", "b")]
    if C.INCLUDE_HOMOGENEITY_FEATURES:
        names += [f"band{k}_bstd" for k in range(1, C.N_BANDS + 1)]
    return names + ["boundary_pos", "boundary_strength", "dL", "da", "db"]


def color_feature_names() -> list[str]:
    """frac_<name> for every config.COLOR_RANGES entry + height_<name> features."""
    return [f"frac_{n}" for n in C.COLOR_RANGES] + [f"height_{n}" for n in C.HEIGHT_FEATURE_COLORS]


def model_feature_names() -> list[str]:
    """All hand features that may be model inputs (colour ones are recomputed per fold)."""
    return fixed_feature_names() + color_feature_names()


SHORTCUT_FEATURES = ["sc_fill_height_frac", "sc_raw_L_frame", "sc_raw_L_background",
                     "sc_raw_rg_background", "sc_raw_bg_background", "sc_surface_rel_tick2"]


def band_index(relh: np.ndarray) -> np.ndarray:
    """Band number 1..N_BANDS of each pixel from its relative height (1 = bottom band)."""
    return np.minimum((relh * C.N_BANDS).astype(int), C.N_BANDS - 1) + 1


def band_features(lab_px, relh) -> dict:
    """Mean L*, a*, b* (and SD of b*) per band, and top-minus-bottom band differences."""
    out = {}
    bidx = band_index(relh)
    for k in range(1, C.N_BANDS + 1):
        sel = bidx == k
        m = lab_px[sel].mean(axis=0) if sel.any() else np.full(3, np.nan)
        out[f"band{k}_L"], out[f"band{k}_a"], out[f"band{k}_b"] = map(float, m)
        if C.INCLUDE_HOMOGENEITY_FEATURES:
            out[f"band{k}_bstd"] = float(lab_px[sel, 2].std()) if sel.any() else np.nan
    for c in ("L", "a", "b"):
        out[f"d{c}"] = out[f"band{C.N_BANDS}_{c}"] - out[f"band1_{c}"]
    return out


def row_mean_profile(lab_px, rows):
    """Mean L*a*b* per image row of the liquid column (empty rows interpolated).

    Returns (profile of shape (n_rows, 3), first row); profile index 0 is the top row.
    """
    r0 = rows.min()
    idx = rows - r0
    n = np.bincount(idx)
    prof = np.stack([np.bincount(idx, weights=lab_px[:, c]) for c in range(3)], 1)
    ok = n > 0
    prof[ok] /= n[ok, None]
    ys = np.arange(len(n))
    for c in range(3):
        prof[:, c] = np.interp(ys, ys[ok], prof[ok, c])
    return prof, r0


def boundary_features(lab_px, rows, surface_y, tip_y) -> dict:
    """Sharpest colour transition of the smoothed vertical profile.

    The ends of the column (config.BOUNDARY_EXCLUDE) are excluded because the meniscus and the
    narrow V-tip produce colour steps that are not layer boundaries.
    """
    prof, r0 = row_mean_profile(lab_px, rows)
    P = gaussian_filter1d(prof, C.PROFILE_SMOOTH_SIGMA, axis=0)
    H = max(tip_y - surface_y, 1)
    half_gap = 3                       # gradient = colour difference between j +/- half_gap
    j_lo = int(C.BOUNDARY_EXCLUDE[1] * H) + half_gap          # surface end (top of profile)
    j_hi = len(P) - 1 - int(C.BOUNDARY_EXCLUDE[0] * H) - half_gap   # tip end
    if j_hi <= j_lo:
        return {"boundary_pos": np.nan, "boundary_strength": 0.0, "boundary_y": np.nan}
    gradient = np.array([np.linalg.norm(P[j + half_gap] - P[j - half_gap])
                         for j in range(j_lo, j_hi + 1)])
    j = j_lo + int(np.argmax(gradient))
    above = P[max(0, j - 10):max(1, j - 2)].mean(axis=0)
    below = P[j + 2:j + 10].mean(axis=0) if j + 2 < len(P) else P[-1]
    y = r0 + j
    return {"boundary_pos": float((tip_y - y) / H),
            "boundary_strength": float(np.linalg.norm(below - above)),
            "boundary_y": int(y)}


def region_height(inside: np.ndarray, rows: np.ndarray, relh: np.ndarray) -> float:
    """Height of the top of the colour region above the tip, as a fraction of the liquid
    height: the highest row in which (smoothed) at least half of the liquid pixels are in the
    colour range; 0 if there is no such row. (Measuring the top, not a chain grown from the
    tip, keeps very dark precipitate or a small gap from cutting the region short.)"""
    if len(rows) == 0 or not inside.any():
        return 0.0
    r0 = rows.min()
    idx = rows - r0
    n = np.bincount(idx)
    n_inside = np.bincount(idx, weights=inside.astype(float))
    ok = n > 0
    ys = np.arange(len(n))
    frac = gaussian_filter1d(np.interp(ys, ys[ok], n_inside[ok] / n[ok]), 2.0)
    u, first = np.unique(idx, return_index=True)
    rel_of_row = np.interp(ys, u, relh[first].astype(float))   # relative height of every row
    on = np.flatnonzero(frac >= 0.5)
    return float(rel_of_row[on.min()]) if len(on) else 0.0     # smallest index = highest row


def color_features(lab_px, relh, rows, ranges: dict) -> dict:
    """frac_<name> for every resolved range; height_<name> for HEIGHT_FEATURE_COLORS."""
    out = {}
    for name, rng in ranges.items():
        inside = in_range(lab_px, rng) if len(lab_px) else np.zeros(0, bool)
        out[f"frac_{name}"] = float(inside.mean()) if len(inside) else 0.0
        if name in C.HEIGHT_FEATURE_COLORS:
            out[f"height_{name}"] = region_height(inside, rows, relh)
    return out


def derive_color_ranges(keys_by_label: dict, px: dict, tag: str = "") -> dict:
    """Resolve config.COLOR_RANGES. Fold-derived entries use LAB percentiles of the chosen band
    of the TRAINING images with the chosen label (equal pixel budget per image, seeded).

    keys_by_label: {label: [(id, aug), ...]}; px: {(id, aug): (lab_px, relh, rows)}.
    """
    out = {}
    for name, spec in C.COLOR_RANGES.items():
        if spec.get("derive") != "fold":
            out[name] = {k: spec[k] for k in ("space", "lo", "hi")}
            continue
        pooled = []
        keys = keys_by_label.get(spec["source_label"], [])
        for key in keys:
            lab_px, relh, _ = px[key]
            sel = band_index(relh) == spec["source_band"]
            p = lab_px[sel]
            # at most 1500 pixels per image, so images with many band pixels do not dominate
            if len(p) > 1500:
                p = p[common.rng_for("range", tag, *key).choice(len(p), 1500, replace=False)]
            pooled.append(p)
        # float64: the CV cache stores pixels as float16, where rounding would overflow
        pooled = np.concatenate(pooled).astype(np.float64) if pooled else np.zeros((0, 3))
        lo_p, hi_p = spec["percentiles"]
        if len(pooled) < 50:
            raise RuntimeError(f"not enough pixels to derive colour range {name!r}")
        out[name] = {"space": "LAB", "lo": np.percentile(pooled, lo_p, axis=0).round(3).tolist(),
                     "hi": np.percentile(pooled, hi_p, axis=0).round(3).tolist(),
                     "derived_from": f"band {spec['source_band']} of {len(keys)} training "
                                     f"images with label {spec['source_label']}"}
    return out


# =============================================================================
# Shortcut-only: position of the printed "2" mark (if detectable)
# =============================================================================
def detect_tick2_y(raw: np.ndarray) -> float:
    """Centre row of the printed '2' numeral on the left of the tube, or NaN.

    Used only for the shortcut check: the surface row relative to this printed mark
    (sc_surface_rel_tick2) approximates the fill volume on the tube's own scale.
    """
    L = to_lab(raw)[..., 0]
    blackhat = cv2.morphologyEx(L, cv2.MORPH_BLACKHAT,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    y0, y1, x0, x1 = 60, 145, 38, 92          # search window where the numeral appears
    roi = (blackhat[y0:y1, x0:x1] > 10).astype(np.uint8)
    n_cc, _, cc_stats, centroids = cv2.connectedComponentsWithStats(roi, 8)
    best, best_area = np.nan, 0
    for k in range(1, n_cc):
        w, h = cc_stats[k, cv2.CC_STAT_WIDTH], cc_stats[k, cv2.CC_STAT_HEIGHT]
        area = cc_stats[k, cv2.CC_STAT_AREA]
        # largest dark blob with the size of a printed numeral
        if 30 <= area <= 400 and 10 <= h <= 34 and 5 <= w <= 26 and area > best_area:
            best, best_area = y0 + centroids[k][1], area
    return float(best)


# =============================================================================
# One image -> one record
# =============================================================================
def extract_record(norm_blanked: np.ndarray, tube: np.ndarray, raw: np.ndarray,
                   raw_stats: dict, img_id: str, aug: int = 0):
    """Return (record dict, pixel arrays (lab, relh, rows)) for one (possibly augmented) view.

    Manual surface overrides apply to the original view (aug == 0) only. Fold-derived colour
    ranges are NOT applied here (see train_cv.py).
    """
    lab = to_lab(norm_blanked)
    override = C.SURFACE_OVERRIDES.get(img_id) if aug == 0 else None
    surface_y, surface_flag, surface_score = detect_surface(lab, tube, override)
    tip_y, tip_flag = detect_tip(lab, tube)
    lab_px, relh, rows = liquid_pixels(lab, tube, surface_y, tip_y)
    rec = {"id": img_id, "aug": aug, "surface_y": surface_y, "surface_flag": surface_flag,
           "surface_score": surface_score, "tip_y": tip_y, "tip_flag": tip_flag,
           "n_liquid_px": int(len(lab_px))}
    rec.update(band_features(lab_px, relh))
    rec.update(boundary_features(lab_px, rows, surface_y, tip_y))
    fixed = {n: spec for n, spec in C.COLOR_RANGES.items() if spec.get("derive") != "fold"}
    rec.update(color_features(lab_px, relh, rows,
                              {n: {k: s[k] for k in ("space", "lo", "hi")}
                               for n, s in fixed.items()}))
    tick = detect_tick2_y(raw)
    rec.update({"sc_fill_height_frac": (tip_y - surface_y) / C.IMAGE_H,
                "sc_raw_L_frame": raw_stats["raw_L_frame"],
                "sc_raw_L_background": raw_stats["raw_L_background"],
                "sc_raw_rg_background": raw_stats["raw_rg_background"],
                "sc_raw_bg_background": raw_stats["raw_bg_background"],
                "sc_surface_rel_tick2": (tick - surface_y) if np.isfinite(tick) else np.nan,
                "diag_tick2_y": tick, "diag_exposure_gain": raw_stats["exposure_gain"],
                "diag_clipped_frac_mask": raw_stats["clipped_frac_mask"]})
    return rec, (lab_px, relh, rows)


# =============================================================================
# OPTIONAL EXTENSIONS -- disabled (commented out). The default run does not use them.
# =============================================================================
# def segment_phases(image, mask, surface_y, tip_y, max_phases=4, min_frac=0.05):
#     """Automatic phase detection along the vertical axis, for systems with an unknown number
#     of layers. Builds the row-mean L*a*b* profile of the liquid column and either finds
#     change-points (pip install ruptures; ruptures.Pelt(model="rbf").fit(P).predict(pen=10))
#     or clusters the rows with k-means in LAB (k = 1..max_phases, pick k by silhouette score,
#     then merge runs shorter than min_frac of the column).
#     Returns the phases from the tip upward:
#         [{"bottom_y": ..., "top_y": ..., "height_frac": ..., "mean_lab": (L, a, b)}, ...]
#     To enable: uncomment, call it in extract_record(), and add the features (n_phases,
#     boundary heights, per-phase mean colour, per-phase volume fraction -- padded to
#     max_phases so that every image has the same columns) to fixed_feature_names().
#     """
#     from sklearn.cluster import KMeans
#     from sklearn.metrics import silhouette_score
#     lab_px, relh, rows = liquid_pixels(to_lab(image), mask, surface_y, tip_y)
#     prof, r0 = row_mean_profile(lab_px, rows)
#     P = gaussian_filter1d(prof, 2.0, axis=0)
#     best = (np.zeros(len(P), int), -1.0)
#     for k in range(2, max_phases + 1):
#         lab_k = KMeans(k, n_init=10, random_state=C.SEED).fit_predict(P)
#         s = silhouette_score(P, lab_k)
#         if s > best[1]:
#             best = (lab_k, s)
#     labels_k = best[0]
#     cuts = [0] + [j for j in range(1, len(P)) if labels_k[j] != labels_k[j - 1]] + [len(P)]
#     phases = []
#     for a, b in zip(cuts[:-1], cuts[1:]):
#         if (b - a) / len(P) >= min_frac:
#             phases.append({"bottom_y": r0 + b - 1, "top_y": r0 + a,
#                            "height_frac": (b - a) / len(P),
#                            "mean_lab": tuple(P[a:b].mean(axis=0).round(2))})
#     return phases[::-1]
#
# def turbidity_features(image, mask, surface_y, tip_y):
#     """Solubility / clarity cues for systems whose label depends on whether a solid
#     dissolved: per band, the local variance of L* (7x7 window), the Canny edge density and
#     the fraction of small dark / bright specks (black-hat / top-hat) inside the liquid.
#     To enable: uncomment, call it in extract_record() and append the names to
#     fixed_feature_names().
#     """
#     L = to_lab(image)[..., 0]
#     mean = cv2.blur(L, (7, 7))
#     local_var = cv2.blur(L * L, (7, 7)) - mean * mean
#     edges = cv2.Canny(np.clip(L * 2.55, 0, 255).astype(np.uint8), 30, 90) > 0
#     k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     specks = (cv2.morphologyEx(L, cv2.MORPH_BLACKHAT, k) > 6) | \
#              (cv2.morphologyEx(L, cv2.MORPH_TOPHAT, k) > 6)
#     _, relh, rows = liquid_pixels(to_lab(image), mask, surface_y, tip_y)
#     sel = mask.copy(); sel[:surface_y + MENISCUS_SKIP_PX] = False; sel[tip_y + 1:] = False
#     yy, xx = np.nonzero(sel)
#     out = {}
#     for k_band in range(1, C.N_BANDS + 1):
#         b = band_index(relh) == k_band
#         out[f"band{k_band}_localvar"] = float(local_var[yy[b], xx[b]].mean())
#         out[f"band{k_band}_edges"] = float(edges[yy[b], xx[b]].mean())
#         out[f"band{k_band}_specks"] = float(specks[yy[b], xx[b]].mean())
#     return out


# =============================================================================
# Drawing (used by this stage and by the progression figure)
# =============================================================================
def render_overlay(norm_blanked, tube, rec, brown_range) -> np.ndarray:
    """Normalized image with brown pixels (red), band limits, surface, tip and boundary."""
    lab = to_lab(norm_blanked)
    s, t = int(rec["surface_y"]), int(rec["tip_y"])
    lab_px, _, _ = liquid_pixels(lab, tube, s, t)
    # same selection as liquid_pixels(), to map its pixels back to image coordinates
    sel = tube.copy()
    sel[:s + MENISCUS_SKIP_PX] = False
    sel[t + 1:] = False
    brown = np.zeros_like(tube)
    yy, xx = np.nonzero(sel)
    brown[yy, xx] = in_range(lab_px, brown_range)
    img = V.overlay(norm_blanked, brown, (255, 0, 0), 0.45)
    H = t - s

    def hline(y, color, th=1, dashed=False):
        """Horizontal line across the mask width (+4 px) at row y."""
        xs = np.where(tube[int(np.clip(y, 0, C.IMAGE_H - 1))])[0]
        x0, x1 = (int(xs.min()) - 4, int(xs.max()) + 4) if len(xs) else (60, 150)
        if dashed:
            for xa in range(x0, x1, 6):
                cv2.line(img, (xa, int(y)), (min(xa + 3, x1), int(y)), color, th)
        else:
            cv2.line(img, (x0, int(y)), (x1, int(y)), color, th)

    for k in range(1, C.N_BANDS):
        hline(t - k * H / C.N_BANDS, (255, 255, 255), 1, dashed=True)
    hline(s, (0, 200, 255), 1)
    hline(t, (255, 0, 255), 1)
    if np.isfinite(rec.get("boundary_y", np.nan)):
        hline(rec["boundary_y"], (0, 255, 0), 2)
    return img


def render_overlay_for_id(img_id: str):
    """render_overlay() from the stage outputs on disk, or None if they do not exist yet."""
    df = common.read_csv(C.FEATURES_HAND_CSV)
    row = df[df["id"] == img_id]
    path = C.PROCESSED_DIR / "normalized" / f"{img_id}.png"
    if row.empty or not path.exists() or not ALLDEV_RANGES_JSON.exists():
        return None
    tube = np.load(C.TUBE_MASK_NPY)
    rng = common.json_load(ALLDEV_RANGES_JSON)["brown"]
    return render_overlay(common.read_rgb(path), tube, row.iloc[0].to_dict(), rng)


# =============================================================================
# Stage
# =============================================================================
def make_figures(fig_dir, df, px, norms, tube, labels, alldev_ranges):
    show = list(C.SHOWCASE_IDS)
    brown = alldev_ranges["brown"]
    # (1) showcase overlays + vertical L/a/b profiles
    fig, axes = plt.subplots(4, 6, figsize=(15, 17), gridspec_kw={"width_ratios": [1, 1.1] * 3})
    for n, i in enumerate(show):
        r, c = divmod(n, 3)
        rec = df[df["id"] == i].iloc[0]
        ax = axes[r, 2 * c]
        ax.imshow(render_overlay(norms[i], tube, rec, brown))
        ax.set_title(f"{i} | {V.label_str(labels[i])}\nsurface {rec['surface_flag']}, "
                     f"brown h={rec['viz_height_brown']:.2f}", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        ax = axes[r, 2 * c + 1]
        lab_px, _, rows = px[(i, 0)]
        prof, r0 = row_mean_profile(lab_px, rows)
        y = np.arange(len(prof)) + r0
        h = (rec["tip_y"] - y) / max(rec["tip_y"] - rec["surface_y"], 1)
        for ch, col, nm in [(0, "k", "L*"), (1, "m", "a*"), (2, "y", "b*")]:
            ax.plot(gaussian_filter1d(prof[:, ch], C.PROFILE_SMOOTH_SIGMA), h, color=col,
                    label=nm, lw=1)
        for k in range(1, C.N_BANDS):
            ax.axhline(k / C.N_BANDS, color="gray", ls=":", lw=0.8)
        if np.isfinite(rec["boundary_pos"]):
            ax.axhline(rec["boundary_pos"], color="g", lw=1.5, label="boundary")
        ax.set_ylim(0, 1)
        ax.set_ylabel("height from tip (fraction)", fontsize=6)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=5, loc="upper right")
        ax.set_title("vertical profile", fontsize=7)
    fig.suptitle("Hand features on the 12 showcase images: bands from the tip (dashed), surface "
                 "(cyan), tip (magenta), brown pixels (red, all-DEV range -- display only), layer "
                 "boundary (green)", fontsize=9)
    fig.tight_layout()
    V.save_fig(fig, fig_dir / "showcase_bands_profiles.png")

    # (2) surface / tip detection QA on ALL dev images
    ids = list(df["id"])
    items, titles, borders = [], [], []
    for i in ids:
        rec = df[df["id"] == i].iloc[0]
        img = norms[i].copy()
        cv2.line(img, (40, int(rec["surface_y"])), (170, int(rec["surface_y"])), (0, 200, 255), 2)
        cv2.line(img, (60, int(rec["tip_y"])), (150, int(rec["tip_y"])), (255, 0, 255), 2)
        items.append(img)
        titles.append(f"{i}\n{rec['surface_flag']}")
        borders.append(None if rec["surface_flag"] == "ok" and rec["tip_flag"] == "ok" else "red")
    V.save_fig(V.show_grid(items, titles, ncols=14, tile_w=1.0, tile_h=1.75, fontsize=5,
                           borders=borders,
                           suptitle="Surface (cyan) and tip (magenta) on ALL DEV images -- red "
                                    "border = flagged (above mask top / weak / tip issue)"),
               fig_dir / "surface_tip_detection_all_dev.png")

    # (3) colour-range check: hue/chroma of all DEV liquid pixels (label-free) + the ranges
    allpx = np.concatenate([px[(i, 0)][0][::4] for i in ids])
    lch = lab_to_space(allpx, "LCH")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist2d(lch[:, 2], lch[:, 1], bins=[180, 60], range=[[0, 360], [0, 90]], cmap="Greys",
              norm=LogNorm())
    for name, rng in alldev_ranges.items():
        if rng["space"] == "LCH":
            ax.add_patch(plt.Rectangle((rng["lo"][2], rng["lo"][1]), rng["hi"][2] - rng["lo"][2],
                                       min(rng["hi"][1], 90) - rng["lo"][1], fill=False,
                                       ec={"yellow": "gold", "green": "green"}.get(name, "blue"),
                                       lw=2, label=f"{name} (fixed)"))
    hue_in_brown = lch[in_range(allpx, alldev_ranges["brown"]), 2]
    if len(hue_in_brown):
        ax.axvspan(*np.percentile(hue_in_brown, [2, 98]), color="red", alpha=0.15,
                   label="hue of pixels in the brown LAB box (all-DEV, display)")
    ax.set_xlim(0, 200)
    ax.set_xlabel("hue angle (deg)")
    ax.set_ylabel("chroma")
    ax.legend(fontsize=7)
    ax.set_title("All DEV liquid pixels (label-free) and the colour ranges", fontsize=9)
    V.save_fig(fig, fig_dir / "color_ranges_check.png")

    # (4) per-class histograms of the model features (brown ones with the all-DEV range)
    feats = fixed_feature_names() + [f"viz_{n}" if n.endswith("brown") else n
                                     for n in color_feature_names()]
    y = df["id"].map(labels).values
    ncol = 6
    nrow = int(np.ceil(len(feats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.0 * nrow), squeeze=False)
    for ax, f in zip(axes.ravel(), feats):
        v = df[f].values.astype(float)
        bins = np.linspace(np.nanmin(v), np.nanmax(v) + 1e-9, 20)
        ax.hist(v[y == 0], bins, alpha=0.55, color="#4c72b0", label="fail")
        ax.hist(v[y == 1], bins, alpha=0.55, color="#dd8452", label="success")
        ax.set_title(f, fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in axes.ravel()[len(feats):]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=6)
    fig.suptitle("Per-class feature histograms (DEV only; viz_ = brown range from all DEV "
                 "successes, display only)", fontsize=9)
    fig.tight_layout()
    V.save_fig(fig, fig_dir / "feature_histograms_by_class.png")

    # (5) correlation heatmap (model features + shortcut features)
    cols = feats + SHORTCUT_FEATURES
    corr = df[cols].astype(float).corr()
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=6)
    ax.set_yticks(range(len(cols)), cols, fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title("Feature correlation (DEV only)", fontsize=10)
    V.save_fig(fig, fig_dir / "feature_correlation.png")


def main():
    fig_dir = common.start_stage(STAGE)
    common.require_mask_approved()
    tube, ref = common.load_masks()
    labels = common.load_dev_labels()
    ids = common.dev_ids()

    records, px, norms = [], {}, {}
    for i in ids:
        raw = common.load_image(i)
        _, norm, _, raw_stats = preprocess.process_raw(raw, tube, ref)
        rec, arrays = extract_record(norm, tube, raw, raw_stats, i, 0)
        records.append(rec)
        px[(i, 0)] = arrays
        norms[i] = norm
    df = pd.DataFrame(records)

    # brown range from ALL dev successes -> display only (viz_ columns, figures, progression)
    succ = [(i, 0) for i in ids if labels[i] == 1]
    alldev = derive_color_ranges({1: succ}, px, tag="alldev")
    C.CACHE_DIR.mkdir(exist_ok=True)
    common.json_dump(alldev, ALLDEV_RANGES_JSON)
    for k, i in enumerate(ids):
        lab_px, relh, rows = px[(i, 0)]
        cf = color_features(lab_px, relh, rows, {"brown": alldev["brown"]})
        df.loc[k, "viz_frac_brown"] = cf["frac_brown"]
        df.loc[k, "viz_height_brown"] = cf["height_brown"]
    df.to_csv(C.FEATURES_HAND_CSV, index=False)
    print(f"wrote features_hand.csv: {len(df)} DEV images x {df.shape[1]} columns")
    print(f"model features ({len(model_feature_names())}): {model_feature_names()}")
    print("  (frac_brown / height_brown are recomputed inside every training fold in stage 07)")
    print(f"shortcut-only features: {SHORTCUT_FEATURES}")
    print(f"brown range from all DEV successes (display only): lo {alldev['brown']['lo']} "
          f"hi {alldev['brown']['hi']}")

    print("\nsurface detection:", df["surface_flag"].value_counts().to_dict())
    print("tip detection:    ", df["tip_flag"].value_counts().to_dict())
    flagged = df[(df["surface_flag"] != "ok") | (df["tip_flag"] != "ok")]
    if len(flagged):
        print(f"FLAGGED images ({len(flagged)}) -- please look at "
              "figures/04_hand_features/surface_tip_detection_all_dev.png (red borders):")
        for _, r in flagged.iterrows():
            print(f"   {r['id']}  surface={r['surface_flag']:15s} y={int(r['surface_y'])}  "
                  f"tip={r['tip_flag']} y={int(r['tip_y'])}")
        print("   'above_mask_top' = no meniscus inside the mask: the liquid reaches above the "
              "mask top, so the column is truncated at the mask top (not guessed).")
    tick_ok = np.isfinite(df["sc_surface_rel_tick2"]).mean()
    print(f"printed '2' mark detected in {tick_ok:.0%} of DEV images (shortcut check only)")

    make_figures(fig_dir, df, px, norms, tube, labels, alldev)
    V.update_progression(STAGE)
    common.stop_here(STAGE, next_cmd="features_clip.py")


if __name__ == "__main__":
    main()
