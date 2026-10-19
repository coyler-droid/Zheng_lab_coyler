"""
Stage 03 -- per-image normalization (nothing is fitted across images).

  1. Gray-world white balance: per-channel gains that make the mean of the WB reference region
     neutral gray            (config.WB_REFERENCE = "frame" | "background" | "mask")
  2. Exposure: one gain that brings the mean luminance of the exposure reference region to the
     fixed target L*         (config.EXPOSURE_REFERENCE, config.EXPOSURE_TARGET_L)
Both are applied in LINEAR RGB to the whole photo (gains are physically meaningful only on
linear intensities); then everything outside the tube mask is blanked to neutral gray. The raw
statistics (brightness, white-balance ratios) are kept for the shortcut check of stage 08 --
they are never model inputs.

Inputs : DEV images, tube_mask.npy and ref_mask.npy (stage 02).
Outputs: processed/normalized/<id>.png (DEV images only before stage 11),
         cache/normalization_options.json, figures/03_normalization/*.png (before/after grid,
         per-image brightness histograms before vs after, comparison of the reference-region
         options).
process_raw() is also imported by the later stages to normalize augmented and test images.

Usage:  python preprocess.py
"""
from __future__ import annotations

import itertools

import cv2
import matplotlib.pyplot as plt
import numpy as np

import common
import config as C
import visualize as V

STAGE = "03_normalization"
RGB_TO_Y = np.array([0.2126, 0.7152, 0.0722], np.float32)    # linear sRGB -> luminance Y


# =============================================================================
# Colour maths
# =============================================================================
def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """sRGB transfer function, inverse (values 0..1 -> linear 0..1, float32)."""
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """sRGB transfer function (linear values are clipped to 0..1 first, float32)."""
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(
        np.float32)


def L_to_Y(L: float) -> float:
    """CIE L* -> relative luminance Y (white = 1)."""
    fy = (L + 16.0) / 116.0
    return fy ** 3 if fy > 6 / 29 else 3 * (6 / 29) ** 2 * (fy - 4 / 29)


def mean_L(rgb_u8: np.ndarray, region: np.ndarray) -> float:
    """Mean CIE L* of a uint8 RGB image over a boolean region."""
    lab = cv2.cvtColor(rgb_u8.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    return float(lab[..., 0][region].mean())


def _region(name: str, tube: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Boolean pixel region for a reference-region name ("frame" | "background" | "mask")."""
    if name == "frame":
        return np.ones_like(tube)
    if name == "background":
        return ref
    if name == "mask":
        return tube
    raise ValueError(f"unknown reference region {name!r}")


# =============================================================================
# Normalization
# =============================================================================
def normalize(raw: np.ndarray, tube: np.ndarray, ref: np.ndarray,
              wb_ref: str | None = None, exp_ref: str | None = None):
    """Return (normalized full-frame RGB uint8, raw statistics dict).

    wb_ref / exp_ref default to config.WB_REFERENCE / config.EXPOSURE_REFERENCE.
    """
    wb_ref = wb_ref or C.WB_REFERENCE
    exp_ref = exp_ref or C.EXPOSURE_REFERENCE
    lin = srgb_to_linear(raw.astype(np.float32) / 255.0)
    # ignore clipped / nearly black pixels when estimating statistics
    usable = (lin.max(axis=2) < 0.98) & (lin @ RGB_TO_Y > 0.005)

    # gray-world white balance: scale each channel to the mean of the three channel means
    wb_pix = lin[_region(wb_ref, tube, ref) & usable]
    channel_mean = wb_pix.mean(axis=0)
    gains = (channel_mean.mean() / np.maximum(channel_mean, 1e-6)).astype(np.float32)
    lin_wb = lin * gains

    # exposure: one gain so that the reference region reaches the target luminance
    exp_pix = lin_wb[_region(exp_ref, tube, ref) & usable]
    y_mean = float((exp_pix @ RGB_TO_Y).mean())
    exposure_gain = L_to_Y(C.EXPOSURE_TARGET_L[exp_ref]) / max(y_mean, 1e-6)
    out_lin = lin_wb * exposure_gain
    out = (linear_to_srgb(out_lin) * 255.0 + 0.5).astype(np.uint8)

    bg = lin[ref & usable].mean(axis=0)            # raw colour cast of the neutral holder
    stats = {
        "raw_L_frame": mean_L(raw, np.ones_like(tube)),
        "raw_L_background": mean_L(raw, ref),
        "raw_L_mask": mean_L(raw, tube),
        "raw_rg_background": float(bg[0] / bg[1]),
        "raw_bg_background": float(bg[2] / bg[1]),
        "wb_gain_r": float(gains[0]), "wb_gain_g": float(gains[1]), "wb_gain_b": float(gains[2]),
        "exposure_gain": float(exposure_gain),
        "clipped_frac_mask": float((out_lin[tube].max(axis=1) >= 1.0).mean()),
    }
    return out, stats


def blank_outside(img: np.ndarray, tube: np.ndarray) -> np.ndarray:
    """Copy of img with every pixel outside the tube mask set to C.NEUTRAL_GRAY."""
    out = img.copy()
    out[~tube] = C.NEUTRAL_GRAY
    return out


def process_raw(raw: np.ndarray, tube: np.ndarray, ref: np.ndarray):
    """raw RGB -> (masked raw, normalized+blanked, normalized full frame, raw stats)."""
    norm_full, stats = normalize(raw, tube, ref)
    return blank_outside(raw, tube), blank_outside(norm_full, tube), norm_full, stats


# =============================================================================
# Stage figures
# =============================================================================
def option_diagnostics(raws, tube, ref):
    """Label-free numbers for each (WB reference, exposure reference) option."""
    rows = []
    for wb, ex in itertools.product(["frame", "background", "mask"], ["mask", "background"]):
        bg_L, bg_chroma, liquid_b, liquid_hue = [], [], [], []
        for raw in raws:
            out, _ = normalize(raw, tube, ref, wb, ex)
            lab = cv2.cvtColor(out.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
            bg_L.append(lab[..., 0][ref].mean())
            bg_chroma.append(np.hypot(lab[..., 1][ref].mean(), lab[..., 2][ref].mean()))
            liquid_b.append(lab[..., 2][tube].mean())
            liquid_hue.append(np.degrees(np.arctan2(lab[..., 2][tube].mean(),
                                                    lab[..., 1][tube].mean())))
        rows.append({"wb_reference": wb, "exposure_reference": ex,
                     "sd_background_L": float(np.std(bg_L)),
                     "mean_background_chroma": float(np.mean(bg_chroma)),
                     "sd_liquid_mean_b": float(np.std(liquid_b)),
                     "mean_liquid_b": float(np.mean(liquid_b)),
                     "sd_liquid_hue": float(np.std(liquid_hue))})
    return rows


def make_figures(fig_dir, ids, raws, norms_full, tube, ref, stats, labels):
    show = list(C.SHOWCASE_IDS)
    idx = {i: k for k, i in enumerate(ids)}
    # (1) before / after grid
    items, titles = [], []
    for i in show:
        k = idx[i]
        items += [raws[k], norms_full[k], blank_outside(norms_full[k], tube)]
        titles += [f"{i} | {V.label_str(labels[i])}\nraw", f"{i}\nwhite-balanced + exposure",
                   f"{i}\nnormalized, blanked"]
    V.save_fig(V.show_grid(items, titles, ncols=9, tile_w=1.5, tile_h=2.45,
                           suptitle=f"Normalization before / after  (WB ref = {C.WB_REFERENCE}, "
                                    f"exposure ref = {C.EXPOSURE_REFERENCE}, target L* "
                                    f"{C.EXPOSURE_TARGET_L[C.EXPOSURE_REFERENCE]})"),
               fig_dir / "before_after_grid.png")

    # (2) brightness histograms before / after (mask region and neutral background region)
    after_mask = [mean_L(n, tube) for n in norms_full]
    after_bg = [mean_L(n, ref) for n in norms_full]
    before_mask = [s["raw_L_mask"] for s in stats]
    before_bg = [s["raw_L_background"] for s in stats]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, before, after, name in [
            (axes[0], before_bg, after_bg, "neutral holder / back-light"),
            (axes[1], before_mask, after_mask, "inside the tube mask")]:
        bins = np.linspace(min(before + after) - 2, max(before + after) + 2, 30)
        ax.hist(before, bins, alpha=0.6, label=f"before (sd {np.std(before):.1f})",
                color="#888888")
        ax.hist(after, bins, alpha=0.6, label=f"after (sd {np.std(after):.1f})",
                color="#1f77b4")
        ax.set_xlabel("per-image mean L*")
        ax.set_ylabel("DEV images")
        ax.set_title(name, fontsize=9)
        ax.legend(fontsize=8)
    fig.suptitle("Lighting variation before vs after normalization (DEV images)", fontsize=10)
    fig.tight_layout()
    V.save_fig(fig, fig_dir / "brightness_histograms.png")

    # (3) option comparison on 6 showcase images
    opts = [("frame", "mask"), ("background", "background"), ("background", "mask"),
            ("mask", "mask")]
    items, titles = [], []
    for i in show[:6]:
        k = idx[i]
        items.append(raws[k])
        titles.append(f"{i} | {V.label_str(labels[i])}\nraw")
        for wb, ex in opts:
            out, _ = normalize(raws[k], tube, ref, wb, ex)
            items.append(blank_outside(out, tube))
            titles.append(f"WB:{wb}\nexposure:{ex}")
    V.save_fig(V.show_grid(items, titles, ncols=len(opts) + 1, tile_w=1.6, tile_h=2.55,
                           suptitle="Reference-region options (columns) -- which pixels the "
                                    "gray-world WB and the exposure are estimated on"),
               fig_dir / "option_comparison.png")


# =============================================================================
# Stage
# =============================================================================
def main():
    fig_dir = common.start_stage(STAGE)
    common.require_mask_approved()
    tube, ref = common.load_masks()
    labels = common.load_dev_labels()
    ids = common.dev_ids()
    print(f"WB_REFERENCE = {C.WB_REFERENCE} | EXPOSURE_REFERENCE = {C.EXPOSURE_REFERENCE} | "
          f"target L* = {C.EXPOSURE_TARGET_L[C.EXPOSURE_REFERENCE]}")

    raws, norms_full, stats = [], [], []
    out_dir = C.PROCESSED_DIR / "normalized"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in ids:
        raw = common.load_image(i)
        _, norm, full, raw_stats = process_raw(raw, tube, ref)
        common.write_rgb(out_dir / f"{i}.png", norm)
        raws.append(raw)
        norms_full.append(full)
        stats.append(raw_stats)
    print(f"wrote processed/normalized/<id>.png for {len(ids)} DEV images")

    bg_before = np.std([s["raw_L_background"] for s in stats])
    bg_after = np.std([mean_L(n, ref) for n in norms_full])
    mask_before = np.std([s["raw_L_mask"] for s in stats])
    mask_after = np.std([mean_L(n, tube) for n in norms_full])
    clipped = np.array([s["clipped_frac_mask"] for s in stats])
    print(f"sd of per-image mean L*  background: {bg_before:.2f} -> {bg_after:.2f} | "
          f"tube mask: {mask_before:.2f} -> {mask_after:.2f}")
    print(f"images with > 1 % clipped tube pixels after normalization: {(clipped > 0.01).sum()}")

    print("\nlabel-free diagnostics of the reference-region options (all DEV images):")
    print("  sd_background_L: lighting variation left (lower = better)")
    print("  mean_background_chroma: colour cast left on the neutral holder (lower = better)")
    print("  sd_liquid_mean_b / sd_liquid_hue: spread of liquid colour between tubes "
          "(the chemistry signal; it collapses if the liquid itself is neutralized)")
    diag = option_diagnostics(raws, tube, ref)
    hdr = f"  {'WB ref':11s} {'exposure':11s} {'sd bg L':>8s} {'bg chroma':>10s} " \
          f"{'sd liq b':>9s} {'mean liq b':>11s} {'sd liq hue':>11s}"
    print(hdr)
    for d in diag:
        print(f"  {d['wb_reference']:11s} {d['exposure_reference']:11s} "
              f"{d['sd_background_L']:8.2f} {d['mean_background_chroma']:10.2f} "
              f"{d['sd_liquid_mean_b']:9.2f} {d['mean_liquid_b']:11.2f} "
              f"{d['sd_liquid_hue']:11.2f}")
    C.CACHE_DIR.mkdir(exist_ok=True)
    common.json_dump({"chosen": {"wb_reference": C.WB_REFERENCE,
                                 "exposure_reference": C.EXPOSURE_REFERENCE,
                                 "target_L": C.EXPOSURE_TARGET_L[C.EXPOSURE_REFERENCE]},
                      "options": diag}, C.CACHE_DIR / "normalization_options.json")

    make_figures(fig_dir, ids, raws, norms_full, tube, ref, stats, labels)
    V.update_progression(STAGE)
    common.stop_here(STAGE, next_cmd="features_hand.py")


if __name__ == "__main__":
    main()
