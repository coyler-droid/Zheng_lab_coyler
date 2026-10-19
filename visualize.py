"""
visualize.py -- plotting helpers shared by the stage scripts (`import visualize as V`).

Provides
  * save_fig / show_grid / render_table: figure saving, image grids and PNG tables;
  * outline / overlay / red_x: drawing a mask contour, a coloured mask overlay, a red cross;
  * original_names: guarded look-up of original file names for captions;
  * update_progression: the pipeline-progression figure (one row per showcase image, one
    column per completed stage), saved as figures/pipeline_progression_after_<stage>.png and
    figures/pipeline_progression.png.

This is the ONLY module allowed to read `original_name` from manifest.csv, and only to caption
images whose labels may be shown at that stage (dev images; test images only after the final
test has locked its predictions).

Uses the non-interactive Agg backend, so it works without a display.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # must precede the pyplot import
import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import common  # noqa: E402
import config as C  # noqa: E402

LABEL_NAMES = {1: "success", 0: "fail"}
DPI = 110   # resolution of every saved figure


def label_str(y) -> str:
    """Caption text for a label, e.g. 'y=1 success'."""
    return f"y={int(y)} {LABEL_NAMES[int(y)]}"


def save_fig(fig, path: Path):
    """Save `fig` (white background, tight bounding box), close it and print its path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {path.relative_to(C.ROOT)}")


# -----------------------------------------------------------------------------
# Original filenames (captions only, guarded)
# -----------------------------------------------------------------------------
def original_names(ids) -> dict:
    """id -> original file name ('?' if unknown). Refuses test ids until labels are visible."""
    ids = list(ids)
    _, test = common._split_sets()
    if set(ids) & test and not common.test_labels_visible():
        raise PermissionError("original filenames of TEST images can only be shown after the "
                              "final test has locked its predictions")
    manifest = common.read_csv(C.MANIFEST_CSV, usecols=["id", "original_name"])
    name_by_id = dict(zip(manifest["id"], manifest["original_name"]))
    return {i: name_by_id.get(i, "?") for i in ids}


# -----------------------------------------------------------------------------
# Image utilities
# -----------------------------------------------------------------------------
def outline(img: np.ndarray, mask: np.ndarray, color=(0, 255, 0), thickness=1) -> np.ndarray:
    """Copy of `img` with the outer contour(s) of `mask` drawn on it."""
    out = img.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, contours, -1, color, thickness)
    return out


def overlay(img: np.ndarray, boolmask: np.ndarray, color=(255, 0, 0), alpha=0.45) -> np.ndarray:
    """Copy of `img` with the pixels of `boolmask` blended towards `color` by `alpha`."""
    out = img.astype(np.float32).copy()
    col = np.array(color, np.float32)
    out[boolmask] = (1 - alpha) * out[boolmask] + alpha * col
    return np.clip(out, 0, 255).astype(np.uint8)


def show_grid(items, titles=None, ncols=6, tile_w=1.55, tile_h=2.55, borders=None,
              suptitle=None, fontsize=6.5, row_labels=None, text_fontsize=8):
    """Grid figure of tiles; returns the figure (save it with save_fig).

    items: list of RGB arrays, str (rendered as a text tile) or None (empty tile).
    titles / borders: optional per-tile title text and border colour (falsy = none).
    tile_w / tile_h: tile size in inches.
    """
    n = len(items)
    nrows = max(1, int(np.ceil(n / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * tile_w, nrows * tile_h),
                             squeeze=False)
    for k, ax in enumerate(axes.ravel()):
        ax.set_xticks([])
        ax.set_yticks([])
        if k >= n or items[k] is None:
            ax.axis("off")
            continue
        item = items[k]
        if isinstance(item, str):
            ax.set_facecolor("#f4f4f4")
            ax.text(0.5, 0.5, item, ha="center", va="center", fontsize=text_fontsize,
                    transform=ax.transAxes, wrap=True)
        else:
            ax.imshow(item)
        if titles is not None and titles[k]:
            ax.set_title(titles[k], fontsize=fontsize)
        if borders is not None and borders[k]:
            for spine in ax.spines.values():
                spine.set_edgecolor(borders[k])
                spine.set_linewidth(4)
    if row_labels is not None:
        for r, lab in enumerate(row_labels):
            axes[r, 0].set_ylabel(lab, fontsize=fontsize + 0.5)
    if suptitle:                       # reserve a fixed ~0.45 inch band for the title
        fig_h = fig.get_figheight()
        fig.suptitle(suptitle, fontsize=10, y=1 - 0.08 / fig_h, va="top")
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / fig_h))
    else:
        fig.tight_layout()
    return fig


def red_x(ax):
    """Draw a red cross over the whole axes (marks a rejected / failed tile)."""
    ax.plot([0, 1], [0, 1], color="red", lw=4, transform=ax.transAxes)
    ax.plot([0, 1], [1, 0], color="red", lw=4, transform=ax.transAxes)


def render_table(rows, path: Path, title: str, col_widths=None, fontsize=8):
    """Render a list of (key, value) or list-of-lists table to a PNG."""
    rows = [[str(c) for c in r] for r in rows]
    fig_h = 0.32 * (len(rows) + 2)
    fig, ax = plt.subplots(figsize=(11, max(2.0, fig_h)))
    ax.axis("off")
    tab = ax.table(cellText=rows, loc="upper left", cellLoc="left", colWidths=col_widths)
    tab.auto_set_font_size(False)
    tab.set_fontsize(fontsize)
    tab.scale(1, 1.25)
    ax.set_title(title, fontsize=11, loc="left")
    save_fig(fig, path)


# -----------------------------------------------------------------------------
# Pipeline progression figure
# -----------------------------------------------------------------------------
def _stage_reached(stage: str, needed: str) -> bool:
    """True if `stage` is at or after `needed` in config.STAGES."""
    return C.STAGES.index(stage) >= C.STAGES.index(needed)


def _prog_columns(stage: str):
    """List of (column title, function id -> image|str|None) available at `stage`.

    A column appears only once its stage is reached AND its output exists, so the figure
    can be rebuilt at any point of the pipeline.
    """
    cols = [("raw", common.load_image)]
    if _stage_reached(stage, "02_mask") and C.TUBE_MASK_NPY.exists():
        mask = np.load(C.TUBE_MASK_NPY)
        cols.append(("mask overlay", lambda i: outline(common.load_image(i), mask)))
        cols.append(("background blanked",
                     lambda i: _maybe(C.PROCESSED_DIR / "masked" / f"{i}.png")))
    if _stage_reached(stage, "03_normalization") and (C.PROCESSED_DIR / "normalized").exists():
        cols.append(("white-balanced / normalized",
                     lambda i: _maybe(C.PROCESSED_DIR / "normalized" / f"{i}.png")))
    if _stage_reached(stage, "04_hand_features") and C.FEATURES_HAND_CSV.exists():
        import features_hand   # local import: features_hand imports this module
        cols.append(("bands + brown + boundary", features_hand.render_overlay_for_id))
    if _stage_reached(stage, "06_augmentation") and (C.CACHE_DIR / "aug_normalized").exists():
        cols.append(("augmented example (aug1)",
                     lambda i: _maybe(C.CACHE_DIR / "aug_normalized" / f"{i}_aug1.png")))
    if _stage_reached(stage, "07_model_comparison") and (C.CV_DIR / "oof_primary.csv").exists():
        cols.append(("OOF probability + verdict", _oof_text))
    return cols


def _maybe(path: Path):
    """The image at `path`, or None (empty tile) if it does not exist."""
    return common.read_rgb(path) if Path(path).exists() else None


def _oof_text(img_id: str):
    """Text tile: the primary model's mean out-of-fold probability and verdict for one id."""
    oof = common.read_csv(C.CV_DIR / "oof_primary.csv")
    probs = oof.loc[oof["id"] == img_id, "prob"]   # one value per CV repeat
    if probs.empty:
        return None
    # Until stage 09 has written thresholds.json, show a provisional 0.5 threshold.
    thr, band, src = 0.5, None, "threshold 0.5 (provisional)"
    tpath = C.CV_DIR / "thresholds.json"
    if tpath.exists():
        t = common.json_load(tpath)["primary"]
        thr, band, src = t["threshold"], t["review_band"], f"threshold {t['threshold']:.3f}"
    p_mean = float(probs.mean())
    verdict = "YES" if p_mean >= thr else "NO"
    if band is not None and band[0] <= p_mean <= band[1]:
        verdict = "REVIEW"
    y = int(common.load_dev_labels()[img_id])
    ok = (p_mean >= thr) == bool(y)
    sel = common.json_load(C.CV_DIR / "selection.json")
    return (f"{sel['primary']['model']}\n{sel['primary']['setting']}\n\n"
            f"OOF p = {p_mean:.2f}\n(+/- {probs.std(ddof=0):.2f} over\n{len(probs)} repeats)\n\n"
            f"{verdict}\n{src}\n\ntrue {label_str(y)}\n{'correct' if ok else 'WRONG'}")


def update_progression(stage: str):
    """Save figures/pipeline_progression_after_<stage>.png and pipeline_progression.png."""
    ids = list(C.SHOWCASE_IDS)
    if not ids:
        print("  (progression figure skipped: config.SHOWCASE_IDS not set yet)")
        return
    common.assert_dev_only(ids, "progression figure")
    labels = common.load_dev_labels()
    cols = _prog_columns(stage)
    items, titles = [], []
    for i in ids:
        for name, fn in cols:
            items.append(fn(i))
            titles.append(f"{i} | {label_str(labels[i])}\n{name}")
    # The same tiles are saved twice: a per-stage snapshot and the "latest" figure.
    fig = show_grid(items, titles, ncols=len(cols), tile_w=1.75, tile_h=2.75, fontsize=6.2,
                    suptitle=f"Pipeline progression after {stage}  (12 fixed DEV showcase images)")
    save_fig(fig, C.FIG_DIR / f"pipeline_progression_after_{stage}.png")
    fig = show_grid(items, titles, ncols=len(cols), tile_w=1.75, tile_h=2.75, fontsize=6.2,
                    suptitle=f"Pipeline progression (latest: {stage})")
    save_fig(fig, C.FIG_DIR / "pipeline_progression.png")
