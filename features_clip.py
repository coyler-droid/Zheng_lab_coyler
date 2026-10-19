"""
Stage 05 -- frozen CLIP image embeddings (no fine-tuning).

What it does
  Every masked + normalized DEV image is turned into three views -- full tube, lower half,
  upper half. Each view is cropped to the tight bounding box of its part of the tube mask,
  padded to a square with neutral gray (aspect ratio preserved, never stretched), resized to
  config.CLIP_IMAGE_SIZE (224x224) and embedded with the frozen open_clip model
  (default config.CLIP_MODEL_NAME / CLIP_PRETRAINED = ViT-B-32 / laion2b_s34b_b79k). Each view
  embedding is L2-normalized and the three are concatenated in config.CLIP_VIEWS order
  (3 x 512 = 1536 numbers per image for ViT-B-32).

Inputs
  tube_mask.npy, processed/normalized/<id>.png, features_hand.csv (stage-order guard),
  DEV labels and CV folds (figures only).

Outputs
  cache/clip/<model>__<pretrained>/<id>_aug<k>.npy   one (3, D) float32 array per image view
                                                     (rows = full, lower, upper); aug 0 =
                                                     original, aug k > 0 written by augment.py
  figures/05_clip/clip_input_views.png               the exact 224x224 inputs of CLIP
  figures/05_clip/clip_embedding_scatter.png         PCA (and UMAP) of the DEV embeddings

The embeddings are never used raw: in stage 07 (train_cv.py) they always pass through a PCA
fitted on the training fold only (config.PCA_COMPONENTS_GRID components).

Usage
  python features_clip.py                                                  # stage 05 (DEV)
  python features_clip.py --model ViT-L-14 --pretrained openai --allow-download
      # optional second model (~0.9 GB download); cached separately, used by train_cv.py
      # only if config.CLIP_MODEL_NAME / CLIP_PRETRAINED are switched to it
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

import common
import config as C
import visualize as V

STAGE = "05_clip"


# =============================================================================
# Views (what CLIP sees)
# =============================================================================
def pad_square(img: np.ndarray) -> np.ndarray:
    """Center `img` on a square neutral-gray canvas (no stretching)."""
    h, w = img.shape[:2]
    side = max(h, w)
    out = np.full((side, side, 3), C.NEUTRAL_GRAY, np.uint8)
    y0, x0 = (side - h) // 2, (side - w) // 2
    out[y0:y0 + h, x0:x0 + w] = img
    return out


def _crop(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Crop `img` to the tight bounding box of the True pixels of `mask`."""
    ys, xs = np.nonzero(mask)
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def make_views(norm_blanked: np.ndarray, tube: np.ndarray) -> dict:
    """{'full','lower','upper'} -> 224x224 RGB uint8, exactly what CLIP sees.

    The tube mask is split at the middle row of its vertical extent; that middle row belongs
    to both halves. Each view is cropped to its own mask bounding box before padding.
    """
    rows = np.nonzero(tube.any(axis=1))[0]
    mid = (rows.min() + rows.max()) // 2
    upper, lower = tube.copy(), tube.copy()
    upper[mid + 1:] = False
    lower[:mid] = False
    views = {}
    for name, mask in (("full", tube), ("lower", lower), ("upper", upper)):
        square = pad_square(_crop(norm_blanked, mask))
        # INTER_AREA for shrinking (avoids aliasing), INTER_CUBIC for enlarging
        interp = cv2.INTER_AREA if square.shape[0] > C.CLIP_IMAGE_SIZE else cv2.INTER_CUBIC
        views[name] = cv2.resize(square, (C.CLIP_IMAGE_SIZE,) * 2, interpolation=interp)
    return views


# =============================================================================
# Model + cache
# =============================================================================
def model_tag(name=None, pretrained=None) -> str:
    """Folder-safe '<model>__<pretrained>' tag (defaults from config)."""
    return f"{name or C.CLIP_MODEL_NAME}__{pretrained or C.CLIP_PRETRAINED}".replace("/", "-")


def cache_dir(name=None, pretrained=None):
    """Embedding cache folder of one CLIP model (defaults from config)."""
    return C.CACHE_DIR / "clip" / model_tag(name, pretrained)


def weights_in_local_cache(name, pretrained):
    """True / False / None (unknown) -- whether the weights are already downloaded.

    Only weights hosted on the Hugging Face hub can be checked; anything else gives None.
    """
    try:
        import open_clip
        from huggingface_hub import try_to_load_from_cache
        cfg = open_clip.get_pretrained_cfg(name, pretrained)
        repo = (cfg.get("hf_hub") or "").strip("/")
        if not repo:
            return None
        for fn in ("open_clip_model.safetensors", "open_clip_pytorch_model.bin"):
            if isinstance(try_to_load_from_cache(repo, fn), str):
                return True
        return False
    except Exception:  # noqa: BLE001
        return None


def _go_offline():
    """Weights are cached -> no network access at all (no update checks)."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        # huggingface_hub reads the env var at import time; set the constant as well in case
        # it is already imported
        import huggingface_hub.constants as hc
        hc.HF_HUB_OFFLINE = True
    except Exception:  # noqa: BLE001
        pass


class ClipEmbedder:
    """Frozen open_clip image encoder (eval mode, no gradients), CPU, seeded."""

    def __init__(self, name=None, pretrained=None, batch_size=48):
        import open_clip
        import torch
        torch.manual_seed(C.SEED)
        torch.set_num_threads(max(1, (os.cpu_count() or 2) - 2))
        self.torch, self.batch_size = torch, batch_size
        self.name, self.pretrained = name or C.CLIP_MODEL_NAME, pretrained or C.CLIP_PRETRAINED
        if weights_in_local_cache(self.name, self.pretrained):
            _go_offline()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.name, pretrained=self.pretrained)
        self.model.eval()

    def embed(self, imgs: list[np.ndarray]) -> np.ndarray:
        """224x224 RGB uint8 images -> (n, D) L2-normalized float32."""
        from PIL import Image
        out = []
        with self.torch.no_grad():
            for start in range(0, len(imgs), self.batch_size):
                batch = self.torch.stack([self.preprocess(Image.fromarray(im))
                                          for im in imgs[start:start + self.batch_size]])
                out.append(self.model.encode_image(batch).float().numpy())
        emb = np.concatenate(out).astype(np.float32)
        return emb / np.linalg.norm(emb, axis=1, keepdims=True)

    def embed_views(self, view_dicts: list[dict]) -> np.ndarray:
        """list of make_views() dicts -> (n, 3, D), views in config.CLIP_VIEWS order."""
        flat = [v[name] for v in view_dicts for name in C.CLIP_VIEWS]
        emb = self.embed(flat)
        return emb.reshape(len(view_dicts), len(C.CLIP_VIEWS), -1)


def cache_path(img_id: str, aug: int, cdir=None):
    """Cache file of one (image id, augmented view index); aug 0 = original."""
    return (cdir or cache_dir()) / f"{img_id}_aug{aug}.npy"


def embed_and_cache(embedder: ClipEmbedder, items, tube, cdir=None, chunk=96) -> int:
    """items: list of (id, aug, norm_blanked). Computes only what is not cached yet.

    Returns the number of newly embedded items. Work is done in chunks of `chunk` images so
    that progress is saved to disk regularly.
    """
    cdir = cdir or cache_dir(embedder.name, embedder.pretrained)
    cdir.mkdir(parents=True, exist_ok=True)
    todo = [it for it in items if not cache_path(it[0], it[1], cdir).exists()]
    for start in range(0, len(todo), chunk):
        part = todo[start:start + chunk]
        emb = embedder.embed_views([make_views(img, tube) for _, _, img in part])
        for (img_id, aug, _), vec in zip(part, emb):
            np.save(cache_path(img_id, aug, cdir), vec.astype(np.float32))
    return len(todo)


def load_cached(keys, cdir=None) -> dict:
    """(id, aug) -> concatenated (3*D,) vector, read from the cache (default model)."""
    out = {}
    for img_id, aug in keys:
        path = cache_path(img_id, aug, cdir)
        if not path.exists():
            raise FileNotFoundError(f"missing CLIP cache {path.name} -- run the CLIP stage first")
        out[(img_id, aug)] = np.load(path).reshape(-1)
    return out


# =============================================================================
# Stage
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=C.CLIP_MODEL_NAME)
    ap.add_argument("--pretrained", default=C.CLIP_PRETRAINED)
    ap.add_argument("--allow-download", action="store_true",
                    help="permit downloading weights that are not in the local cache")
    args = ap.parse_args()
    fig_dir = common.start_stage(STAGE)
    common.require_mask_approved()
    common.require_files(C.FEATURES_HAND_CSV, hint="run features_hand.py first")

    cached = weights_in_local_cache(args.model, args.pretrained)
    print(f"CLIP model {args.model} / {args.pretrained}: weights in local cache = {cached}")
    if cached is False and not args.allow_download:
        print("The weights must be DOWNLOADED first. Re-run with --allow-download if that is OK.")
        sys.exit(2)
    if cached:
        _go_offline()                                       # no network needed
    second_run = (args.model, args.pretrained) != (C.CLIP_MODEL_NAME, C.CLIP_PRETRAINED)
    if second_run:
        print("Optional second run: embeddings are cached separately and are not used by "
              "train_cv.py unless config.CLIP_MODEL_NAME / CLIP_PRETRAINED are switched.")

    tube = np.load(C.TUBE_MASK_NPY)
    labels = common.load_dev_labels()
    ids = common.dev_ids()
    norms = {i: common.read_rgb(C.PROCESSED_DIR / "normalized" / f"{i}.png") for i in ids}
    embedder = ClipEmbedder(args.model, args.pretrained)
    cdir = cache_dir(args.model, args.pretrained)
    n_new = embed_and_cache(embedder, [(i, 0, norms[i]) for i in ids], tube, cdir)
    vecs = load_cached([(i, 0) for i in ids], cdir)
    X = np.stack([vecs[(i, 0)] for i in ids])
    print(f"embedded {n_new} new / {len(ids)} DEV images -> {X.shape[1]} numbers each "
          f"({len(C.CLIP_VIEWS)} views x {X.shape[1] // len(C.CLIP_VIEWS)}), cache: "
          f"{cdir.relative_to(C.ROOT)}")
    if second_run:
        common.stop_here(STAGE)
        return

    # (1) the exact images fed to CLIP for the showcase images
    items, titles = [], []
    for i in C.SHOWCASE_IDS:
        views = make_views(norms[i], tube)
        for name in C.CLIP_VIEWS:
            items.append(views[name])
            titles.append(f"{i} | {V.label_str(labels[i])}\nCLIP view: {name} (224x224)")
    V.save_fig(V.show_grid(items, titles, ncols=6, tile_w=1.9, tile_h=2.1, fontsize=6,
                           suptitle="Exactly what CLIP sees: full / lower half / upper half, "
                                    "padded to square with neutral gray, 224x224"),
               fig_dir / "clip_input_views.png")

    # (2) PCA (and UMAP) of the DEV embeddings, colour = label, marker = fold of repeat 0.
    # Visual check only: these projections are fitted on all DEV images and are not used by
    # any model.
    folds = common.load_folds()
    fold_of_id = folds[folds["repeat"] == 0].set_index("id")["fold"]
    y = np.array([labels[i] for i in ids])
    fold = np.array([fold_of_id[i] for i in ids])
    Z = PCA(n_components=2, svd_solver="full").fit(X).transform(X)
    panels = [("PCA (fit on DEV only)", Z)]
    try:
        import umap
        U = umap.UMAP(n_neighbors=15, min_dist=0.2, random_state=C.SEED,
                      n_jobs=1).fit_transform(X)
        panels.append(("UMAP (fit on DEV only)", U))
    except ImportError:
        print("umap-learn not installed -> PCA scatter only")
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 5.2), squeeze=False)
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    for ax, (title, coords) in zip(axes[0], panels):
        for f in sorted(set(fold)):
            for lab, col in [(0, "#4c72b0"), (1, "#dd8452")]:
                sel = (fold == f) & (y == lab)
                ax.scatter(coords[sel, 0], coords[sel, 1], c=col,
                           marker=markers[f % len(markers)],
                           s=28, alpha=0.8, edgecolor="k", linewidth=0.3,
                           label=f"{V.LABEL_NAMES[lab]}, fold {f}")
        ax.set_title(title, fontsize=10)
    axes[0, 0].legend(fontsize=6, ncol=2)
    fig.suptitle(f"CLIP {args.model} embeddings of the {len(ids)} DEV images (colour = label, "
                 "marker = held-out fold in CV repeat 0) -- visual check only", fontsize=9)
    fig.tight_layout()
    V.save_fig(fig, fig_dir / "clip_embedding_scatter.png")

    print("Reminder: in stage 07 the CLIP vectors always go through a PCA fitted on the "
          f"training fold only (n_components in {list(C.PCA_COMPONENTS_GRID)}).")
    V.update_progression(STAGE)
    common.stop_here(STAGE, next_cmd="augment.py")


if __name__ == "__main__":
    main()
