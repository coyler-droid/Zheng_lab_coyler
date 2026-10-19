"""
Stage 11 -- the FINAL TEST (run once).

 1. frozen_config.json must exist and the manifest / mask / model hashes must still match.
 2. Prints the frozen settings (primary and secondary) and asks:
        "Run the final test now? Nothing can be tuned afterward."
    An explicit 'yes' is needed (typed in a terminal, or passed as `--answer yes` from a
    non-interactive shell).
 3. LOCKS PREDICTIONS FIRST: the test ids come from splits.csv; the test images are featurized
    now (no label file is open); both models predict; every test id must have a probability
    and a YES/NO/REVIEW call. test_predictions_locked.csv is written and its SHA-256 printed
    and recorded in final_test_lock.json.
 4. ONLY THEN are the sealed labels loaded, through _load_sealed_test_labels() -- the single
    dedicated loader, defined here and imported by no other module (enforced by assertions and
    by common.assert_sealed_isolation) -- and validated: ids == test ids of splits.csv, labels
    0/1 without blanks, success/failure counts == splits_meta.json. Any failure aborts.
 5. Labels are joined to the locked predictions by id -> test_report.csv (primary),
    test_report_secondary.csv, and test_report_with_names.csv (original filenames, for manual
    inspection only), all in config.REPORTS_DIR.
 6. Primary: success-class precision, recall, F1, PR-AUC, confusion matrix with 95 % bootstrap
    CIs (config.BOOTSTRAP_N resamples of condition GROUPS; resamples without a success are
    skipped and counted). Secondary: same, separately, plus paired bootstrap differences in F1
    and PR-AUC. The results are written to final_test_done.json, figures to
    figures/11_final_test/.
 7. Labels, ids, predictions and the threshold are never edited afterwards; the primary model
    stays primary whatever the secondary scores.

If test_predictions_locked.csv already exists (and matches its recorded hash), the run
re-scores the locked predictions without predicting again.

Usage:  python final_test.py                 (python final_test.py --answer yes)
Exits with code 3 when a hash, prediction or label check fails, and 0 when the run is declined.
"""
from __future__ import annotations

import argparse
import datetime
import inspect
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common
import config as C
import evaluate as E
import features_hand as FH
import train_cv as TC
import visualize as V

STAGE = "11_final_test"
LOCKED_CSV = C.ROOT / "test_predictions_locked.csv"
LOCK_JSON = C.ROOT / "final_test_lock.json"
METRICS = ("precision", "recall", "f1", "pr_auc")
ROLES = ("primary", "secondary")


# =============================================================================
# THE dedicated loader of the sealed test labels (the only one in the project)
# =============================================================================
def _load_sealed_test_labels() -> pd.DataFrame:
    assert __name__ == "__main__", \
        "the sealed test labels can only be loaded by running final_test.py itself"
    caller = inspect.stack()[1]
    assert Path(caller.filename).resolve() == Path(__file__).resolve() and \
        caller.function == "main", "the sealed labels may only be loaded from final_test.main()"
    assert LOCKED_CSV.exists() and LOCK_JSON.exists(), \
        "predictions must be locked BEFORE the sealed labels are loaded"
    assert common.sha256_file(LOCKED_CSV) == common.json_load(LOCK_JSON)["sha256"], \
        "test_predictions_locked.csv was modified after locking"
    return pd.read_csv(C.SEALED_TEST_LABELS_CSV, dtype={"id": str, "label": str},
                       keep_default_na=False)


# =============================================================================
# Helpers
# =============================================================================
def abort(msg: str):
    print("\n" + "!" * 78 + f"\nABORT: {msg}\n" + "!" * 78)
    sys.exit(3)


def verify_frozen():
    """Load frozen_config.json and abort if the manifest, a mask or a model file changed."""
    if not C.FROZEN_CONFIG_JSON.exists():
        abort("frozen_config.json does not exist -- run freeze.py first.")
    frozen = common.json_load(C.FROZEN_CONFIG_JSON)
    if frozen["manifest_sha256"] != common.sha256_file(C.MANIFEST_CSV):
        abort("manifest.csv changed since the freeze. Rerun data_split.py from scratch.")
    if frozen["mask"]["sha256"] != common.sha256_file(C.TUBE_MASK_NPY) or \
            frozen["mask"]["reference_sha256"] != common.sha256_file(C.REF_MASK_NPY):
        abort("a mask file changed since the freeze.")
    for model_info in frozen["models"].values():
        if common.sha256_file(C.ROOT / model_info["file"]) != model_info["sha256"]:
            abort(f"{model_info['file']} changed since the freeze.")
    return frozen


def print_frozen(frozen):
    print("FROZEN SETTINGS")
    for role in ROLES:
        m = frozen["models"][role]
        tag = "" if role == "primary" else "   (information only)"
        print(f"  {role.upper():9s} {m['model']} | augmentation {m['augmentation_setting']} | "
              f"{m['hyperparameters']} | threshold {m['threshold']:.3f} "
              f"({m['threshold_objective']}){tag}")
    print(f"  review band {frozen['review_band'] or 'none (every tube YES/NO)'} | CLIP "
          f"{frozen['clip']['model']}/{frozen['clip']['pretrained']}"
          f" | normalization WB {frozen['normalization']['wb_reference']}, exposure "
          f"{frozen['normalization']['exposure_reference']} | split {frozen['split_mode']} seed "
          f"{frozen['split_seed']} | frozen at {frozen['frozen_at']}")


def lock_predictions(frozen):
    """Featurize the test images and predict with both frozen models. No labels involved.

    Writes test_predictions_locked.csv and final_test_lock.json (its SHA-256), saves the
    masked/normalized test images for the figures and adds the test rows to features_hand.csv.
    """
    common.unlock_test_images()
    tube, ref = common.load_masks()
    ids = common.test_ids()
    bundles = {r: joblib.load(C.ROOT / frozen["models"][r]["file"]) for r in ROLES}
    embedder = None
    if any(TC.bundle_needs_clip(b) for b in bundles.values()):
        import features_clip as FC
        embedder = FC.ClipEmbedder(frozen["clip"]["model"], frozen["clip"]["pretrained"])
    rows, hand_records = [], []
    for img_id in ids:
        feat = TC.featurize_raw(common.load_image(img_id), tube, ref, img_id, embedder)
        common.write_rgb(C.PROCESSED_DIR / "masked" / f"{img_id}.png", feat["masked"])
        common.write_rgb(C.PROCESSED_DIR / "normalized" / f"{img_id}.png", feat["norm"])
        hand_records.append(feat["rec"])
        row = {"id": img_id}
        for role, bundle in bundles.items():
            hand_row = TC.bundle_hand_row(bundle, feat["rec"], feat["px"])
            clip_rows = [feat["clip"]] if TC.bundle_needs_clip(bundle) else None
            prob = float(TC.bundle_predict(bundle, [hand_row], clip_rows)[0])
            call, yesno, review = TC.decide(prob, bundle["threshold"], bundle["review_band"])
            row.update({f"{role}_prob": round(prob, 6), f"{role}_yesno": yesno,
                        f"{role}_review": review, f"{role}_call": call})
        rows.append(row)
    locked = pd.DataFrame(rows)
    for role in bundles:
        if locked[f"{role}_prob"].isna().any() or set(locked["id"]) != set(ids) or \
                not locked[f"{role}_call"].isin(["YES", "NO", "REVIEW"]).all():
            abort(f"incomplete {role} predictions -- nothing was locked.")
    locked.to_csv(LOCKED_CSV, index=False)
    digest = common.sha256_file(LOCKED_CSV)
    common.json_dump({"sha256": digest, "locked_at": datetime.datetime.now().isoformat(
        timespec="seconds"), "n_test": len(locked)}, LOCK_JSON)
    # replace any earlier test rows of features_hand.csv by the freshly extracted ones
    hand_csv = common.read_csv(C.FEATURES_HAND_CSV)
    pd.concat([hand_csv[~hand_csv["id"].isin(ids)], pd.DataFrame(hand_records)],
              ignore_index=True).to_csv(C.FEATURES_HAND_CSV, index=False)
    print(f"LOCKED {len(locked)} test predictions (primary + secondary) -> {LOCKED_CSV.name}")
    print(f"SHA-256 of {LOCKED_CSV.name}: {digest}")
    return locked


def validate_labels(labels: pd.DataFrame, meta: dict) -> pd.Series:
    """Check the sealed labels against splits.csv / splits_meta.json; return id -> 0/1."""
    test = common.test_ids()
    if labels["id"].duplicated().any():
        dup = sorted(labels.loc[labels["id"].duplicated(), "id"])
        abort(f"duplicate ids in the sealed label file: {dup}")
    missing = sorted(set(test) - set(labels["id"]))
    extra = sorted(set(labels["id"]) - set(test))
    if missing or extra:
        abort(f"sealed label ids != test ids of splits.csv (missing {missing}, extra {extra})")
    if not labels["label"].isin(["0", "1"]).all():
        abort("sealed labels contain values other than 0/1 or blanks")
    y = labels.set_index("id")["label"].astype(int)
    n_success, n_fail = int(y.sum()), int((y == 0).sum())
    if n_success != meta["test_successes"] or n_fail != meta["test_failures"]:
        abort(f"label counts ({n_success} successes / {n_fail} failures) differ "
              f"from splits_meta.json ({meta['test_successes']} / {meta['test_failures']})")
    print(f"sealed labels validated: {len(y)} ids == test ids of splits.csv, 0/1 only, "
          f"{n_success} successes / {n_fail} failures as in splits_meta.json")
    return y


def group_bootstrap(df: pd.DataFrame, thresholds: dict, n=C.BOOTSTRAP_N):
    """Paired bootstrap over condition groups for both models.

    Each resample draws len(groups) groups with replacement and takes all their images; both
    models are scored on the same resamples so that differences are paired. Resamples without
    a success are skipped. Returns {role: [metric dict per resample]} and the skip count.
    """
    rng = np.random.default_rng(C.SEED)
    groups = df["group"].unique()
    rows_of_group = {g: np.flatnonzero(df["group"].to_numpy() == g) for g in groups}
    out = {r: [] for r in thresholds}
    skipped = 0
    for _ in range(n):
        idx = np.concatenate([rows_of_group[g]
                              for g in rng.choice(groups, len(groups), replace=True)])
        y = df["y"].to_numpy()[idx]
        if y.sum() == 0:
            skipped += 1
            continue
        for role, thr in thresholds.items():
            out[role].append(E.binary_metrics(y, df[f"{role}_prob"].to_numpy()[idx], thr))
    return out, skipped


def ci(vals):
    """95 % percentile interval of the finite values (NaN, NaN if there are none)."""
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if len(v) else (np.nan, np.nan)


def report_model(role, df, thr, boot):
    """Print the test metrics of one model with bootstrap CIs.

    Returns (point-estimate metric dict, {metric or confusion cell: (estimate, lo, hi)});
    confusion-cell CIs are for the cell's fraction of the resample.
    """
    m = E.binary_metrics(df["y"], df[f"{role}_prob"], thr)
    print(f"  threshold {thr:.3f} | n = {m['n']} ({m['n_pos']} successes)")
    results = {}
    for k in METRICS:
        lo, hi = ci([b[k] for b in boot])
        results[k] = (m[k], lo, hi)
        print(f"  {k:10s} {m[k]:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
    print("  confusion matrix at the threshold (rows true, cols predicted):")
    print("                pred NO  pred YES")
    print(f"     true fail  {m['tn']:7d}  {m['fp']:8d}")
    print(f"     true succ  {m['fn']:7d}  {m['tp']:8d}")
    for cell in ("tn", "fp", "fn", "tp"):
        lo, hi = ci([b[cell] / b["n"] for b in boot])
        results[cell] = (m[cell], lo, hi)
    print("  cell fractions 95% CI: " + ", ".join(
        f"{c.upper()} [{results[c][1]:.2f}, {results[c][2]:.2f}]"
        for c in ("tn", "fp", "fn", "tp")))
    n_review = int(df[f"{role}_review"].sum())
    print(f"  REVIEW flags: {n_review} of {len(df)} test images")
    return m, results


# =============================================================================
# Figures
# =============================================================================
def make_figures(fig_dir, df, frozen, res, tube):
    """Test-set figures; `res` is {role: report_model() output}."""
    prim = frozen["models"]["primary"]
    names = V.original_names(list(df["id"]))
    thr = prim["threshold"]

    # all test images, sorted by primary probability
    items, titles, borders = [], [], []
    for _, r in df.sort_values("primary_prob").iterrows():
        items.append(common.load_image(r["id"]))
        correct = (r["primary_yesno"] == "YES") == bool(r["y"])
        titles.append(f"{r['id']} | true {V.label_str(r['y'])}\np = {r['primary_prob']:.2f} -> "
                      f"{r['primary_call']}\n{names[r['id']]}")
        borders.append("green" if correct else "red")
    V.save_fig(V.show_grid(items, titles, ncols=10, tile_w=1.6, tile_h=2.75, fontsize=5.5,
                           borders=borders,
                           suptitle=f"All TEST images, PRIMARY model ({prim['model']}"
                                    f" / {prim['augmentation_setting']}), "
                                    f"threshold {thr:.3f}: green = correct, red = wrong"),
               fig_dir / "test_images_grid.png")

    # confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, role in zip(axes, ROLES):
        m = res[role][0]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for (row, col), v in np.ndenumerate(cm):
            ax.text(col, row, str(v), ha="center", va="center", fontsize=14)
        ax.set_xticks([0, 1], ["pred NO", "pred YES"])
        ax.set_yticks([0, 1], ["true fail", "true success"])
        ax.set_title(f"{role}{' (information only)' if role == 'secondary' else ''}\n"
                     f"{frozen['models'][role]['model']} / "
                     f"{frozen['models'][role]['augmentation_setting']}",
                     fontsize=8)
    fig.suptitle("Confusion matrices on the TEST set", fontsize=10)
    fig.tight_layout()
    V.save_fig(fig, fig_dir / "confusion_matrices.png")

    # metrics with bootstrap CIs
    fig, ax = plt.subplots(figsize=(8, 4))
    for k, role in enumerate(ROLES):
        x = np.arange(len(METRICS)) + (k - 0.5) * 0.3
        est = [res[role][1][mm][0] for mm in METRICS]
        lo = [res[role][1][mm][1] for mm in METRICS]
        hi = [res[role][1][mm][2] for mm in METRICS]
        ax.errorbar(x, est, yerr=[np.subtract(est, lo), np.subtract(hi, est)], fmt="o",
                    capsize=4, label=role + (" (information only)" if role == "secondary" else ""))
    ax.set_xticks(range(len(METRICS)), METRICS)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"TEST metrics with 95% group-bootstrap CIs ({C.BOOTSTRAP_N} resamples)",
                 fontsize=10)
    ax.legend(fontsize=8)
    V.save_fig(fig, fig_dir / "bootstrap_ci.png")

    # primary vs secondary: metric bars and per-image probabilities
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    width = 0.38
    for k, role in enumerate(ROLES):
        axes[0].bar(np.arange(len(METRICS)) + (k - 0.5) * width,
                    [res[role][1][mm][0] for mm in METRICS], width,
                    label=f"{role}: {frozen['models'][role]['augmentation_setting']}")
    axes[0].set_xticks(range(len(METRICS)), METRICS)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=8)
    axes[0].set_title("Primary vs secondary (augmentation vs none) on TEST", fontsize=9)
    for lab, col in [(0, "#4c72b0"), (1, "#dd8452")]:
        sub = df[df["y"] == lab]
        axes[1].scatter(sub["secondary_prob"], sub["primary_prob"], c=col, s=30,
                        label=V.LABEL_NAMES[lab], edgecolor="k", linewidth=0.3)
    axes[1].plot([0, 1], [0, 1], "k:", lw=1)
    axes[1].set_xlabel("secondary probability "
                       f"({frozen['models']['secondary']['augmentation_setting']})")
    axes[1].set_ylabel(f"primary probability ({prim['augmentation_setting']})")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    V.save_fig(fig, fig_dir / "primary_vs_secondary.png")

    # pipeline progression for 12 random test images
    rng = np.random.default_rng(C.SEED)
    pick = sorted(rng.choice(df["id"].to_numpy(), min(12, len(df)), replace=False))
    brown = prim["color_ranges"]["brown"]
    hand_csv = common.read_csv(C.FEATURES_HAND_CSV).set_index("id")
    items, titles = [], []
    for img_id in pick:
        r = df[df["id"] == img_id].iloc[0]
        norm = common.read_rgb(C.PROCESSED_DIR / "normalized" / f"{img_id}.png")
        rec = hand_csv.loc[img_id].to_dict()
        masked = common.read_rgb(C.PROCESSED_DIR / "masked" / f"{img_id}.png")
        for name, im in [("raw", common.load_image(img_id)),
                         ("mask overlay", V.outline(common.load_image(img_id), tube)),
                         ("background blanked", masked),
                         ("normalized", norm),
                         ("bands + brown + boundary", FH.render_overlay(norm, tube, rec, brown))]:
            items.append(im)
            titles.append(f"{img_id} | {V.label_str(r['y'])}\n{name}")
        items.append(f"PRIMARY\np = {r['primary_prob']:.2f}\n{r['primary_call']}\n\n"
                     f"true {V.label_str(r['y'])}\n"
                     f"{'correct' if (r['primary_yesno'] == 'YES') == bool(r['y']) else 'WRONG'}")
        titles.append(f"{img_id}\nlocked prediction")
    V.save_fig(V.show_grid(items, titles, ncols=6, tile_w=1.75, tile_h=2.75, fontsize=6,
                           suptitle="Pipeline progression for 12 TEST images (after locking)"),
               fig_dir / "test_progression_12.png")


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--answer", choices=["yes"], help="explicit yes from a non-interactive shell")
    args = ap.parse_args()
    fig_dir = common.start_stage(STAGE)
    frozen = verify_frozen()
    meta = common.json_load(C.SPLITS_META_JSON)
    print_frozen(frozen)

    # --- 1. lock the predictions (or reuse an existing, hash-verified lock) -------------------
    if LOCKED_CSV.exists():
        lock = common.json_load(LOCK_JSON)
        if common.sha256_file(LOCKED_CSV) != lock["sha256"]:
            abort("test_predictions_locked.csv does not match its recorded hash.")
        print(f"\nPredictions were already LOCKED at {lock['locked_at']} "
              f"(sha256 {lock['sha256']}).")
        print("Re-scoring from the locked file; nothing is re-predicted or re-tuned.")
        common.unlock_test_images()
        locked = common.read_csv(LOCKED_CSV)
    else:
        ok = common.confirm("\nRun the final test now? Nothing can be tuned afterward.",
                            args.answer == "yes", "--answer yes")
        if not ok:
            print("Final test NOT run.")
            sys.exit(0)
        locked = lock_predictions(frozen)

    # --- 2. only now: load and validate the sealed labels -------------------------------------
    labels = _load_sealed_test_labels()
    y = validate_labels(labels, meta)
    common.mark_test_labels_unsealed()

    # --- 3. per-image reports ------------------------------------------------------------------
    splits = common.load_splits().set_index("id")
    df = locked.copy()
    df["y"] = df["id"].map(y).astype(int)
    df["group"] = df["id"].map(splits["group"])
    for role, fname in (("primary", "test_report.csv"), ("secondary", "test_report_secondary.csv")):
        is_correct = (df[f"{role}_yesno"] == "YES") == df["y"].astype(bool)
        rep = pd.DataFrame({"id": df["id"], "probability": df[f"{role}_prob"],
                            "yes_no": df[f"{role}_yesno"], "review_flag": df[f"{role}_review"],
                            "call": df[f"{role}_call"], "true_label": df["y"],
                            "correct": np.where(is_correct, "correct", "wrong")})
        rep.to_csv(C.REPORTS_DIR / fname, index=False)
    names = V.original_names(list(df["id"]))
    pd.read_csv(C.REPORTS_DIR / "test_report.csv", dtype={"id": str}).assign(
        original_name=lambda d: d["id"].map(names)).to_csv(
        C.REPORTS_DIR / "test_report_with_names.csv", index=False)
    print("wrote test_report.csv, test_report_secondary.csv, test_report_with_names.csv "
          "(filenames: for your inspection only)")

    # --- 4. metrics with group-bootstrap CIs ---------------------------------------------------
    thresholds = {r: frozen["models"][r]["threshold"] for r in ROLES}
    boot, skipped = group_bootstrap(df, thresholds)
    print(f"\nbootstrap: {C.BOOTSTRAP_N} resamples of {df['group'].nunique()} condition groups; "
          f"{skipped} skipped (no success in the resample)")
    res = {}
    print("\n" + "=" * 78 + "\nPRIMARY MODEL  " + f"{frozen['models']['primary']['model']} / "
          f"{frozen['models']['primary']['augmentation_setting']}\n" + "=" * 78)
    res["primary"] = report_model("primary", df, thresholds["primary"], boot["primary"])
    print("\n" + "-" * 78 + "\nSECONDARY MODEL (information only)  "
          f"{frozen['models']['secondary']['model']} / "
          f"{frozen['models']['secondary']['augmentation_setting']}\n" + "-" * 78)
    res["secondary"] = report_model("secondary", df, thresholds["secondary"], boot["secondary"])
    print("\nPAIRED DIFFERENCE primary - secondary (same bootstrap resamples):")
    diffs = {}
    for k in ("f1", "pr_auc"):
        d = [a[k] - b[k] for a, b in zip(boot["primary"], boot["secondary"])]
        est = res["primary"][0][k] - res["secondary"][0][k]
        lo, hi = ci(d)
        diffs[k] = (est, lo, hi)
        print(f"  d{k:7s} {est:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]")

    # --- 5. figures and the completion record --------------------------------------------------
    tube, _ = common.load_masks()
    make_figures(fig_dir, df, frozen, res, tube)
    common.json_dump({"completed_at": datetime.datetime.now().isoformat(timespec="seconds"),
                      "locked_sha256": common.json_load(LOCK_JSON)["sha256"],
                      "bootstrap_skipped": skipped,
                      "primary": {k: list(v) for k, v in res["primary"][1].items()},
                      "secondary": {k: list(v) for k, v in res["secondary"][1].items()},
                      "difference": {k: list(v) for k, v in diffs.items()}},
                     common.FINAL_TEST_DONE_JSON)
    print(f"\nTest has ~{int(df['y'].sum())} successes, so intervals are wide. The secondary "
          "result is for information only. Do NOT switch models or tune anything based on "
          "test results.")
    print("\nFINAL SUMMARY")
    p = res["primary"][1]
    print(f"  PRIMARY {frozen['models']['primary']['model']} / "
          f"{frozen['models']['primary']['augmentation_setting']}: F1 {p['f1'][0]:.3f} "
          f"[{p['f1'][1]:.3f}, {p['f1'][2]:.3f}], PR-AUC {p['pr_auc'][0]:.3f} "
          f"[{p['pr_auc'][1]:.3f}, {p['pr_auc'][2]:.3f}], precision {p['precision'][0]:.3f}, "
          f"recall {p['recall'][0]:.3f}")
    print("\n" + common.resplit_reminder())
    common.stop_here(STAGE, next_cmd="predict.py --demo")


if __name__ == "__main__":
    main()
