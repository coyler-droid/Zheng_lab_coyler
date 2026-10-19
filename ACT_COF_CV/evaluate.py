"""
Metric helpers + Stage 08 -- error analysis of the selected model's out-of-fold predictions.

Metric helpers (imported by train_cv.py, threshold.py, final_test.py)
  pr_auc, roc_auc        NaN when undefined (no positives / single class)
  binary_metrics         success-class precision / recall / F1 at a threshold, PR-AUC,
                         ROC-AUC and confusion counts
  corrected_ci           Nadeau-Bengio corrected interval for paired repeated-CV differences

Stage 08 (DEV only; the test set is not touched)
  Inputs: cv_results/selection.json and cv_results/oof_primary.csv (train_cv.py),
  features_hand.csv, DEV labels and CV folds.
  * every misclassified OOF image -- mean OOF probability over the CV repeats vs 0.5 (the
    operating threshold is chosen only in stage 09): id, true label, probability
  * image grid of the misclassified images, captioned with the original filenames (DEV
    images only; for visual inspection)
  * shortcut checks for (a) liquid fill height, (b) raw brightness + raw white balance,
    (c) surface position relative to the printed "2" mark (only if detected in >= 80 % of
    DEV images): point-biserial correlation with the label, and a logistic regression on
    ONLY those features (same CV folds) compared with chance and with a label-permutation
    null; plus the selected model's OOF probability vs each shortcut, overall and within
    each class
  * review_labels.csv: confident wrong predictions (possible label errors) and borderline
    images. Labels are NEVER changed by this code.
  Outputs: cv_results/shortcut_checks.csv, cv_results/shortcut_within_class.csv,
  review_labels.csv, figures/08_error_analysis/*.png

Usage:  python evaluate.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

import common
import config as C

STAGE = "08_error_analysis"


# =============================================================================
# Metrics
# =============================================================================
def pr_auc(y, p) -> float:
    """Average precision of the success class; NaN if there is no positive."""
    y = np.asarray(y)
    if y.sum() == 0:
        return np.nan
    return float(average_precision_score(y, p))


def roc_auc(y, p) -> float:
    """ROC-AUC; NaN if only one class is present."""
    y = np.asarray(y)
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    return float(roc_auc_score(y, p))


def binary_metrics(y, p, thr: float = 0.5) -> dict:
    """Success-class precision / recall / F1 at `thr`, PR-AUC, ROC-AUC, confusion counts.

    Conventions: precision = 0 when nothing is predicted positive, recall = NaN when there
    is no positive, F1 = 0 when there is no true positive.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    yhat = (p >= thr).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * prec * rec / (prec + rec) if (tp > 0) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "pr_auc": pr_auc(y, p),
            "roc_auc": roc_auc(y, p), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "n": len(y), "n_pos": int(y.sum())}


def corrected_ci(diffs, test_frac: float | None = None, alpha: float = 0.05):
    """Mean and 95 % interval of repeated-CV paired differences with the Nadeau & Bengio
    (2003) variance correction (folds share training data, so the naive t interval is too
    narrow). NaN differences are dropped; `test_frac` defaults to 1 / config.N_FOLDS.
    Returns (mean, sd, lo, hi); sd / lo / hi are NaN with fewer than two differences."""
    d = np.asarray(diffs, float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 2:
        return (float(d.mean()) if n else np.nan), np.nan, np.nan, np.nan
    test_frac = test_frac if test_frac is not None else 1.0 / C.N_FOLDS
    sd = d.std(ddof=1)
    se = np.sqrt((1.0 / n + test_frac / (1 - test_frac)) * sd ** 2)
    t = stats.t.ppf(1 - alpha / 2, n - 1)
    return float(d.mean()), float(sd), float(d.mean() - t * se), float(d.mean() + t * se)


# =============================================================================
# Stage 08
# =============================================================================
def _cv_shortcut_score(df: pd.DataFrame, cols, labels, folds, n_perm=100):
    """Logistic regression on shortcut columns only, same repeated CV folds.

    Missing values are imputed with the training-fold median. Returns the mean (and SD) of
    the per-fold ROC-AUC, the mean per-fold PR-AUC, and a label-permutation null for the
    ROC-AUC (`n_perm` permutations of all DEV labels, same folds) with its p-value.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X_all = df.set_index("id")[cols].astype(float)

    def cv_scores(y_series):
        rocs, prs = [], []
        for _, _, tr, te in common.iter_cv_folds(folds):
            Xtr, Xte = X_all.loc[tr].copy(), X_all.loc[te].copy()
            med = Xtr.median()
            Xtr, Xte = Xtr.fillna(med), Xte.fillna(med)
            clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced",
                                                                     max_iter=2000))
            clf.fit(Xtr.values, y_series.loc[tr].values)
            prob = clf.predict_proba(Xte.values)[:, 1]
            rocs.append(roc_auc(y_series.loc[te].values, prob))
            prs.append(pr_auc(y_series.loc[te].values, prob))
        return float(np.nanmean(rocs)), float(np.nanmean(prs)), float(np.nanstd(rocs, ddof=1))

    roc, pr, roc_sd = cv_scores(labels)
    rng = np.random.default_rng(C.SEED)
    null = []
    for _ in range(n_perm):
        null.append(cv_scores(pd.Series(rng.permutation(labels.values), index=labels.index))[0])
    # add-one permutation p-value (never exactly 0)
    pval = (1 + sum(v >= roc for v in null)) / (1 + n_perm)
    return {"roc_auc": roc, "roc_auc_sd": roc_sd, "pr_auc": pr,
            "null_roc_mean": float(np.mean(null)), "null_roc_95": float(np.percentile(null, 95)),
            "perm_p": float(pval)}


def main():
    import matplotlib.pyplot as plt

    import features_hand as FH
    import visualize as V
    fig_dir = common.start_stage(STAGE)
    common.require_files(C.CV_DIR / "selection.json", C.CV_DIR / "oof_primary.csv",
                         hint="run train_cv.py first")
    sel = common.json_load(C.CV_DIR / "selection.json")
    labels = common.load_dev_labels()
    folds = common.load_folds()
    oof = common.read_csv(C.CV_DIR / "oof_primary.csv")
    prim = sel["primary"]
    print(f"selected model: {prim['model']} | augmentation {prim['setting']} | "
          f"params {prim['params']}  (confirmed: {sel.get('confirmed', False)})")

    # ---- per-image OOF summary over the CV repeats ---------------------------------------
    per = oof.groupby("id")["prob"].agg(["mean", "std", "count"]).rename(
        columns={"mean": "prob_mean", "std": "prob_sd", "count": "n_repeats"})
    per["label"] = labels.reindex(per.index).values
    wrong_rep = oof.assign(w=((oof["prob"] >= 0.5).astype(int) != oof["y"]))
    per["n_repeats_wrong"] = wrong_rep.groupby("id")["w"].sum()
    per["pred_05"] = (per["prob_mean"] >= 0.5).astype(int)
    mis = per[per["pred_05"] != per["label"]].sort_values("prob_mean")
    print(f"\nmisclassified at 0.5 (mean OOF probability over {int(per['n_repeats'].max())} "
          f"repeats): {len(mis)} of {len(per)} DEV images")
    print(f"  {'id':12s} {'true':>4s} {'p_mean':>7s} {'p_sd':>6s} {'wrong in':>9s}")
    for i, r in mis.iterrows():
        print(f"  {i:12s} {int(r['label']):4d} {r['prob_mean']:7.3f} {r['prob_sd']:6.3f} "
              f"{int(r['n_repeats_wrong']):4d}/{int(r['n_repeats'])}")

    # ---- misclassified grid (DEV only, captioned with the original filenames) -----------
    names = V.original_names(list(mis.index))
    items, titles, borders = [], [], []
    for i, r in mis.iterrows():
        items += [common.load_image(i), FH.render_overlay_for_id(i)]
        cap = (f"{i} | true {V.label_str(r['label'])}\np = {r['prob_mean']:.2f} +/- "
               f"{r['prob_sd']:.2f}\n{names[i]}")
        titles += [cap, "features overlay"]
        borders += ["red", None]
    if items:
        V.save_fig(V.show_grid(items, titles, ncols=8, tile_w=1.7, tile_h=2.9, fontsize=5.5,
                               borders=borders,
                               suptitle=f"Misclassified OOF images (DEV only) -- {prim['model']} "
                                        f"/ {prim['setting']}"),
                   fig_dir / "misclassified_oof_grid.png")

    # ---- shortcut checks ---------------------------------------------------------------
    fh = common.read_csv(C.FEATURES_HAND_CSV)
    fh = fh[fh["id"].isin(labels.index)]
    # The printed-scale check is only meaningful if the "2" mark is found in most images.
    tick_rate = float(np.isfinite(fh["sc_surface_rel_tick2"]).mean())
    sets = {"fill height": ["sc_fill_height_frac"],
            "raw brightness + raw white balance": ["sc_raw_L_frame", "sc_raw_L_background",
                                                   "sc_raw_rg_background",
                                                   "sc_raw_bg_background"]}
    if tick_rate >= 0.8:
        sets["surface vs printed '2' mark"] = ["sc_surface_rel_tick2"]
    else:
        print(f"\nprinted-scale position detected in only {tick_rate:.0%} of DEV images -> "
              "that shortcut check is skipped (not reliably detectable)")
    y = labels.reindex(fh["id"]).values
    prev = float(y.mean())
    print(f"\nSHORTCUT CHECKS (chance: ROC-AUC 0.5, PR-AUC = prevalence {prev:.2f}):")
    rows = []
    for name, cols in sets.items():
        res = _cv_shortcut_score(fh, cols, labels, folds)
        corr = {c: stats.pointbiserialr(y[np.isfinite(fh[c])], fh[c][np.isfinite(fh[c])])[0]
                for c in cols}
        # warning rule: clearly predictive, or significant and moderately predictive
        warn = res["roc_auc"] > 0.65 or (res["perm_p"] < 0.05 and res["roc_auc"] > 0.6)
        rows.append({"set": name, **res, "warning": warn})
        print(f"  {name:36s} ROC-AUC {res['roc_auc']:.3f} (+/- {res['roc_auc_sd']:.3f}) "
              f"PR-AUC {res['pr_auc']:.3f} | permutation null ROC mean "
              f"{res['null_roc_mean']:.3f}, 95th pct {res['null_roc_95']:.3f}, p = "
              f"{res['perm_p']:.3f}" + ("   <-- WARNING: well above chance" if warn else ""))
        for c, r in corr.items():
            print(f"      corr({c}, label) = {r:+.3f}")
    pd.DataFrame(rows).to_csv(C.CV_DIR / "shortcut_checks.csv", index=False)
    if any(r["warning"] for r in rows):
        print("WARNING: at least one shortcut-only feature set predicts the label well above "
              "chance. The models could exploit it -- look at the scatter plots below "
              "(OOF probability vs the shortcut) before trusting the comparison.")

    # OOF probability vs every shortcut feature
    sc_cols = [c for cols in sets.values() for c in cols]
    merged = fh.set_index("id").join(per[["prob_mean"]], how="inner")
    fig, axes = plt.subplots(1, len(sc_cols), figsize=(3.2 * len(sc_cols), 3.2), squeeze=False)
    for ax, c in zip(axes[0], sc_cols):
        for lab, col in [(0, "#4c72b0"), (1, "#dd8452")]:
            s = merged[labels.reindex(merged.index).values == lab]
            ax.scatter(s[c], s["prob_mean"], s=14, c=col, alpha=0.75, label=V.LABEL_NAMES[lab])
        ok = np.isfinite(merged[c])
        rho = stats.spearmanr(merged[c][ok], merged["prob_mean"][ok])[0]
        ax.set_title(f"{c}\nSpearman(OOF p, feature) = {rho:+.2f}", fontsize=7)
        ax.set_ylabel("mean OOF probability", fontsize=7)
        ax.tick_params(labelsize=6)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle("Shortcut check: does the selected model's OOF probability follow a "
                 "nuisance variable? (DEV only)", fontsize=9)
    fig.tight_layout()
    V.save_fig(fig, fig_dir / "shortcut_scatter.png")

    # Within-class check: an overall correlation can simply mean that the label and the
    # nuisance happen to overlap in this data set. Inside each class the label is constant, so
    # a correlation there means the model's output itself moves with the nuisance.
    yv = labels.reindex(merged.index).to_numpy()
    wc_rows = []
    print("\nWITHIN-CLASS check (Spearman rho between the selected model's mean OOF probability "
          "and each shortcut variable, computed inside each class):")
    for c in sc_cols:
        res = {}
        for lab in (0, 1):
            s = merged[yv == lab]
            ok = np.isfinite(s[c].to_numpy(float))
            rho, pv = stats.spearmanr(s[c][ok], s["prob_mean"][ok])
            res[lab] = (float(rho), float(pv), int(ok.sum()))
        follows = any(abs(r) >= 0.3 and p < 0.05 for r, p, _ in res.values())
        wc_rows.append({"variable": c, "rho_fail": res[0][0], "p_fail": res[0][1],
                        "n_fail": res[0][2], "rho_success": res[1][0], "p_success": res[1][1],
                        "n_success": res[1][2], "model_follows_nuisance": follows})
        print(f"  {c:24s} fails: rho {res[0][0]:+.2f} (p {res[0][1]:.2f}, n {res[0][2]}) | "
              f"successes: rho {res[1][0]:+.2f} (p {res[1][1]:.2f}, n {res[1][2]})"
              + ("   <-- model output follows this nuisance" if follows else ""))
    pd.DataFrame(wc_rows).to_csv(C.CV_DIR / "shortcut_within_class.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 3.4))
    xs = np.arange(len(rows))
    ax.bar(xs, [r["roc_auc"] for r in rows], yerr=[r["roc_auc_sd"] for r in rows],
           color=["#c44e52" if r["warning"] else "#8c8c8c" for r in rows], capsize=4)
    ax.scatter(xs, [r["null_roc_95"] for r in rows], marker="_", s=600, color="k",
               label="95th pct of label-permutation null", zorder=3)
    ax.axhline(0.5, color="k", ls=":", lw=1, label="chance")
    ax.set_xticks(xs, [r["set"] for r in rows], fontsize=7)
    ax.set_ylabel("CV ROC-AUC (shortcut features only)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    V.save_fig(fig, fig_dir / "shortcut_scores.png")

    # ---- review list (labels are NOT changed) --------------------------------------------
    conf_wrong = per[((per["label"] == 1) & (per["prob_mean"] < 0.2)) |
                     ((per["label"] == 0) & (per["prob_mean"] > 0.8))].assign(
        reason="confident wrong prediction (possible label error or atypical image)")
    border = per[(per["prob_mean"] >= 0.4) & (per["prob_mean"] <= 0.6)].assign(
        reason="borderline (mean OOF probability 0.4-0.6)")
    review = pd.concat([conf_wrong, border])
    review = review[~review.index.duplicated()].sort_values("prob_mean")
    review.index.name = "id"
    review[["label", "prob_mean", "prob_sd", "n_repeats_wrong", "n_repeats", "reason"]].round(
        4).to_csv(C.REPORTS_DIR / "review_labels.csv")
    print(f"\nwrote review_labels.csv: {len(conf_wrong)} confident-wrong + {len(border)} "
          "borderline image(s) for manual inspection. No label was changed.")
    V.update_progression(STAGE)
    common.stop_here(STAGE, next_cmd="threshold.py")


if __name__ == "__main__":
    main()
