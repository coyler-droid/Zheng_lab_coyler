"""
Stage 07 -- model comparison by repeated StratifiedGroupKFold on the DEV set (test set unused).

Models
  Six models (config.MODELS) = {hand features, CLIP+PCA, hand + CLIP+PCA} x {logistic
  regression (LR), random forest (RF)}, named "<feature set>_<LR|RF>" (e.g. "hand+clip_RF").
  Each runs under three augmentation settings (config.AUG_SETTINGS = NONE, LIGHT, HEAVY) on
  IDENTICAL folds, so every comparison is paired. Two baselines (setting NONE only):
  "always_fail" (probability 0 for every image) and "brown_only_LR", a one-feature logistic
  regression on the brown fraction.

Leakage control -- fitted inside every training fold only (never on held-out or test images):
  * the brown colour range (LAB percentiles of the lowest band of the fold's training
    successes) -> frac_brown / height_brown for every image of that fold
  * median imputation, StandardScaler, PCA on the CLIP vectors, the classifier
  * augmented views come from the fold's training images only; held-out images are never
    augmented
  class_weight="balanced" everywhere; no SMOTE / synthetic oversampling.

Selection rules (fixed before looking at the results)
  1. per model and setting: the hyper-parameters with the best mean per-fold PR-AUC
     (ties: mean F1, then the simpler hyper-parameters, see complexity())
  2. per model, the augmentation decision: the setting with the best mean per-fold PR-AUC
     (ties: NONE); a LIGHT or HEAVY winner is kept only if its mean paired per-fold gain over
     NONE is at least one SD of those paired differences, otherwise NONE
  3. between models (each at its rule-2 setting): best mean PR-AUC, then F1; a simpler model
     (config.MODEL_SIMPLICITY_ORDER, simplest first) is preferred when the mean paired per-fold
     PR-AUC difference to the best model is smaller than its SD
  PRIMARY = the result. SECONDARY = same model and hyper-parameters, opposite augmentation
  (NONE if the primary is augmented, otherwise HEAVY); reported for information only.

Optional nested CV (config.NESTED_CV, default off) adds an honest estimate in which the
hyper-parameters are chosen by an inner StratifiedGroupKFold inside every outer training fold.
It does not influence the selection.

Inputs
  features_hand.csv, features_hand_aug.csv, the normalized originals and augmented views,
  the CLIP cache (features_clip.py / augment.py), DEV labels, splits and CV folds.

Outputs (cv_results/)
  color_ranges_per_fold.json        fold-derived colour ranges
  oof_predictions.csv.gz            every held-out prediction of every model / setting / params
  fold_metrics.csv                  metrics per (model, setting, params, repeat, fold)
  summary_all_params.csv            mean / SD over folds, pooled-per-repeat PR-AUC / F1, counts
  augmentation_comparison.csv       paired LIGHT/HEAVY - NONE differences per model
  selection.json                    PRIMARY / SECONDARY and the evidence behind them
  oof_primary.csv, oof_secondary.csv
  oof_predictions_nested.csv.gz, nested_param_choices.csv     (only with NESTED_CV)
  figures/07_model_comparison/      model_comparison_bars.png, pr_curves_oof.png,
                                    augmentation_vs_none.png

This module also provides the final-model API used by freeze.py, final_test.py and
predict.py (DevData, fit_bundle, featurize_raw, bundle_* helpers, decide).

Usage:
  python train_cv.py                       # full comparison (a few minutes on CPU)
  python train_cv.py --confirm-selection   # mark the printed selection as confirmed
"""
from __future__ import annotations

import argparse
import datetime
import itertools
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import common
import config as C
import evaluate as E
import features_clip as FC
import features_hand as FH

STAGE = "07_model_comparison"
MODELS = list(C.MODELS)
SEL_JSON = C.CV_DIR / "selection.json"

# Naming used throughout:
#   fs = feature set ("hand" | "clip" | "hand+clip"), mt = model type ("LR" | "RF")
#   key = (image id, aug) with aug 0 = original image, aug k > 0 = k-th augmented view
#   r, f = CV repeat and fold; s / setting = augmentation setting
#   p = hyper-parameter dict, pk = its canonical JSON string (params_key)
#   result row = (model, setting, params_key, repeat, fold, held-out probabilities)


# =============================================================================
# Model zoo: feature sets, hyper-parameter grids, estimators
# =============================================================================
def split_model(name: str):
    """'hand+clip_RF' -> ('hand+clip', 'RF')."""
    fs, mt = name.rsplit("_", 1)
    return fs, mt


def uses_clip(fs: str) -> bool:
    return fs in ("clip", "hand+clip")


def uses_hand(fs: str) -> bool:
    return fs in ("hand", "hand+clip")


def needs_clip_image(model: str) -> bool:
    return uses_clip(split_model(model)[0])


def param_grid(model: str) -> list[dict]:
    """All hyper-parameter combinations of a model; CLIP models also search n_pca."""
    fs, mt = split_model(model)
    if mt == "LR":
        base = [{"C": c} for c in C.LR_C_GRID]
    else:
        g = C.RF_GRID
        base = [dict(n_estimators=n, max_depth=d, min_samples_leaf=m) for n, d, m in
                itertools.product(g["n_estimators"], g["max_depth"], g["min_samples_leaf"])]
    if uses_clip(fs):
        return [dict(p, n_pca=k) for p in base for k in C.PCA_COMPONENTS_GRID]
    return base


def params_key(p: dict) -> str:
    """Canonical string of a hyper-parameter dict (used as the 'params' column)."""
    return json.dumps(p, sort_keys=True)


def complexity(p: dict):
    """Smaller = simpler (only used to break exact ties).

    Order of importance: fewer PCA components, stronger LR regularization (smaller C), fewer
    trees, shallower trees (None = unlimited counts as 99), larger leaves.
    """
    return (p.get("n_pca", 0), p.get("C", 0), p.get("n_estimators", 0),
            p.get("max_depth") or 99, -p.get("min_samples_leaf", 0))


def make_estimator(mt: str, p: dict):
    """Unfitted pipeline: median imputation (+ scaling for LR) and a balanced classifier."""
    if mt == "LR":
        clf = LogisticRegression(C=p["C"], class_weight="balanced", max_iter=5000)
        return Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler()), ("clf", clf)])
    clf = RandomForestClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                                 min_samples_leaf=p["min_samples_leaf"],
                                 class_weight="balanced", random_state=C.SEED, n_jobs=1)
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("clf", clf)])


def assemble(fs, p, task, Z):
    """Feature matrices of one feature set; Z = (train, held-out) PCA scores of the CLIP
    vectors (fitted on the training rows). The first p['n_pca'] columns are used."""
    tr, te = [], []
    if uses_hand(fs):
        tr.append(task["Xh_tr"])
        te.append(task["Xh_te"])
    if uses_clip(fs):
        tr.append(Z[0][:, :p["n_pca"]])
        te.append(Z[1][:, :p["n_pca"]])
    return np.hstack(tr), np.hstack(te)


# =============================================================================
# Data: DEV rows (originals + augmented views) and per-fold tasks
# =============================================================================
def norm_path(key):
    """Normalized image of a key: originals in processed/, augmented views in cache/."""
    i, k = key
    if k == 0:
        return C.PROCESSED_DIR / "normalized" / f"{i}.png"
    return C.CACHE_DIR / "aug_normalized" / f"{i}_aug{k}.png"


def views_for(setting: str, label: int):
    """Augmented view indices a training image of `label` uses under `setting`."""
    return range(1, C.AUG_VIEWS_PER_CLASS[setting][int(label)] + 1)


def training_keys(train_ids, labels, setting: str):
    """Training rows: all originals first, then each image's augmented views."""
    keys = [(i, 0) for i in train_ids]
    for i in train_ids:
        keys += [(i, k) for k in views_for(setting, labels[i])]
    return keys


class DevData:
    """Everything the CV needs, loaded once: labels, fixed hand features, liquid pixels
    (for the fold-derived colour features) and CLIP vectors, for originals + augmented views.

    need_aug=False loads the originals only; need_clip=False skips the CLIP cache.
    """

    def __init__(self, need_aug=True, need_clip=True):
        self.labels = common.load_dev_labels()
        self.tube = np.load(C.TUBE_MASK_NPY)
        frames = [common.read_csv(C.FEATURES_HAND_CSV)]
        if need_aug:
            frames.append(common.read_csv(C.FEATURES_HAND_AUG_CSV))
        recs = pd.concat(frames, ignore_index=True)
        recs = recs[recs["id"].isin(self.labels.index)].copy()
        recs["aug"] = recs["aug"].astype(int)
        self.keys = list(zip(recs["id"], recs["aug"]))
        self.index = pd.MultiIndex.from_tuples(self.keys, names=["id", "aug"])
        self.fixed = pd.DataFrame(recs[FH.fixed_feature_names()].to_numpy(float),
                                  index=self.index, columns=FH.fixed_feature_names())
        # liquid pixels per key, kept compact (float16) because they are reused for every fold
        self.px = {}
        for key, surface_y, tip_y in zip(self.keys, recs["surface_y"], recs["tip_y"]):
            lab = FH.to_lab(common.read_rgb(norm_path(key)))
            lab_px, relh, rows = FH.liquid_pixels(lab, self.tube, int(surface_y), int(tip_y))
            self.px[key] = (lab_px.astype(np.float16), relh.astype(np.float16),
                            rows.astype(np.int32))
        # colour features of the label-free (fixed) ranges can be computed once
        fixed_ranges = {n: {k: r[k] for k in ("space", "lo", "hi")}
                        for n, r in C.COLOR_RANGES.items() if r.get("derive") != "fold"}
        self.fixed_color = pd.DataFrame([FH.color_features(*self.px[k], fixed_ranges)
                                         for k in self.keys], index=self.index)
        self.clip = FC.load_cached(self.keys) if need_clip else {}

    def hand_matrix(self, ranges: dict, keys=None) -> pd.DataFrame:
        """Model hand features for `keys` (default: all) with the given resolved ranges."""
        keys = self.keys if keys is None else keys
        derived = {n: r for n, r in ranges.items() if C.COLOR_RANGES[n].get("derive") == "fold"}
        idx = pd.MultiIndex.from_tuples(keys, names=["id", "aug"])
        fold_part = pd.DataFrame([FH.color_features(*self.px[k], derived) for k in keys],
                                 index=idx)
        out = pd.concat([self.fixed.loc[keys], self.fixed_color.loc[keys], fold_part], axis=1)
        return out[FH.model_feature_names()]

    def ranges_for(self, train_ids, tag: str) -> dict:
        """Colour ranges derived from the ORIGINAL images of `train_ids` only."""
        by_label = {lab: [(i, 0) for i in train_ids if self.labels[i] == lab] for lab in (0, 1)}
        return FH.derive_color_ranges(by_label, self.px, tag=tag)


def make_task(data: DevData, hand: pd.DataFrame, r, f, train_ids, held_ids, setting):
    """Arrays of one (repeat, fold, setting): training rows (with augmented views per
    `setting`) and held-out originals. `hand` must use the fold's own colour ranges."""
    tr = training_keys(train_ids, data.labels, setting)
    te = [(i, 0) for i in held_ids]
    return {"r": r, "f": f, "setting": setting, "ids_te": list(held_ids),
            "Xh_tr": hand.loc[tr].to_numpy(float), "Xh_te": hand.loc[te].to_numpy(float),
            "Xc_tr": np.stack([data.clip[k] for k in tr]),
            "Xc_te": np.stack([data.clip[k] for k in te]),
            "y_tr": np.array([data.labels[i] for i, _ in tr]),
            "y_te": np.array([data.labels[i] for i in held_ids]),
            "brown_tr": hand.loc[tr, "frac_brown"].to_numpy(float),
            "brown_te": hand.loc[te, "frac_brown"].to_numpy(float)}


# =============================================================================
# Fitting: one CV task (runs inside a joblib worker)
# =============================================================================
def pca_parts(task: dict):
    """PCA fitted on the task's TRAINING rows only (first k components = PCA(k)).

    One PCA with the largest grid value is fitted per task and truncated per n_pca; with
    svd_solver="full" the leading components are the same as those of a smaller PCA.
    """
    pca = PCA(n_components=max(C.PCA_COMPONENTS_GRID), svd_solver="full").fit(task["Xc_tr"])
    return pca.transform(task["Xc_tr"]), pca.transform(task["Xc_te"])


def fit_predict(task: dict, model: str, p: dict, Z) -> np.ndarray:
    """Fit one model on the task's training rows; success probabilities of held-out rows."""
    fs, mt = split_model(model)
    Xtr, Xte = assemble(fs, p, task, Z)
    return make_estimator(mt, p).fit(Xtr, task["y_tr"]).predict_proba(Xte)[:, 1]


def run_grid(task: dict, models=None, baselines=True):
    """Fit every (model, hyper-parameter) on the task's training rows; predict held-out.

    Returns a list of result rows; the baselines are added for setting NONE only.
    """
    models = MODELS if models is None else models
    out = []
    Z = pca_parts(task) if any(uses_clip(split_model(m)[0]) for m in models) else (None, None)
    for model in models:
        for p in param_grid(model):
            out.append((model, task["setting"], params_key(p), task["r"], task["f"],
                        fit_predict(task, model, p, Z)))
    if baselines and task["setting"] == "NONE":
        out.append(("always_fail", "NONE", "{}", task["r"], task["f"],
                    np.zeros(len(task["y_te"]))))
        est = make_estimator("LR", {"C": 1.0}).fit(task["brown_tr"][:, None], task["y_tr"])
        out.append(("brown_only_LR", "NONE", params_key({"C": 1.0}), task["r"], task["f"],
                    est.predict_proba(task["brown_te"][:, None])[:, 1]))
    return out


def run_nested(outer: dict, inner_tasks: list[dict]):
    """Nested CV: choose hyper-parameters per model on inner folds, refit on the outer
    training part, predict the outer held-out fold.

    Returns (result rows with params "nested", {model: chosen params}).
    """
    out, chosen = [], {}
    inner_scores = {}
    for it in inner_tasks:
        for model, _, pk, _, _, prob in run_grid(it, baselines=False):
            inner_scores.setdefault((model, pk), []).append(E.pr_auc(it["y_te"], prob))
    Z = pca_parts(outer)
    for model in MODELS:
        cands = [(np.nanmean(inner_scores[(model, params_key(p))]), p) for p in param_grid(model)]
        best = sorted(cands, key=lambda c: (-c[0], complexity(c[1])))[0][1]
        chosen[model] = best
        out.append((model, outer["setting"], "nested", outer["r"], outer["f"],
                    fit_predict(outer, model, best, Z)))
    return out, chosen


def predictions_frame(rows, tasks_by_rf):
    """Result rows -> long table, one line per held-out image and result row."""
    recs = []
    for model, setting, pk, r, f, prob in rows:
        ids = tasks_by_rf[(r, f)]["ids_te"]
        y = tasks_by_rf[(r, f)]["y_te"]
        recs.append(pd.DataFrame({"model": model, "setting": setting, "params": pk, "repeat": r,
                                  "fold": f, "id": ids, "y": y, "prob": prob}))
    return pd.concat(recs, ignore_index=True)


# =============================================================================
# Metrics + selection rules
# =============================================================================
def metric_tables(pred: pd.DataFrame):
    """(per-fold metrics, per-repeat pooled metrics, summary per model/setting/params).

    The summary holds mean and SD over folds, the mean and SD over repeats of the PR-AUC / F1
    of each repeat's pooled OOF predictions, and confusion counts summed over all folds.
    """
    keys = ["model", "setting", "params"]
    per_fold = [{**dict(zip(keys + ["repeat", "fold"], k)), **E.binary_metrics(g["y"], g["prob"])}
                for k, g in pred.groupby(keys + ["repeat", "fold"], sort=True)]
    per_fold = pd.DataFrame(per_fold)
    per_repeat = [{**dict(zip(keys + ["repeat"], k)), **E.binary_metrics(g["y"], g["prob"])}
                  for k, g in pred.groupby(keys + ["repeat"], sort=True)]
    per_repeat = pd.DataFrame(per_repeat)
    agg = {f"{m}_{s}": (m, "mean" if s == "mean" else (lambda x: x.std(ddof=1)))
           for m in ("pr_auc", "f1", "precision", "recall", "roc_auc") for s in ("mean", "sd")}
    summ = per_fold.groupby(keys).agg(**agg)
    pooled = per_repeat.groupby(keys).agg(pooled_pr_auc_mean=("pr_auc", "mean"),
                                          pooled_pr_auc_sd=("pr_auc", lambda x: x.std(ddof=1)),
                                          pooled_f1_mean=("f1", "mean"),
                                          pooled_f1_sd=("f1", lambda x: x.std(ddof=1)))
    summ = summ.join(pooled).reset_index()
    counts = per_fold.groupby(keys)[["tn", "fp", "fn", "tp"]].sum().reset_index()
    return per_fold, per_repeat, summ.merge(counts, on=keys)


def fold_vec(fm, model, setting, pk, metric="pr_auc"):
    """Per-fold metric vector in (repeat, fold) order -> pairs across models / settings."""
    g = fm[(fm["model"] == model) & (fm["setting"] == setting) & (fm["params"] == pk)]
    return g.sort_values(["repeat", "fold"])[metric].to_numpy(float)


def select(fm: pd.DataFrame, summ: pd.DataFrame):
    """Apply selection rules 1-3 (module docstring).

    Returns (selection dict for selection.json, augmentation comparison table,
    best_p = {(model, setting): best params_key}).
    """
    sub = summ[summ["model"].isin(MODELS)]

    # rule 1: best hyper-parameters per (model, setting)
    best_p = {}
    for (m, s), g in sub.groupby(["model", "setting"]):
        g = g.assign(cx=[complexity(json.loads(k)) if k != "nested" else ()
                         for k in g["params"]])
        g = g.sort_values(["pr_auc_mean", "f1_mean", "cx"], ascending=[False, False, True])
        best_p[(m, s)] = g.iloc[0]["params"]

    def row(m, s):
        return sub[(sub["model"] == m) & (sub["setting"] == s) &
                   (sub["params"] == best_p[(m, s)])].iloc[0]

    # rule 2: augmentation decision per model (paired per-fold differences vs NONE)
    aug_rows, chosen = [], {}
    for m in MODELS:
        pr_folds = {s: fold_vec(fm, m, s, best_p[(m, s)]) for s in C.AUG_SETTINGS}
        f1_folds = {s: fold_vec(fm, m, s, best_p[(m, s)], "f1") for s in C.AUG_SETTINGS}
        means = {s: float(np.mean(pr_folds[s])) for s in C.AUG_SETTINGS}
        best_s = max(C.AUG_SETTINGS, key=lambda s: (means[s], s == "NONE"))
        decision = "NONE"
        for s in C.AUG_SETTINGS:
            if s == "NONE":
                continue
            d_pr = pr_folds[s] - pr_folds["NONE"]
            d_f1 = f1_folds[s] - f1_folds["NONE"]
            mpr, sdpr, lo, hi = E.corrected_ci(d_pr)
            mf1, sdf1, lo1, hi1 = E.corrected_ci(d_f1)
            aug_rows.append({"model": m, "comparison": f"{s} - NONE",
                             "d_pr_auc_mean": mpr, "d_pr_auc_sd": sdpr, "d_pr_auc_lo95": lo,
                             "d_pr_auc_hi95": hi, "d_f1_mean": mf1, "d_f1_sd": sdf1,
                             "d_f1_lo95": lo1, "d_f1_hi95": hi1})
            if s == best_s:
                decision = s if mpr >= sdpr else "NONE"
        chosen[m] = decision

    # rule 3: best model, then prefer a simpler one within one SD of the paired differences
    cands = {m: row(m, chosen[m]) for m in MODELS}
    best = max(MODELS, key=lambda m: (cands[m]["pr_auc_mean"], cands[m]["f1_mean"]))
    v_best = fold_vec(fm, best, chosen[best], best_p[(best, chosen[best])])
    selected, why = best, f"{best} has the best mean PR-AUC"
    for m in C.MODEL_SIMPLICITY_ORDER:
        if m == best:
            break
        d = v_best - fold_vec(fm, m, chosen[m], best_p[(m, chosen[m])])
        if d.mean() < d.std(ddof=1):
            selected = m
            why = (f"{m} is simpler than the best model {best} and within one SD of the paired "
                   f"per-fold PR-AUC differences (mean diff {d.mean():.3f} < SD "
                   f"{d.std(ddof=1):.3f})")
            break
    prim_setting = chosen[selected]
    sec_setting = "NONE" if prim_setting != "NONE" else "HEAVY"
    pk = best_p[(selected, prim_setting)]

    def info(m, s, p):
        r = sub[(sub["model"] == m) & (sub["setting"] == s) & (sub["params"] == p)].iloc[0]
        return {"model": m, "setting": s, "params": json.loads(p) if p != "nested" else p,
                "params_key": p, "pr_auc_mean": r["pr_auc_mean"], "pr_auc_sd": r["pr_auc_sd"],
                "f1_mean": r["f1_mean"], "f1_sd": r["f1_sd"],
                "pooled_pr_auc_mean": r["pooled_pr_auc_mean"]}

    # alternative readings of the rules, reported for transparency only
    none_best = max(MODELS, key=lambda m: (row(m, "NONE")["pr_auc_mean"],
                                           row(m, "NONE")["f1_mean"]))
    joint = max(((m, s) for m in MODELS for s in C.AUG_SETTINGS),
                key=lambda ms: (row(*ms)["pr_auc_mean"], row(*ms)["f1_mean"]))
    return {
        "primary": info(selected, prim_setting, pk),
        "secondary": info(selected, sec_setting, pk),
        "reason": why,
        "best_by_pr_auc": best,
        "aug_decision_per_model": chosen,
        "best_params": {f"{m}|{s}": json.loads(p) if p != "nested" else p
                        for (m, s), p in best_p.items()},
        "alternative_readings": {"best_model_under_NONE_only": none_best,
                                 "best_joint_model_and_setting": list(joint)},
        "rules": "rule1 best params by mean per-fold PR-AUC; rule2 augmentation: best mean "
                 "PR-AUC, NONE unless the gain >= 1 SD of paired per-fold differences; rule3 "
                 "best model by PR-AUC then F1, simpler model preferred within 1 SD of paired "
                 "per-fold differences",
    }, pd.DataFrame(aug_rows), best_p


# =============================================================================
# Figures
# =============================================================================
def make_figures(fig_dir, pred, summ, aug_df, sel, best_p):
    """Model comparison bars, pooled OOF PR curves, augmentation vs NONE."""
    import visualize as V
    chosen = sel["aug_decision_per_model"]

    # (1) every model at its rule-2 setting + baselines
    labels_x = MODELS + ["brown_only_LR", "always_fail"]
    rows = []
    for m in MODELS:
        s = chosen[m]
        rows.append(summ[(summ["model"] == m) & (summ["setting"] == s) &
                         (summ["params"] == best_p[(m, s)])].iloc[0])
    for b in ("brown_only_LR", "always_fail"):
        rows.append(summ[summ["model"] == b].iloc[0])
    prev = float(pred.drop_duplicates(["id"])["y"].mean())
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, met in zip(axes, ("pr_auc", "f1")):
        vals = [r[f"{met}_mean"] for r in rows]
        sds = [r[f"{met}_sd"] for r in rows]
        cols = ["#dd8452" if m == sel["primary"]["model"] else
                ("#4c72b0" if m in MODELS else "#9a9a9a") for m in labels_x]
        ax.bar(range(len(rows)), vals, yerr=sds, color=cols, capsize=3)
        ax.set_xticks(range(len(rows)),
                      [f"{m}\n({chosen[m]})" if m in MODELS else m for m in labels_x],
                      fontsize=6.5, rotation=30, ha="right")
        if met == "pr_auc":
            ax.axhline(prev, color="k", ls=":", lw=1, label=f"chance = prevalence {prev:.2f}")
            ax.legend(fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{met.upper()} (mean +/- SD over {C.N_REPEATS} repeats x {C.N_FOLDS} "
                     "folds)", fontsize=9)
    fig.suptitle(f"Model comparison on DEV (orange = selected primary: "
                 f"{sel['primary']['model']} / {sel['primary']['setting']})", fontsize=10)
    fig.tight_layout()
    V.save_fig(fig, fig_dir / "model_comparison_bars.png")

    # (2) PR curves (pooled OOF over all repeats)
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    for m in MODELS + ["brown_only_LR"]:
        s = chosen.get(m, "NONE")
        pk = best_p.get((m, s), params_key({"C": 1.0}))
        g = pred[(pred["model"] == m) & (pred["setting"] == s) & (pred["params"] == pk)]
        pr, rc, _ = precision_recall_curve(g["y"], g["prob"])
        lw = 2.5 if m == sel["primary"]["model"] else 1.2
        ls = "--" if m == "brown_only_LR" else "-"
        ax.plot(rc, pr, lw=lw, ls=ls, label=f"{m} ({s}) AP={E.pr_auc(g['y'], g['prob']):.2f}")
    ax.axhline(prev, color="k", ls=":", lw=1, label=f"always fail / chance ({prev:.2f})")
    ax.set_xlabel("recall (success)")
    ax.set_ylabel("precision (success)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=6.5, loc="lower left")
    ax.set_title("Pooled OOF precision-recall curves (DEV)", fontsize=10)
    V.save_fig(fig, fig_dir / "pr_curves_oof.png")

    # (3) augmentation vs no augmentation (bars assume the three settings NONE/LIGHT/HEAVY)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    w = 0.26
    colors = {"NONE": "#8c8c8c", "LIGHT": "#64b5cd", "HEAVY": "#4c72b0"}
    for ax, met in zip(axes[:2], ("pr_auc", "f1")):
        for k, s in enumerate(C.AUG_SETTINGS):
            vals, sds = [], []
            for m in MODELS:
                r = summ[(summ["model"] == m) & (summ["setting"] == s) &
                         (summ["params"] == best_p[(m, s)])].iloc[0]
                vals.append(r[f"{met}_mean"])
                sds.append(r[f"{met}_sd"])
            ax.bar(np.arange(len(MODELS)) + (k - 1) * w, vals, w, yerr=sds, capsize=2,
                   color=colors[s], label=s)
        ax.set_xticks(range(len(MODELS)), [f"{m}\nchosen: {chosen[m]}" for m in MODELS],
                      fontsize=6.5)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{met.upper()} by augmentation setting (paired, identical folds)",
                     fontsize=9)
        ax.legend(fontsize=7)
    ax = axes[2]
    for k, comp in enumerate(["LIGHT - NONE", "HEAVY - NONE"]):
        d = aug_df[aug_df["comparison"] == comp].set_index("model").loc[MODELS]
        x = np.arange(len(MODELS)) + (k - 0.5) * 0.3
        ax.errorbar(x, d["d_pr_auc_mean"],
                    yerr=[d["d_pr_auc_mean"] - d["d_pr_auc_lo95"],
                          d["d_pr_auc_hi95"] - d["d_pr_auc_mean"]],
                    fmt="o", capsize=3, label=f"{comp} (95% CI, corrected)",
                    color=colors[comp.split()[0]])
        ax.scatter(x, d["d_pr_auc_sd"], marker="_", s=120, color="red",
                   label="1 SD of paired diffs" if k == 0 else None)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(MODELS)), MODELS, fontsize=6.5, rotation=20)
    ax.set_title("Paired per-fold PR-AUC difference vs NONE", fontsize=9)
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    V.save_fig(fig, fig_dir / "augmentation_vs_none.png")


# =============================================================================
# Final models -- shared by freeze.py, final_test.py and predict.py
# =============================================================================
def fit_bundle(data: DevData, train_ids, model: str, setting: str, params: dict) -> dict:
    """Fit colour ranges, PCA, scaler and classifier on `train_ids` (+ their augmented views).

    Same recipe as one CV training fold; here the PCA is fitted directly with
    params['n_pca'] components.
    """
    fs, mt = split_model(model)
    ranges = data.ranges_for(train_ids, tag="final")
    keys = training_keys(train_ids, data.labels, setting)
    y = np.array([data.labels[i] for i, _ in keys])
    parts, pca = [], None
    if uses_hand(fs):
        parts.append(data.hand_matrix(ranges, keys).to_numpy(float))
    if uses_clip(fs):
        Xc = np.stack([data.clip[k] for k in keys])
        pca = PCA(n_components=params["n_pca"], svd_solver="full").fit(Xc)
        parts.append(pca.transform(Xc))
    est = make_estimator(mt, params).fit(np.hstack(parts), y)
    return {"model": model, "setting": setting, "params": params,
            "uses_hand": uses_hand(fs), "uses_clip": uses_clip(fs),
            "hand_cols": FH.model_feature_names() if uses_hand(fs) else [],
            "pca": pca, "estimator": est, "color_ranges": ranges,
            "clip_model": [C.CLIP_MODEL_NAME, C.CLIP_PRETRAINED],
            "n_train_rows": len(keys), "n_train_images": len(train_ids),
            "n_train_success_images": int(sum(data.labels[i] for i in train_ids))}


def featurize_raw(raw: np.ndarray, tube, ref, img_id: str, embedder=None) -> dict:
    """Exactly the DEV preprocessing for one raw image: normalize + blank, hand record,
    liquid pixels and (if an embedder is given) the CLIP vector."""
    import preprocess
    if raw.shape[:2] != (C.IMAGE_H, C.IMAGE_W):
        raise ValueError(f"image is {raw.shape[1]}x{raw.shape[0]}, expected "
                         f"{C.IMAGE_W}x{C.IMAGE_H} (the fixed mask needs identical framing)")
    masked, norm, _, st = preprocess.process_raw(raw, tube, ref)
    rec, px = FH.extract_record(norm, tube, raw, st, img_id, 0)
    clipv = None
    if embedder is not None:
        clipv = embedder.embed_views([FC.make_views(norm, tube)])[0].reshape(-1)
    return {"masked": masked, "norm": norm, "rec": rec, "px": px, "clip": clipv}


def bundle_hand_row(bundle: dict, rec: dict, px) -> np.ndarray:
    """Hand-feature row of one image, using the bundle's frozen colour ranges."""
    vals = {**rec, **FH.color_features(*px, bundle["color_ranges"])}
    return np.array([vals[c] for c in bundle["hand_cols"]], float)


def bundle_needs_clip(bundle: dict) -> bool:
    return bool(bundle["uses_clip"])


def bundle_predict(bundle: dict, X_hand, X_clip) -> np.ndarray:
    """Success probabilities from hand rows and/or raw CLIP vectors (PCA applied here)."""
    parts = []
    if bundle["uses_hand"]:
        parts.append(np.asarray(X_hand, float))
    if bundle["uses_clip"]:
        parts.append(bundle["pca"].transform(np.asarray(X_clip, np.float32)))
    return bundle["estimator"].predict_proba(np.hstack(parts))[:, 1]


def decide(prob: float, threshold: float, band) -> tuple[str, str, bool]:
    """(call YES/NO/REVIEW, YES/NO at the threshold, REVIEW flag). band None = no REVIEW."""
    yesno = "YES" if prob >= threshold else "NO"
    review = bool(band is not None and band[0] <= prob <= band[1])
    return ("REVIEW" if review else yesno), yesno, review


# =============================================================================
# CV driver
# =============================================================================
def run_cv(data: DevData):
    """Build all (repeat, fold, setting) tasks, fit the full grid in parallel, and
    optionally run the nested CV.

    Returns (result rows, nested result rows, nested parameter choices, tasks by (r, f),
    fold colour ranges).
    """
    folds = common.load_folds()
    splits = common.load_splits().set_index("id")
    tasks, fold_ranges = [], {}
    t0 = time.time()
    for r, f, train, held in common.iter_cv_folds(folds):
        ranges = data.ranges_for(train, tag=f"cv{r}-{f}")
        fold_ranges[f"repeat{r}_fold{f}"] = ranges
        hand = data.hand_matrix(ranges)
        for s in C.AUG_SETTINGS:
            tasks.append(make_task(data, hand, r, f, train, held, s))
    print(f"built {len(tasks)} fold x setting tasks ({time.time() - t0:.0f} s, fold-specific "
          "brown ranges derived from each fold's training successes)")
    # held-out ids / labels are the same for every setting of a fold, so any task will do
    by_rf = {(t["r"], t["f"]): t for t in tasks}
    n_fits = sum(len(param_grid(m)) for m in MODELS) * len(tasks)
    print(f"fitting {n_fits} models on {C.N_JOBS} joblib workers ...")
    t0 = time.time()
    results = Parallel(n_jobs=C.N_JOBS)(delayed(run_grid)(t) for t in tasks)
    print(f"done in {time.time() - t0:.0f} s")
    rows = [row for res in results for row in res]
    nested_rows, nested_choice = [], []
    if C.NESTED_CV:
        print("NESTED_CV = True: inner StratifiedGroupKFold for every outer fold ...")
        jobs = []
        for r, f, train, held in common.iter_cv_folds(folds):
            inner_cv = StratifiedGroupKFold(C.NESTED_INNER_FOLDS, shuffle=True,
                                            random_state=C.SEED + 100 * r + f)
            tr_arr = np.array(train)
            inner_def = []
            for j, (inner_tr, inner_te) in enumerate(inner_cv.split(
                    tr_arr, data.labels[tr_arr].to_numpy(),
                    splits.loc[tr_arr, "group"].to_numpy())):
                # inner folds get their own colour ranges (from inner training images only)
                inner_ranges = data.ranges_for(list(tr_arr[inner_tr]), tag=f"nested{r}-{f}-{j}")
                inner_def.append((list(tr_arr[inner_tr]), list(tr_arr[inner_te]),
                                  data.hand_matrix(inner_ranges)))
            outer_hand = data.hand_matrix(fold_ranges[f"repeat{r}_fold{f}"])
            for s in C.AUG_SETTINGS:
                outer = make_task(data, outer_hand, r, f, train, held, s)
                inner = [make_task(data, inner_hand, r, f, tr_ids, te_ids, s)
                         for tr_ids, te_ids, inner_hand in inner_def]
                jobs.append((outer, inner))
        res = Parallel(n_jobs=C.N_JOBS)(delayed(run_nested)(o, i) for o, i in jobs)
        for (o, _), (nrows, ch) in zip(jobs, res):
            nested_rows += nrows
            nested_choice.append({"repeat": o["r"], "fold": o["f"], "setting": o["setting"],
                                  **{m: params_key(p) for m, p in ch.items()}})
    return rows, nested_rows, nested_choice, by_rf, fold_ranges


# =============================================================================
# Main
# =============================================================================
def confirm_selection():
    """Mark selection.json as confirmed (required by the later stages)."""
    common.require_files(SEL_JSON, hint="run `python train_cv.py` first")
    sel = common.json_load(SEL_JSON)
    sel["confirmed"] = True
    sel["confirmed_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    common.json_dump(sel, SEL_JSON)
    print(f"selection CONFIRMED: primary {sel['primary']['model']} / "
          f"{sel['primary']['setting']} {sel['primary']['params']}; secondary "
          f"{sel['secondary']['model']} / {sel['secondary']['setting']} (information only)")


def main():
    import visualize as V
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--confirm-selection", action="store_true")
    args = ap.parse_args()
    fig_dir = common.start_stage(STAGE)
    if args.confirm_selection:
        confirm_selection()
        common.stop_here(STAGE, next_cmd="evaluate.py")
        return
    common.require_mask_approved()
    common.require_files(C.FEATURES_HAND_CSV, C.FEATURES_HAND_AUG_CSV,
                         hint="run features_hand.py, features_clip.py and augment.py first")
    C.CV_DIR.mkdir(exist_ok=True)

    # ---------------- cross-validation + selection ----------------
    print("loading DEV data (originals + augmented views, liquid pixels, CLIP vectors) ...")
    data = DevData()
    print(f"  {len(data.keys)} rows ({data.labels.size} DEV images), "
          f"{len(FH.model_feature_names())} hand features, CLIP dim "
          f"{len(next(iter(data.clip.values())))}")
    rows, nested_rows, nested_choice, by_rf, fold_ranges = run_cv(data)
    common.json_dump(fold_ranges, C.CV_DIR / "color_ranges_per_fold.json")

    pred = predictions_frame(rows, by_rf)
    pred.to_csv(C.CV_DIR / "oof_predictions.csv.gz", index=False)
    fm, _, summ = metric_tables(pred)
    fm.to_csv(C.CV_DIR / "fold_metrics.csv", index=False)
    summ.to_csv(C.CV_DIR / "summary_all_params.csv", index=False)
    sel, aug_df, best_p = select(fm, summ)
    aug_df.to_csv(C.CV_DIR / "augmentation_comparison.csv", index=False)

    # ---------------- report ----------------
    pd.set_option("display.width", 200)
    print("\nPER MODEL AND AUGMENTATION SETTING (best hyper-parameters; mean +/- SD over "
          f"{C.N_REPEATS}x{C.N_FOLDS} folds; pooled = OOF of a whole repeat):")
    print(f"  {'model':14s} {'setting':7s} {'PR-AUC':>15s} {'F1@0.5':>15s} {'prec':>6s} "
          f"{'recall':>6s} {'pooled PR-AUC':>14s}  params")
    for m in MODELS:
        for s in C.AUG_SETTINGS:
            r = summ[(summ["model"] == m) & (summ["setting"] == s) &
                     (summ["params"] == best_p[(m, s)])].iloc[0]
            mark = "  <- rule 2" if sel["aug_decision_per_model"][m] == s else ""
            print(f"  {m:14s} {s:7s} {r['pr_auc_mean']:.3f} +/- {r['pr_auc_sd']:.3f} "
                  f"{r['f1_mean']:.3f} +/- {r['f1_sd']:.3f} {r['precision_mean']:6.3f} "
                  f"{r['recall_mean']:6.3f} {r['pooled_pr_auc_mean']:7.3f} +/- "
                  f"{r['pooled_pr_auc_sd']:.3f}  {best_p[(m, s)]}{mark}")
    print("\nBASELINES (NONE):")
    for b in ("always_fail", "brown_only_LR"):
        r = summ[summ["model"] == b].iloc[0]
        print(f"  {b:14s} PR-AUC {r['pr_auc_mean']:.3f} +/- {r['pr_auc_sd']:.3f} | F1 "
              f"{r['f1_mean']:.3f} +/- {r['f1_sd']:.3f}")
    print("\nAUGMENTATION vs NO AUGMENTATION (paired per-fold differences, 95% CI with the "
          "Nadeau-Bengio correction):")
    for _, r in aug_df.iterrows():
        print(f"  {r['model']:14s} {r['comparison']:13s} dPR-AUC {r['d_pr_auc_mean']:+.3f} "
              f"(SD {r['d_pr_auc_sd']:.3f}, 95% CI {r['d_pr_auc_lo95']:+.3f}.."
              f"{r['d_pr_auc_hi95']:+.3f})"
              f" | dF1 {r['d_f1_mean']:+.3f} (95% CI {r['d_f1_lo95']:+.3f}.."
              f"{r['d_f1_hi95']:+.3f})")
    print("  decision (rule 2) per model:", sel["aug_decision_per_model"])
    if C.NESTED_CV and nested_rows:
        npred = predictions_frame(nested_rows, by_rf)
        npred.to_csv(C.CV_DIR / "oof_predictions_nested.csv.gz", index=False)
        pd.DataFrame(nested_choice).to_csv(C.CV_DIR / "nested_param_choices.csv", index=False)
        _, _, nsum = metric_tables(npred)
        print("\nNESTED CV (hyper-parameters chosen inside each outer training fold):")
        for _, r in nsum.iterrows():
            print(f"  {r['model']:14s} {r['setting']:6s} PR-AUC {r['pr_auc_mean']:.3f} +/- "
                  f"{r['pr_auc_sd']:.3f} | F1 {r['f1_mean']:.3f} +/- {r['f1_sd']:.3f}")

    p, s2 = sel["primary"], sel["secondary"]
    print("\nSELECTION")
    print(f"  best by PR-AUC: {sel['best_by_pr_auc']} -> selected: {p['model']} "
          f"({sel['reason']})")
    print(f"  PRIMARY  : {p['model']} | augmentation {p['setting']} | {p['params']} | PR-AUC "
          f"{p['pr_auc_mean']:.3f} +/- {p['pr_auc_sd']:.3f} | F1 {p['f1_mean']:.3f} +/- "
          f"{p['f1_sd']:.3f}")
    print(f"  SECONDARY: {s2['model']} | augmentation {s2['setting']} | same hyper-parameters | "
          f"PR-AUC {s2['pr_auc_mean']:.3f} +/- {s2['pr_auc_sd']:.3f} (information only)")
    alt = sel["alternative_readings"]
    if (alt["best_model_under_NONE_only"] != p["model"]
            or alt["best_joint_model_and_setting"][0] != p["model"]):
        print(f"  note: other readings of the rules would give: best under NONE only = "
              f"{alt['best_model_under_NONE_only']}; best joint (model, setting) = "
              f"{alt['best_joint_model_and_setting']}")
    print("\nWARNINGS")
    print(f"  * DEV has only {int(data.labels.sum())} successes: differences smaller than the "
          "fold-to-fold SD are not meaningful.")
    print("  * OOF metrics that were used for selection are slightly optimistic; the sealed "
          "test set (stage 11) is the honest estimate. Optional nested CV: config.NESTED_CV.")
    print("  * Augmentation adds robustness to lighting and tube placement, NOT new chemistry.")

    # ---------------- outputs ----------------
    sel.update({"confirmed": False,
                "created": datetime.datetime.now().isoformat(timespec="seconds"),
                "nested_cv": C.NESTED_CV})
    common.json_dump(sel, SEL_JSON)
    for role in ("primary", "secondary"):
        info = sel[role]
        g = pred[(pred["model"] == info["model"]) & (pred["setting"] == info["setting"]) &
                 (pred["params"] == info["params_key"])]
        g[["id", "repeat", "fold", "y", "prob"]].to_csv(C.CV_DIR / f"oof_{role}.csv", index=False)
    make_figures(fig_dir, pred, summ, aug_df, sel, best_p)
    V.update_progression(STAGE)
    print("\nPLEASE CONFIRM the decision above before anything is frozen:")
    print("   python train_cv.py --confirm-selection")
    common.stop_here(STAGE, next_cmd="train_cv.py --confirm-selection   (then evaluate.py)")


if __name__ == "__main__":
    main()
