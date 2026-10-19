"""
Stage 09 -- operating threshold from pooled out-of-fold (OOF) predictions (DEV only).

Objective (config.THRESHOLD_OBJECTIVE, with X = config.THRESHOLD_TARGET):
  "max_f1"        maximize F1 (default)
  "recall>=X"     the best precision among thresholds with recall >= X
  "precision>=X"  the best recall among thresholds with precision >= X
The PRIMARY model's threshold comes from its pooled OOF predictions (all CV repeats); the
SECONDARY model gets its own threshold from its own OOF predictions, the same way.
Probabilities inside config.REVIEW_BAND are output as "REVIEW".

Inputs : cv_results/selection.json (confirmed), cv_results/oof_primary.csv and
         cv_results/oof_secondary.csv
Outputs: cv_results/thresholds.json; figures/09_threshold/ (threshold tables, PR curves with the
         operating points, probability histograms)

Note: the final models are trained on all DEV images (stage 10); they can be calibrated
slightly differently from the fold models whose OOF predictions set this threshold. The
probabilities are not recalibrated. A calibrator (Platt or isotonic) fitted on the pooled
OOF predictions would have to be applied before choose_threshold() here, saved with the
model in freeze.py and applied after bundle_predict() in final_test.py and predict.py.

Usage:  python threshold.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

import common
import config as C
import evaluate as E
import visualize as V

STAGE = "09_threshold"
# Operating points reported (and plotted) next to the configured objective, for comparison.
ALTERNATIVES = ["max_f1", "recall>=0.9", "recall>=0.8", "precision>=0.9", "precision>=0.8"]


def parse_objective(objective: str):
    """'max_f1' -> ('max_f1', None); 'recall>=0.8' -> ('recall', 0.8); 'X' -> THRESHOLD_TARGET."""
    if objective == "max_f1":
        return "max_f1", None
    kind, target = objective.split(">=")
    return kind, (C.THRESHOLD_TARGET if target.strip().upper() == "X" else float(target))


def choose_threshold(y, p, objective: str) -> dict:
    """Pick the threshold on the precision-recall curve of (y, p) that meets `objective`.

    If a recall/precision target cannot be reached, the threshold with the highest attainable
    value is used instead and "feasible" is False. Remaining ties go to the threshold closest
    to 0.5.
    """
    kind, target = parse_objective(objective)
    precision, recall, thresholds = precision_recall_curve(y, p)
    # precision_recall_curve returns one more precision/recall point than thresholds
    precision, recall = precision[:-1], recall[:-1]
    denom = precision + recall
    f1 = np.where(denom > 0, 2 * precision * recall / np.maximum(denom, 1e-12), 0.0)
    feasible = True
    if kind == "max_f1":
        candidates = np.flatnonzero(f1 == f1.max())
    else:
        constrained, optimized = (recall, precision) if kind == "recall" else (precision, recall)
        meets = np.flatnonzero(constrained >= target)
        if len(meets) == 0:                              # unattainable: closest possible
            feasible = False
            meets = np.flatnonzero(constrained == constrained.max())
        candidates = meets[optimized[meets] == optimized[meets].max()]
    best = candidates[np.argmin(np.abs(thresholds[candidates] - 0.5))]
    return {"objective": objective, "threshold": float(thresholds[best]),
            "precision": float(precision[best]), "recall": float(recall[best]),
            "f1": float(f1[best]), "feasible": feasible}


def band_fractions(oof, band):
    """Fraction of pooled OOF predictions, and of DEV images (mean OOF probability per image),
    that fall inside the REVIEW band."""
    inside = (oof["prob"] >= band[0]) & (oof["prob"] <= band[1])
    per_img = oof.groupby("id")["prob"].mean()
    return float(inside.mean()), float(((per_img >= band[0]) & (per_img <= band[1])).mean())


def main():
    fig_dir = common.start_stage(STAGE)
    common.require_files(C.CV_DIR / "selection.json", hint="run train_cv.py first")
    selection = common.json_load(C.CV_DIR / "selection.json")
    if not selection.get("confirmed"):
        print("ABORT: the model selection of stage 07 is not confirmed yet "
              "(python train_cv.py --confirm-selection).")
        raise SystemExit(2)
    band = list(C.REVIEW_BAND) if C.REVIEW_BAND is not None else None   # None = no REVIEW
    out = {"review_band": band, "objective_config": C.THRESHOLD_OBJECTIVE,
           "note": "thresholds come from pooled OOF predictions of fold models; the final "
                   "all-DEV models (stage 10) may be calibrated slightly differently"}
    fig_pr, axes_pr = plt.subplots(1, 2, figsize=(12, 4.6))
    fig_hist, axes_hist = plt.subplots(1, 2, figsize=(12, 3.8))
    for col, role in enumerate(("primary", "secondary")):
        info = selection[role]
        oof = common.read_csv(C.CV_DIR / f"oof_{role}.csv")
        y, p = oof["y"].to_numpy(), oof["prob"].to_numpy()
        chosen = choose_threshold(y, p, C.THRESHOLD_OBJECTIVE)
        alternatives = {name: choose_threshold(y, p, name) for name in ALTERNATIVES}
        f_pooled, f_img = band_fractions(oof, band) if band is not None else (0.0, 0.0)
        chosen.update({"model": info["model"], "setting": info["setting"],
                       "review_band": band, "frac_in_band_pooled": f_pooled,
                       "frac_in_band_images": f_img, "pr_auc_pooled": E.pr_auc(y, p),
                       "alternatives": alternatives})
        out[role] = chosen

        # --- console summary -----------------------------------------------------------------
        print(f"\n{role.upper()}  {info['model']} / {info['setting']}  (pooled OOF: {len(p)} "
              f"predictions = {oof['id'].nunique()} images x {oof['repeat'].nunique()} repeats)")
        print(f"  chosen ({C.THRESHOLD_OBJECTIVE}): threshold {chosen['threshold']:.3f} -> "
              f"precision {chosen['precision']:.3f}, recall {chosen['recall']:.3f}, "
              f"F1 {chosen['f1']:.3f}")
        print("  alternatives:")
        for name, alt in alternatives.items():
            flag = "" if alt["feasible"] else "  (target not reachable -> closest)"
            print(f"    {name:15s} threshold {alt['threshold']:.3f}  "
                  f"precision {alt['precision']:.3f}"
                  f"  recall {alt['recall']:.3f}  F1 {alt['f1']:.3f}{flag}")
        if band is None:
            print("  no REVIEW band: every image gets a YES/NO call at the threshold")
        else:
            print(f"  review band {band}: {f_pooled:.1%} of pooled OOF predictions, {f_img:.1%} "
                  "of DEV images (mean OOF probability) would be REVIEW")
            if not band[0] <= chosen["threshold"] <= band[1]:
                print(f"  WARNING: the threshold {chosen['threshold']:.3f} lies OUTSIDE the "
                      f"review band {band}: some YES/NO decisions near the threshold are not "
                      "flagged REVIEW.")

        # --- metrics on a 0.1 threshold grid plus the chosen threshold ------------------------
        rows = [["threshold", "precision", "recall", "F1", "n predicted YES"]]
        print(f"  {'thr':>6s} {'prec':>6s} {'recall':>6s} {'F1':>6s} {'nYES':>5s}")
        grid = set(np.round(np.arange(0.1, 0.91, 0.1), 2)) | {round(chosen['threshold'], 3)}
        for t in sorted(grid):
            m = E.binary_metrics(y, p, t)
            mark = "  <- chosen" if abs(t - chosen["threshold"]) < 1e-3 else ""
            print(f"  {t:6.3f} {m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f} "
                  f"{m['tp'] + m['fp']:5d}{mark}")
            rows.append([f"{t:.3f}{' (chosen)' if mark else ''}", f"{m['precision']:.3f}",
                         f"{m['recall']:.3f}", f"{m['f1']:.3f}", m["tp"] + m["fp"]])
        V.render_table(rows, fig_dir / f"threshold_table_{role}.png",
                       f"{role}: {info['model']} / {info['setting']} -- pooled OOF precision / "
                       "recall at several thresholds (DEV)")

        # --- PR curve with the operating points -----------------------------------------------
        curve_prec, curve_rec, _ = precision_recall_curve(y, p)
        ax = axes_pr[col]
        ax.plot(curve_rec, curve_prec, color="#4c72b0", lw=1.5)
        for name, alt in alternatives.items():
            ax.scatter(alt["recall"], alt["precision"], s=22, zorder=3,
                       label=f"{name} ({alt['threshold']:.2f})")
        ax.scatter(chosen["recall"], chosen["precision"], s=140, marker="*", color="red",
                   zorder=4, label=f"CHOSEN {C.THRESHOLD_OBJECTIVE} ({chosen['threshold']:.2f})")
        ax.set_xlabel("recall (success)")
        ax.set_ylabel("precision (success)")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=6.5, loc="lower left")
        ax.set_title(f"{role}: {info['model']} / {info['setting']}", fontsize=9)

        # --- OOF probability histograms per class ---------------------------------------------
        ax = axes_hist[col]
        bins = np.linspace(0, 1, 26)
        ax.hist(p[y == 0], bins, alpha=0.6, color="#4c72b0", label="fail (pooled OOF)")
        ax.hist(p[y == 1], bins, alpha=0.6, color="#dd8452", label="success (pooled OOF)")
        if band is not None:
            ax.axvspan(band[0], band[1], color="gold", alpha=0.25, label=f"REVIEW band {band}")
        ax.axvline(chosen["threshold"], color="red", lw=2,
                   label=f"threshold {chosen['threshold']:.3f}")
        ax.set_xlabel("predicted probability of success")
        ax.set_title(f"{role}: {info['model']} / {info['setting']}", fontsize=9)
        ax.legend(fontsize=7)
    fig_pr.suptitle("PR curves (pooled OOF, DEV) with the operating points", fontsize=10)
    fig_pr.tight_layout()
    V.save_fig(fig_pr, fig_dir / "pr_curve_operating_point.png")
    fig_hist.tight_layout()
    V.save_fig(fig_hist, fig_dir / "probability_histograms.png")
    common.json_dump(out, C.CV_DIR / "thresholds.json")
    print("\nNote: the final models are trained on ALL DEV images (stage 10) and may be "
          "calibrated slightly differently from the fold models used here.")
    V.update_progression(STAGE)
    common.stop_here(STAGE, next_cmd="freeze.py")


if __name__ == "__main__":
    main()
