"""
Label review sheet -- for re-assessing ambiguous tubes AFTER the final test (stage 11).

The tubes whose photos the model found ambiguous are mixed with randomly chosen control tubes
and shuffled, and the sheet shows NO model output, so that the re-assessment is not pulled
towards the model's opinion. The model's view is written to a separate key file that should be
opened only after the review form is filled in.

Ambiguous tubes:
  * DEV  -- every entry of review_labels.csv (confident-wrong and borderline out-of-fold
            predictions) and every misclassified out-of-fold image of the primary model
  * TEST -- every wrong YES/NO of the frozen primary model (test_report.csv)
Controls: N_CONTROLS other tubes drawn at random (seed config.SEED), half successes, half
failures.  With --all, EVERY tube is on the (equally blind) sheet instead.

Writes label_review/ (label_review_all/ with --all):
  review_sheet_01.png, ...  printable pages, PER_PAGE (12) tubes per page (the raw photo, large)
  review_form.csv           fill in your_label (1 = success, 0 = fail), confidence, notes
  model_key.csv             the model's opinion -- open only AFTER finishing the form
  README.txt

If labels are changed afterwards: edit manifest.csv, run `python data_split.py
--confirm-rebuild` (archives every old output, makes a new split) and every later stage again.

Usage:  python label_review.py [--all]
"""
import argparse

import numpy as np
import pandas as pd

import common
import config as C
import visualize as V

N_CONTROLS = 20
PER_PAGE = 12


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true",
                    help="blind review of EVERY tube (label_review_all/) instead of the "
                         "ambiguous tubes + controls (label_review/)")
    args = ap.parse_args()
    out_dir = C.ROOT / ("label_review_all" if args.all else "label_review")
    common.start_stage("label_review")
    if not common.final_test_completed():
        print("ABORT: run the final test first -- before it, test labels are sealed.")
        raise SystemExit(2)
    # current labels: dev from labels_dev.csv, test from the final-test report
    dev_lab = common.load_dev_labels()
    test_rep = common.read_csv(C.REPORTS_DIR / "test_report.csv")
    test_lab = pd.Series(test_rep["true_label"].astype(int).to_numpy(), index=test_rep["id"])
    labels = pd.concat([dev_lab, test_lab])
    splits = common.load_splits().set_index("id")
    thr = common.json_load(C.FROZEN_CONFIG_JSON)["models"]["primary"]["threshold"]

    # the model's view (goes to the key file only): mean out-of-fold probability over the CV
    # repeats for dev, the frozen model's probability for test
    oof = common.read_csv(C.CV_DIR / "oof_primary.csv").groupby("id")["prob"].mean()
    prob = pd.concat([oof, pd.Series(test_rep["probability"].to_numpy(), index=test_rep["id"])])

    # why each flagged tube is on the sheet (id -> reason)
    reason = {}
    review = common.read_csv(C.REPORTS_DIR / "review_labels.csv")
    for _, r in review.iterrows():
        reason[r["id"]] = "dev: " + r["reason"].split(" (")[0]
    for i, p in oof.items():
        if (p >= 0.5) != bool(dev_lab[i]) and i not in reason:
            reason[i] = "dev: misclassified out-of-fold"
    for _, r in test_rep[test_rep["correct"] == "wrong"].iterrows():
        reason[r["id"]] = f"test: wrong {'YES' if r['yes_no'] == 'YES' else 'NO'} call"
    flagged = sorted(reason)

    # one generator for both the control draw and the shuffle, so the sheet is reproducible
    rng = np.random.default_rng(C.SEED)
    controls = []
    if args.all:                       # every tube, in random order, same blind sheet
        ids = sorted(labels.index)
        for photo_id in ids:
            reason.setdefault(photo_id, "not flagged")
    else:
        rest = labels.drop(flagged)
        for lab in (1, 0):             # N_CONTROLS // 2 successes, then as many failures
            pool = sorted(rest.index[rest == lab])
            controls += [str(c) for c in rng.choice(pool, N_CONTROLS // 2, replace=False)]
        for c in controls:
            reason[c] = "control (not flagged)"
        ids = flagged + controls
    order = [ids[k] for k in rng.permutation(len(ids))]
    names = V.original_names(order)

    # one form row (no model output) and one key row (model output) per tube
    rows, key = [], []
    width = 3 if len(order) > 99 else 2          # R01..R99 or R001..
    for n, photo_id in enumerate(order, 1):
        group = splits.loc[photo_id, "group"]
        partners = [j for j in splits.index[splits["group"] == group] if j != photo_id]
        partner_txt = ", ".join(f"R{order.index(j) + 1:0{width}d}" if j in order
                                else f"{j} (not on sheet)" for j in partners)
        rows.append({"review_no": f"R{n:0{width}d}", "id": photo_id,
                     "original_name": names[photo_id],
                     "condition": group, "current_label": int(labels[photo_id]),
                     "replicate_of": partner_txt, "your_label": "", "confidence_1to3": "",
                     "notes": ""})
        model_yes = prob[photo_id] >= thr
        key.append({"review_no": f"R{n:0{width}d}", "id": photo_id,
                    "set": splits.loc[photo_id, "split"],
                    "why_on_sheet": reason[photo_id], "current_label": int(labels[photo_id]),
                    "model_probability": round(float(prob[photo_id]), 3),
                    "model_call": "YES" if model_yes else "NO",
                    "model_agrees_with_label": model_yes == bool(labels[photo_id])})
    out_dir.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "review_form.csv", index=False)
    pd.DataFrame(key).to_csv(out_dir / "model_key.csv", index=False)

    pages = int(np.ceil(len(rows) / PER_PAGE))
    for pg in range(pages):
        chunk = rows[pg * PER_PAGE:(pg + 1) * PER_PAGE]
        items = [common.load_image(r["id"]) for r in chunk]
        titles = [f"{r['review_no']}   current label: {r['current_label']} "
                  f"({V.LABEL_NAMES[r['current_label']]})\n{r['original_name']}"
                  + (f"\nreplicate of {r['replicate_of']}" if r["replicate_of"] else "")
                  for r in chunk]
        fig = V.show_grid(items, titles, ncols=6, tile_w=2.3, tile_h=4.5, fontsize=7.5,
                          suptitle=f"Label review sheet {pg + 1}/{pages} -- no model output "
                                   "shown; record your judgement in review_form.csv")
        V.save_fig(fig, out_dir / f"review_sheet_{pg + 1:02d}.png")

    content = (f"ALL {len(order)} tubes" if args.all else
               f"{len(flagged)} tubes the model found ambiguous + {len(controls)} random "
               "control tubes")
    (out_dir / "README.txt").write_text(
        "LABEL REVIEW\n\n"
        f"{content}, shuffled (R..). The sheets show NO model output on purpose.\n\n"
        "1. Before looking, write down the criterion that defines success for this chemistry\n"
        "   (ideally one that does not come from the photo itself, e.g. the measurement the\n"
        "   original Y/N came from).\n"
        "2. For every R-number, fill in review_form.csv: your_label (1 = success, 0 = fail),\n"
        "   confidence_1to3 and notes. Judge every tube the same way, controls included.\n"
        "3. Only afterwards open model_key.csv (why each tube is on the sheet, the model's\n"
        "   probability).\n"
        "4. To use corrected labels: edit the label column of manifest.csv (keep a list of\n"
        "   the changes and reasons), then run `python data_split.py --confirm-rebuild` and\n"
        "   every later stage again. The old outputs are archived, not deleted.\n"
        "   Caveat: the new test set is drawn from the same 151 photos, so it is no longer a\n"
        "   fully independent estimate -- new photos are the clean check.\n",
        encoding="utf-8")
    n_dev = sum(1 for i in flagged if splits.loc[i, "split"] != "test")
    if args.all:
        print(f"label review: ALL {len(order)} tubes (of which {len(flagged)} were flagged "
              f"ambiguous) -> {pages} sheet(s) in {out_dir.name}/")
    else:
        print(f"label review: {len(flagged)} ambiguous tubes ({n_dev} dev, "
              f"{len(flagged) - n_dev} test) + {len(controls)} controls -> {pages} sheet(s) in "
              f"{out_dir.name}/")
    print("  review_form.csv  -> fill in your_label / confidence_1to3 / notes")
    print("  model_key.csv    -> open only AFTER the form is filled")


if __name__ == "__main__":
    main()
