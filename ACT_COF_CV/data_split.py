"""
Stage 01 -- grouped + stratified split, repeated StratifiedGroupKFold, label sealing.

All photos of one condition (group) always land on the same side of every split, so replicates
of a condition never appear in both training and evaluation data.

SPLIT_MODE "cv_plus_test" (default):
  * TEST = TEST_FRACTION (20 %) of images, stratified by label, grouped by condition
  * DEV  = the rest; N_FOLDS x N_REPEATS (5 x 5) StratifiedGroupKFold, own shuffle seed per
    repeat
  * seeds are tried from config.SEED upward until the test set has TEST_SUCCESS_RANGE (9-12)
    successes AND every CV held-out fold has >= MIN_SUCCESS_PER_FOLD (6) successes; the
    accepted seed is recorded in splits_meta.json
SPLIT_MODE "fixed_60_20_20": train/val/test, val = 25 % of dev (grouped, stratified).
SPLIT_MODE "fixed_test": the test set is exactly the ids in fixed_test_ids.txt (written by
  `python data_split.py --save-fixed-test-ids`: the current test ids + new photos); every other
  photo is DEV with the same repeated StratifiedGroupKFold.

Writes splits.csv, folds.csv (NO labels), labels_dev.csv, labels_test_SEALED.csv and
splits_meta.json (incl. the SHA-256 of manifest.csv), plus two figures. This is the LAST script
allowed to parse the label column of manifest.csv. It never reads the sealed file back.

Usage:
  python data_split.py                        # first run
  python data_split.py --confirm-rebuild      # splits exist: archive ALL outputs, then rebuild
  python data_split.py --save-fixed-test-ids  # write fixed_test_ids.txt (before a rebuild)
"""
import argparse
import datetime
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

import common
import config as C
import visualize as V

STAGE = "01_split"


def archive_previous_outputs():
    """Move every split/fold/feature/model/figure/report output to archive/<timestamp>/."""
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = C.ARCHIVE_DIR / stamp
    dest.mkdir(parents=True, exist_ok=False)
    items = [C.SPLITS_CSV, C.FOLDS_CSV, C.LABELS_DEV_CSV, C.SEALED_TEST_LABELS_CSV,
             C.SPLITS_META_JSON, C.TUBE_MASK_NPY, C.REF_MASK_NPY, C.MASK_META_JSON,
             C.ROOT / "showcase_candidates.json",
             C.FEATURES_HAND_CSV, C.FEATURES_HAND_AUG_CSV,
             C.FIG_DIR, C.PROCESSED_DIR,
             C.CACHE_DIR, C.CV_DIR, C.MODELS_DIR, C.FROZEN_CONFIG_JSON,
             C.REPORTS_DIR / "review_labels.csv", common.FINAL_TEST_DONE_JSON,
             C.ROOT / "test_predictions_locked.csv", C.ROOT / "final_test_lock.json"]
    items += sorted(C.REPORTS_DIR.glob("test_report*.csv"))
    moved = []
    for p in items:
        if p.exists():
            shutil.move(str(p), str(dest / p.name))
            moved.append(p.name)
    shutil.copy2(C.ROOT / "config.py", dest / "config.py.snapshot")
    print(f"archived {len(moved)} item(s) to {dest.relative_to(C.ROOT)}: {moved}")


def test_split(df: pd.DataFrame, seed: int):
    """(dev_idx, test_idx): the first fold of a round(1 / TEST_FRACTION)-fold
    StratifiedGroupKFold is the test set."""
    n_splits = int(round(1 / C.TEST_FRACTION))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dev_idx, test_idx = next(sgkf.split(np.zeros(len(df)), df["y"], df["group"]))
    return dev_idx, test_idx


def cv_folds(dev: pd.DataFrame, seed: int):
    """N_REPEATS x N_FOLDS StratifiedGroupKFold on DEV; repeat r uses seed + 1 + r.

    Returns (folds, repeat_seeds); folds has one row per (repeat, held-out id) with the
    columns id, repeat, fold, repeat_seed.
    """
    rows, seeds = [], [seed + 1 + r for r in range(C.N_REPEATS)]
    for repeat, repeat_seed in enumerate(seeds):
        sgkf = StratifiedGroupKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=repeat_seed)
        fold_iter = sgkf.split(np.zeros(len(dev)), dev["y"], dev["group"])
        for fold, (_, held) in enumerate(fold_iter):
            rows += [{"id": dev["id"].iloc[i], "repeat": repeat, "fold": fold,
                      "repeat_seed": repeat_seed}
                     for i in held]
    return pd.DataFrame(rows), seeds


def fixed_folds(dev: pd.DataFrame, seed: int):
    """fixed_60_20_20: one repeat; fold 0 = val (held out), fold -1 = train."""
    sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed + 1)
    train_idx, val_idx = next(sgkf.split(np.zeros(len(dev)), dev["y"], dev["group"]))
    rows = [{"id": dev["id"].iloc[i], "repeat": 0, "fold": -1, "repeat_seed": seed + 1}
            for i in train_idx]
    rows += [{"id": dev["id"].iloc[i], "repeat": 0, "fold": 0, "repeat_seed": seed + 1}
             for i in val_idx]
    return pd.DataFrame(rows), [seed + 1]


def find_split(df: pd.DataFrame):
    """Try seeds SEED, SEED + 1, ... until the test set and every held-out fold meet the
    success-count constraints (see the module docstring); exit 1 if none does.

    Returns (seed, n_attempts, dev, test, folds, repeat_seeds).
    """
    lo, hi = C.TEST_SUCCESS_RANGE
    for attempt in range(C.MAX_SPLIT_ATTEMPTS):
        seed = C.SEED + attempt
        dev_idx, test_idx = test_split(df, seed)
        n_succ = int(df["y"].iloc[test_idx].sum())
        if not lo <= n_succ <= hi:
            print(f"  seed {seed}: rejected (test successes = {n_succ}, need {lo}-{hi})")
            continue
        dev = df.iloc[dev_idx].reset_index(drop=True)
        make_folds = cv_folds if C.SPLIT_MODE == "cv_plus_test" else fixed_folds
        folds, seeds = make_folds(dev, seed)
        held = folds[folds["fold"] >= 0].merge(dev[["id", "y"]], on="id")
        per_fold = held.groupby(["repeat", "fold"])["y"].sum()
        if per_fold.min() < C.MIN_SUCCESS_PER_FOLD:
            print(f"  seed {seed}: rejected (a held-out fold has {per_fold.min()} successes, "
                  f"need >= {C.MIN_SUCCESS_PER_FOLD})")
            continue
        print(f"  seed {seed}: accepted (test successes = {n_succ}, min successes per held-out "
              f"fold = {per_fold.min()})")
        return seed, attempt + 1, dev, df.iloc[test_idx].reset_index(drop=True), folds, seeds
    print("ABORT: no seed satisfied the split constraints.")
    sys.exit(1)


# -----------------------------------------------------------------------------
# SPLIT_MODE "fixed_test": the test set is a given list of ids (not drawn at random)
# -----------------------------------------------------------------------------
def read_fixed_test_ids() -> list[str]:
    """Ids listed in FIXED_TEST_IDS_FILE, one per line ('#' starts a comment); exit 2 if the
    file is missing."""
    if not C.FIXED_TEST_IDS_FILE.exists():
        print(f"ABORT: SPLIT_MODE 'fixed_test' needs {C.FIXED_TEST_IDS_FILE.name}; write it with "
              "`python data_split.py --save-fixed-test-ids` before re-splitting.")
        sys.exit(2)
    ids = []
    for line in C.FIXED_TEST_IDS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.append(line)
    return ids


def out_of_study_ids() -> set[str]:
    """ids in unlabeled.csv / excluded.csv: photos without a label, left out of the study."""
    ids = set()
    for p in (C.ROOT / "unlabeled.csv", C.EXCLUDED_CSV):
        if p.exists():
            ids |= set(common.read_csv(p, usecols=["id"])["id"])
    return ids


def find_split_fixed(df: pd.DataFrame):
    """SPLIT_MODE "fixed_test": test = the listed ids, dev = every other photo; seeds are tried
    only for the CV folds on dev. Exit 2 if the list is inconsistent (duplicate or unknown ids,
    or a condition with photos on both sides), exit 1 if no seed works.

    Returns (seed, n_attempts, dev, test, folds, repeat_seeds), like find_split.
    """
    ids = read_fixed_test_ids()
    dup = sorted({i for i in ids if ids.count(i) > 1})
    left_out = sorted((set(ids) - set(df["id"])) & out_of_study_ids())
    unknown = sorted(set(ids) - set(df["id"]) - set(left_out))
    if dup or unknown:
        print(f"ABORT: {C.FIXED_TEST_IDS_FILE.name}: duplicate ids {dup}; unknown ids {unknown} "
              "(in none of manifest.csv, unlabeled.csv, excluded.csv -- a typo?).")
        sys.exit(2)
    if left_out:
        print(f"  {len(left_out)} id(s) on {C.FIXED_TEST_IDS_FILE.name} are left out of the study "
              f"(no label: in unlabeled.csv / excluded.csv), so they are not tested: {left_out}")
    is_test = df["id"].isin(ids)
    test, dev = df[is_test].reset_index(drop=True), df[~is_test].reset_index(drop=True)
    spanning = sorted(set(test["group"]) & set(dev["group"]))
    if spanning:
        print("ABORT: these conditions have photos in BOTH the fixed test set and dev, which would "
              "leak the condition into training:")
        for g in spanning:
            print(f"   {g}: test {sorted(test.loc[test['group'] == g, 'id'])}, "
                  f"dev {sorted(dev.loc[dev['group'] == g, 'id'])}")
        print("Add the dev photo(s) of those conditions to the fixed test list, or leave the new "
              "photo out of the study (decide before running, and write it down).")
        sys.exit(2)
    print(f"  fixed test set: {len(test)} photos ({int(test['y'].sum())} successes) from "
          f"{C.FIXED_TEST_IDS_FILE.name}; {len(dev)} DEV photos")
    for attempt in range(C.MAX_SPLIT_ATTEMPTS):
        seed = C.SEED + attempt
        folds, seeds = cv_folds(dev, seed)
        held = folds.merge(dev[["id", "y"]], on="id")
        per_fold = held.groupby(["repeat", "fold"])["y"].sum()
        if per_fold.min() < C.MIN_SUCCESS_PER_FOLD:
            print(f"  seed {seed}: rejected (a held-out fold has {per_fold.min()} successes, "
                  f"need >= {C.MIN_SUCCESS_PER_FOLD})")
            continue
        print(f"  seed {seed}: accepted for the CV folds (min successes per held-out fold = "
              f"{per_fold.min()})")
        return seed, attempt + 1, dev, test, folds, seeds
    print("ABORT: no seed satisfied the CV-fold constraints.")
    sys.exit(1)


def save_fixed_test_ids():
    """Write FIXED_TEST_IDS_FILE = the CURRENT test ids of splits.csv (sealed while the pipeline
    was developed) + every photo that is not in splits.csv yet (new photos). Run it BEFORE
    `--confirm-rebuild`, because the rebuild archives the current splits.csv."""
    common.require_files(C.SPLITS_CSV, hint="the current split is needed for the test ids")
    if C.FIXED_TEST_IDS_FILE.exists():
        print(f"ABORT: {C.FIXED_TEST_IDS_FILE.name} already exists -- delete it yourself if you "
              "really want to write it again.")
        sys.exit(2)
    splits = common.load_splits()
    old_test = sorted(splits.loc[splits["split"] == "test", "id"])
    pool = list(common.read_csv(C.MANIFEST_CSV, usecols=["id"])["id"])
    unlabeled_csv = C.ROOT / "unlabeled.csv"
    if unlabeled_csv.exists():
        pool += list(common.read_csv(unlabeled_csv, usecols=["id"])["id"])
    new = sorted(set(pool) - set(splits["id"]))
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    lines = [f"# fixed test set for SPLIT_MODE = 'fixed_test' (written {stamp})",
             "# original test photos -- sealed during the whole development:", *old_test,
             "# new photos -- never used for development:", *new]
    C.FIXED_TEST_IDS_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    new_file = C.FIXED_TEST_IDS_FILE.with_name("fixed_test_ids_new.txt")
    new_file.write_text("\n".join(new) + ("\n" if new else ""), encoding="utf-8")
    print(f"wrote {C.FIXED_TEST_IDS_FILE.name}: {len(old_test)} original test ids + {len(new)} "
          f"new photo ids (the new ones also in {new_file.name}).")


def run_assertions(splits, folds, dev_ids, test_ids, labels_dev_ids):
    """Leakage checks on the new split, run before anything is written."""
    assert splits["id"].is_unique, "an id appears in more than one split"
    with_flag = splits.assign(t=splits["split"].eq("test"))
    assert (with_flag.groupby("group")["t"].nunique() == 1).all(), \
        "a condition group spans dev and test"
    group_of = splits.set_index("id")["group"]
    n_iter = 0
    for repeat, fold, train, held in common.iter_cv_folds(folds):
        shared = set(group_of.loc[train]) & set(group_of.loc[held])
        assert not shared, \
            f"group(s) {shared} span train and held-out (repeat {repeat}, fold {fold})"
        assert not (set(train) | set(held)) & set(test_ids), "a test id is inside the CV folds"
        n_iter += 1
    # in fixed_60_20_20 the single repeat holds train (fold -1) + val (fold 0) = all dev ids
    for repeat, repeat_rows in folds.groupby("repeat"):
        assert repeat_rows["id"].is_unique and sorted(repeat_rows["id"]) == sorted(dev_ids), \
            f"repeat {repeat} does not contain every dev id exactly once"
    assert not set(labels_dev_ids) & set(test_ids), "a test id appears in labels_dev.csv"
    print(f"assertions passed: unique ids, no group spans dev/test, no group spans train/held-out"
          f" in any of {n_iter} CV iterations, no test id in labels_dev.csv")


def count_table(df: pd.DataFrame, split_name: str) -> dict:
    """One row of the per-split count table (also stored in splits_meta.json)."""
    return {"split": split_name, "images": len(df), "successes": int(df["y"].sum()),
            "failures": int((df["y"] == 0).sum()), "groups": int(df["group"].nunique())}


def make_figures(fig_dir, counts, folds, dev):
    """split_counts.png and fold_success_counts.png in fig_dir."""
    # (1) images / successes / failures / groups per split
    cdf = pd.DataFrame(counts).set_index("split")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    keys = ["images", "successes", "failures", "groups"]
    x = np.arange(len(cdf))
    w = 0.2
    colors = ["#4c72b0", "#dd8452", "#55a868", "#8172b3"]
    for k, (key, col) in enumerate(zip(keys, colors)):
        bars = ax.bar(x + (k - 1.5) * w, cdf[key], w, label=key, color=col)
        ax.bar_label(bars, fontsize=8)
    ax.set_xticks(x, cdf.index)
    ax.set_ylabel("count")
    ax.set_title(f"Split ({C.SPLIT_MODE}): images / successes / failures / groups")
    ax.legend(fontsize=8)
    V.save_fig(fig, fig_dir / "split_counts.png")

    # (2) per-fold success counts for every CV repeat
    held = folds[folds["fold"] >= 0].merge(dev[["id", "y"]], on="id")
    succ = held.pivot_table(index="repeat", columns="fold", values="y", aggfunc="sum")
    n = held.pivot_table(index="repeat", columns="fold", values="y", aggfunc="count")
    fig, ax = plt.subplots(figsize=(1.3 * succ.shape[1] + 2.5, 0.7 * succ.shape[0] + 1.6))
    im = ax.imshow(succ.values, cmap="YlOrBr", vmin=0, vmax=max(12, succ.values.max()))
    for (i, j), v in np.ndenumerate(succ.values):
        ax.text(j, i, f"{int(v)} succ\n{int(n.values[i, j])} img", ha="center", va="center",
                fontsize=8)
    ax.set_xticks(range(succ.shape[1]), [f"fold {c}" for c in succ.columns])
    ax.set_yticks(range(succ.shape[0]), [f"repeat {r}" for r in succ.index])
    ax.set_title(f"Held-out successes per CV fold (DEV only; min required "
                 f"{C.MIN_SUCCESS_PER_FOLD})", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8, label="successes")
    V.save_fig(fig, fig_dir / "fold_success_counts.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--confirm-rebuild", action="store_true",
                    help="splits already exist: archive every previous output and rebuild")
    ap.add_argument("--save-fixed-test-ids", action="store_true",
                    help="write fixed_test_ids.txt (current test ids + new photos) and exit")
    args = ap.parse_args()

    fig_dir = common.start_stage(STAGE, needs_split=False)
    if args.save_fixed_test_ids:
        save_fixed_test_ids()
        return
    print(common.resplit_reminder() + "\n")

    if C.SPLITS_CSV.exists():
        ok = common.confirm(
            "splits.csv already exists. Rebuilding moves ALL previous splits, folds, features, "
            "models, figures and reports to archive/<timestamp>/ and you must rerun every later "
            "stage.", args.confirm_rebuild, "--confirm-rebuild")
        if not ok:
            print("Nothing changed.")
            sys.exit(0)
        archive_previous_outputs()
        fig_dir.mkdir(parents=True, exist_ok=True)

    manifest = common.load_manifest_with_labels()
    manifest["y"] = manifest["label"].astype(int)
    assert set(manifest["y"]) <= {0, 1}
    print(f"manifest: {len(manifest)} images, {int(manifest['y'].sum())} successes, "
          f"{manifest['group'].nunique()} groups | SPLIT_MODE = {C.SPLIT_MODE}")

    print("searching split seeds:")
    finder = find_split_fixed if C.SPLIT_MODE == "fixed_test" else find_split
    seed, attempts, dev, test, folds, rep_seeds = finder(manifest)

    cv_mode = C.SPLIT_MODE in ("cv_plus_test", "fixed_test")   # else fixed_60_20_20
    if cv_mode:
        dev_split_name = pd.Series("dev", index=dev.index)
    else:
        val_ids = set(folds.loc[folds["fold"] == 0, "id"])
        dev_split_name = dev["id"].map(lambda i: "val" if i in val_ids else "train")
    splits = pd.concat([
        pd.DataFrame({"id": dev["id"], "split": dev_split_name, "group": dev["group"]}),
        pd.DataFrame({"id": test["id"], "split": "test", "group": test["group"]}),
    ]).sort_values("id").reset_index(drop=True)

    labels_dev = dev[["id", "y"]].rename(columns={"y": "label"}).sort_values("id")
    labels_test = test[["id", "y"]].rename(columns={"y": "label"}).sort_values("id")
    run_assertions(splits, folds, list(dev["id"]), list(test["id"]), list(labels_dev["id"]))

    splits.to_csv(C.SPLITS_CSV, index=False)
    folds.sort_values(["repeat", "fold", "id"]).to_csv(C.FOLDS_CSV, index=False)
    labels_dev.to_csv(C.LABELS_DEV_CSV, index=False)
    labels_test.to_csv(C.SEALED_TEST_LABELS_CSV, index=False)   # written once, never read here
    common._split_sets.cache_clear()   # the cached dev/test id sets belong to the old split

    if cv_mode:
        counts = [count_table(dev, "dev"), count_table(test, "test")]
    else:
        train = dev[dev_split_name == "train"]
        val = dev[dev_split_name == "val"]
        counts = [count_table(train, "train"), count_table(val, "val"),
                  count_table(test, "test")]
    meta = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "base_seed": C.SEED, "split_seed": seed, "attempts": attempts,
        "split_mode": C.SPLIT_MODE, "test_fraction": C.TEST_FRACTION,
        "n_folds": C.N_FOLDS if cv_mode else 1,
        "n_repeats": C.N_REPEATS if cv_mode else 1,
        "repeat_seeds": rep_seeds, "counts": counts,
        "test_successes": int(test["y"].sum()), "test_failures": int((test["y"] == 0).sum()),
        "manifest_sha256": common.sha256_file(C.MANIFEST_CSV),
        "sklearn_version": sklearn.__version__,
    }
    if C.SPLIT_MODE == "fixed_test":
        meta["fixed_test_ids_sha256"] = common.sha256_file(C.FIXED_TEST_IDS_FILE)
        meta["fixed_test_ids_left_out"] = sorted(set(read_fixed_test_ids()) - set(manifest["id"]))
    common.json_dump(meta, C.SPLITS_META_JSON)

    # reset auto-managed state that depended on the previous split
    common.set_auto_config("SHOWCASE_IDS", [])
    common.set_auto_config("MASK_APPROVED", False)

    print("\nper-split counts:")
    print(pd.DataFrame(counts).to_string(index=False))
    held = folds[folds["fold"] >= 0].merge(dev[["id", "y"]], on="id")
    tab = held.pivot_table(index="repeat", columns="fold", values="y", aggfunc="sum")
    print("\nheld-out successes per fold (rows = repeat, cols = fold), DEV only:")
    print(tab.to_string())
    imgs = held.pivot_table(index="repeat", columns="fold", values="y", aggfunc="count")
    print("\nheld-out images per fold:")
    print(imgs.to_string())
    print(f"\nrepeat shuffle seeds: {rep_seeds}  (accepted split seed {seed} after "
          f"{attempts} attempt(s))")
    print(f"manifest.csv SHA-256: {meta['manifest_sha256']}")
    print("wrote splits.csv, folds.csv, labels_dev.csv, labels_test_SEALED.csv (sealed), "
          "splits_meta.json")

    make_figures(fig_dir, counts, folds, dev)
    print("\n" + common.resplit_reminder())
    common.stop_here(STAGE, next_cmd="mask.py")


if __name__ == "__main__":
    main()
