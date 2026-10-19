"""
Stage 0 -- data checks on manifest.csv, excluded.csv and anon_images/ (text output, no figures).

Checks:
  * anon_images/ holds exactly one IMAGE_EXT file per manifest id and nothing else
  * every image is readable and IMAGE_W x IMAGE_H (config.py); no two ids share identical pixels
  * no duplicate ids; no id or condition (group) of excluded.csv in manifest.csv / anon_images/
  * labels are 0/1 only; no group has mixed labels; no row has an empty group
Also prints the label counts, the number of groups, the group-size distribution and every group
with more than one image (replicate tags and labels).

Exits with code 1 if any check fails; the problems must be resolved (or a decision on them
recorded) before data_split.py is run.

Usage:  python data_checks.py
"""
import hashlib
import sys
from collections import Counter

import common
import config as C


def main():
    common.start_stage("00_data_checks", needs_split=False)
    problems = []

    manifest = common.load_manifest_with_labels()
    excluded = common.read_csv(C.EXCLUDED_CSV, usecols=["id", "group"])
    print(f"manifest.csv: {len(manifest)} rows | excluded.csv: {len(excluded)} rows")

    # --- ids <-> files --------------------------------------------------------------
    files = sorted(p for p in C.IMAGES_DIR.iterdir() if p.is_file())
    file_ids = [p.stem for p in files if p.suffix.lower() == C.IMAGE_EXT]
    other_files = [p.name for p in files if p.suffix.lower() != C.IMAGE_EXT]
    man_ids = list(manifest["id"])
    missing = sorted(set(man_ids) - set(file_ids))
    extra = sorted(set(file_ids) - set(man_ids))
    print(f"anon_images/: {len(file_ids)} {C.IMAGE_EXT} files, {len(other_files)} other files")
    if missing:
        problems.append(f"{len(missing)} manifest ids have no image: {missing[:10]}")
    if extra:
        problems.append(f"{len(extra)} images are not in the manifest: {extra[:10]}")
    if other_files:
        problems.append(f"unexpected non-{C.IMAGE_EXT} files: {other_files[:10]}")

    # --- duplicates -----------------------------------------------------------------
    dup_ids = sorted(k for k, v in Counter(man_ids).items() if v > 1)
    if dup_ids:
        problems.append(f"duplicate ids in manifest: {dup_ids}")

    # --- excluded ids ---------------------------------------------------------------
    exc_in_man = sorted(set(excluded["id"]) & set(man_ids))
    exc_in_files = sorted(set(excluded["id"]) & set(file_ids))
    exc_groups_in_man = sorted(set(excluded["group"]) & set(manifest["group"]))
    if exc_in_man:
        problems.append(f"excluded ids present in manifest: {exc_in_man}")
    if exc_in_files:
        problems.append(f"excluded ids present in anon_images/: {exc_in_files}")
    if exc_groups_in_man:
        problems.append(f"excluded condition(s) present in manifest: {exc_groups_in_man}")

    # --- images: readable and the right size ----------------------------------------
    sizes, unreadable, ids_by_pixel_hash = Counter(), [], {}
    for photo_id in man_ids:
        if photo_id in missing:
            continue
        try:
            img = common.read_rgb(common.image_path(photo_id))
        except Exception as e:  # noqa: BLE001 -- any decoding error counts as unreadable
            unreadable.append(f"{photo_id} ({e})")
            continue
        h, w = img.shape[:2]
        sizes[(w, h)] += 1
        pixel_hash = hashlib.md5(img.tobytes()).hexdigest()
        ids_by_pixel_hash.setdefault(pixel_hash, []).append(photo_id)
    print(f"image sizes (w x h): {dict(sizes)}")
    if unreadable:
        problems.append(f"unreadable images: {unreadable}")
    wrong_size = {k: v for k, v in sizes.items() if k != (C.IMAGE_W, C.IMAGE_H)}
    if wrong_size:
        problems.append(f"images not {C.IMAGE_W}x{C.IMAGE_H}: {wrong_size}")
    same_pixels = [v for v in ids_by_pixel_hash.values() if len(v) > 1]
    if same_pixels:
        problems.append(f"pixel-identical images under different ids: {same_pixels}")

    # --- labels ---------------------------------------------------------------------
    bad_labels = sorted(set(manifest["label"]) - {"0", "1"})
    if bad_labels:
        problems.append(f"labels other than 0/1 (or blank): {bad_labels}")
    else:
        y = manifest["label"].astype(int)
        print(f"label counts: success(1)={int((y == 1).sum())}  fail(0)={int((y == 0).sum())}"
              f"  total={len(y)}")

    # --- groups ---------------------------------------------------------------------
    group_sizes = manifest.groupby("group").size()
    size_distribution = dict(sorted(Counter(int(v) for v in group_sizes.values).items()))
    print(f"groups: {manifest['group'].nunique()} | group-size distribution "
          f"(size: n_groups): {size_distribution}")
    multi = group_sizes[group_sizes > 1]
    print(f"groups with >1 image ({len(multi)}):")
    for group in multi.index:
        sub = manifest[manifest["group"] == group]
        print(f"   {group:28s} n={len(sub)}  tags={list(sub['replicate_tag'])}  "
              f"labels={list(sub['label'])}")
    if not bad_labels:
        mixed = [g for g, s in manifest.groupby("group") if s["label"].nunique() > 1]
        print(f"groups with mixed labels: {mixed if mixed else 'none'}")
        if mixed:
            problems.append(f"groups with mixed labels: {mixed}")
    if manifest["group"].isna().any() or (manifest["group"] == "").any():
        problems.append("some rows have an empty group")

    # --- verdict --------------------------------------------------------------------
    print()
    if problems:
        print("DATA CHECKS FAILED:")
        for p in problems:
            print("  -", p)
        print("STOP: resolve these problems (or record a decision on them) before running "
              "anything else.")
        sys.exit(1)
    print("ALL DATA CHECKS PASSED.  Next:  python data_split.py")


if __name__ == "__main__":
    main()
