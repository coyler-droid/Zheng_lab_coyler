"""
Step 1 -- photo ingestion: anonymise the tube photos and build the label tables.

Every .jpg / .jpeg / .png in PHOTO_FOLDER whose file name follows the synthesis-condition
pattern

    p<mmol>mmol_<ul>ul_<hr>[hr]_<dc>dc[_<tag>...][_Y|_N]       e.g. p6mmol_100ul_1p5hr_200dc_v2_Y

gets an anonymous id and is copied to OUT_FOLDER/anon_images/<id><ext> (extension lower-cased).
Inside a field "p" is the decimal point (1p5 -> 1.5, p5 -> 0.5); the leading "p" of the file
name is not part of the mmol value (p75mmol -> group "75mmol...").  The unit "hr" is sometimes
missing (e.g. ..._p5_150dc).
The optional tags (_v2, _new, _p2, ...) mark replicates of the same condition; the final _Y / _N
is the label (Y = success = 1, N = fail = 0).

Outputs in OUT_FOLDER, all with the columns  id, label, group, replicate_tag, original_name:
  manifest.csv    labelled photos
  unlabeled.csv   photos without _Y / _N (label left empty)
  excluded.csv    photos of the conditions in EXCLUDE_GROUPS; their images are copied to
                  anon_images_unused/ instead of anon_images/ (apply_label_changes.py --formula
                  can bring them back into the study)
Rows are in sorted file-name order.  "group" is the normalised condition, e.g.
6mmol_100ul_1.5hr_200dc; replicates share a group.

Ids are STABLE across re-runs: a photo already listed (by original file name) in an existing
manifest.csv / unlabeled.csv / excluded.csv in OUT_FOLDER keeps its id; only photos seen for the
first time get a new random id (uuid4().hex[:10], unique among all ids).

Printed: exact duplicate files (same bytes), label / group counts, groups with mixed labels,
unlabeled photos, how many ids were kept or newly made, and file names the pattern did not match
(those photos are skipped).

Usage:  python step1.py <photo folder> <out folder>
"""
import csv
import hashlib
import re
import shutil
import sys
import uuid
from collections import Counter
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
CSV_COLUMNS = ["id", "label", "group", "replicate_tag", "original_name"]
# the three tables that together list every photo (and therefore every id ever assigned)
TABLE_NAMES = ("manifest.csv", "unlabeled.csv", "excluded.csv")

# p6mmol_100ul_1p5hr_200dc[_v2|_new|_p2...]_Y   ("hr" sometimes missing, e.g. p5_150dc)
NAME_PATTERN = re.compile(
    r"^p(?P<mmol>[\dp]+)mmol_(?P<ul>[\dp]+)ul_(?P<hr>[\dp]+?)(?:hr)?_(?P<dc>\d+)dc"
    r"(?P<extra>(?:_[A-Za-z0-9]+)*?)(?:_(?P<lab>[YN]))?$",
    re.I,
)

# Manual curation decisions:
#   LABEL_OVERRIDES: original file name -> label that replaces the one in the file name
#   EXCLUDE_GROUPS:  conditions whose replicates carry conflicting Y/N labels; left out of the
#                    study for now (listed in excluded.csv, images in anon_images_unused/)
LABEL_OVERRIDES = {"p75mmol_0ul_1p5hr_200dc.jpg": 0, "p9mmol_100ul_1hr_250dc.jpg": 1}
EXCLUDE_GROUPS = {"6mmol_100ul_1hr_200dc"}


def parse_number(text):
    """Number written with "p" as the decimal point: p75 -> 0.75, 1p5 -> 1.5, 100 -> 100.0."""
    return float(text.lower().replace("p", "."))


def read_known_ids(out_dir):
    """original_name -> id from the tables of a previous run in `out_dir` (empty if none).

    The tables are read in TABLE_NAMES order, so a name listed twice gets the id of the later
    table.
    """
    known = {}
    for name in TABLE_NAMES:
        if (out_dir / name).exists():
            with open(out_dir / name, newline="") as fh:
                known.update({r["original_name"]: r["id"] for r in csv.DictReader(fh)})
    return known


class IdAllocator:
    """Hands out new random ids that collide with no id already in use."""

    def __init__(self, used_ids):
        self.used = set(used_ids)
        self.new_names = []           # original names that received a new id, in order

    def new_id(self, original_name):
        while (photo_id := uuid.uuid4().hex[:10]) in self.used:
            pass
        self.used.add(photo_id)
        self.new_names.append(original_name)
        return photo_id


def make_row(path, fields, photo_id):
    """The table row of one photo (before LABEL_OVERRIDES are applied)."""
    label = "" if not fields["lab"] else int(fields["lab"].upper() == "Y")
    group = (f"{parse_number(fields['mmol']):g}mmol_{parse_number(fields['ul']):g}ul_"
             f"{parse_number(fields['hr']):g}hr_{fields['dc']}dc")
    return {"id": photo_id,
            "label": label,
            "group": group,
            "replicate_tag": (fields["extra"] or "").strip("_"),
            "original_name": path.name}


def write_table(path, rows):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(labeled, unlabeled, excluded, new_names, unparsed):
    n_success = sum(r["label"] for r in labeled)
    group_sizes = Counter(r["group"] for r in labeled)
    print(f"labeled={len(labeled)} success={n_success} fail={len(labeled) - n_success} | "
          f"unlabeled={len(unlabeled)} | excluded={len(excluded)} | groups={len(group_sizes)}")
    print("group sizes:", dict(Counter(group_sizes.values())))
    mixed = [g for g in group_sizes
             if len({r["label"] for r in labeled if r["group"] == g}) > 1]
    print("groups with mixed labels:", mixed)
    if unlabeled:
        print("UNLABELED:", [r["original_name"] for r in unlabeled])
    n_total = len(labeled) + len(unlabeled) + len(excluded)
    print(f"ids: {n_total - len(new_names)} kept, "
          f"{len(new_names)} new: {new_names}")
    if unparsed:
        print("UNPARSED:", unparsed)


def main():
    src_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    images_dir = out_dir / "anon_images"
    images_dir.mkdir(parents=True, exist_ok=True)
    unused_dir = out_dir / "anon_images_unused"

    known_ids = read_known_ids(out_dir)
    allocator = IdAllocator(known_ids.values())

    labeled, unlabeled, excluded, unparsed = [], [], [], []
    first_name_by_digest = {}         # md5 of the file bytes -> file name (exact duplicates)
    for path in sorted(src_dir.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        match = NAME_PATTERN.match(path.stem)
        if not match:
            unparsed.append(path.name)
            continue

        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest in first_name_by_digest:
            print("EXACT DUPLICATE:", path.name, "==", first_name_by_digest[digest])
        first_name_by_digest[digest] = path.name

        photo_id = known_ids.get(path.name) or allocator.new_id(path.name)
        row = make_row(path, match.groupdict(), photo_id)
        if path.name in LABEL_OVERRIDES:
            row["label"] = LABEL_OVERRIDES[path.name]
        image_name = f"{row['id']}{path.suffix.lower()}"

        if row["group"] in EXCLUDE_GROUPS:
            # parked outside the study; apply_label_changes.py --formula can bring it back
            excluded.append(row)
            unused_dir.mkdir(exist_ok=True)
            shutil.copy(path, unused_dir / image_name)
            continue
        (unlabeled if row["label"] == "" else labeled).append(row)
        shutil.copy(path, images_dir / image_name)

    for name, rows in zip(TABLE_NAMES, (labeled, unlabeled, excluded)):
        write_table(out_dir / name, rows)

    print_summary(labeled, unlabeled, excluded, allocator.new_names, unparsed)


if __name__ == "__main__":
    main()
