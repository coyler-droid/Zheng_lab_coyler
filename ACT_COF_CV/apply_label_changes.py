"""
Apply new labels to manifest.csv -- data curation done BEFORE a re-split, never in the middle
of a run.

>>> The new PASS/FAIL values for the re-test are typed into ONE file:
>>>     retest_protocol/new_labels.csv
>>> Step by step: README_RETEST.md, top section "HOW TO ENTER THE NEW PASS/FAIL VALUES".

Two modes:

1. The sheet (--template, then --formula): EVERY photo gets its label from the sheet
   retest_protocol/new_labels.csv, one row per photo:
       original_name | condition | crystallinity | pass_fail | note
   * with --cutoff X: PASS (1) if crystallinity >= X, else FAIL (0)
   * without --cutoff: the pass_fail column (PASS / FAIL; 1 / 0 and Y / N also work)
   * NA -> the photo leaves the study: moved to unlabeled.csv, image moved to
     anon_images_unused/ (excluded photos stay in excluded.csv)
   Every photo of manifest.csv, unlabeled.csv and excluded.csv needs a row. A listed photo whose
   image sits in anon_images_unused/ (left out earlier, or excluded by step1.py) is moved back
   to anon_images/. The crystallinity / pass_fail columns go only into the change log, never
   into manifest.csv.

2. Individual changes (no option): reads retest_protocol/label_changes.csv (id, original_name,
   old_label, new_label, reason, reviewer, date), checks every row and rewrites ONLY the label
   column of those rows.

--formula and mode 2 only check and report unless --apply is given (--template always writes
the sheet). With --apply they back up the CSV files into retest_protocol/ and write with the
plain csv module (ids stay text, unlike in Excel). Afterwards every pipeline script refuses to
run until `python data_split.py --confirm-rebuild` makes a new split.

Usage:
  python apply_label_changes.py --template                       # make / update the sheet
  python apply_label_changes.py --formula --cutoff 0.6           # check the sheet (dry run)
  python apply_label_changes.py --formula --cutoff 0.6 --apply   # write the new labels
  python apply_label_changes.py --formula [--apply]              # sheet filled with PASS/FAIL
  python apply_label_changes.py --formula OTHER.csv ...          # a sheet stored elsewhere
  python apply_label_changes.py [--apply]                        # individual changes
"""
import argparse
import csv
import datetime
import io
import math
import shutil
import sys
from pathlib import Path

import config as C

PROTO = C.ROOT / "retest_protocol"
SHEET = PROTO / "new_labels.csv"                 # THE file where the new values are typed
SHEET_FIELDS = ["original_name", "condition", "crystallinity", "pass_fail", "note"]
CHANGES = PROTO / "label_changes.csv"
FIELDS = ["id", "original_name", "old_label", "new_label", "reason", "reviewer", "date"]
NA_VALUES = {"NA", "N/A", "#N/A", "NAN", "NONE"}     # no value: the photo leaves the study
PASS_VALUES = {"PASS", "1", "Y", "YES", "SUCCESS"}
FAIL_VALUES = {"FAIL", "0", "N", "NO", "FAILURE"}


def read_rows(path):
    """(fieldnames, rows) of a CSV written by this pipeline; (None, []) if it does not exist."""
    if not path.exists():
        return None, []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return reader.fieldnames, list(reader)


def write_rows(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_sheet(path):
    """(columns, rows) of a sheet saved by Excel or a text editor: UTF-8 with or without BOM,
    or the Windows encoding; comma- or semicolon-separated. Empty rows are dropped."""
    try:
        data = path.read_bytes()
    except PermissionError:          # Windows: the file is open (locked) in Excel
        print(f"ABORT: cannot read {path.name} -- close it in Excel and run again.")
        sys.exit(1)
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("cp1252", errors="replace")
    # Excel in locales with a decimal comma saves CSV with ";" as the separator
    header = text.split("\n", 1)[0]
    reader = csv.DictReader(io.StringIO(text),
                            delimiter=";" if ";" in header and "," not in header else ",")
    reader.fieldnames = [(c or "").strip() for c in (reader.fieldnames or [])]
    rows = [r for r in reader if any(isinstance(v, str) and v.strip() for v in r.values())]
    return reader.fieldnames, rows


def warn_if_excel_file(path):
    """Excel 'Save As' can silently produce new_labels.xlsx; only the .csv is read."""
    for ext in (".xlsx", ".xls"):
        xl_path = path.with_suffix(ext)
        if xl_path.exists() and (not path.exists()
                                 or xl_path.stat().st_mtime > path.stat().st_mtime):
            print(f"WARNING: {xl_path.name} is newer than {path.name}, but only the .csv is read. "
                  f"In Excel: File > Save As > 'CSV UTF-8 (Comma delimited) (*.csv)', name "
                  f"{path.stem}.")


def pass_fail_code(raw):
    """'1' (PASS), '0' (FAIL), 'NA', '' (empty) or None (not understood)."""
    value = (raw or "").strip().upper()
    if value in NA_VALUES:
        return "NA"
    if value in PASS_VALUES:
        return "1"
    if value in FAIL_VALUES:
        return "0"
    return "" if not value else None


def write_template():
    """Make or update SHEET: one row per photo (manifest, unlabeled, excluded), sorted by
    condition. Values already typed into an existing sheet are kept; photos without a row yet
    (e.g. new photos added with step1.py) get empty rows."""
    warn_if_excel_file(SHEET)
    old_cols, old_rows = read_sheet(SHEET) if SHEET.exists() else ([], [])
    old = {(r.get("original_name") or "").strip(): r for r in old_rows}
    # extra columns added by hand to the sheet are carried over
    extra_cols = [c for c in old_cols if c and c not in SHEET_FIELDS and c not in ("id", "label")]

    def sheet_row(old_row, name, condition, note):
        # an older sheet layout used a "label" column instead of "pass_fail"
        return {"original_name": name, "condition": condition,
                "crystallinity": (old_row.get("crystallinity") or "").strip(),
                "pass_fail": (old_row.get("pass_fail") or old_row.get("label") or "").strip(),
                "note": (old_row.get("note") or "").strip() or note,
                **{c: old_row.get(c) or "" for c in extra_cols}}

    unused = C.ROOT / "anon_images_unused"
    rows, known, added = [], set(), []
    for src, csv_path in (("manifest", C.MANIFEST_CSV), ("unlabeled", C.ROOT / "unlabeled.csv"),
                          ("excluded", C.EXCLUDED_CSV)):
        for r in read_rows(csv_path)[1]:
            name = r["original_name"]
            known.add(name)
            note = "excluded in the first run" if src == "excluded" else ""
            if src == "unlabeled" and (unused / f"{r['id']}{C.IMAGE_EXT}").exists():
                note = "left out earlier (NA)"
            if name not in old:
                added.append(name)
            rows.append(sheet_row(old.get(name, {}), name, r["group"], note))
    rows.sort(key=lambda r: (r["condition"], r["original_name"]))
    # rows of the old sheet that match no photo are kept (at the end) and flagged
    stray = [n for n in old if n and n not in known]
    for n in stray:
        rows.append({**sheet_row(old[n], n, old[n].get("condition") or "", ""),
                     "note": "NOT A KNOWN PHOTO -- fix the name or delete this row"})

    PROTO.mkdir(exist_ok=True)
    if SHEET.exists():
        backup_dir = PROTO / "sheet_backups"
        backup_dir.mkdir(exist_ok=True)
        shutil.copy2(SHEET,
                     backup_dir / f"new_labels_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv")
    try:
        write_rows(SHEET, SHEET_FIELDS + extra_cols, rows)
    except PermissionError:
        print(f"ABORT: cannot write {SHEET.name} -- close it in Excel and run again.")
        sys.exit(1)
    filled = sum(bool(r["crystallinity"] or r["pass_fail"]) for r in rows)
    print(f"{'updated' if old_rows else 'made'} {SHEET.relative_to(C.ROOT)}: {len(rows)} rows "
          f"({filled} filled in, {len(rows) - filled} still empty)")
    if old_rows and added:
        print(f"new empty rows added ({len(added)}): {added}")
    if stray:
        print(f"WARNING: {len(stray)} row(s) match no photo (marked in the note column): {stray}")
    print("Next: open it in Excel, fill in every row, save it as CSV under the same name, then\n"
          "  python apply_label_changes.py --formula --cutoff <your cutoff>     (crystallinity)\n"
          "  python apply_label_changes.py --formula                            (PASS/FAIL)\n"
          "See README_RETEST.md, top section.")


def formula_mode(path, apply: bool, cutoff=None):
    """Relabel EVERY photo from the sheet (see the module docstring). With a cutoff, label =
    1 if crystallinity >= cutoff else 0, computed here for every row.

    Nothing is written unless every row passes every check (exit 1 otherwise).
    """
    files = {"manifest": C.MANIFEST_CSV, "unlabeled": C.ROOT / "unlabeled.csv",
             "excluded": C.EXCLUDED_CSV}
    # pool: id -> (source table, row) over all three tables; order: their original row order
    fields, pool, order = None, {}, []
    for src, csv_path in files.items():
        table_fields, rows = read_rows(csv_path)
        fields = fields or table_fields
        for r in rows:
            pool[r["id"]] = (src, r)
            order.append(r["id"])
    by_name = {r["original_name"]: photo_id for photo_id, (_, r) in pool.items()}
    warn_if_excel_file(path)
    if not path.exists():
        print(f"ABORT: {path} does not exist -- make it with `python apply_label_changes.py "
              "--template`.")
        sys.exit(1)
    sheet_cols, sheet_rows = read_sheet(path)
    pf_col = "pass_fail" if "pass_fail" in sheet_cols else "label"
    if not ({"id", "original_name"} & set(sheet_cols)):
        print("ABORT: the sheet needs an `original_name` (or `id`) column.")
        sys.exit(1)
    if cutoff is not None and "crystallinity" not in sheet_cols:
        print("ABORT: --cutoff needs a `crystallinity` column in the sheet.")
        sys.exit(1)
    if cutoff is None and pf_col not in sheet_cols:
        print("ABORT: without --cutoff the sheet needs a `pass_fail` column.")
        sys.exit(1)
    # one clear hint instead of a hundred row errors when the command does not fit the sheet
    n_cryst = sum(bool((r.get("crystallinity") or "").strip()) for r in sheet_rows)
    n_pf = sum(bool((r.get(pf_col) or "").strip()) for r in sheet_rows)
    if not n_cryst and not n_pf:
        print(f"ABORT: {path.name} has no values yet -- fill in the crystallinity (or pass_fail) "
              "column and save it as CSV (README_RETEST.md, top section).")
        sys.exit(1)
    if cutoff is None and not n_pf:
        print("ABORT: the sheet has crystallinity numbers but no pass_fail values -- add your "
              "cutoff:  python apply_label_changes.py --formula --cutoff <your cutoff>")
        sys.exit(1)
    if cutoff is not None and not n_cryst:
        print("ABORT: the sheet has pass_fail values but no crystallinity numbers -- run it "
              "without the cutoff:  python apply_label_changes.py --formula")
        sys.exit(1)

    # --- check every sheet row; collect the new label of every photo that stays in the study
    problems, labels, extra, seen = [], {}, {}, set()
    for line, r in enumerate(sheet_rows, start=2):           # line 1 is the header
        key_id = (r.get("id") or "").strip()
        name = (r.get("original_name") or "").strip()
        photo_id = key_id if key_id else by_name.get(name)
        if not photo_id or photo_id not in pool:
            problems.append(f"line {line}: {key_id or name!r} is not a known photo")
            continue
        if photo_id in seen:
            problems.append(f"line {line}: {pool[photo_id][1]['original_name']} has two rows")
        seen.add(photo_id)
        shown = pool[photo_id][1]["original_name"]
        pf_raw = (r.get(pf_col) or "").strip()
        pf_code = pass_fail_code(pf_raw)
        if pf_code is None:
            problems.append(f"line {line}: {shown}: pass_fail {pf_raw!r} not understood -- "
                            "use PASS, FAIL or NA")
            continue
        if cutoff is not None:
            raw = (r.get("crystallinity") or "").strip()
            if raw.upper() in NA_VALUES:
                if pf_code not in ("", "NA"):
                    problems.append(f"line {line}: {shown}: crystallinity is NA but pass_fail "
                                    f"is {pf_raw!r}")
                continue                      # no value on purpose: leaves the study
            try:
                value = float(raw)
            except ValueError:
                value = math.nan
            if not math.isfinite(value):
                problems.append(f"line {line}: {shown}: crystallinity {raw!r} is not a number "
                                "-- enter a number (decimal point, no %) or NA")
                continue
            label = "1" if value >= cutoff else "0"
            # a pass_fail value typed next to the number must agree with the cutoff
            if pf_code not in ("", label):
                problems.append(f"line {line}: {shown}: pass_fail {pf_raw!r} disagrees with "
                                f"crystallinity {value} and cutoff {cutoff}")
        else:
            if pf_code == "NA":
                continue                      # no value on purpose: leaves the study
            if pf_code == "":
                problems.append(f"line {line}: {shown}: pass_fail is empty -- enter PASS, "
                                "FAIL or NA")
                continue
            label = pf_code
        labels[photo_id] = label
        # the remaining sheet columns (crystallinity, pass_fail, note, ...) go to the change log
        extra[photo_id] = {k: v for k, v in r.items()
                           if k and k not in ("id", "original_name", "label")
                           and isinstance(v, str)}
        if cutoff is not None:
            extra[photo_id]["cutoff"] = cutoff
    no_row = [pool[photo_id][1]["original_name"] for photo_id in order if photo_id not in seen]
    if no_row:
        problems.append(f"{len(no_row)} photo(s) have no row in the sheet: {no_row} -- run "
                        "`python apply_label_changes.py --template` (adds the rows and keeps "
                        "your values), then give each a value or NA")

    # --- every labelled photo needs its image in anon_images/ (or one to restore) -----------
    unused = C.ROOT / "anon_images_unused"
    restore, missing_img = [], []
    for photo_id in labels:
        if (C.IMAGES_DIR / f"{photo_id}{C.IMAGE_EXT}").exists():
            continue
        if (unused / f"{photo_id}{C.IMAGE_EXT}").exists():
            restore.append(photo_id)          # left out earlier / excluded: moved back on --apply
        else:
            missing_img.append(photo_id)
    for photo_id in missing_img:
        hint = ("re-run step1.py on the photo folder (it parks excluded photos in "
                "anon_images_unused/), see README_RETEST.md step 1"
                if pool[photo_id][0] == "excluded" else "was the image file deleted or renamed?")
        problems.append(f"{pool[photo_id][1]['original_name']}: no image anon_images/{photo_id}"
                        f"{C.IMAGE_EXT} -- {hint}")
    if problems:
        print(f"NOT APPLIED -- fix these in {path.name} first (nothing was changed):")
        for p in problems:
            print("  -", p)
        sys.exit(1)

    # --- new tables: labelled -> manifest; NA -> unlabeled (excluded rows stay excluded) ----
    log, new_man, new_unl, new_exc, to_unused, park = [], [], [], [], [], []
    for photo_id in order:
        src, r = pool[photo_id]
        if photo_id in labels:
            old = r["label"]
            log.append({"id": photo_id, "original_name": r["original_name"], "source": src,
                        "old_label": old, "new_label": labels[photo_id],
                        "changed": old != labels[photo_id], **extra[photo_id]})
            new_man.append({**r, "label": labels[photo_id]})
        elif src == "excluded":
            new_exc.append(r)
            park.append(photo_id)             # stays excluded: image (if any) out of the study
        else:
            new_unl.append({**r, "label": ""})
            to_unused.append(photo_id)
    was = [e for e in log if e["source"] == "manifest"]
    n_pass = sum(int(e["new_label"]) for e in log)
    print(f"new labels for {len(log)} photos: PASS {n_pass}, FAIL {len(log) - n_pass}")
    print(f"  already in the study: {len(was)} ({sum(e['changed'] for e in was)} labels changed: "
          f"{sum(e['changed'] and e['old_label'] == '0' for e in was)} FAIL->PASS, "
          f"{sum(e['changed'] and e['old_label'] == '1' for e in was)} PASS->FAIL)")
    print(f"  added from unlabeled.csv: {sum(e['source'] == 'unlabeled' for e in log)} | "
          f"back from excluded.csv: {sum(e['source'] == 'excluded' for e in log)}")
    left = [pool[photo_id][1]["original_name"] for photo_id in to_unused] + \
           [r["original_name"] for r in new_exc]
    if left:
        print(f"  NA -> NOT in the study ({len(left)}): {left}")
    if restore:
        print(f"  images moved back from anon_images_unused/ ({len(restore)}): "
              f"{[pool[photo_id][1]['original_name'] for photo_id in restore]}")
    if not apply:
        print("\nCheck only -- nothing written. If this looks right, run the same command "
              "again with --apply at the end.")
        return

    # --- --apply: back up, move images, write the tables and the change log ----------------
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = PROTO / f"backup_before_new_labels_{stamp}"
    backup_dir.mkdir(parents=True)
    for csv_path in files.values():
        if csv_path.exists():
            shutil.copy2(csv_path, backup_dir / csv_path.name)
    shutil.copy2(path, backup_dir / f"sheet_used_{path.name}")
    for photo_id in restore:
        shutil.move(str(unused / f"{photo_id}{C.IMAGE_EXT}"),
                    str(C.IMAGES_DIR / f"{photo_id}{C.IMAGE_EXT}"))
    write_rows(C.MANIFEST_CSV, fields, new_man)
    write_rows(files["unlabeled"], fields, new_unl)
    write_rows(C.EXCLUDED_CSV, fields, new_exc)
    for photo_id in to_unused + park:
        img = C.IMAGES_DIR / f"{photo_id}{C.IMAGE_EXT}"
        if img.exists():
            unused.mkdir(exist_ok=True)
            shutil.move(str(img), str(unused / img.name))
    log_path = PROTO / f"label_changes_new_labels_{stamp}.csv"
    log_fields = list(dict.fromkeys(k for e in log for k in e))   # union of keys, in order
    write_rows(log_path, log_fields, log)
    print(f"\nWRITTEN. Backups in {backup_dir.relative_to(C.ROOT)}, change log "
          f"{log_path.relative_to(C.ROOT)}.")
    print("Next: python data_checks.py (README_RETEST.md, step 4).")


def individual_changes_mode(apply: bool):
    """Mode 2: apply the rows of CHANGES (label_changes.csv) to the label column of
    manifest.csv. Creates an empty CHANGES file (exit 0) if there is none; exit 1 if any row
    fails a check."""
    if not CHANGES.exists():
        with open(CHANGES, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(FIELDS)
        print(f"Created an empty {CHANGES.relative_to(C.ROOT)} -- add one row per label change "
              "(edit it with a text editor, not Excel), then run this script again.\n"
              "For the re-test with new PASS/FAIL values use --template instead "
              "(README_RETEST.md, top section).")
        sys.exit(0)

    with open(C.MANIFEST_CSV, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields, rows = reader.fieldnames, list(reader)
    by_id = {r["id"]: r for r in rows}
    with open(CHANGES, newline="", encoding="utf-8-sig") as fh:
        changes = [c for c in csv.DictReader(fh) if any((v or "").strip() for v in c.values())]

    problems, seen = [], set()
    for line, change in enumerate(changes, start=2):          # line 1 is the header
        photo_id = (change.get("id") or "").strip()
        if photo_id in seen:
            problems.append(f"line {line}: id {photo_id} appears twice")
        seen.add(photo_id)
        if photo_id not in by_id:
            problems.append(f"line {line}: id {photo_id!r} is not in manifest.csv "
                            "(typo or Excel-mangled?)")
            continue
        if (change.get("old_label") or "").strip() != by_id[photo_id]["label"]:
            problems.append(f"line {line}: {photo_id} old_label {change.get('old_label')!r} != "
                            f"manifest label {by_id[photo_id]['label']!r}")
        if (change.get("new_label") or "").strip() not in ("0", "1"):
            problems.append(f"line {line}: {photo_id} new_label must be 0 or 1")
        elif change["new_label"].strip() == by_id[photo_id]["label"]:
            problems.append(f"line {line}: {photo_id} new_label equals the current label "
                            "(no change)")
        if not (change.get("reason") or "").strip():
            problems.append(f"line {line}: {photo_id} has no reason -- every change needs one")
    if problems:
        print("NOT APPLIED -- fix these rows in label_changes.csv first:")
        for p in problems:
            print("  -", p)
        sys.exit(1)

    print(f"{len(changes)} label change(s):")
    for change in changes:
        r = by_id[change["id"].strip()]
        print(f"  {r['id']}  {r['original_name']:40s} {r['label']} -> "
              f"{change['new_label'].strip()}   ({change['reason'].strip()})")
    n_success = sum(int(r["label"]) for r in rows)
    delta = sum(int(c["new_label"]) - int(by_id[c["id"].strip()]["label"]) for c in changes)
    print(f"successes: {n_success} -> {n_success + delta} of {len(rows)}")
    if not apply:
        print("\nDry run only. Run `python apply_label_changes.py --apply` to write manifest.csv.")
        return

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = PROTO / f"manifest_before_label_changes_{stamp}.csv"
    shutil.copy2(C.MANIFEST_CSV, backup)
    for change in changes:
        by_id[change["id"].strip()]["label"] = change["new_label"].strip()
    with open(C.MANIFEST_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nmanifest.csv updated (backup: {backup.relative_to(C.ROOT)}).")
    print("Next: python data_checks.py, then python data_split.py --confirm-rebuild "
          "(see README_RETEST.md).")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="write the files")
    ap.add_argument("--template", action="store_true",
                    help=f"make or update {SHEET.relative_to(C.ROOT)} (keeps typed values)")
    ap.add_argument("--formula", nargs="?", const=str(SHEET), metavar="CSV",
                    help="relabel EVERY photo from the sheet "
                         f"(default {SHEET.relative_to(C.ROOT)})")
    ap.add_argument("--cutoff", type=float,
                    help="with --formula: PASS if crystallinity >= CUTOFF, else FAIL")
    args = ap.parse_args()
    if args.cutoff is not None and not args.formula:
        print("ABORT: --cutoff only works together with --formula.")
        sys.exit(1)
    PROTO.mkdir(exist_ok=True)
    if args.template:
        write_template()
        return
    if args.formula:
        formula_mode(Path(args.formula), args.apply, args.cutoff)
        return
    individual_changes_mode(args.apply)


if __name__ == "__main__":
    main()
