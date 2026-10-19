"""
common.py -- helpers shared by every stage script (`import common`).

Provides
  * run log: everything printed is also appended to run_log.txt (config.RUN_LOG);
  * stage start banner / STOP banner, and the re-split reminder;
  * guards: manifest-hash check (aborts, no override), sealed-label isolation check (static
    scan of every project .py file), and a block on test images / test labels before the
    final test (stage 11);
  * safe CSV reading (ids are ALWAYS strings), split / fold / dev-label loaders;
  * image reading/writing, with images loaded BY ID ONLY (file names are never parsed);
  * small utilities: JSON, hashing, deterministic RNGs, rewriting the auto-managed lines of
    config.py, and yes/no confirmations that work interactively and from the command line.

Reads config.py settings and the split files written by data_split.py; writes run_log.txt and
(via set_auto_config) the "# [auto]" lines of config.py.
"""
from __future__ import annotations

import datetime as _dt
import functools
import hashlib
import inspect
import json
import os
import platform
import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import config as C

FINAL_TEST_DONE_JSON = C.ROOT / "final_test_done.json"   # written by final_test.py at the end


# -----------------------------------------------------------------------------
# Run log
# -----------------------------------------------------------------------------
class _Tee:
    """Write to the console AND to run_log.txt."""

    def __init__(self, stream, fh):
        self._stream, self._fh = stream, fh

    def write(self, s):
        self._stream.write(s)
        self._fh.write(s)
        return len(s)

    def flush(self):
        self._stream.flush()
        self._fh.flush()

    def isatty(self):
        return getattr(self._stream, "isatty", lambda: False)()

    def __getattr__(self, name):
        return getattr(self._stream, name)


_LOG_FH = None


def _start_log():
    """Redirect stdout/stderr through _Tee into run_log.txt (idempotent)."""
    global _LOG_FH
    if _LOG_FH is not None:
        return
    # Avoid UnicodeEncodeError on consoles with a narrow code page (e.g. Windows cp1252).
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass
    _LOG_FH = open(C.RUN_LOG, "a", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, _LOG_FH)
    sys.stderr = _Tee(sys.stderr, _LOG_FH)


def seed_everything(seed: int = C.SEED):
    """Seed Python's and NumPy's global RNGs (and PYTHONHASHSEED for child processes)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def rng_for(*keys) -> np.random.Generator:
    """Deterministic RNG for (SEED, *keys) -- independent of call order."""
    words = [C.SEED] + [int(hashlib.md5(str(k).encode()).hexdigest()[:8], 16) for k in keys]
    return np.random.default_rng(words)


# -----------------------------------------------------------------------------
# Stage start / stop
# -----------------------------------------------------------------------------
def start_stage(stage: str, needs_split: bool = True) -> Path | None:
    """Open the run log, print a header, run the global guards, return the figure folder.

    `needs_split=False` is for stages that run before (or independently of) data_split.py.
    Returns figures/<stage>/ for the numbered stages in config.STAGES, otherwise None.
    """
    _start_log()
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    script = Path(sys.argv[0]).name
    print("\n" + "=" * 78)
    print(f"STAGE {stage}   |   {script} {' '.join(sys.argv[1:])}   |   {now}")
    import sklearn   # imported here only to report its version
    print(f"python {platform.python_version()} | numpy {np.__version__} | pandas {pd.__version__}"
          f" | sklearn {sklearn.__version__} | opencv {cv2.__version__}")
    print(f"seed {C.SEED} | SPLIT_MODE {C.SPLIT_MODE} | profile {C.ACTIVE_PROFILE}")
    print("=" * 78)
    seed_everything()
    assert_sealed_isolation()
    if needs_split:
        require_files(C.SPLITS_CSV, C.FOLDS_CSV, C.LABELS_DEV_CSV, C.SPLITS_META_JSON,
                      hint="run data_split.py first")
        check_manifest_hash()
    if stage in C.STAGES:
        fig_dir = C.FIG_DIR / stage
        fig_dir.mkdir(parents=True, exist_ok=True)
        return fig_dir
    return None


def stop_here(stage: str, next_cmd: str | None = None):
    """Print the end-of-stage banner and, if given, the command for the next stage."""
    print("\n" + "-" * 78)
    if stage in C.STAGES:
        print(f"STOP after stage {stage}.  Figures: figures/{stage}/   "
              f"(progression: figures/pipeline_progression.png)")
    else:
        print(f"STOP after stage {stage}.")
    if next_cmd:
        print(f"Review the images and numbers, then type 'continue' to run:  python {next_cmd}")
    print("-" * 78)


def resplit_reminder() -> str:
    """The re-split banner, built from excluded.csv (a label change can change its contents)."""
    excluded = read_csv(C.EXCLUDED_CSV) if C.EXCLUDED_CSV.exists() else pd.DataFrame()
    if excluded.empty:
        return ("NOTE: no images are excluded (excluded.csv is empty). If images are added to\n"
                "manifest.csv later, you MUST rerun data_split.py and everything after it from\n"
                "scratch. Do not append them to existing splits.")
    groups = sorted(excluded["group"].unique())
    return (f"NOTE: {len(excluded)} images (condition{'s' if len(groups) > 1 else ''} "
            f"{', '.join(groups)}, see excluded.csv) are excluded. If\n"
            "you add them back to manifest.csv, you MUST rerun data_split.py and everything after\n"
            "it from scratch. Do not append them to existing splits.")


def require_files(*paths, hint: str = ""):
    """Exit with code 2 if any of `paths` is missing."""
    missing = [str(Path(p).name) for p in paths if not Path(p).exists()]
    if missing:
        print(f"ABORT: missing {', '.join(missing)}. {hint}")
        sys.exit(2)


def require_mask_approved():
    """Exit with code 2 unless stage 02 is complete (mask approved and showcase ids set)."""
    if not C.MASK_APPROVED or not C.TUBE_MASK_NPY.exists():
        print("ABORT: the tube mask has not been approved yet (stage 02). "
              "Review figures/02_mask/ and run `python mask.py --approve`.")
        sys.exit(2)
    if len(C.SHOWCASE_IDS) != 2 * C.SHOWCASE_N_RANDOM_PER_CLASS + C.SHOWCASE_N_USER:
        print("ABORT: config.SHOWCASE_IDS is not set yet (stage 02). "
              "Run `python mask.py --set-showcase <id id id id>` or `--set-showcase random`.")
        sys.exit(2)


# -----------------------------------------------------------------------------
# Hashes, JSON, CSV
# -----------------------------------------------------------------------------
def sha256_file(path) -> str:
    """Hex SHA-256 of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(o):
    """json.dumps `default=` hook for NumPy scalars/arrays, Paths, sets and tuples."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def json_dump(obj, path):
    """Write `obj` as indented UTF-8 JSON."""
    Path(path).write_text(json.dumps(obj, indent=2, default=_jsonable), encoding="utf-8")


def json_load(path):
    """Read a UTF-8 JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


# Columns always read as strings (ids can be all digits, e.g. '5390632840').
_STR_COLS = {"id": str, "group": str, "replicate_tag": str, "original_name": str,
             "split": str, "setting": str, "model": str, "params": str}


def read_csv(path, **kw) -> pd.DataFrame:
    """pd.read_csv that keeps ids and other text columns (_STR_COLS) as strings."""
    dtype = dict(_STR_COLS)
    dtype.update(kw.pop("dtype", {}))
    df = pd.read_csv(path, dtype=dtype, **kw)
    if "replicate_tag" in df.columns:
        df["replicate_tag"] = df["replicate_tag"].fillna("")
    return df


# -----------------------------------------------------------------------------
# Manifest guards
# -----------------------------------------------------------------------------
def check_manifest_hash():
    """Exit with code 3 if manifest.csv changed after the split was made (no override)."""
    meta = json_load(C.SPLITS_META_JSON)
    current = sha256_file(C.MANIFEST_CSV)
    if current != meta["manifest_sha256"]:
        print("\n" + "!" * 78)
        print("ABORT: manifest.csv has changed since the split was made.")
        print(f"  stored hash : {meta['manifest_sha256']}")
        print(f"  current hash: {current}")
        print("You must rerun `python data_split.py` FROM SCRATCH (it archives every previous")
        print("output) and then every later stage. There is no override flag.")
        print("!" * 78)
        sys.exit(3)


def load_manifest_with_labels() -> pd.DataFrame:
    """Only data_checks.py (stage 0) and data_split.py may parse the manifest label column."""
    caller = Path(inspect.stack()[1].filename).name
    if caller not in ("data_checks.py", "data_split.py"):
        raise PermissionError(f"{caller}: after data_split.py no script may parse the label "
                              "column of manifest.csv -- read labels_dev.csv instead.")
    return read_csv(C.MANIFEST_CSV, usecols=["id", "label", "group", "replicate_tag"],
                    dtype={"label": str})


# -----------------------------------------------------------------------------
# Sealed-label isolation (static check of every project .py file, run at every stage)
# -----------------------------------------------------------------------------
_SEALED_TOKEN = "labels_test_" + "SEALED"          # built by concatenation so that this
_SEALED_CONST = "SEALED_TEST_" + "LABELS_CSV"      # checker does not match itself
_SEALED_LOADER = "_load_sealed_" + "test_labels"


def assert_sealed_isolation():
    """Fail unless only config.py, data_split.py and final_test.py mention the sealed file,
    only final_test.py mentions the sealed-label loader, no script imports final_test, and
    data_split.py never reads the sealed file (it may only write/archive it)."""
    problems = []
    for py in sorted(C.ROOT.glob("*.py")):
        src = py.read_text(encoding="utf-8", errors="ignore")
        name = py.name
        if (_SEALED_TOKEN in src or _SEALED_CONST in src) and name not in (
                "config.py", "data_split.py", "final_test.py"):
            problems.append(f"{name} refers to the sealed test-label file")
        if _SEALED_LOADER in src and name != "final_test.py":
            problems.append(f"{name} refers to the sealed-label loader")
        if name != "final_test.py" and re.search(
                r"^\s*(import\s+final_test|from\s+final_test\s+import)", src, re.M):
            problems.append(f"{name} imports final_test")
        if name == "data_split.py" and re.search(
                r"(read_csv|open|read_text|loadtxt|read_bytes)\s*\(\s*C\." + _SEALED_CONST, src):
            problems.append("data_split.py READS the sealed file (it may only write/archive it)")
    assert not problems, "Sealed-label isolation violated:\n  " + "\n  ".join(problems)


# -----------------------------------------------------------------------------
# Splits, folds, dev labels
# -----------------------------------------------------------------------------
def load_splits() -> pd.DataFrame:
    """splits.csv: id, split, group (no labels)."""
    return read_csv(C.SPLITS_CSV)


@functools.lru_cache(maxsize=1)
def _split_sets():
    """(dev ids, test ids) as frozensets; both empty before data_split.py has run.
    Cached for the life of the process."""
    if not C.SPLITS_CSV.exists():
        return frozenset(), frozenset()
    splits = load_splits()
    return (frozenset(splits.loc[splits["split"] != "test", "id"]),
            frozenset(splits.loc[splits["split"] == "test", "id"]))


def dev_ids() -> list[str]:
    """Sorted ids of every non-test image (CV / train / val)."""
    splits = load_splits()
    return sorted(splits.loc[splits["split"] != "test", "id"])


def test_ids() -> list[str]:
    """Sorted ids of the sealed test images."""
    splits = load_splits()
    return sorted(splits.loc[splits["split"] == "test", "id"])


def load_folds() -> pd.DataFrame:
    """folds.csv: id, repeat, fold, ... (no labels)."""
    return read_csv(C.FOLDS_CSV)


def load_dev_labels() -> pd.Series:
    """id -> 0/1 for DEV ids only (the only label source after data_split.py)."""
    df = read_csv(C.LABELS_DEV_CSV)
    return pd.Series(df["label"].astype(int).values, index=df["id"].values, name="label")


def iter_cv_folds(folds: pd.DataFrame | None = None):
    """Yield (repeat, fold, train_ids, heldout_ids). fold == -1 means 'never held out'
    (the train part of SPLIT_MODE fixed_60_20_20)."""
    folds = load_folds() if folds is None else folds
    for rep in sorted(folds["repeat"].unique()):
        rep_folds = folds[folds["repeat"] == rep]
        for fold in sorted(rep_folds.loc[rep_folds["fold"] >= 0, "fold"].unique()):
            held = sorted(rep_folds.loc[rep_folds["fold"] == fold, "id"])
            train = sorted(rep_folds.loc[rep_folds["fold"] != fold, "id"])
            yield int(rep), int(fold), train, held


# -----------------------------------------------------------------------------
# Test-set access guards
# -----------------------------------------------------------------------------
_TEST_IMAGES_UNLOCKED = False
_TEST_LABELS_UNSEALED = False


def final_test_completed() -> bool:
    """True once final_test.py has finished (its marker file exists)."""
    return FINAL_TEST_DONE_JSON.exists()


def unlock_test_images():
    """Only final_test.py (after the operator confirms with 'yes') may featurize test images."""
    global _TEST_IMAGES_UNLOCKED
    caller = Path(inspect.stack()[1].filename).name
    if caller != "final_test.py":
        raise PermissionError(f"{caller} may not unlock test images")
    _TEST_IMAGES_UNLOCKED = True


def mark_test_labels_unsealed():
    """Called by final_test.py only AFTER predictions are locked and labels loaded."""
    global _TEST_LABELS_UNSEALED
    caller = Path(inspect.stack()[1].filename).name
    if caller != "final_test.py":
        raise PermissionError(f"{caller} may not unseal test labels")
    _TEST_LABELS_UNSEALED = True


def test_images_allowed() -> bool:
    """Test images may be loaded during or after the final test."""
    return _TEST_IMAGES_UNLOCKED or final_test_completed()


def test_labels_visible() -> bool:
    """Test labels / original names may be shown after predictions are locked."""
    return _TEST_LABELS_UNSEALED or final_test_completed()


def assert_dev_only(ids, what: str = "this step"):
    """Raise PermissionError if `ids` contains test ids while test images are still blocked."""
    _, test = _split_sets()
    bad = sorted(set(ids) & test)
    if bad and not test_images_allowed():
        raise PermissionError(f"{what}: {len(bad)} TEST id(s) requested before stage 11")


# -----------------------------------------------------------------------------
# Images (loaded BY ID ONLY -- filenames are never parsed)
# -----------------------------------------------------------------------------
def read_rgb(path) -> np.ndarray:
    """Read an image as RGB uint8. np.fromfile + imdecode also handles non-ASCII paths."""
    data = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"unreadable image: {Path(path).name}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_rgb(path, rgb: np.ndarray):
    """Write an RGB uint8 image (format from the suffix), creating parent folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise IOError(f"could not encode {path.name}")
    buf.tofile(str(path))


def image_path(img_id: str) -> Path:
    """anon_images/<id>.jpg -- no guard; use load_image() inside the pipeline."""
    return C.IMAGES_DIR / f"{img_id}{C.IMAGE_EXT}"


def load_image(img_id: str) -> np.ndarray:
    """Raw RGB uint8 image for an id. Test images are blocked before stage 11."""
    _, test = _split_sets()
    if img_id in test and not test_images_allowed():
        raise PermissionError(f"test image {img_id} requested before stage 11")
    return read_rgb(image_path(img_id))


def load_masks():
    """(tube mask, reference mask), both bool HxW arrays from stage 02."""
    return np.load(C.TUBE_MASK_NPY), np.load(C.REF_MASK_NPY)


# -----------------------------------------------------------------------------
# Auto-managed config lines and confirmations
# -----------------------------------------------------------------------------
def set_auto_config(name: str, value):
    """Rewrite the line `NAME = ...  # [auto] ...` in config.py (keeps the comment) and
    update the already-imported config module."""
    path = C.ROOT / "config.py"
    src = path.read_text(encoding="utf-8")
    pat = re.compile(rf"^{re.escape(name)} = .*?(  # \[auto\].*)$", re.M)
    if not pat.search(src):
        raise RuntimeError(f"auto-managed line for {name} not found in config.py")
    src = pat.sub(lambda m: f"{name} = {value!r}{m.group(1)}", src, count=1)
    path.write_text(src, encoding="utf-8")
    setattr(C, name, value)


def confirm(question: str, flag_given: bool, flag_name: str, word: str = "yes") -> bool:
    """Interactive prompt in a terminal; from a non-interactive shell the flag is the answer."""
    print(question)
    if flag_given:
        print(f"  -> answered '{word}' via {flag_name}")
        return True
    if sys.stdin is not None and sys.stdin.isatty():
        try:
            return input(f"Type '{word}' to confirm: ").strip().lower() == word
        except EOFError:       # e.g. Windows NUL stdin reports isatty() but has no input
            print()
    print(f"  (non-interactive run: re-run with {flag_name} to answer '{word}')")
    return False
