"""Fine-tune the plate OCR on Philippine plates: split, prepare, train, export, compare.

    python tools/ocr_data/finetune.py split   [--labels datasets/plates/labels.csv] [--seed 1]
                                              [--holdout-source NAME] [--reassign]
    python tools/ocr_data/finetune.py prepare --name ph1
    python tools/ocr_data/finetune.py train   --name ph1 [--run] [--epochs 60]
    python tools/ocr_data/finetune.py export  --name ph1 --new-name cct-ph-v1
    python tools/ocr_data/finetune.py compare --new cct-ph-v1 --out runs/ph1/compare
    python tools/ocr_data/finetune.py --synthetic-smoke   (pipeline check on fake plates)

Needs the training extras (fast-plate-ocr[train]; Keras 3 + TensorFlow, CPU).

Pretrained weights: the shipped cct-xs models exist here only as ONNX. The
trainer's --weights-path wants Keras weights (.keras / .weights.h5), and ONNX
cannot be turned back into them, so by default we train FROM SCRATCH on a small
CCT. That needs far more plates than fine-tuning would (thousands, not hundreds);
treat results on a small set with suspicion, and let `compare` decide.

CPU time (8 cores, measured): the "small" model trains at ~0.4 s per 32-plate step
(plus augmentation, which runs on the loader threads), so 3000 training plates is
about a minute per epoch and a 60-epoch run 1-2 hours. "tiny" is ~6x faster.
A 1-epoch end-to-end smoke run on 200 fake plates takes under a minute.

compare reads every held-out crop through the real PlateEngine path and ends
with ADOPT only if the new model (alone or in the ensemble) beats the shipped
ensemble by >= 3 points exact match, is not worse on either vehicle type, and
the test split has >= 50 plates.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

COLUMNS = ["image_path", "plate_text", "suggested_text", "suggested_conf", "vehicle", "source",
           "license", "orig_image", "box", "verified", "split"]
SPLITS = ("train", "val", "test")
RATIOS = (0.8, 0.1, 0.1)
MIN_TOTAL = 300           # fewer verified plates than this: results are not trustworthy
MIN_SPLIT = 30            # fewer plates than this in one split: its numbers are noise
ALPHABET = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
MAX_SLOTS = 10
BASE_PLATE_YAML = "cct-xs-v2-global-model"
SHIPPED = ["cct-xs-v1-global-model", "cct-xs-v2-global-model"]
DETECTOR = "yolo-v9-t-384-license-plate-end2end"
MIN_TEST, MIN_GAIN = 50, 3.0
DEFAULT_LABELS = "datasets/plates/labels.csv"


# ----- labels.csv -----------------------------------------------------------

def read_labels(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or COLUMNS)
        return fields, [{k: (v or "") for k, v in row.items() if k is not None} for row in reader]


def write_labels(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    """Write via a temp file in the same folder, so a crash never leaves half a CSV."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def is_verified(row: dict[str, str]) -> bool:
    return row.get("verified", "").strip() == "1" and bool(row.get("plate_text", "").strip())


def clean_text(row: dict[str, str]) -> str:
    return row.get("plate_text", "").strip().upper()


# ----- (1) split ------------------------------------------------------------

def assign_splits(rows: list[dict[str, str]], seed: int = 1, ratios=RATIOS,
                  holdout_source: str | None = None, reassign: bool = False) -> int:
    """Fill `split` on verified rows; returns how many rows were newly assigned.

    Rows sharing a plate text form a group (same vehicle) and are never separated.
    Groups are dealt out per vehicle type to hit the ratios, and val/test each get
    at least one group of a type that has 3+ groups. Existing splits are kept
    (unless reassign), and a new photo of an already-placed plate follows it.
    """
    elig = [r for r in rows if is_verified(r)]
    if reassign:
        for r in elig:
            r["split"] = ""
    placed: dict[str, str] = {}          # plate text -> split already used
    for r in elig:
        if r.get("split") in SPLITS:
            placed.setdefault(clean_text(r), r["split"])
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in elig:
        if r.get("split") not in SPLITS:
            groups[clean_text(r)].append(r)

    votes: dict[str, Counter] = defaultdict(Counter)
    for r in elig:
        if r.get("vehicle"):
            votes[clean_text(r)][r["vehicle"]] += 1

    def vehicle(text: str) -> str:
        return votes[text].most_common(1)[0][0] if votes[text] else ""

    counts: dict[str, Counter] = defaultdict(Counter)   # vehicle -> split -> rows
    for r in elig:
        if r.get("split") in SPLITS:
            counts[vehicle(clean_text(r))][r["split"]] += 1

    todo: dict[str, list[str]] = defaultdict(list)       # vehicle -> new plate texts
    assigned = 0

    def put(text: str, split: str, veh: str) -> None:
        nonlocal assigned
        for r in groups[text]:
            r["split"] = split
        counts[veh][split] += len(groups[text])
        assigned += len(groups[text])

    held = (holdout_source or "").strip()
    for text in sorted(groups):
        veh = vehicle(text)
        sources = {s for r in groups[text] for s in r.get("source", "").split("|")}
        if text in placed:
            put(text, placed[text], veh)
        elif held and held in sources:
            put(text, "test", veh)
        else:
            todo[veh].append(text)

    rng = random.Random(seed)
    for veh in sorted(todo):
        texts = todo[veh]
        rng.shuffle(texts)
        total = sum(len(groups[t]) for t in texts) + sum(counts[veh].values())
        n_groups = len(texts) + sum(1 for t in placed if vehicle(t) == veh)
        if n_groups >= 3:                                  # make val and test non-empty first
            for s in ("test", "val"):
                if not counts[veh][s] and texts:
                    put(texts.pop(), s, veh)
        for text in texts:
            need = [ratios[i] * total - counts[veh][s] for i, s in enumerate(SPLITS)]
            put(text, SPLITS[need.index(max(need))], veh)
    return assigned


def split_summary(rows: list[dict[str, str]]) -> tuple[list[str], list[str]]:
    """Report lines and WARN lines for the verified, split rows."""
    lines, warns = [], []
    ver = [r for r in rows if is_verified(r)]
    done = [r for r in ver if r.get("split") in SPLITS]
    lines.append(f"verified plates: {len(ver)} ({len({clean_text(r) for r in ver})} distinct texts), "
                 f"with a split: {len(done)}")
    for s in SPLITS:
        sub = [r for r in done if r["split"] == s]
        veh = Counter(r.get("vehicle") or "?" for r in sub)
        src = Counter(r.get("source") or "?" for r in sub)
        lines.append(f"  {s:<5} {len(sub):>5}  vehicle {dict(sorted(veh.items()))}  source {dict(sorted(src.items()))}")
        if len(sub) < MIN_SPLIT:
            warns.append(f"the {s} split has only {len(sub)} plates (< {MIN_SPLIT}): its numbers are noise")
        for v in ("car", "motorcycle"):
            if sub and veh[v] == 0 and any(r.get("vehicle") == v for r in done):
                warns.append(f"the {s} split has no {v} plates")
    if len(ver) < MIN_TOTAL:
        warns.append(f"only {len(ver)} verified plates (< {MIN_TOTAL}): too few to trust a trained model "
                     "or its comparison")
    return lines, warns


def print_warnings(warns: list[str]) -> None:
    for w in warns:
        print(f"\n*** WARNING: {w} ***", file=sys.stderr)
    sys.stderr.flush()


def cmd_split(a: argparse.Namespace) -> int:
    path = Path(a.labels)
    fields, rows = read_labels(path)
    for c in COLUMNS:
        if c not in fields:
            fields.append(c)
    n = assign_splits(rows, a.seed, holdout_source=a.holdout_source, reassign=a.reassign)
    if not a.dry_run:
        write_labels(path, fields, rows)
    lines, warns = split_summary(rows)
    print(f"{'would assign' if a.dry_run else 'assigned'} {n} rows (seed {a.seed})")
    print("\n".join(lines))
    print_warnings(warns)
    return 0


# ----- (2) prepare ----------------------------------------------------------

CCT_SIZES = {
    # name: (conv filters, transformer layers, projection dim); last filters == dim; input 64x128 is pooled 8x to 8x16 tokens
    "tiny": ([16, 32, 48], 1, 48),
    "small": ([32, 64, 128], 3, 128),
}


def model_config_yaml(size: str) -> str:
    """A CCT model config (library schema) small enough to train on a CPU.

    silu, not gelu: exact GELU exports an Erfc op that ONNX Runtime does not have.
    """
    filters, layers, dim = CCT_SIZES[size]
    blocks = []
    for f in filters:
        blocks.append(f"      - {{layer: Conv2D, filters: {f}, kernel_size: 3, strides: 1, padding: same, "
                      "activation: relu}\n      - {layer: MaxPooling2D, pool_size: 2}")
    return ("# Small CCT for CPU training (see tools/ocr_data/finetune.py)\n"
            "model: cct\nrescaling: {scale: 0.00392156862745098, offset: 0.0}\ntokenizer:\n"
            "  blocks:\n" + "\n".join(blocks) + "\n  patch_size: 1\n  positional_emb: true\n"
            f"transformer_encoder:\n  layers: {layers}\n  heads: 4\n  projection_dim: {dim}\n"
            f"  units: [{dim * 2}, {dim}]\n  activation: silu\n  stochastic_depth: 0.05\n"
            "  attention_dropout: 0.1\n  mlp_dropout: 0.1\n  head_mlp_dropout: 0.2\n"
            "  token_reducer_heads: 2\n  normalization: layer_norm\n")


def build_augmentation():
    """Webcam-style augmentation (albumentations; saved with A.save for --augmentation-path)."""
    import albumentations as A
    import cv2
    return A.Compose([
        A.Affine(translate_percent=(-0.03, 0.03), scale=(0.8, 1.1), rotate=(-10, 10), shear=(-6, 6),
                 border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.7),
        A.Perspective(scale=(0.02, 0.08), fill=0, p=0.5),
        A.OneOf([A.MotionBlur(blur_limit=(3, 9), p=1), A.GaussianBlur(blur_limit=(3, 7), p=1),
                 A.Defocus(radius=(1, 3), p=1)], p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.7),
        A.RandomGamma(gamma_limit=(70, 140), p=0.3),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=10, p=0.25),
        A.ToGray(p=0.1),
        A.OneOf([A.GaussNoise(std_range=(0.02, 0.1), p=1), A.ISONoise(intensity=(0.01, 0.05), p=1)], p=0.4),
        A.Downscale(scale_range=(0.4, 0.9), p=0.4),
        A.ImageCompression(quality_range=(30, 90), p=0.4),
    ])


def resolve_image(labels_dir: Path, image_path: str) -> Path:
    p = Path(image_path)
    if p.is_absolute():
        return p
    return labels_dir / p if (labels_dir / p).exists() else Path.cwd() / p


def annotation_rows(rows: list[dict[str, str]], split: str, labels_dir: Path, out_dir: Path
                    ) -> tuple[list[tuple[str, str]], int]:
    """(image path relative to out_dir, text) for one split, and how many rows were skipped.

    The trainer joins the CSV's folder and image_path, so paths must be relative.
    """
    out, skipped = [], 0
    for r in rows:
        if not is_verified(r) or r.get("split") != split:
            continue
        text, img = clean_text(r), resolve_image(labels_dir, r.get("image_path", ""))
        if not text or len(text) > MAX_SLOTS or not set(text) <= ALPHABET or not img.is_file():
            skipped += 1
            continue
        try:
            rel = os.path.relpath(img, out_dir)
        except ValueError:                                  # other drive: copy next to the CSV
            dest = out_dir / "images" / img.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, dest)
            rel = os.path.relpath(dest, out_dir)
        out.append((rel, text))
    return out, skipped


def write_annotations(path: Path, pairs: list[tuple[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "plate_text"])
        w.writerows(pairs)


def run_dir(a: argparse.Namespace) -> Path:
    return Path(a.runs) / a.name


def cmd_prepare(a: argparse.Namespace) -> int:
    labels = Path(a.labels)
    _, rows = read_labels(labels)
    out = run_dir(a)
    out.mkdir(parents=True, exist_ok=True)
    for split, fname in (("train", "train.csv"), ("val", "val.csv")):
        pairs, skipped = annotation_rows(rows, split, labels.parent, out)
        if not pairs:
            print(f"no verified {split} rows: run `split` first", file=sys.stderr)
            return 1
        write_annotations(out / fname, pairs)
        print(f"{fname}: {len(pairs)} plates" + (f" ({skipped} skipped: bad text or missing image)" if skipped else ""))
    src = next((Path(a.models_dir) / "alpr" / BASE_PLATE_YAML).glob("*.yaml"), None)
    if src is None:
        print(f"missing plate config under {Path(a.models_dir) / 'alpr' / BASE_PLATE_YAML}", file=sys.stderr)
        return 1
    shutil.copy2(src, out / "plate_config.yaml")           # unchanged, so the model drops into PlateEngine
    (out / "model_config.yaml").write_text(model_config_yaml(a.size), encoding="utf-8")
    build_augmentation_file(out / "augmentation.yaml")
    print(f"wrote {out} (plate_config.yaml copied from {src.name}; model '{a.size}')")
    return 0


def build_augmentation_file(path: Path) -> None:
    import albumentations as A
    A.save(build_augmentation(), path, data_format="yaml")


# ----- (3) train ------------------------------------------------------------

def train_command(a: argparse.Namespace) -> list[str]:
    out = run_dir(a)
    cmd = [sys.executable, "-m", "fast_plate_ocr.cli.train",
           "--model-config-file", str(out / "model_config.yaml"),
           "--plate-config-file", str(out / "plate_config.yaml"),
           "--annotations", str(out / "train.csv"), "--val-annotations", str(out / "val.csv"),
           "--augmentation-path", str(out / "augmentation.yaml"),
           "--output-dir", str(out / "train"),
           "--epochs", str(a.epochs), "--batch-size", str(a.batch_size),
           "--workers", str(a.workers), "--lr", str(a.lr),
           "--early-stopping-patience", str(a.patience), "--early-stopping-metric", "val_plate_acc",
           "--validate-dataset", "warn", "--seed", str(a.seed)]
    if a.weights:
        cmd += ["--weights-path", str(a.weights)]
    return cmd


def cmd_train(a: argparse.Namespace) -> int:
    out = run_dir(a)
    for need in ("train.csv", "val.csv", "plate_config.yaml", "model_config.yaml", "augmentation.yaml"):
        if not (out / need).is_file():
            print(f"{out / need} missing: run `prepare --name {a.name}` first", file=sys.stderr)
            return 1
    cmd = train_command(a)
    print("=" * 78)
    if a.weights:
        print(f"NOTE: starting from Keras weights {a.weights} (loaded with skip_mismatch; it must match the\n"
              f"model config in {out / 'model_config.yaml'}).")
    else:
        print("NOTE: TRAINING FROM SCRATCH. The shipped OCR models are ONNX-only and the trainer's\n"
              "--weights-path needs Keras weights (.keras / .weights.h5); ONNX cannot be converted back,\n"
              "and the original Keras checkpoints are not available offline. A from-scratch model on a few\n"
              "hundred plates will usually LOSE to the shipped ones: trust only `compare`.")
    print("=" * 78)
    print("command:\n  " + subprocess.list2cmdline(cmd))
    if not a.run:
        print("\n(not run; add --run). CPU time: about 0.4 s per 32-plate step for the small model.")
        return 0
    env = dict(os.environ, PYTHONUNBUFFERED="1", TF_CPP_MIN_LOG_LEVEL="2", TF_ENABLE_ONEDNN_OPTS="1")
    t0 = time.time()
    code = subprocess.call(cmd, env=env, cwd=ROOT)
    print(f"training finished in {time.time() - t0:.0f} s (exit {code})")
    return code


# ----- (4) export -----------------------------------------------------------

def find_checkpoint(out: Path) -> Path | None:
    found = sorted((out / "train").glob("*/best.keras"))
    return found[-1] if found else None


# The library's exporter reopens a NamedTemporaryFile that is still open, which Windows forbids;
# this shim swaps in a closed temp file, then runs the library's own export CLI.
EXPORT_SHIM = """
import os, sys, tempfile
import fast_plate_ocr.cli.export as ex
class _Tmp:
    def __init__(self, suffix=""):
        fd, self.name = tempfile.mkstemp(suffix=suffix); os.close(fd)
    def __enter__(self): return self
    def __exit__(self, *exc):
        try: os.unlink(self.name)
        except OSError: pass
ex.NamedTemporaryFile = _Tmp
ex.export(prog_name="export")
"""


def cmd_export(a: argparse.Namespace) -> int:
    dest = Path(a.models_dir) / "alpr" / a.new_name
    if dest.exists():
        print(f"refusing: {dest} already exists (pick another --new-name; shipped models are never overwritten)",
              file=sys.stderr)
        return 1
    out = run_dir(a)
    ckpt = Path(a.checkpoint) if a.checkpoint else find_checkpoint(out)
    if ckpt is None or not ckpt.is_file():
        print(f"no best.keras under {out / 'train'}: train first", file=sys.stderr)
        return 1
    stage = out / "export"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    cfg = out / "plate_config.yaml"
    cmd = [sys.executable, "-c", EXPORT_SHIM, "-m", str(ckpt), "-f", "onnx",
           "--plate-config-file", str(cfg), "--save-dir", str(stage)]
    code = subprocess.call(cmd, cwd=ROOT, env=dict(os.environ, TF_CPP_MIN_LOG_LEVEL="2"))
    onnx = next(stage.glob("*.onnx"), None)
    if code or onnx is None:
        print("export failed", file=sys.stderr)
        return code or 1
    dest.mkdir(parents=True)
    shutil.copy2(onnx, dest / f"{a.new_name}.onnx")
    shutil.copy2(cfg, dest / f"{a.new_name}_plate_config.yaml")
    print(f"exported {ckpt} -> {dest}")
    return 0


# ----- (5) compare ----------------------------------------------------------

def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score(records: list[dict]) -> dict[str, float]:
    """n, exact-match %, character error rate % (edits / true characters), mean confidence."""
    n = len(records)
    if not n:
        return {"n": 0, "exact": 0.0, "cer": 0.0, "conf": 0.0}
    edits = sum(edit_distance(r["pred"], r["truth"]) for r in records)
    chars = sum(len(r["truth"]) for r in records) or 1
    return {"n": n, "exact": 100.0 * sum(r["pred"] == r["truth"] for r in records) / n,
            "cer": 100.0 * edits / chars, "conf": sum(r["conf"] for r in records) / n}


def group_scores(records: list[dict], key: str) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[r.get(key) or "?"].append(r)
    return {k: score(v) for k, v in sorted(groups.items())}


def decide(by_model: dict[str, list[dict]], base: str, candidates: list[str]) -> tuple[str, str]:
    """(verdict line, reason). ADOPT needs +3 points exact, no vehicle type worse, >= 50 test plates."""
    rule = (f"rule: ADOPT only if a new model (alone or in the ensemble) beats '{base}' by >= {MIN_GAIN:g} "
            f"points exact match, is not worse on either vehicle type, and the test split has >= {MIN_TEST} plates")
    n = len(by_model[base])
    if n < MIN_TEST:
        return f"NOT PROVEN BETTER: only {n} test plates (< {MIN_TEST}).", rule
    b = score(by_model[base])["exact"]
    bveh = {k: v["exact"] for k, v in group_scores(by_model[base], "vehicle").items()}
    best = None
    for c in candidates:
        s = score(by_model[c])["exact"]
        cveh = {k: v["exact"] for k, v in group_scores(by_model[c], "vehicle").items()}
        worse = [k for k in ("car", "motorcycle") if k in bveh and cveh.get(k, 0.0) < bveh[k]]
        if s - b >= MIN_GAIN and not worse and (best is None or s > best[1]):
            best = (c, s)
    if best:
        return f"ADOPT {best[0]}: exact {best[1]:.1f}% vs {b:.1f}% for the shipped ensemble.", rule
    return "NOT PROVEN BETTER: no candidate cleared the rule on the held-out test split.", rule


def load_crop(path: Path):
    import cv2
    data = np.fromfile(str(path), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def build_engine(models_dir: Path, ocr: str):
    from platescanner.config import load_config
    from platescanner.vision.alpr import PlateEngine
    cfg = load_config().ocr
    eng = PlateEngine(models_dir, DETECTOR, ocr, cfg.detector_confidence, cfg.plate_layouts, cfg.deblur,
                      cfg.deskew, cfg.enhance)
    eng.load()
    return eng


def read_all(engine, rows: list[dict[str, str]], labels_dir: Path) -> list[dict]:
    out = []
    for r in rows:
        crop = load_crop(resolve_image(labels_dir, r.get("image_path", "")))
        res = engine.read(crop) if crop is not None else None
        out.append({"image": r.get("image_path", ""), "truth": clean_text(r), "vehicle": r.get("vehicle", ""),
                    "source": r.get("source", ""), "pred": res.text if res else "",
                    "conf": res.confidence if res else 0.0})
    return out


def fmt_row(name: str, s: dict[str, float]) -> str:
    return f"| {name} | {s['n']} | {s['exact']:.1f} | {s['cer']:.1f} | {s['conf']:.3f} |"


def report_markdown(by_model: dict[str, list[dict]], base: str, verdict: str, rule: str) -> str:
    head = "| | plates | exact % | CER % | mean conf |\n|---|---|---|---|---|"
    lines = ["# OCR comparison on the held-out test split", "", f"**{verdict}**", "", f"_{rule}_", "",
             "## Overall", "", head]
    lines += [fmt_row(m, score(rs)) for m, rs in by_model.items()]
    for key, title in (("vehicle", "By vehicle type"), ("source", "By source")):
        lines += ["", f"## {title}", "", head]
        for m, rs in by_model.items():
            lines += [fmt_row(f"{m} / {k}", s) for k, s in group_scores(rs, key).items()]
    base_rows = by_model[base]
    for m, rs in by_model.items():
        if m == base:
            continue
        worse = [(b, r) for b, r in zip(base_rows, rs)
                 if edit_distance(r["pred"], r["truth"]) > edit_distance(b["pred"], b["truth"])]
        lines += ["", f"## Where `{m}` is worse than `{base}` ({len(worse)})", "",
                  "| image | truth | base | new |\n|---|---|---|---|"]
        lines += [f"| {r['image']} | {r['truth']} | {b['pred']} | {r['pred']} |" for b, r in worse[:60]]
    return "\n".join(lines) + "\n"


def write_rows_csv(path: Path, by_model: dict[str, list[dict]]) -> None:
    models = list(by_model)
    first = by_model[models[0]]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "plate_text", "vehicle", "source"]
                   + [f"{m}:{x}" for m in models for x in ("pred", "conf")])
        for i, r in enumerate(first):
            w.writerow([r["image"], r["truth"], r["vehicle"], r["source"]]
                       + [v for m in models for v in (by_model[m][i]["pred"], f"{by_model[m][i]['conf']:.3f}")])


def cmd_compare(a: argparse.Namespace) -> int:
    labels = Path(a.labels)
    _, rows = read_labels(labels)
    test = [r for r in rows if is_verified(r) and r.get("split") == "test"]
    if not test:
        print("no verified test rows: run `split` first", file=sys.stderr)
        return 1
    base = a.base
    candidates = [a.new, f"{base},{a.new}"] if a.new else []
    names = list(dict.fromkeys([*a.models, base, *candidates]))
    models_dir = Path(a.models_dir)
    by_model = {}
    for m in names:
        print(f"reading {len(test)} test crops with {m} ...")
        by_model[m] = read_all(build_engine(models_dir, m), test, labels.parent)
    verdict, rule = decide(by_model, base, candidates)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(report_markdown(by_model, base, verdict, rule), encoding="utf-8")
    write_rows_csv(out / "rows.csv", by_model)
    for m, rs in by_model.items():
        s = score(rs)
        print(f"{m:<60} exact {s['exact']:5.1f}%  CER {s['cer']:5.1f}%  conf {s['conf']:.3f}")
    print(f"wrote {out / 'report.md'} and {out / 'rows.csv'}")
    print(f"\nRECOMMENDATION: {verdict}\n({rule})")
    return 0


# ----- synthetic smoke test -------------------------------------------------

def synthetic_dataset(root: Path, n: int, seed: int) -> Path:
    """~n tiny rendered plates (cars LLLDDDD, motorcycles DDDLLL; repeats share a text) + labels.csv."""
    import cv2
    sys.path.insert(0, str(ROOT / "tools"))
    import bench_conditions as bc
    rng = random.Random(seed)
    img_dir = root / "images"
    img_dir.mkdir(parents=True)
    texts, rows = [], []
    for i in range(n):
        if texts and rng.random() < 0.2:
            text, veh = rng.choice(texts)                   # another photo of the same vehicle
        else:
            veh = rng.choice(["car", "motorcycle"])
            text = ("".join(rng.choices("ABCDEFGHJKLMNPRSTUVWXYZ", k=3)) + "".join(rng.choices("0123456789", k=4))
                    if veh == "car" else
                    "".join(rng.choices("0123456789", k=3)) + "".join(rng.choices("ABCDEFGHJKLMNPRSTUVWXYZ", k=3)))
            texts.append((text, veh))
        img = bc.combined(rng, bc.render_plate(rng, text)) if rng.random() < 0.5 else bc.render_plate(rng, text)
        rel = f"images/{i:04d}.jpg"
        cv2.imwrite(str(root / rel), img)
        rows.append({"image_path": rel, "plate_text": text, "vehicle": veh, "verified": "1",
                     "source": "synthetic_a" if i % 2 else "synthetic_b", "license": "n/a"})
    path = root / "labels.csv"
    write_labels(path, COLUMNS, [{c: r.get(c, "") for c in COLUMNS} for r in rows])
    return path


def cmd_smoke(a: argparse.Namespace) -> int:
    """Run split -> prepare -> train (1 epoch) -> export -> compare on fake plates in a temp dir."""
    t_all = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="finetune-smoke-"))
    try:
        labels = synthetic_dataset(tmp / "data", a.n, 7)
        models = tmp / "models"
        shutil.copytree(Path(a.models_dir) / "alpr" / BASE_PLATE_YAML, models / "alpr" / BASE_PLATE_YAML)
        shutil.copy2(next((Path(a.models_dir) / "alpr").glob("yolo-v9-t-384*.onnx")), models / "alpr")
        common = dict(labels=str(labels), runs=str(tmp / "runs"), name="smoke", models_dir=str(models))
        steps = [("split", cmd_split, dict(common, seed=1, holdout_source=None, reassign=False, dry_run=False)),
                 ("prepare", cmd_prepare, dict(common, size="tiny"))]
        for name, fn, kw in steps:
            if fn(argparse.Namespace(**kw)):
                print(f"smoke FAILED at {name}")
                return 1
        n_train = sum(1 for _ in open(tmp / "runs" / "smoke" / "train.csv")) - 1
        t0 = time.time()
        targs = argparse.Namespace(**dict(common, epochs=a.epochs, batch_size=16, workers=2, lr=0.002,
                                          patience=5, seed=1, weights=None, run=True))
        if cmd_train(targs):
            print("smoke FAILED at train")
            return 1
        t_train = time.time() - t0
        steps_n = -(-n_train // 16) * a.epochs
        print(f"\nSMOKE train: {t_train:.0f} s total for {a.epochs} epoch(s) ~ {steps_n} steps of 16 plates "
              f"(includes start-up, TF graph build and validation)")
        if cmd_export(argparse.Namespace(**dict(common, new_name="cct-smoke", checkpoint=None))):
            print("smoke FAILED at export")
            return 1
        if cmd_compare(argparse.Namespace(**dict(common, new="cct-smoke", base=BASE_PLATE_YAML, models=[],
                                                 out=str(tmp / "compare")))):
            print("smoke FAILED at compare")
            return 1
        print(f"\nSMOKE OK in {time.time() - t_all:.0f} s (work dir {tmp})")
        return 0
    finally:
        if not a.keep:
            shutil.rmtree(tmp, ignore_errors=True)


# ----- CLI ------------------------------------------------------------------

def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser, name=False, labels=True):
        if labels:
            p.add_argument("--labels", default=DEFAULT_LABELS)
        p.add_argument("--models-dir", default="models")
        if name:
            p.add_argument("--name", required=True, help="run name: files go under runs/<name>/")
            p.add_argument("--runs", default="runs")

    p = sub.add_parser("split", help="assign train/val/test to verified rows (group-aware)")
    common(p)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--holdout-source", help="put this whole source dataset in test only")
    p.add_argument("--reassign", action="store_true", help="redo splits that are already assigned")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_split)

    p = sub.add_parser("prepare", help="write annotation CSVs and configs into runs/<name>/")
    common(p, name=True)
    p.add_argument("--size", choices=sorted(CCT_SIZES), default="small", help="model size (default small)")
    p.set_defaults(fn=cmd_prepare)

    p = sub.add_parser("train", help="build (and with --run, execute) the training command")
    common(p, name=True, labels=False)
    p.add_argument("--run", action="store_true")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--patience", type=int, default=15, help="early stopping, in epochs")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--weights", help="Keras weights (.keras/.weights.h5) to start from; default: scratch")
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("export", help="best checkpoint -> models/alpr/<new-name>/ (never overwrites)")
    common(p, name=True, labels=False)
    p.add_argument("--new-name", required=True)
    p.add_argument("--checkpoint", help="a .keras file (default: newest best.keras of the run)")
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("compare", help="score model folders on the test split with the real PlateEngine")
    common(p)
    p.add_argument("--new", help="exported model folder name; also tested inside the shipped ensemble")
    p.add_argument("--base", default=",".join(SHIPPED), help="the model/ensemble to beat (default: shipped)")
    p.add_argument("--models", nargs="*", default=SHIPPED, help="other ocr_model strings to include")
    p.add_argument("--out", default="runs/compare")
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("synthetic-smoke", help=argparse.SUPPRESS)
    p.add_argument("--models-dir", default="models")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--keep", action="store_true")
    p.set_defaults(fn=cmd_smoke)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    argv = ["synthetic-smoke" if x == "--synthetic-smoke" else x for x in (sys.argv[1:] if argv is None else argv)]
    args = parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
