from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path

TARGETS = ("Alligator", "Bee", "Corals")
RF_SUFFIX = re.compile(r"_jpg\.rf\.[0-9a-f]+\.jpg$", re.I)
FLICKR_ORIGINAL = re.compile(r"^(?P<photo_id>\d+)_(?P<secret>[0-9a-f]+)_o_jpg\.rf\.[0-9a-f]+\.jpg$", re.I)


def read_manifest(path: Path):
    if not path.exists():
        return []
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def classify_path(p: str):
    norm = p.replace("\\", "/")
    for cls in TARGETS:
        token = f"/dataset/{cls}/"
        if token in norm:
            return cls
    return None


def derive_split(p: str):
    norm = p.replace("\\", "/")
    for split in ("train", "valid", "val", "test"):
        if f"/{split}/images/" in norm:
            return "valid" if split == "val" else split
    return "unknown"


def original_stub(name: str):
    return RF_SUFFIX.sub("", name)


def main():
    ap = argparse.ArgumentParser(description="Audit/recover missing S.A.V.E. source images.")
    ap.add_argument("--dataset-root", required=True, help="Current dataset root, e.g. /content/drive/MyDrive/dataset")
    ap.add_argument("--train-manifest", required=True, help="Original train.txt")
    ap.add_argument("--valid-manifest", required=True, help="Original valid.txt")
    ap.add_argument("--source-root", default=None, help="Optional recovered source export to copy exact matching files from")
    ap.add_argument("--output-dir", default="research_eval/outputs/recovery")
    args = ap.parse_args()

    root = Path(args.dataset_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for manifest_path, nominal_split in [
        (Path(args.train_manifest), "train"),
        (Path(args.valid_manifest), "valid"),
    ]:
        for raw in read_manifest(manifest_path):
            cls = classify_path(raw)
            if not cls:
                continue
            split = derive_split(raw) or nominal_split
            filename = Path(raw).name
            expected = root / cls / split / "images" / filename
            m = FLICKR_ORIGINAL.match(filename)
            rows.append({
                "class": cls,
                "split": split,
                "filename": filename,
                "original_stub": original_stub(filename),
                "expected_path": str(expected),
                "present": expected.exists(),
                "flickr_photo_id_candidate": m.group("photo_id") if m else "",
                "flickr_secret_candidate": m.group("secret") if m else "",
            })

    # Optional exact-filename recovery from another local/exported source tree.
    recovered = 0
    if args.source_root:
        source_root = Path(args.source_root)
        index = {}
        for f in source_root.rglob("*"):
            if f.is_file():
                index.setdefault(f.name, f)
        for row in rows:
            if row["present"]:
                continue
            src = index.get(row["filename"])
            if src:
                dst = Path(row["expected_path"])
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                row["present"] = True
                recovered += 1

    csv_path = out / "missing_image_manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "class", "split", "filename", "original_stub", "expected_path", "present",
            "flickr_photo_id_candidate", "flickr_secret_candidate"
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    summary = []
    for cls in TARGETS:
        sub = [r for r in rows if r["class"] == cls]
        present = sum(bool(r["present"]) for r in sub)
        missing = len(sub) - present
        summary.append((cls, len(sub), present, missing))

    summary_path = out / "recovery_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("S.A.V.E. missing-image recovery audit\n")
        f.write("=================================\n")
        for cls, total, present, missing in summary:
            f.write(f"{cls}: expected={total}, present={present}, missing={missing}\n")
        f.write(f"TOTAL: expected={len(rows)}, present={sum(r['present'] for r in rows)}, missing={sum(not r['present'] for r in rows)}\n")
        if recovered:
            f.write(f"Recovered from --source-root: {recovered}\n")
        f.write("\nNote: Coral filenames that match Flickr's <photo_id>_<secret>_o pattern are flagged as source candidates.\n")
        f.write("Do not treat those candidates as verified source URLs until checked against the original image content/license.\n")

    print(summary_path.read_text())
    print(f"Manifest: {csv_path}")


if __name__ == "__main__":
    main()
