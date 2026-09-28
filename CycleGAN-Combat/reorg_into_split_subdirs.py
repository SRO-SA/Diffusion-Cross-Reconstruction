#!/usr/bin/env python
"""Move a FLAT cyclegan_target12 export into per-split subfolders using
reconstruction_index.csv, and update the CSV's file_path column to match.

    python reorg_into_split_subdirs.py [OUT_DIR] [--dry-run]

OUT_DIR defaults to ~/bigdata/cyclegan_target12. Safe to re-run.
"""
import csv, os, shutil, sys

pos = [a for a in sys.argv[1:] if not a.startswith("-")]
DRY = "--dry-run" in sys.argv
OUT_DIR = os.path.expanduser(pos[0] if pos else "~/bigdata/cyclegan_target12")
CSV = os.path.join(OUT_DIR, "reconstruction_index.csv")
if not os.path.isfile(CSV):
    sys.exit(f"CSV not found: {CSV}")

with open(CSV, newline="") as f:
    rows = list(csv.DictReader(f))
fieldnames = list(rows[0].keys()) if rows else []
print(f"{len(rows)} rows | OUT_DIR={OUT_DIR}{'  [DRY-RUN]' if DRY else ''}")

moved = already = missing = 0
for r in rows:
    split = r["split"]                                   # ood_val / ood_test
    gidx = int(r["global_idx"])
    fname = f"recon_{gidx:07d}.nii.gz"
    dst_dir = os.path.join(OUT_DIR, split)
    dst = os.path.join(dst_dir, fname)
    src_flat = os.path.join(OUT_DIR, fname)              # old flat location
    if os.path.isfile(dst):
        already += 1
        if os.path.isfile(src_flat) and os.path.abspath(src_flat) != os.path.abspath(dst) and not DRY:
            os.remove(src_flat)                          # drop leftover duplicate
    else:
        src = src_flat if os.path.isfile(src_flat) else r.get("file_path", "")
        if src and os.path.isfile(src):
            if not DRY:
                os.makedirs(dst_dir, exist_ok=True)
                shutil.move(src, dst)
            moved += 1
        else:
            missing += 1
            print(f"  MISSING source for {fname} (split={split})")
    r["file_path"] = dst                                 # point CSV at new location

# also move flat QC pngs:  qc/<split>_<gidx>.png -> <split>/qc/<split>_<gidx>.png
flat_qc = os.path.join(OUT_DIR, "qc")
if os.path.isdir(flat_qc):
    for png in sorted(os.listdir(flat_qc)):
        if png.endswith(".png") and "_" in png:
            split = png.rsplit("_", 1)[0]
            if not DRY:
                os.makedirs(os.path.join(OUT_DIR, split, "qc"), exist_ok=True)
                shutil.move(os.path.join(flat_qc, png),
                            os.path.join(OUT_DIR, split, "qc", png))
    if not DRY and os.path.isdir(flat_qc) and not os.listdir(flat_qc):
        os.rmdir(flat_qc)

if not DRY and rows:
    with open(CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)

print(f"moved={moved}, already_in_place={already}, missing={missing}")
print("CSV file_path updated." if not DRY else "DRY-RUN: nothing moved, CSV unchanged.")