#!/usr/bin/env python
"""
combat_harmonize.py
===================

Official ComBat (neuroCombat) harmonization of OpenBHB CAT12 VBM gray-matter maps
toward reference site 12, saved in the SAME per-split layout as the CycleGAN export
so the SAME evaluator works on it.

ComBat is a STATISTICAL, transductive method: it needs the site label of EVERY
image (including OOD) and is fit on all of them at once. We therefore harmonize
train + ood_val + ood_test TOGETHER, using ``original_site`` as the batch and
``age`` (+ sex if present) as preserved covariates, with ``ref_batch = site 12``
(mirrors cyclegan_target12). Site-12 images stay ~unchanged; others map toward 12.

Outputs (under --out-dir, default the harmonized data dir):
    train/recon_{gidx:07d}.nii.gz, ood_val/..., ood_test/...
    reconstruction_index.csv   (source = --source-name, default 'combat')

Runs on CPU (numpy). Needs: neuroCombat, nibabel, pandas, and OpenBHBDataset
(torch+wilds) importable. No TensorFlow / GPU.

Install:  pip install neuroCombat   (repo: github.com/Jfortin1/neuroCombat)
"""
import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd

# neuroCombat's ref_batch path uses the removed np.int alias (NumPy >= 1.24).
if not hasattr(np, "int"):
    np.int = int          # noqa: NPY001  (compat shim)
if not hasattr(np, "float"):
    np.float = float      # noqa: NPY001

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openbhb_bridge as ob  # dataset / index / meta / load helpers (no TF)

AFFINE_MNI15 = np.array([[-1.5, 0.0, 0.0, 90.0],
                         [0.0, 1.5, 0.0, -126.0],
                         [0.0, 0.0, 1.5, -72.0],
                         [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
CSV_COLUMNS = ["global_idx", "split", "source", "file_path",
               "age", "domain_site", "original_site"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataloader-dir", required=True,
                   help="Directory containing the user's DataLoader.py.")
    p.add_argument("--openbhb-root", default=None, help="OpenBHB data root (real .npy).")
    p.add_argument("--out-dir",
                   default="/rhome/ssafa013/bigdata/data/openBHB_v1.1_harmonized",
                   help="Export root for the harmonized recon_*.nii.gz + CSV.")
    p.add_argument("--source-name", default="combat",
                   help="Value written to the CSV 'source' column / used by the evaluator.")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                   help="WILDS split keys to harmonize together (ComBat is transductive).")
    p.add_argument("--ref-site", default="12",
                   help="Reference batch (target site), e.g. 12. 'none' = standard "
                        "grand-mean ComBat (domain-neutral) instead of target-site.")
    p.add_argument("--mask-thresh", type=float, default=0.05,
                   help="Group GM mask = voxels whose MEAN GM across subjects > thresh.")
    p.add_argument("--chunk-voxels", type=int, default=50000,
                   help="Run ComBat in voxel-chunks of this size to bound CPU memory "
                        "(0 = whole matrix at once). neuroCombat allocates several "
                        "float64 (features x samples) arrays, so the full matrix needs "
                        "~tens of GB. EB priors pool within each chunk; with tens of "
                        "thousands of voxels this matches the full run. ~50000 keeps "
                        "peak RAM near ~14 GB for 3500 subjects.")
    p.add_argument("--keep-outside-mask", action="store_true",
                   help="Outside the GM mask keep the ORIGINAL voxel values (default: 0).")
    p.add_argument("--no-clip-negative", action="store_true",
                   help="Do NOT clip harmonized GM at 0 (VBM GM is >= 0).")
    p.add_argument("--affine", choices=["mni15", "identity"], default="mni15")
    p.add_argument("--limit", type=int, default=None,
                   help="Debug: at most N subjects per split (may break ComBat's "
                        ">=2-per-site requirement; use only as an I/O smoke test).")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    from neuroCombat import neuroCombat
    import nibabel as nib

    os.makedirs(args.out_dir, exist_ok=True)
    affine = AFFINE_MNI15 if args.affine == "mni15" else np.eye(4, dtype=np.float64)

    ds = ob.load_openbhb_dataset(args.dataloader_dir, args.openbhb_root)
    m = ds.metadata

    # ---- gather subjects across requested splits (order defines matrix columns) ----
    idx_by_split, all_idx = {}, []
    for sp in args.splits:
        idxs = ob.get_split_indices(ds, sp)
        if args.limit:
            idxs = idxs[:args.limit]
        idx_by_split[sp] = [int(i) for i in idxs]
        all_idx.extend(int(i) for i in idxs)
    all_idx = np.asarray(all_idx, dtype=int)
    N = len(all_idx)
    if N == 0:
        raise SystemExit("No subjects selected.")
    print(f"[combat] {N} subjects across splits {args.splits}")

    # ---- covariates: site (batch), age (+ sex if available) ----
    site = np.asarray([int(round(float(m["original_site"].iloc[i]))) for i in all_idx])
    age = np.asarray([float(m["age"].iloc[i]) for i in all_idx], dtype=float)
    covars_dict = {"SITE": site, "age": age}
    cat_cols = []
    if "sex" in m.columns:
        covars_dict["sex"] = np.asarray([m["sex"].iloc[i] for i in all_idx])
        cat_cols = ["sex"]
        print("[combat] including 'sex' as a preserved categorical covariate")
    covars = pd.DataFrame(covars_dict)

    uniq, cnts = np.unique(site, return_counts=True)
    print(f"[combat] {len(uniq)} sites present; min per-site count = {int(cnts.min())}")
    small = uniq[cnts < 2]
    if len(small):
        raise SystemExit(
            f"Sites with <2 subjects (ComBat cannot estimate their variance): "
            f"{small.tolist()}. Remove --limit or drop/merge those sites.")

    if str(args.ref_site).lower() == "none":
        ref_batch = None
        print("[combat] standard ComBat (grand-mean; domain-neutral, NOT target-site)")
    else:
        ref_batch = int(round(float(args.ref_site)))
        if ref_batch not in set(uniq.tolist()):
            raise SystemExit(f"--ref-site {ref_batch} not present (sites={uniq.tolist()}).")
        print(f"[combat] target-site ComBat: ref_batch = site {ref_batch}")

    # ---- pass 1/2: group GM mask (mean GM > thresh) ----
    shape = ob.load_raw_volume(ds, int(all_idx[0])).shape
    print(f"[combat] pass 1/2: building group GM mask (mean>{args.mask_thresh}); shape={shape}")
    acc = np.zeros(shape, dtype=np.float64)
    for k, i in enumerate(all_idx, 1):
        acc += ob.load_raw_volume(ds, int(i))
        if k % 250 == 0 or k == N:
            print(f"    mask {k}/{N}", flush=True)
    mask = (acc / N) > args.mask_thresh
    n_vox = int(mask.sum())
    if n_vox == 0:
        raise SystemExit("Empty GM mask; lower --mask-thresh.")
    print(f"[combat] mask voxels = {n_vox} / {int(np.prod(shape))}")

    # ---- pass 2/2: data matrix (features x samples), float32 ----
    print(f"[combat] pass 2/2: loading masked matrix ({n_vox} x {N}) ...")
    dat = np.empty((n_vox, N), dtype=np.float32)
    for j, i in enumerate(all_idx):
        dat[:, j] = ob.load_raw_volume(ds, int(i))[mask].astype(np.float32)
        if (j + 1) % 250 == 0 or (j + 1) == N:
            print(f"    load {j + 1}/{N}", flush=True)

    # ---- run ComBat (chunked over voxels to bound CPU memory) ----
    import gc

    def _combat(block):
        # neuroCombat does not modify its input in place; returns a new array.
        r = neuroCombat(dat=block, covars=covars, batch_col="SITE",
                        categorical_cols=cat_cols, continuous_cols=["age"],
                        ref_batch=ref_batch)
        out = np.asarray(r["data"], dtype=np.float32)
        del r
        return out

    cs = args.chunk_voxels
    if cs and cs > 0 and n_vox > cs:
        n_chunks = (n_vox + cs - 1) // cs
        print(f"[combat] running neuroCombat (ref_batch={ref_batch}) in {n_chunks} "
              f"voxel-chunks of {cs} (peak RAM ~14 GB)")
        for ci, start in enumerate(range(0, n_vox, cs), 1):
            end = min(start + cs, n_vox)
            dat[start:end] = _combat(dat[start:end])   # harmonize in place
            gc.collect()
            print(f"    combat chunk {ci}/{n_chunks} (voxels {start}:{end})", flush=True)
    else:
        print(f"[combat] running neuroCombat (ref_batch={ref_batch}) on the full matrix ...")
        dat[:] = _combat(dat)
        gc.collect()

    harm = dat  # harmonized in place; same (n_vox, N) column order as all_idx
    print(f"[combat] harmonized matrix {harm.shape}; "
          f"range [{float(harm.min()):.4f}, {float(harm.max()):.4f}]")

    # ---- write per subject into per-split subfolders (same order as all_idx) ----
    print("[combat] writing volumes ...")
    index_rows, seen, pos = [], set(), 0
    for sp in args.splits:
        label = ob.SPLIT_LABELS.get(sp, sp)
        split_dir = os.path.join(args.out_dir, label)
        os.makedirs(split_dir, exist_ok=True)
        for i in idx_by_split[sp]:
            meta = ob.subject_meta(ds, i)
            gidx = meta["global_idx"]
            if args.keep_outside_mask:
                vol = ob.load_raw_volume(ds, i).astype(np.float32).copy()
            else:
                vol = np.zeros(shape, dtype=np.float32)
            vol[mask] = harm[:, pos]
            if not args.no_clip_negative:
                vol = np.maximum(vol, 0.0)
            fname = f"recon_{gidx:07d}.nii.gz"
            fpath = os.path.join(split_dir, fname)
            img = nib.Nifti1Image(vol, affine)
            img.set_data_dtype(np.float32)
            nib.save(img, fpath)
            if gidx in seen:
                raise SystemExit(f"duplicate global_idx {gidx}")
            seen.add(gidx)
            index_rows.append({"global_idx": gidx, "split": label,
                               "source": args.source_name, "file_path": fpath,
                               "age": meta["age"], "domain_site": meta["domain_site"],
                               "original_site": meta["original_site"]})
            pos += 1
        print(f"    {label}: {len(idx_by_split[sp])} volumes -> {split_dir}", flush=True)

    index_path = os.path.join(args.out_dir, "reconstruction_index.csv")
    ordered = sorted(index_rows, key=lambda r: (str(r["split"]), int(r["global_idx"])))
    with open(index_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in ordered:
            w.writerow({c: r[c] for c in CSV_COLUMNS})

    print(f"[combat] wrote {len(ordered)} rows -> {index_path}")
    print(f"[combat] DONE. Export root: {args.out_dir}  source='{args.source_name}'")


if __name__ == "__main__":
    main()
