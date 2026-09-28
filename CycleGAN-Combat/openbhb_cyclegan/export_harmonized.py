#!/usr/bin/env python
"""
export_harmonized.py
====================

Apply the trained ``generator_AtoB`` (site-target CycleGAN, ``cyclegan_target12``)
to OOD subjects and save harmonized volumes in the format the downstream
evaluator expects.

For each requested split (default: OOD val + OOD test) and each subject:
  1. load the native CAT12 VBM volume through the user's ``OpenBHBDataset``
  2. normalize (x/S - 1), centre-pad to 192^3
  3. run G_AtoB, keep background at -1
  4. crop back to the native shape, invert normalization ((y+1)*S), clip >= 0
  5. save ``recon_{global_idx:07d}.nii.gz``

Outputs (under --out-dir, default .../proposed_split/cyclegan_target12/):
  * recon_{global_idx:07d}.nii.gz          (one per subject)
  * reconstruction_index.csv               (global_idx, split, source, file_path,
                                            age, domain_site, original_site)
  * sanity_metrics.csv                     (per-volume MAE/MSE + shape checks)
  * qc/{split}_{global_idx:07d}.png        (real | harmonized | |difference|)

Site labels are NOT used here: G_AtoB is applied blindly to every OOD subject,
which is exactly the harmonization hypothesis being tested.

Runs on GPU or CPU; on the server, use the GPU.  See ``README_openbhb.md``.
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openbhb_bridge as ob  # noqa: E402

SOURCE_NAME = "cyclegan_target12"

# Standard MNI152 1.5 mm affine for a 121x145x121 grid (CAT12 VBM default space).
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
    p.add_argument("--openbhb-root", default=None, help="OpenBHB data root.")
    p.add_argument("--repo-harmo-dir", required=True,
                   help="Path to 3d_cyclegan_mri_harmonization/harmonization.")
    p.add_argument("--ckpt-dir", required=True,
                   help="Directory holding generator_AtoB.h5 and norm_stats.json.")
    p.add_argument("--gen-weights", default=None,
                   help="Explicit path to generator_AtoB.h5 "
                        "(default: <ckpt-dir>/generator_AtoB.h5).")
    p.add_argument("--out-dir",
                   default=os.path.expanduser("~/bigdata/cyclegan_target12"),
                   help="Destination directory for recon_*.nii.gz and the CSVs. "
                        "Default: ~/bigdata/cyclegan_target12 (a private export root -- "
                        "NOT the shared simple_unet_exports/proposed_split tree). Copy or "
                        "symlink into proposed_split/cyclegan_target12 only for final eval.")
    p.add_argument("--splits", nargs="+", default=["val", "test"],
                   help="WILDS split keys to export. val=OOD Val, test=OOD Test, "
                        "id_test, id_val, train are also valid.")
    p.add_argument("--norm-scale", type=float, default=None,
                   help="Override S (else read from <ckpt-dir>/norm_stats.json).")
    p.add_argument("--affine", choices=["mni15", "identity"], default="mni15",
                   help="Affine to write when no --ref-nii is given.")
    p.add_argument("--ref-nii", default=None,
                   help="Copy affine/header from this existing NIfTI (e.g. an "
                        "existing recon from another source) for exact compatibility.")
    p.add_argument("--no-clip-negative", action="store_true",
                   help="Do NOT clip harmonized output at 0 (VBM GM is >= 0).")
    p.add_argument("--n-vis", type=int, default=6,
                   help="Number of QC PNG grids to save per split (0 disables).")
    p.add_argument("--split-subdirs", action="store_true", default=True,
                   help="Write recons into per-split subfolders "
                        "(<out-dir>/ood_val/, <out-dir>/ood_test/). Default: on.")
    p.add_argument("--no-split-subdirs", dest="split_subdirs", action="store_false",
                   help="Flat layout: all recons directly in <out-dir> (matches the "
                        "refined/unrefined/cross convention some evaluators expect).")
    p.add_argument("--append", action="store_true",
                   help="Merge into an existing reconstruction_index.csv instead "
                        "of overwriting (dedupe by global_idx).")
    p.add_argument("--limit", type=int, default=None,
                   help="Debug: process at most N subjects per split.")
    p.add_argument("--mixed-precision", action="store_true",
                   help="Enable mixed_float16 (GPU only; faster, slightly less precise).")
    return p.parse_args()


def build_generator(args):
    import tensorflow as tf
    if args.mixed_precision and tf.config.list_physical_devices("GPU"):
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
        print("[tf] mixed_float16 enabled")
    sys.path.insert(0, os.path.abspath(args.repo_harmo_dir))
    from model_architectures import Generator  # noqa: E402
    gen = Generator()
    weights = args.gen_weights or os.path.join(args.ckpt_dir, "generator_AtoB.h5")
    if not os.path.isfile(weights):
        raise SystemExit(f"Generator weights not found: {weights}")
    gen.load_weights(weights)
    print(f"[gen] loaded G_AtoB from {weights}")
    return gen, tf


def get_affine(args):
    # Affine only (never copy a foreign header: a copied int/scaled header would
    # silently round/rescale the float VBM values on save).
    if args.ref_nii:
        import nibabel as nib
        ref = nib.load(args.ref_nii)
        print(f"[nii] copying affine from {args.ref_nii}")
        return np.asarray(ref.affine, dtype=np.float64)
    if args.affine == "identity":
        return np.eye(4, dtype=np.float64)
    return AFFINE_MNI15


def harmonize_volume(gen, tf, ds, idx, scale):
    """Return (harmonized_native_3d, raw_native_3d, checks_dict)."""
    net_in, info = ob.load_normed_padded(ds, idx, scale, ob.NET_SHAPE)  # (192,192,192,1)
    raw = info["raw"]                                                    # native VBM (no re-read)
    x = tf.convert_to_tensor(net_in[np.newaxis, ...], dtype=tf.float32)  # (1,192,192,192,1)
    y = gen(x, training=False).numpy().squeeze().astype(np.float32)      # (192,192,192)

    y_native_grid = ob.unpad(y, info["pad_info"], info["orig_shape"])    # normalized space
    brain = info["brain_mask"]
    y_native_grid[~brain] = ob.BG_NORM                                   # background -> -1
    out = ob.denormalize(y_native_grid, scale)                          # -> VBM scale
    out[~brain] = 0.0                                                    # exact background
    checks = {"orig_shape": info["orig_shape"], "out_shape": tuple(out.shape)}
    return out, raw, checks


def per_volume_metrics(harm, raw):
    diff = harm - raw
    brain = raw > 0
    mae_whole = float(np.mean(np.abs(diff)))
    mse_whole = float(np.mean(diff ** 2))
    if brain.any():
        mae_brain = float(np.mean(np.abs(diff[brain])))
        mse_brain = float(np.mean(diff[brain] ** 2))
    else:
        mae_brain = mse_brain = float("nan")
    return mae_whole, mse_whole, mae_brain, mse_brain


def save_qc_grid(path, raw, harm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = np.abs(harm - raw)
    cx, cy, cz = [s // 2 for s in raw.shape]
    planes = [
        ("axial", lambda v: v[:, :, cz]),
        ("coronal", lambda v: v[:, cy, :]),
        ("sagittal", lambda v: v[cx, :, :]),
    ]
    vmax = float(np.percentile(raw[raw > 0], 99)) if (raw > 0).any() else 1.0
    dmax = float(np.percentile(d[d > 0], 99)) if (d > 0).any() else 1.0
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for r, (_, sl) in enumerate(planes):
        for c, (img, title, vm) in enumerate([
                (raw, "real", vmax), (harm, "harmonized", vmax), (d, "|diff|", dmax)]):
            ax = axes[r, c]
            ax.imshow(np.rot90(sl(img)), cmap="gray" if c < 2 else "magma",
                      vmin=0, vmax=vm)
            ax.set_title(title if r == 0 else "", fontsize=11)
            ax.axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=90)
    plt.close(fig)


def load_existing_csv(path):
    rows = {}
    if os.path.isfile(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                rows[int(row["global_idx"])] = row
    return rows


def main():
    args = parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)  # live logs under nohup/SLURM
    except Exception:
        pass
    os.makedirs(args.out_dir, exist_ok=True)

    import nibabel as nib
    gen, tf = build_generator(args)
    affine = get_affine(args)

    ds = ob.load_openbhb_dataset(args.dataloader_dir, args.openbhb_root)

    # normalization scale
    if args.norm_scale is not None:
        scale = float(args.norm_scale)
    else:
        stats_path = os.path.join(args.ckpt_dir, "norm_stats.json")
        if not os.path.isfile(stats_path):
            raise SystemExit(f"norm_stats.json not found in {args.ckpt_dir}; "
                             f"pass --norm-scale explicitly.")
        scale, _ = ob.load_norm_stats(stats_path)
    print(f"[norm] using S = {scale:.6f}")

    index_rows = load_existing_csv(os.path.join(args.out_dir, "reconstruction_index.csv")) \
        if args.append else {}
    sanity_rows = []
    counts = {}
    all_gidx_ok = True
    all_shape_ok = True
    seen_gidx = {}  # gidx -> (split, file) to detect collisions

    for split in args.splits:
        label = ob.SPLIT_LABELS.get(split, split)
        idxs = ob.get_split_indices(ds, split)
        if args.limit:
            idxs = idxs[:args.limit]
        # Per-split subfolder by default (ood_val/, ood_test/); --no-split-subdirs
        # gives the flat <out-dir>/recon_*.nii.gz layout.
        split_dir = os.path.join(args.out_dir, label) if args.split_subdirs else args.out_dir
        qc_dir = os.path.join(split_dir, "qc")
        os.makedirs(split_dir, exist_ok=True)
        print(f"\n=== Exporting split '{split}' (label='{label}') -> {split_dir} : "
              f"{len(idxs)} subjects ===")
        n_vis_left = args.n_vis
        for k, idx in enumerate(idxs, 1):
            meta = ob.subject_meta(ds, idx)
            gidx = meta["global_idx"]
            harm, raw, checks = harmonize_volume(gen, tf, ds, idx, scale)

            # ---- verifications ----
            shape_ok = checks["out_shape"] == checks["orig_shape"] == tuple(raw.shape)
            all_shape_ok = all_shape_ok and shape_ok

            fname = f"recon_{gidx:07d}.nii.gz"
            fpath = os.path.join(split_dir, fname)
            # global_idx integrity: name round-trips to the same int, is unique,
            # and the file actually lands on disk.
            parsed = int(fname[len("recon_"):-len(".nii.gz")])
            if gidx in seen_gidx and seen_gidx[gidx][0] != label:
                raise SystemExit(
                    f"global_idx collision: {gidx} appears in both "
                    f"'{seen_gidx[gidx][0]}' and '{label}'. Splits must be disjoint.")
            seen_gidx[gidx] = (label, fpath)

            if not args.no_clip_negative:
                harm = np.maximum(harm, 0.0)

            img = nib.Nifti1Image(harm.astype(np.float32), affine)
            img.set_data_dtype(np.float32)
            nib.save(img, fpath)
            gidx_ok = (parsed == gidx) and os.path.isfile(fpath)
            all_gidx_ok = all_gidx_ok and gidx_ok

            mae_w, mse_w, mae_b, mse_b = per_volume_metrics(harm, raw)
            index_rows[int(gidx)] = {
                "global_idx": gidx, "split": label, "source": SOURCE_NAME,
                "file_path": fpath, "age": meta["age"],
                "domain_site": meta["domain_site"],
                "original_site": meta["original_site"],
            }
            sanity_rows.append({
                "global_idx": gidx, "split": label,
                "mae_whole": mae_w, "mse_whole": mse_w,
                "mae_brain": mae_b, "mse_brain": mse_b,
                "in_shape": tuple(raw.shape), "out_shape": checks["out_shape"],
                "shape_ok": shape_ok, "gidx_ok": gidx_ok, "file_path": fpath,
            })

            if n_vis_left > 0:
                save_qc_grid(os.path.join(qc_dir, f"{label}_{gidx:07d}.png"), raw, harm)
                n_vis_left -= 1

            if k % 20 == 0 or k == len(idxs):
                print(f"  {k}/{len(idxs)}  gidx={gidx:07d}  "
                      f"MAE_brain={mae_b:.5f}  shape_ok={shape_ok}", flush=True)
        counts[label] = len(idxs)
        print()

    # ---- write reconstruction_index.csv (exact schema) ----
    index_path = os.path.join(args.out_dir, "reconstruction_index.csv")
    ordered = sorted(index_rows.values(), key=lambda r: (str(r["split"]), int(r["global_idx"])))
    with open(index_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in ordered:
            w.writerow({c: r[c] for c in CSV_COLUMNS})

    # ---- write sanity_metrics.csv ----
    sanity_path = os.path.join(args.out_dir, "sanity_metrics.csv")
    with open(sanity_path, "w", newline="") as f:
        cols = ["global_idx", "split", "mae_whole", "mse_whole", "mae_brain",
                "mse_brain", "in_shape", "out_shape", "shape_ok", "gidx_ok", "file_path"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sanity_rows:
            w.writerow(r)

    # ---- summary ----
    print("\n=== Export summary ===")
    for label, n in counts.items():
        print(f"  {label}: {n} volumes exported")
    print(f"  total rows in reconstruction_index.csv: {len(ordered)}")
    if sanity_rows:
        mb = np.array([r["mae_brain"] for r in sanity_rows], dtype=float)
        print(f"  MAE(brain) vs real: mean={np.nanmean(mb):.5f}, "
              f"min={np.nanmin(mb):.5f}, max={np.nanmax(mb):.5f}")
    print(f"  every filename matches its global_idx: {all_gidx_ok}")
    print(f"  every output shape matches input shape: {all_shape_ok}")
    layout = ("per-split subfolders: " + ", ".join(
        f"{ob.SPLIT_LABELS.get(s, s)}/" for s in args.splits)) \
        if args.split_subdirs else "flat (all recons directly in out-dir)"
    print(f"  recon layout: {layout}")
    print(f"\n  reconstruction_index.csv -> {index_path}")
    print(f"  sanity_metrics.csv       -> {sanity_path}")
    print(f"  recons + QC grids        -> under {args.out_dir}")
    if not (all_gidx_ok and all_shape_ok):
        raise SystemExit("Sanity check FAILED (see flags above).")


if __name__ == "__main__":
    main()
