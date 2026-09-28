# `cyclegan_target12` — 3D CycleGAN target-site harmonization for OpenBHB CAT12 VBM

A target-site harmonization **baseline** that maps non-target CAT12 VBM
gray-matter maps toward the **reference site `original_site == 12.0`**, then
harmonizes OOD val/test subjects with the trained `G_AtoB` generator.

This adapts the **older 3D CycleGAN** repo
(`gitlab.com/RocaV/3d_cyclegan_mri_harmonization`) — a clean two-domain CycleGAN,
which matches the Domain A (site≠12) / Domain B (site==12) setup exactly. The
generator/discriminator architectures and the LSGAN + cycle losses are imported
from that repo unchanged; we add an **identity loss** (both domains are CAT12
VBM). IGUANe was rejected as heavier to adapt (multi-source universal generator,
TFRecords, bias-sampling, T1 preprocessing pipeline).

> **This is not the DDPM "cross" method.** `cross` is target-free / domain-neutral;
> `cyclegan_target12` explicitly harmonizes *toward site 12*.

---

## 0. Files

**Added** (this folder — copy the whole `openbhb_cyclegan/` folder to the server):

| File | Purpose |
|---|---|
| `openbhb_bridge.py` | The only file that touches `OpenBHBDataset`. Builds Domain A/B indices, CAT12-VBM normalization (`x/S−1`), 121×145×121 ⇄ 192³ padding, and the `tf.data` feeds. |
| `train_cyclegan_target12.py` | Training (LSGAN + cycle + identity). Saves `generator_AtoB.h5`, `norm_stats.json`, `stats.json`. |
| `export_harmonized.py` | Applies `G_AtoB` to OOD splits → `recon_{global_idx:07d}.nii.gz`, `reconstruction_index.csv`, `sanity_metrics.csv`, QC PNGs. |
| `README_openbhb.md` | This file. |

**Changed in the cloned repo:** none. Everything is additive; the repo's
`harmonization/model_architectures.py` is imported, not modified.

---

## 1. Environment (server)

Both candidate repos are **TensorFlow**; your `OpenBHBDataset` is **PyTorch/WILDS**.
So one environment must have *both*. Torch is only used to load `.npy` files on
CPU, so install the **CPU-only** torch wheel to avoid clashing with TF's CUDA:

```bash
conda create -n cyclegan_vbm python=3.9 -y
conda activate cyclegan_vbm
# TF stack (pinned to the versions the repos were built against)
pip install tensorflow==2.12.0 numpy==1.23.5 nibabel==5.1.0 pandas==2.0.0 h5py==3.8.0 scipy matplotlib
# Your dataset stack (CPU torch — data loading only)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install wilds
```

Set the GPU library paths as the IGUANe README notes (only needed for TF/GPU):

```bash
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib
```

> If `wilds`/`torch` pull an incompatible NumPy, force `pip install numpy==1.23.5`
> last (TF 2.12 needs `numpy < 1.24`).

Clone the CycleGAN repo on the server (for `model_architectures.py`):

```bash
git clone https://gitlab.com/RocaV/3d_cyclegan_mri_harmonization.git
```

---

## 2. Paths you will pass

| Placeholder | Meaning | Example |
|---|---|---|
| `DATALOADER_DIR` | folder containing your `DataLoader.py` | `/rhome/ssafa013/.../project` |
| `OPENBHB_ROOT` | OpenBHB root (has `images/metadata.tsv`) | `/rhome/ssafa013/bigdata/data` (the `OpenBHBDataset` default) |
| `REPO_HARMO_DIR` | `<clone>/harmonization` | `/rhome/ssafa013/.../3d_cyclegan_mri_harmonization/harmonization` |
| `CKPT_DIR` | where checkpoints/`norm_stats.json` go | `/rhome/ssafa013/bigdata/cyclegan_target12_ckpt` |
| `OUT_DIR` | export target (evaluator reads here) | `/rhome/ssafa013/bigdata/simple_unet_exports/proposed_split/cyclegan_target12` |

---

## 3. Smoke test first (tiny, ~minutes)

Validate the whole path end-to-end before the real run:

```bash
cd /path/to/openbhb_cyclegan
python train_cyclegan_target12.py \
  --dataloader-dir DATALOADER_DIR \
  --openbhb-root  OPENBHB_ROOT \
  --repo-harmo-dir REPO_HARMO_DIR \
  --dest-dir CKPT_DIR \
  --epochs-base 1 --epochs-decay 0 --steps-per-epoch 5 --disc-buf-size 4
```

```bash
python export_harmonized.py \
  --dataloader-dir DATALOADER_DIR \
  --openbhb-root  OPENBHB_ROOT \
  --repo-harmo-dir REPO_HARMO_DIR \
  --ckpt-dir CKPT_DIR \
  --out-dir OUT_DIR \
  --splits val --limit 4 --n-vis 2
```

Confirm the prints: `|A|`, `|B|`, native shape `(121,145,121)`, normalized range,
and (export) `every output shape matches input shape: True`.

---

## 4. Full training (GPU, server)

```bash
cd /path/to/openbhb_cyclegan
python train_cyclegan_target12.py \
  --dataloader-dir DATALOADER_DIR \
  --openbhb-root  OPENBHB_ROOT \
  --repo-harmo-dir REPO_HARMO_DIR \
  --dest-dir CKPT_DIR \
  --epochs-base 150 --epochs-decay 150 --steps-per-epoch 200 \
  --lambda-cyc-init 200 --lambda-cyc-end 100 --identity-frac 0.5
```

Notes:
- Defaults reproduce the repo schedule (300 epochs). For a quicker baseline try
  `--epochs-base 60 --epochs-decay 40`.
- `--gpu-mem-limit 21000` caps the logical GPU memory (the repo hard-codes 21 GB).
- If XLA errors on your GPU, tell me and I'll flip `jit_compile` off.
- Reduce `--disc-buf-size` (default 50 → ~1.4 GB) on smaller GPUs.

**Prints emitted** (as requested): number of A_train and B_train subjects, native
and network image shape, normalized intensity range, and steps/batches per epoch.

---

## 5. Harmonize OOD val + test (GPU or CPU)

```bash
cd /path/to/openbhb_cyclegan
python export_harmonized.py \
  --dataloader-dir DATALOADER_DIR \
  --openbhb-root  OPENBHB_ROOT \
  --repo-harmo-dir REPO_HARMO_DIR \
  --ckpt-dir CKPT_DIR \
  --out-dir OUT_DIR \
  --splits val test \
  --n-vis 6
```

Optional in-distribution export (your "id_full"): there is **no** `id_full` WILDS
split here. Use whichever you mean:
- site-35 in-distribution holdout → `--splits id_test --append`
- all training subjects        → `--splits train --append`

`--append` merges into the existing `reconstruction_index.csv` (dedupe by `global_idx`).

For **exact** NIfTI compatibility with your other sources, point at one of their
existing recons to copy its affine:

```bash
  --ref-nii /rhome/ssafa013/bigdata/simple_unet_exports/proposed_split/refined/recon_0000000.nii.gz
```

(Without it, a standard MNI152 1.5 mm affine for 121×145×121 is written.)

---

## 6. Outputs (in `OUT_DIR`)

```
recon_0000000.nii.gz ...            # one per OOD subject, float32, native 121x145x121
reconstruction_index.csv            # global_idx, split, source, file_path, age, domain_site, original_site
sanity_metrics.csv                  # per-volume MAE/MSE (whole + brain) + shape/idx checks
qc/ood_val_0000000.png ...          # real | harmonized | |difference| (3 planes)
```

`source` is always `cyclegan_target12`; `split` is `ood_val` / `ood_test`
(`val`/`test` → those labels). Then add `"cyclegan_target12"` to `DATASET_SOURCES`
in your downstream script — the on-disk layout matches `refined/`, `unrefined/`,
`cross/`.

The export **fails loudly** (`SystemExit`) if any output shape ≠ input shape or any
`global_idx` filename is wrong/duplicated.

---

## 7. Assumptions & limitations

1. **Framework bridge.** Both repos are TF; `OpenBHBDataset` is PyTorch. Images are
   loaded through your dataset on CPU, converted to NumPy, fed to `tf.data`. One env
   needs both TF and torch+wilds (§1).
2. **No T1 preprocessing.** No skull-strip / N4 / registration / T1 intensity scaling —
   the CAT12 VBM maps are already registered. The only transforms are an intensity
   rescale and geometric padding, both inverted before saving.
3. **Normalization.** `x/S − 1` with `S` = median of nonzero GM voxels over Domain B
   (the reference), the VBM analogue of the repo's fixed `/500` for T1. Saved in
   `norm_stats.json`; export reuses the identical `S`. Background (exactly 0) → −1.
   *Assumes background is exactly 0* (unsmoothed modulated GM, `mwp1`-style). If your
   maps are smoothed and background is slightly nonzero, tell me and I'll add a mask
   threshold.
4. **Geometry.** The repo's generator is fixed at **192³**, so 121×145×121 is
   centre-padded to 192³ (background −1) and cropped back on export. Larger than the
   data's native grid → more compute but no information loss. (IGUANe's generator is
   shape-flexible; if 192³ is too heavy we can swap it in and pad only to 128×160×128.)
5. **Affine.** The `.npy` carry no affine. Default = MNI152 1.5 mm; use `--ref-nii`
   to copy the affine your other sources use. Voxel data/orientation are never
   transposed, so grids line up regardless.
6. **`global_idx`.** = the full-dataset row index (the arg to `OpenBHBDataset.get_input`,
   what WILDS subsets store in `.indices`). If your metadata has an explicit
   `global_idx` column, it is honored instead.
7. **Baseline, not SOTA.** Single A↔B CycleGAN, batch size 1, from scratch on VBM.
   The mask-aware losses and buffer follow the repo. No bias-sampling / age matching.
8. **Not run here.** Written and reviewed on a machine without Python/GPU; run the
   smoke test (§3) first on the server to catch any environment-specific issue.
```
