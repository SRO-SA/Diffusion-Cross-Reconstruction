# Diffusion cross-reconstruction for OpenBHB brain VBM maps

Code used to study site predictability, image fidelity, and brain-age prediction with Cross, Unrefined, Refined, CycleGAN, and ComBat images. Cross inverts a CAT12 gray-matter VBM map with the unrefined diffusion model and reconstructs it with the refined model. The paper reports results on OpenBHB, including evaluation sites excluded from model training.

This repository contains research scripts. The paths and switches used in an experiment are still set inside some scripts; read the notes below before running them. The dataset, trained checkpoints, and exported images are not included.

## Code map

| File or folder | Purpose |
| --- | --- |
| `DataLoader.py` | OpenBHB data loading and site splits. |
| `simple_diffusion_test.py` | Diffusion and site-discriminator training. |
| `simple_diffusion_test_export.py` | Export Unrefined, Refined, and Cross reconstructions. |
| `discriminator_loss_metric_updated.py` | Site, fidelity, and brain-age evaluations. |
| `site_discrminator.py`, `ukbb_brain_age_model.py`, `helper.py` | Local model and loading code imported by the evaluation script. The spelling `site_discrminator.py` matches the current import. |
| `CycleGAN-Combat/openbhb_cyclegan/` | OpenBHB CycleGAN training and export, ComBat harmonization, and supporting scripts. |
| `CycleGAN-Combat/3d_cyclegan_mri_harmonization/harmonization/model_architectures.py` | CycleGAN generator and discriminator architectures from the upstream project. |
| `CycleGAN-Combat/evaluate_cyclegan_target12.py`, `CycleGAN-Combat/probe_eval.py` | Baseline evaluation code. |

The `openbhb_cyclegan/README_openbhb.md` file gives more detail on the CycleGAN run. Its older installation example uses TensorFlow 2.12; `requirements-cyclegan.txt` records the TensorFlow 2.15.1 environment actually reported for this project.

## Data

Obtain the OpenBHB CAT12 voxel-based morphometry gray-matter maps through the dataset's access process. Set up the OpenBHB dataset root so that `OpenBHBDataset` can find `images/metadata.tsv` and files named like `images/sub-<participant_id>_preproc-cat12vbm_desc-gm_T1w.npy`. These are processed gray-matter maps, not raw T1 scans.

`DataLoader.py` currently defaults to a machine-specific data root. Change its `root_dir` default or pass a root where supported. It uses the proposed site split stored in `_SPLIT_DATA_PROPOSED_BALANCED`. The code subtracts 1 from metadata site IDs; the site IDs used by the scripts are zero-based. Site 35 is excluded from the main train/validation/test split in that configuration. Check that your metadata and split match the paper before running.

Keep the same metadata row order throughout the workflow. Exported filenames such as `recon_0000123.nii.gz` use the full dataset row index (`global_idx`) to pair generated images with the real images. Do not reorder `metadata.tsv` after export.

## Environments

Two environments were used. The files here list the package versions reported from those environments; installation in a new environment has not been verified.

| Workflow | Python | Main framework |
| --- | --- | --- |
| Diffusion, ComBat, and evaluation | 3.10.11 | PyTorch 2.0.1, CUDA build 11.8, cuDNN 8.7.0 |
| CycleGAN training and export | 3.9.25 | TensorFlow 2.15.1, CUDA build 12.2, cuDNN build 8; CPU PyTorch for data loading |

For the diffusion environment, install a PyTorch 2.0.1 / torchvision 0.15.2 build suitable for your machine, then install `requirements-diffusion.txt`:

```bash
conda create -n dgddpm python=3.10.11 -y
conda activate dgddpm
# Install the matching PyTorch/CUDA 11.8 build using the PyTorch installation instructions.
python -m pip install -r requirements-diffusion.txt
```

For CycleGAN, create a separate environment. Its PyTorch CPU wheels require the PyTorch CPU package index:

```bash
conda create -n cyclegan_vbm python=3.9.25 -y
conda activate cyclegan_vbm
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-cyclegan.txt
```

The recorded TensorFlow build detected a GPU on the original server. Check GPU access in any new installation before a full run. `neuroCombat` runs in the diffusion/evaluation environment and does not require TensorFlow.

## Workflow

1. Prepare OpenBHB maps and metadata, preserving their row order.
2. Train the unrefined diffusion model and a discriminator on real training images, then refine the diffusion model. `simple_diffusion_test.py` currently has the discriminator and unrefined stages disabled in its `__main__` settings. Enable them one at a time and set checkpoint paths before running. The enabled refined stage expects both earlier checkpoints. The script also creates the dataset at import time, so the configured dataset must be available even when another script imports its classes.
3. Run `simple_diffusion_test_export.py` to create `unrefined/`, `refined/`, and `cross/` reconstructions. It currently has `RUN_RECONSTRUCTION_EXPORT = True` and fixed checkpoint paths in its `__main__` block. `EXPORT_SPLIT=all` selects train, validation, and test. Set paths to your data, checkpoints, and output folders first.
4. Run `discriminator_loss_metric_updated.py` with `RUN_MODE` set to the needed evaluation. Available modes include `cross_training`, `brain_age_multiseed`, and `paired_psnr_ssim`. `RESULTS_ROOT` can be set with an environment variable; `EXPORT_ROOT` is currently a fixed path in the script. The five-seed brain-age run uses seeds 1337–1341.
5. Train and export CycleGAN images in the CycleGAN environment. The commands below use the scripts and argument names in this repository. `REPO` means the top-level repository folder; replace the other paths with your own.

```bash
cd "$REPO/CycleGAN-Combat/openbhb_cyclegan"
python train_cyclegan_target12.py \
  --dataloader-dir "$REPO/CycleGAN-Combat" \
  --openbhb-root "$OPENBHB_ROOT" \
  --repo-harmo-dir "$REPO/CycleGAN-Combat/3d_cyclegan_mri_harmonization/harmonization" \
  --dest-dir "$CYCLEGAN_CKPT_DIR" \
  --epochs-base 150 --epochs-decay 150 --steps-per-epoch 200 \
  --lambda-cyc-init 200 --lambda-cyc-end 100 --identity-frac 0.5
python export_harmonized.py \
  --dataloader-dir "$REPO/CycleGAN-Combat" \
  --openbhb-root "$OPENBHB_ROOT" \
  --repo-harmo-dir "$REPO/CycleGAN-Combat/3d_cyclegan_mri_harmonization/harmonization" \
  --ckpt-dir "$CYCLEGAN_CKPT_DIR" \
  --out-dir "$CYCLEGAN_EXPORT_DIR" \
  --splits train val test
```

6. Run ComBat in the diffusion/evaluation environment. It fits the train, validation, and test cohorts together using their site labels and ages. The values shown are script defaults; record the exact options used for any published run, especially the mask threshold, voxel chunk size, and clipping setting.

```bash
cd "$REPO/CycleGAN-Combat/openbhb_cyclegan"
python combat_harmonize.py \
  --dataloader-dir "$REPO/CycleGAN-Combat" \
  --openbhb-root "$OPENBHB_ROOT" \
  --out-dir "$COMBAT_EXPORT_DIR" \
  --splits train val test --ref-site 12 \
  --mask-thresh 0.05 --chunk-voxels 50000
```

7. Evaluate CycleGAN and ComBat exports using `CycleGAN-Combat/evaluate_cyclegan_target12.py`. Set `SOURCE`, `EXPORT_ROOT`, `RESULTS_ROOT`, and `RUN_MODE` for each source. `brain_age_combined`, `domain_confusion`, and `all` require generated training images in the export. See the script for the available modes.

The CycleGAN and ComBat results have different access to evaluation data: trained CycleGAN inference does not need evaluation-site labels, whereas the ComBat fit includes labeled validation and test cohorts. A drop in accuracy for a classifier trained on real images shows that its learned site cues are less detectable; it does not establish that scanner effects alone were removed.

## Before publishing results from a new setup

Check that the dataset has the expected splits and image shape. Run a small data-loading, reconstruction, and evaluation test in a fresh environment with no access to the original code folders. The environment files record installed versions; they are not a guarantee that a full experiment has been reproduced on another machine.

## Upstream CycleGAN code

The CycleGAN architectures come from the included `3d_cyclegan_mri_harmonization` project. Keep its attribution and any applicable license terms when publishing this repository. Only the `harmonization/model_architectures.py` file is imported by the OpenBHB CycleGAN training/export scripts here.
