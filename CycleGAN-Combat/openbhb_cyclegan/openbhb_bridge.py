"""
openbhb_bridge.py
=================

Bridge between the user's PyTorch/WILDS ``OpenBHBDataset`` (defined in
``DataLoader.py``) and the TensorFlow 3D CycleGAN training/inference code taken
from ``3d_cyclegan_mri_harmonization``.

This module is the *only* place that touches ``OpenBHBDataset``.  Everything the
CycleGAN code needs is expressed here as plain NumPy arrays / index lists, so the
rest of the pipeline never has to know about torch or WILDS.

What it provides
----------------
* ``load_openbhb_dataset(...)``        -> an ``OpenBHBDataset`` instance
* ``get_domain_indices(...)``          -> Domain A / Domain B global-index arrays
* ``get_split_indices(...)``           -> global indices for any WILDS split
* ``subject_meta(...)``                -> dict(global_idx, age, domain_site, ...)
* ``compute_norm_scale(...)``          -> S = median of nonzero GM voxels (VBM)
* ``normalize`` / ``denormalize``      -> x/S-1  <->  (y+1)*S
* ``pad_to`` / ``unpad``               -> 121x145x121 <-> 192^3 (undoable)
* ``load_normed_padded(...)``          -> ready-to-feed (H,W,D,1) float32 volume
* ``make_tf_dataset(...)``             -> infinite tf.data pipeline for a domain

Design notes
------------
* CAT12 VBM gray-matter maps are already registered/preprocessed.  We do NOT do
  any T1 preprocessing (no skull strip, no N4, no registration).  The only
  transform is an intensity rescale + geometric padding, both fully invertible.
* Intensity convention matches the CycleGAN repo: background -> -1, brain median
  -> 0.  The repo does this for T1 with ``x/500 - 1``; for VBM we replace the
  fixed ``500`` with ``S`` = median of nonzero (in-brain) voxels.  Inversion is
  ``(y + 1) * S`` and background is re-masked to native 0.
* The network is a fixed-size 192^3 U-Net, so volumes are centre-padded to 192^3
  with the background value (-1 after normalization) and cropped back afterwards.

Torch / TF are imported lazily inside functions so this file can be imported for
inspection on a machine without a GPU or without one of the two frameworks.
"""

import json
import os
import sys

import numpy as np

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

TARGET_SITE = 12.0                 # reference / majority site (Domain B)
NET_SHAPE = (192, 192, 192)        # fixed input size of the repo's Generator
BG_NORM = -1.0                     # background value in normalized space

# WILDS split keys -> human labels used in the export CSV "split" column.
SPLIT_LABELS = {
    "train": "train",
    "id_val": "id_val",
    "id_test": "id_test",
    "val": "ood_val",
    "test": "ood_test",
}


# --------------------------------------------------------------------------- #
# Dataset construction
# --------------------------------------------------------------------------- #

def load_openbhb_dataset(dataloader_dir, root_dir=None, split_scheme="official"):
    """Import ``OpenBHBDataset`` from ``DataLoader.py`` and instantiate it.

    Parameters
    ----------
    dataloader_dir : str
        Directory that contains ``DataLoader.py`` (the user's file).
    root_dir : str or None
        OpenBHB data root (folder that contains ``images/metadata.tsv`` and the
        ``.npy`` files).  If ``None``, ``OpenBHBDataset``'s own default is used.
    split_scheme : str
        Passed straight through; 'official' maps to the site-based proposed split.
    """
    dataloader_dir = os.path.abspath(dataloader_dir)
    if dataloader_dir not in sys.path:
        sys.path.insert(0, dataloader_dir)
    from DataLoader import OpenBHBDataset  # noqa: E402  (deliberate late import)

    kwargs = dict(split_scheme=split_scheme, download=False)
    if root_dir is not None:
        kwargs["root_dir"] = root_dir
    ds = OpenBHBDataset(**kwargs)
    return ds


# --------------------------------------------------------------------------- #
# Index / metadata helpers
# --------------------------------------------------------------------------- #

def _split_dict(ds):
    return getattr(ds, "_split_dict",
                   {"train": 0, "id_val": 1, "id_test": 2, "val": 3, "test": 4})


def _split_array(ds):
    return np.asarray(ds._split_array).astype(np.int64)


def _original_site(ds):
    return np.asarray(ds.metadata["original_site"]).astype(float)


def resolve_global_idx(ds, idx):
    """Return the global index used for output filenames.

    If the metadata carries an explicit ``global_idx`` column we honour it;
    otherwise the positional row index (== the argument passed to
    ``OpenBHBDataset.get_input``) is the global index.
    """
    if "global_idx" in ds.metadata.columns:
        return int(ds.metadata["global_idx"].iloc[int(idx)])
    return int(idx)


def get_domain_indices(ds, target_site=TARGET_SITE):
    """Return ``(A_idx, B_idx)`` positional indices within the TRAIN split.

    A = train subjects with ``original_site != target_site``
    B = train subjects with ``original_site == target_site``
    """
    split = _split_array(ds)
    sites = _original_site(ds)
    train_mask = split == _split_dict(ds)["train"]
    A_idx = np.where(train_mask & (sites != target_site))[0]
    B_idx = np.where(train_mask & (sites == target_site))[0]
    return A_idx, B_idx


def get_split_indices(ds, split_key):
    """Positional indices of every subject in the WILDS split ``split_key``."""
    sd = _split_dict(ds)
    if split_key not in sd:
        raise KeyError(f"Unknown split '{split_key}'. Known: {sorted(sd)}")
    split = _split_array(ds)
    return np.where(split == sd[split_key])[0]


def subject_meta(ds, idx):
    """Metadata dict for a positional index (used to build the export CSV)."""
    m = ds.metadata
    i = int(idx)
    return {
        "global_idx": resolve_global_idx(ds, i),
        "participant_id": m["participant_id"].iloc[i],
        "age": float(m["age"].iloc[i]),
        "domain_site": int(m["domain_site"].iloc[i]),
        "original_site": float(m["original_site"].iloc[i]),
    }


# --------------------------------------------------------------------------- #
# Image loading + normalization
# --------------------------------------------------------------------------- #

def load_raw_volume(ds, idx):
    """Load one subject's native VBM volume via ``OpenBHBDataset.get_input``.

    Returns a 3D float32 NumPy array (singleton channel dims squeezed out).
    """
    img = ds.get_input(int(idx))
    if hasattr(img, "detach"):          # torch.Tensor
        img = img.detach().cpu().numpy()
    else:
        img = np.asarray(img)
    img = np.asarray(img, dtype=np.float32)
    img = np.squeeze(img)
    if img.ndim == 4:
        # e.g. (C, H, W, D) with C>1 -> keep first channel and warn once.
        img = img[0]
    if img.ndim != 3:
        raise ValueError(
            f"Expected a 3D volume after squeeze, got shape {img.shape} for idx {idx}"
        )
    return img


def normalize(vol, scale):
    """VBM -> network space:  x/S - 1  (native background 0 -> -1)."""
    return vol / float(scale) - 1.0


def denormalize(vol, scale):
    """Network space -> VBM:  (y + 1) * S  (network background -1 -> 0)."""
    return (vol + 1.0) * float(scale)


def compute_norm_scale(ds, idxs, max_subjects=60, verbose=True):
    """S = median of nonzero (in-brain) voxels over a sample of ``idxs``.

    This is the VBM analogue of the repo's fixed ``500`` for T1: after
    ``x/S - 1`` the in-brain median sits near 0 and background sits at -1.
    Computed on the reference domain (site 12) so the target style defines the
    scale; saved to JSON so training and export share an identical transform.
    """
    idxs = np.asarray(idxs)
    if len(idxs) == 0:
        raise ValueError("compute_norm_scale received an empty index list.")
    if len(idxs) > max_subjects:
        # Deterministic, evenly spaced subsample (no RNG -> reproducible).
        sel = np.linspace(0, len(idxs) - 1, max_subjects).round().astype(int)
        idxs = idxs[np.unique(sel)]
    medians = []
    for i in idxs:
        vol = load_raw_volume(ds, i)
        nz = vol[vol > 0]
        if nz.size:
            medians.append(float(np.median(nz)))
    if not medians:
        raise ValueError("All sampled volumes were empty; cannot compute scale.")
    scale = float(np.median(medians))
    if scale <= 0:
        raise ValueError(f"Computed non-positive normalization scale S={scale}.")
    if verbose:
        print(f"[norm] S (median nonzero, n={len(medians)} subjects) = {scale:.6f}")
    return scale


def save_norm_stats(path, scale, extra=None):
    stats = {"scale": float(scale), "convention": "x/S - 1 ; invert (y+1)*S",
             "target_site": TARGET_SITE, "net_shape": list(NET_SHAPE)}
    if extra:
        stats.update(extra)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    return path


def load_norm_stats(path):
    with open(path) as f:
        stats = json.load(f)
    return float(stats["scale"]), stats


# --------------------------------------------------------------------------- #
# Geometry: pad to 192^3 and undo
# --------------------------------------------------------------------------- #

def pad_to(vol3d, shape=NET_SHAPE, cval=BG_NORM):
    """Centre-pad a 3D volume up to ``shape``.

    Returns ``(padded, pad_info)`` where ``pad_info`` is a list of
    ``(before, after)`` per axis so the padding can be undone exactly.
    Raises if any dimension already exceeds the target (this repo's generator is
    fixed at 192^3 and our data is 121x145x121, so padding-only is expected).
    """
    vol3d = np.asarray(vol3d)
    pad_info = []
    for cur, tgt in zip(vol3d.shape, shape):
        if cur > tgt:
            raise ValueError(
                f"Volume dim {cur} exceeds network dim {tgt}; central cropping "
                f"would remove brain voxels. Increase NET_SHAPE.")
        before = (tgt - cur) // 2
        after = tgt - cur - before
        pad_info.append((before, after))
    padded = np.pad(vol3d, pad_info, mode="constant", constant_values=cval)
    return padded, pad_info


def unpad(vol3d, pad_info, orig_shape):
    """Invert :func:`pad_to`, returning a volume of shape ``orig_shape``."""
    sl = tuple(slice(b, b + s) for (b, _), s in zip(pad_info, orig_shape))
    out = np.asarray(vol3d)[sl]
    if tuple(out.shape) != tuple(orig_shape):
        raise ValueError(f"unpad produced {out.shape}, expected {orig_shape}")
    return out


def load_normed_padded(ds, idx, scale, shape=NET_SHAPE):
    """Full load path for one subject.

    Returns
    -------
    net_in : (H, W, D, 1) float32   -- normalized, padded, channel-added
    info : dict with keys ``orig_shape``, ``pad_info``, ``brain_mask``
        ``brain_mask`` is the native-space (unpadded) boolean in-brain mask.
    """
    raw = load_raw_volume(ds, idx)                 # native VBM, 3D
    brain_mask = raw > 0                           # VBM background is exactly 0
    normed = normalize(raw, scale)                 # native 0 -> -1
    padded, pad_info = pad_to(normed, shape, cval=BG_NORM)
    net_in = padded.astype(np.float32)[..., np.newaxis]
    info = {"orig_shape": tuple(raw.shape), "pad_info": pad_info,
            "brain_mask": brain_mask, "raw": raw}
    return net_in, info


# --------------------------------------------------------------------------- #
# Simple, invertible augmentation (random integer shift with -1 fill)
# --------------------------------------------------------------------------- #

def random_shift(vol3d, shift_max, cval=BG_NORM, rng=None):
    """Translate a 3D volume by a random integer offset in each axis.

    Exposed edges are filled with ``cval`` (matches the repo's ``random_shift``
    which fills with -1).  ``vol3d`` is expected to be in normalized space.
    """
    if rng is None:
        rng = np.random
    shifts = [int(rng.integers(-shift_max, shift_max + 1)) if hasattr(rng, "integers")
              else int(rng.randint(-shift_max, shift_max + 1)) for _ in range(3)]
    out = np.full_like(vol3d, cval)
    src, dst = [], []
    for s, sz in zip(shifts, vol3d.shape):
        if s >= 0:
            src.append(slice(0, sz - s)); dst.append(slice(s, sz))
        else:
            src.append(slice(-s, sz)); dst.append(slice(0, sz + s))
    out[tuple(dst)] = vol3d[tuple(src)]
    return out


# --------------------------------------------------------------------------- #
# tf.data pipeline for one domain
# --------------------------------------------------------------------------- #

def make_tf_dataset(ds, idxs, scale, shape=NET_SHAPE, augment=True,
                    shift_max=5, seed=0):
    """Build an infinite ``tf.data.Dataset`` yielding ``[1, *shape, 1]`` batches.

    Index order is reshuffled every pass in Python (cheap), so we do NOT use a
    tf.data shuffle buffer over full 192^3 volumes (which would be very heavy).
    Images are loaded on demand through ``OpenBHBDataset`` inside the generator.
    """
    import tensorflow as tf

    idxs = np.asarray(idxs).astype(np.int64)
    if len(idxs) == 0:
        raise ValueError("make_tf_dataset received an empty index list.")

    def _gen():
        rng = np.random.default_rng(seed)
        order = idxs.copy()
        while True:
            rng.shuffle(order)
            for i in order:
                raw = load_raw_volume(ds, int(i))
                normed = normalize(raw, scale)
                padded, _ = pad_to(normed, shape, cval=BG_NORM)
                if augment:
                    padded = random_shift(padded, shift_max, cval=BG_NORM, rng=rng)
                yield padded.astype(np.float32)[..., np.newaxis]

    sig = tf.TensorSpec(shape=tuple(shape) + (1,), dtype=tf.float32)
    dset = tf.data.Dataset.from_generator(_gen, output_signature=sig)
    dset = dset.batch(1).prefetch(tf.data.AUTOTUNE)
    return dset


# --------------------------------------------------------------------------- #
# Small reporting helper
# --------------------------------------------------------------------------- #

def report_intensity_range(ds, idxs, scale, n=8):
    """Print min/max/mean of the *normalized* volumes over a small sample."""
    idxs = np.asarray(idxs)
    sample = idxs[np.linspace(0, len(idxs) - 1, min(n, len(idxs))).round().astype(int)]
    lo, hi, means = np.inf, -np.inf, []
    for i in np.unique(sample):
        normed = normalize(load_raw_volume(ds, int(i)), scale)
        lo = min(lo, float(normed.min()))
        hi = max(hi, float(normed.max()))
        means.append(float(normed.mean()))
    print(f"[norm] normalized intensity range over {len(np.unique(sample))} "
          f"subjects: min={lo:.4f}, max={hi:.4f}, mean~{np.mean(means):.4f}")
    return lo, hi
