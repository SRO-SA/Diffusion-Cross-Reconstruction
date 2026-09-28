"""
Two-model DDIM reconstruction exporter for OpenBHB MRI volumes.

This version intentionally reuses the DDIM methods already implemented in
simple_diffusion_test.py / MRIObjectivePack:

    - pack.invert_x0_to_xT(...)
    - pack.decode_xT_to_x0(...)
    - pack.reconstruct_roundtrip(...)

No DDIM inverse/forward equations are reimplemented here. This keeps the export
script consistent with the training/evaluation code.

Requested outputs:
  export_root/refined/recon_XXXXXXX.nii.gz
  export_root/unrefined/recon_XXXXXXX.nii.gz
  export_root/cross/recon_XXXXXXX.nii.gz
  export_root/reconstruction_index.csv

CSV columns:
  global_idx, split,
  file_path_ref, file_path_unref, file_path_cross,
  mae_refined, mae_unrefined, mae_cross,
  mse_refined, mse_unrefined, mse_cross
"""

import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from wilds.common.data_loaders import get_eval_loader

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DataLoader import OpenBHBDataset
from simple_diffusion_test import (
    DiffusionSchedule,
    MRIObjectivePack,
    SimpleUNet3D,
    WithGlobalIndex,
)

# Optional imports. These are only needed when you want to instantiate/check the
# refined discriminator from the refined checkpoint. Reconstruction itself only
# uses the diffusion model inside MRIObjectivePack.
try:
    from simple_diffusion_test import StrongDiffusionImageDiscriminator3D
except Exception:
    StrongDiffusionImageDiscriminator3D = None

try:
    from simple_diffusion_test import MRIObjectiveFlexiblePack
except Exception:
    MRIObjectiveFlexiblePack = None

from tqdm.auto import tqdm

# =========================================================
# FIXED layout helpers
# Dataset layout: (B, C, H, W, D)
# Model layout  : (B, C, D, H, W)
# Keep these exactly as in your current script.
# =========================================================
def to_model_layout(x_bchwd):
    # no intensity/spatial changes, only dimension order
    x_bchwd = x_bchwd.squeeze(1)  # (B,1,H,W,D)
    return x_bchwd.permute(0, 1, 4, 2, 3).contiguous()


def from_model_layout(x_bcdhw):
    assert x_bcdhw.ndim == 5, f"Expected [B,C,D,H,W], got {tuple(x_bcdhw.shape)}"
    return x_bcdhw.permute(0, 1, 3, 4, 2).contiguous()


def _compute_per_sample_metrics(x_hat_np: np.ndarray, x_real_np: np.ndarray, *, set_name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns one MAE/MSE scalar per item in the batch.
    Both arrays must be in dataset layout: (B, C, H, W, D).
    """
    if x_hat_np.shape != x_real_np.shape:
        raise ValueError(
            f"Shape mismatch for {set_name}: "
            f"x_hat_np.shape={x_hat_np.shape}, x_real_np.shape={x_real_np.shape}. "
            "This usually means one tensor is still in model layout (B,C,D,H,W) "
            "while the other is in dataset layout (B,C,H,W,D)."
        )

    B = x_real_np.shape[0]
    diff = (x_hat_np.astype(np.float32, copy=False) - x_real_np.astype(np.float32, copy=False)).reshape(B, -1)
    mae = np.mean(np.abs(diff), axis=1)
    mse = np.mean(diff ** 2, axis=1)
    return mae, mse
# =========================================================
# NIfTI helpers
# =========================================================
def save_single_nifti(vol_hwd, out_path, affine=None):
    """
    vol_hwd: numpy array with shape (H, W, D)
    """
    out_path = str(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if affine is None:
        affine = np.eye(4, dtype=np.float32)
    nii = nib.Nifti1Image(vol_hwd.astype(np.float32), affine)
    nib.save(nii, out_path)


def load_reconstruction_by_global_idx(
    global_idx: int,
    export_root: str | Path = "/rhome/ssafa013/bigdata/simple_unet_exports",
    set_name: str = "refined",
):
    export_root = Path(export_root)
    path = export_root / set_name / f"recon_{int(global_idx):07d}.nii.gz"
    if not path.exists():
        raise FileNotFoundError(f"No saved {set_name} reconstruction for global_idx={global_idx} at {path}")
    nii = nib.load(str(path))
    return nii.get_fdata(), str(path)


# =========================================================
# Shape helper for discriminator construction
# =========================================================
def infer_model_input_shape_from_loader(loader):
    """
    Infer model-space input shape (C, D, H, W) from one batch in the loader.
    Safe to call before export/training starts.
    """
    first_batch = next(iter(loader))
    x = first_batch[0] if isinstance(first_batch, (tuple, list)) else first_batch
    x = x.float()
    x = to_model_layout(x)         # -> (B, C, D, H, W)
    return tuple(x.shape[1:])      # (C, D, H, W)


# =========================================================
# Checkpoint loading helpers
# =========================================================
def _is_probably_state_dict(obj) -> bool:
    return isinstance(obj, dict) and any(
        isinstance(k, str) and (k.endswith("weight") or k.endswith("bias"))
        for k in obj.keys()
    )


def _extract_state_dict(checkpoint, possible_keys: Sequence[str]):
    """
    Return the first state_dict-like object found in checkpoint.

    Handles both:
      1) plain model state_dict
      2) training checkpoint dict with keys like model_state_dict/discriminator_state_dict
    """
    if isinstance(checkpoint, dict):
        for key in possible_keys:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]

        if _is_probably_state_dict(checkpoint):
            return checkpoint

    return checkpoint


def _strip_prefix_if_present(state_dict: Mapping, prefix: str) -> Dict:
    if not isinstance(state_dict, Mapping):
        return state_dict
    if not any(str(k).startswith(prefix) for k in state_dict.keys()):
        return dict(state_dict)
    return {str(k)[len(prefix):] if str(k).startswith(prefix) else k: v for k, v in state_dict.items()}


def _strip_common_prefixes(state_dict: Mapping) -> Dict:
    if not isinstance(state_dict, Mapping):
        return state_dict

    cleaned = dict(state_dict)
    for prefix in (
        "module.",
        "model.",
        "model_v.",
        "diffusion_model.",
        "unet.",
        "net.",
    ):
        cleaned = _strip_prefix_if_present(cleaned, prefix)
    return cleaned


def _load_unet_from_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path, device, strict: bool = True):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = _extract_state_dict(
        checkpoint,
        possible_keys=(
            "model",
            "model_state_dict",
            "state_dict",
            "model_v",
            "unet",
            "ema_model",
            "diffusion_model",
        ),
    )
    state = _strip_common_prefixes(state)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if missing or unexpected:
        print(f"[load] {checkpoint_path}")
        print(f"       missing keys   : {len(missing)}")
        print(f"       unexpected keys: {len(unexpected)}")
    return checkpoint


def _maybe_build_and_load_discriminator(
    *,
    checkpoint,
    device,
    input_shape: Optional[Tuple[int, int, int, int]],
    num_domains: int = 15,
    use_time_conditioning: bool = False,
):
    """
    Optional discriminator load/check for refined checkpoints.

    This function is not required for reconstruction. It is useful when your
    refined checkpoint stores discriminator weights and you want to verify the
    checkpoint is being interpreted correctly.
    """
    if StrongDiffusionImageDiscriminator3D is None:
        print("[load] StrongDiffusionImageDiscriminator3D is not importable; skipping discriminator.")
        return None

    if input_shape is None:
        print("[load] input_shape is None; skipping discriminator.")
        return None

    discriminator = StrongDiffusionImageDiscriminator3D(
        input_shape=input_shape,
        in_ch=input_shape[0],
        n_domains=num_domains,
        hidden_size=1024,
        batch_norm=False,
        dropout=0.1,
        use_conv5=True,
        use_instancenorm=True,
        t_emb_dim=64,
        grl_alpha=0.5,
        use_time_conditioning=use_time_conditioning,
        tau_D=0.0,
        s_D=1.0,
    ).to(device)

    disc_state = _extract_state_dict(
        checkpoint,
        possible_keys=(
            "discriminator",
            "discriminator_state_dict",
            "disc",
            "disc_state_dict",
            "D",
        ),
    )

    # If _extract_state_dict returns the whole checkpoint, it probably did not
    # find discriminator weights.
    if disc_state is checkpoint:
        print("[load] No discriminator state_dict found; discriminator constructed but not loaded.")
        return discriminator.eval()

    disc_state = _strip_common_prefixes(disc_state)
    missing, unexpected = discriminator.load_state_dict(disc_state, strict=False)
    print(f"[load] discriminator missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    return discriminator.eval()


# =========================================================
# Pack loading
# =========================================================
def load_pretrained_v_pack(
    model_path: str | Path = "./med-ddpm/results/mri_tests/mri_vpred_unet3d.pt",
    device=None,
    T: int = 1000,
    in_ch: int = 1,
    base: int = 16,
    t_dim: int = 128,
    objective: str = "v",
    strict: bool = True,
):
    """
    Same idea as your old loader, but with slightly more robust checkpoint parsing.
    It returns MRIObjectivePack so DDIM reconstruction uses pack methods.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sched = DiffusionSchedule(T=T, device=device)
    model = SimpleUNet3D(in_ch=in_ch, base=base, t_dim=t_dim).to(device)
    _load_unet_from_checkpoint(model, model_path, device, strict=strict)
    model.eval()
    return MRIObjectivePack(model, sched, device, objective)


def load_pretrained_v_packs(
    *,
    refined_model_path: str | Path,
    unrefined_model_path: str | Path,
    device=None,
    T: int = 1000,
    in_ch: int = 1,
    base: int = 16,
    t_dim: int = 128,
    objective: str = "v",
    strict_unet: bool = True,
    input_shape: Optional[Tuple[int, int, int, int]] = None,
    load_refined_discriminator: bool = True,
    num_domains: int = 15,
    discriminator_use_time_conditioning: bool = False,
):
    """
    Load refined and unrefined models and return MRIObjectivePack objects.

    Important:
      - unrefined uses the old SimpleUNet3D + MRIObjectivePack path.
      - refined also returns an MRIObjectivePack for reconstruction.
      - if the refined checkpoint has discriminator weights, the discriminator is
        optionally constructed/loaded, but it is not used for DDIM reconstruction.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Unrefined model/pack.
    unrefined_sched = DiffusionSchedule(T=T, device=device)
    unrefined_model = SimpleUNet3D(in_ch=in_ch, base=base, t_dim=t_dim).to(device)
    _load_unet_from_checkpoint(unrefined_model, unrefined_model_path, device, strict=strict_unet)
    unrefined_model.eval()
    unrefined_pack = MRIObjectivePack(unrefined_model, unrefined_sched, device, objective)

    # Refined model/pack.
    refined_sched = DiffusionSchedule(T=T, device=device)
    refined_model = SimpleUNet3D(in_ch=in_ch, base=base, t_dim=t_dim).to(device)
    refined_ckpt = _load_unet_from_checkpoint(refined_model, refined_model_path, device, strict=strict_unet)
    refined_model.eval()
    refined_pack = MRIObjectivePack(refined_model, refined_sched, device, objective)

    refined_discriminator = None
    if load_refined_discriminator:
        refined_discriminator = _maybe_build_and_load_discriminator(
            checkpoint=refined_ckpt,
            device=device,
            input_shape=input_shape,
            num_domains=num_domains,
            use_time_conditioning=discriminator_use_time_conditioning,
        )

    return {"refined": refined_pack, "unrefined": unrefined_pack}, refined_discriminator


# =========================================================
# Pack-based reconstruction logic
# =========================================================
def _is_cross_alpha_set_name(set_name: str) -> bool:
    return str(set_name).startswith("cross-alpha-")


def _parse_alpha_from_set_name(set_name: str) -> float:
    """
    Parse names like:
      cross-alpha-0.5
      cross-alpha-0.50
      cross-alpha-0.625
    """
    prefix = "cross-alpha-"
    if not str(set_name).startswith(prefix):
        raise ValueError(f"Not a cross-alpha set name: {set_name}")

    alpha_str = str(set_name)[len(prefix):]
    alpha = float(alpha_str)

    if alpha < 0.0 or alpha > 1.0:
        raise ValueError(f"Alpha must be in [0, 1], got {alpha} from {set_name}")

    return alpha


def _alpha_set_name(alpha: float) -> str:
    """
    Folder-friendly alpha name.
    Example:
      0.5   -> cross-alpha-0.5
      0.625 -> cross-alpha-0.625
    """
    alpha_str = f"{float(alpha):.6g}"
    return f"cross-alpha-{alpha_str}"


def _validate_generate_sets(generate_sets: Sequence[str]) -> Tuple[str, ...]:
    allowed = {"refined", "unrefined", "cross"}
    generate_sets = tuple(str(x) for x in generate_sets)

    unknown = []
    for set_name in generate_sets:
        if set_name in allowed:
            continue
        if _is_cross_alpha_set_name(set_name):
            _parse_alpha_from_set_name(set_name)
            continue
        unknown.append(set_name)

    if unknown:
        raise ValueError(
            f"Unknown generate_sets entries: {sorted(unknown)}. "
            f"Allowed: {sorted(allowed)} plus names like cross-alpha-0.5"
        )

    return generate_sets

@torch.no_grad()
def reconstruct_requested_sets_with_packs(
    *,
    x0_bchwd: torch.Tensor,
    refined_pack: MRIObjectivePack,
    unrefined_pack: MRIObjectivePack,
    generate_sets: Sequence[str] = ("refined", "unrefined", "cross"),
    steps: int = 1000,
    seed: int = 0,
    clip_inverse: bool = False,
    clip_decode: bool = True,
    show_pbar: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Produce requested reconstructions using pack methods only.

    Supported generate_sets:
      refined
      unrefined
      cross
      cross-alpha-0.5
      cross-alpha-0.625
      ...

    Alpha path:
      noise_unrefined = unrefined inverse(real)
      noise_refined   = refined inverse(real)
      noise_alpha = noise_unrefined + alpha * (noise_refined - noise_unrefined)
      output = refined forward(noise_alpha)

    Special cases:
      cross-alpha-0.0 == cross
      cross-alpha-1.0 == refined-style forward from refined inverse
    """
    generate_sets = _validate_generate_sets(generate_sets)
    outputs: Dict[str, torch.Tensor] = {}

    need_noise_unrefined = (
        ("unrefined" in generate_sets)
        or ("cross" in generate_sets)
        or any(_is_cross_alpha_set_name(s) for s in generate_sets)
    )

    need_noise_refined = (
        ("refined" in generate_sets)
        or any(_is_cross_alpha_set_name(s) for s in generate_sets)
    )

    noise_refined = None
    noise_unrefined = None

    if need_noise_refined:
        noise_refined = refined_pack.invert_x0_to_xT(
            x0_bchwd,
            steps=steps,
            seed=seed,
            clip_x0=clip_inverse,
            show_pbar=show_pbar,
        )

    if need_noise_unrefined:
        noise_unrefined = unrefined_pack.invert_x0_to_xT(
            x0_bchwd,
            steps=steps,
            seed=seed,
            clip_x0=clip_inverse,
            show_pbar=show_pbar,
        )

    if "refined" in generate_sets:
        outputs["refined"] = refined_pack.decode_xT_to_x0(
            noise_refined,
            steps=steps,
            clip_x0=clip_decode,
            show_pbar=show_pbar,
        )

    if "unrefined" in generate_sets:
        outputs["unrefined"] = unrefined_pack.decode_xT_to_x0(
            noise_unrefined,
            steps=steps,
            clip_x0=clip_decode,
            show_pbar=show_pbar,
        )

    if "cross" in generate_sets:
        outputs["cross"] = refined_pack.decode_xT_to_x0(
            noise_unrefined,
            steps=steps,
            clip_x0=clip_decode,
            show_pbar=show_pbar,
        )

    for set_name in generate_sets:
        if not _is_cross_alpha_set_name(set_name):
            continue

        alpha = _parse_alpha_from_set_name(set_name)
        noise_alpha = noise_unrefined + alpha * (noise_refined - noise_unrefined)

        outputs[set_name] = refined_pack.decode_xT_to_x0(
            noise_alpha,
            steps=steps,
            clip_x0=clip_decode,
            show_pbar=show_pbar,
        )

    return outputs

# =========================================================
# CSV helpers
# =========================================================
def _empty_record(global_idx: int, split_name: str) -> Dict:
    return {
        "global_idx": int(global_idx),
        "split": split_name,
        
        "file_path_ref": np.nan,
        "file_path_unref": np.nan,
        "file_path_cross": np.nan,
        
        "mae_refined": np.nan,
        "mae_unrefined": np.nan,
        "mae_cross": np.nan,
        
        "mse_refined": np.nan,
        "mse_unrefined": np.nan,
        "mse_cross": np.nan,
    }


def _safe_col_suffix(set_name: str) -> str:
    """
    Make set name safe for CSV column names.
    Example:
      cross-alpha-0.5 -> cross_alpha_0p5
    """
    return (
        str(set_name)
        .replace("-", "_")
        .replace(".", "p")
    )


def _fill_paths(record: Dict, out_paths: Mapping[str, Sequence[Path]], batch_item_index: int):
    if "refined" in out_paths:
        record["file_path_ref"] = str(out_paths["refined"][batch_item_index])

    if "unrefined" in out_paths:
        record["file_path_unref"] = str(out_paths["unrefined"][batch_item_index])

    if "cross" in out_paths:
        record["file_path_cross"] = str(out_paths["cross"][batch_item_index])

    # Generic path columns for alpha sets.
    for set_name in out_paths:
        if _is_cross_alpha_set_name(set_name):
            suffix = _safe_col_suffix(set_name)
            record[f"file_path_{suffix}"] = str(out_paths[set_name][batch_item_index])
            

def _set_metric(record: Dict, set_name: str, mae: float, mse: float):
    if set_name == "refined":
        record["mae_refined"] = float(mae)
        record["mse_refined"] = float(mse)

    elif set_name == "unrefined":
        record["mae_unrefined"] = float(mae)
        record["mse_unrefined"] = float(mse)

    elif set_name == "cross":
        record["mae_cross"] = float(mae)
        record["mse_cross"] = float(mse)

    elif _is_cross_alpha_set_name(set_name):
        suffix = _safe_col_suffix(set_name)
        alpha = _parse_alpha_from_set_name(set_name)

        record[f"alpha_{suffix}"] = float(alpha)
        record[f"mae_{suffix}"] = float(mae)
        record[f"mse_{suffix}"] = float(mse)

    else:
        raise ValueError(f"Unknown set_name={set_name}")

def _write_index_csv(records: Sequence[Dict], csv_path: str | Path) -> pd.DataFrame:
    df = pd.DataFrame(records).drop_duplicates(subset=["global_idx"], keep="last")
    df = df.sort_values("global_idx").reset_index(drop=True)

    fixed_cols = [
        "global_idx",
        "split",

        "file_path_ref",
        "file_path_unref",
        "file_path_cross",

        "mae_refined",
        "mae_unrefined",
        "mae_cross",

        "mse_refined",
        "mse_unrefined",
        "mse_cross",
    ]

    for col in fixed_cols:
        if col not in df.columns:
            df[col] = np.nan

    extra_cols = [c for c in df.columns if c not in fixed_cols]
    extra_cols = sorted(extra_cols)

    df = df[fixed_cols + extra_cols]
    df.to_csv(csv_path, index=False)
    return df


import re

def _extract_global_idx_from_recon_path(path: Path) -> Optional[int]:
    """
    Extract global_idx from recon_XXXXXXX.nii.gz.
    """
    m = re.match(r"^recon_(\d+)\.nii\.gz$", path.name)
    if m is None:
        return None
    return int(m.group(1))


def _existing_global_indices_in_dir(folder: Path) -> set[int]:
    """
    Return all global_idx values that already exist in one output folder.
    Ignores temporary/incomplete files.
    """
    if not folder.exists():
        return set()

    out = set()
    for p in folder.glob("recon_*.nii.gz"):
        gidx = _extract_global_idx_from_recon_path(p)
        if gidx is not None:
            out.add(gidx)
    return out


def _infer_resume_start_global_idx(
    *,
    set_to_dir: Mapping[str, Path],
    generate_sets: Sequence[str],
    resume_overlap: int = 2,
) -> Optional[int]:
    """
    Infer where to resume from existing final images.

    Conservative rule:
      - Look at each requested output folder.
      - Find the largest generated global_idx in each folder.
      - Use the minimum of those largest indices.
      - Step back by resume_overlap.

    Example:
      refined has up to 1000
      unrefined has up to 997
      cross has up to 996
      => resume from 996 - resume_overlap

    This avoids assuming all folders completed equally.
    """
    last_by_set = {}

    for set_name in generate_sets:
        existing = _existing_global_indices_in_dir(set_to_dir[set_name])
        if len(existing) == 0:
            last_by_set[set_name] = None
        else:
            last_by_set[set_name] = max(existing)

    print("[resume] existing last global_idx by folder:")
    for set_name, last_gidx in last_by_set.items():
        print(f"         {set_name:9s}: {last_gidx}")

    valid_last = [v for v in last_by_set.values() if v is not None]
    if len(valid_last) == 0:
        print("[resume] no existing final images found. Starting from the beginning.")
        return None

    conservative_last = min(valid_last)
    resume_start = max(0, conservative_last - int(resume_overlap))

    print(
        f"[resume] conservative_last={conservative_last}, "
        f"resume_overlap={resume_overlap}, resume_start_global_idx={resume_start}"
    )

    return resume_start


def _load_existing_index_records(csv_path: Path) -> list[Dict]:
    """
    Load previous CSV rows so a resumed run does not erase earlier records.
    """
    if not csv_path.exists():
        return []

    df_old = pd.read_csv(csv_path)
    if "global_idx" not in df_old.columns:
        print(f"[resume] existing CSV has no global_idx column; ignoring {csv_path}")
        return []

    print(f"[resume] loaded {len(df_old)} old rows from {csv_path}")
    return df_old.to_dict("records")


def _count_existing_recons(folder: Path) -> int:
    if not folder.exists():
        return 0
    return len(list(folder.glob("recon_*.nii.gz")))


def _infer_resume_position_by_counts(
    *,
    set_to_dir: Mapping[str, Path],
    generate_sets: Sequence[str],
    resume_overlap: int = 2,
) -> int:
    counts = {
        set_name: _count_existing_recons(set_to_dir[set_name])
        for set_name in generate_sets
    }

    print("[resume] existing file counts:")
    for set_name, count in counts.items():
        print(f"         {set_name:9s}: {count}")

    if len(counts) == 0:
        return 0

    completed = min(counts.values())
    resume_position = max(0, completed - int(resume_overlap))

    print(
        f"[resume] completed_min={completed}, "
        f"resume_overlap={resume_overlap}, "
        f"resume_position={resume_position}"
    )

    return resume_position


# =========================================================
# Noise-space analysis helpers
# =========================================================
def _tensor_to_list_int(x):
    if torch.is_tensor(x):
        return x.long().detach().cpu().tolist()
    return [int(v) for v in x]


def _extract_age_vector(y: torch.Tensor) -> np.ndarray:
    """
    y is usually shape [B, 1] for OpenBHB age.
    Returns shape [B].
    """
    if torch.is_tensor(y):
        y_np = y.detach().cpu().float().view(y.shape[0], -1)[:, 0].numpy()
    else:
        y_np = np.asarray(y).reshape(len(y), -1)[:, 0].astype(np.float32)
    return y_np.astype(np.float32)


def _extract_site_vector(metadata: torch.Tensor) -> list[int]:
    """
    Your metadata_array is [site, age, study, participant_id].
    So metadata[:, 0] is site/domain.
    """
    if torch.is_tensor(metadata):
        return metadata[:, 0].long().detach().cpu().tolist()
    return [int(row[0]) for row in metadata]


def _infer_train_age_mean_std(train_split) -> tuple[float, float]:
    """
    Infer train age distribution from the train subset without loading images
    when possible. Falls back to iterating if y_array is unavailable.
    """
    y_arr = getattr(train_split, "y_array", None)

    if y_arr is not None:
        y_np = y_arr.detach().cpu().float().view(y_arr.shape[0], -1)[:, 0].numpy()
        return float(np.mean(y_np)), float(np.std(y_np, ddof=1))

    # Fallback: iterate through subset.
    ages = []
    for i in range(len(train_split)):
        _, y, _ = train_split[i]
        if torch.is_tensor(y):
            ages.append(float(y.detach().cpu().view(-1)[0]))
        else:
            ages.append(float(np.asarray(y).reshape(-1)[0]))

    ages = np.asarray(ages, dtype=np.float32)
    return float(np.mean(ages)), float(np.std(ages, ddof=1))


def _age_bucket_from_seen_range(age: float, low: float, high: float) -> tuple[str, str]:
    """
    Returns:
      age_bucket: readable bucket
      seen_or_unseen: seen_age_range / outside_seen_range
    """
    if age < low:
        return f"below_seen_<{low:.2f}", "outside_seen_range"
    if age > high:
        return f"above_seen_>{high:.2f}", "outside_seen_range"
    return f"seen_[{low:.2f},{high:.2f}]", "seen_age_range"


def _per_sample_noise_stats(
    noise_unrefined: torch.Tensor,
    noise_refined: torch.Tensor,
    eps: float = 1e-8,
) -> dict[str, np.ndarray]:
    """
    Compute per-sample metrics between unrefined/refined DDIM-inverted noises.

    Inputs can be any shape [B, ...]. They are flattened per sample.
    """
    if noise_unrefined.shape != noise_refined.shape:
        raise ValueError(
            f"Noise shape mismatch: "
            f"noise_unrefined={tuple(noise_unrefined.shape)}, "
            f"noise_refined={tuple(noise_refined.shape)}"
        )

    u = noise_unrefined.detach().float().flatten(start_dim=1)
    r = noise_refined.detach().float().flatten(start_dim=1)

    diff = u - r

    noise_mae = diff.abs().mean(dim=1)
    noise_mse = (diff ** 2).mean(dim=1)
    noise_rmse = torch.sqrt(noise_mse + eps)

    norm_unrefined = torch.linalg.vector_norm(u, dim=1)
    norm_refined = torch.linalg.vector_norm(r, dim=1)

    cosine_similarity = torch.sum(u * r, dim=1) / ((norm_unrefined * norm_refined) + eps)
    cosine_distance = 1.0 - cosine_similarity

    norm_ratio = norm_unrefined / (norm_refined + eps)

    return {
        "noise_mae": noise_mae.detach().cpu().numpy(),
        "noise_rmse": noise_rmse.detach().cpu().numpy(),
        "noise_cosine_similarity": cosine_similarity.detach().cpu().numpy(),
        "noise_cosine_distance": cosine_distance.detach().cpu().numpy(),
        "norm_unrefined": norm_unrefined.detach().cpu().numpy(),
        "norm_refined": norm_refined.detach().cpu().numpy(),
        "norm_ratio": norm_ratio.detach().cpu().numpy(),
    }


# =========================================================
# Main noise metrics function
# =========================================================
@torch.no_grad()
def export_refined_unrefined_noise_metrics(
    *,
    refined_pack: MRIObjectivePack,
    unrefined_pack: MRIObjectivePack,
    splits: Mapping[str, torch.utils.data.Dataset],
    export_root: str | Path = "/rhome/ssafa013/bigdata/simple_unet_exports",
    batch_size: int = 2,
    steps: int = 1000,
    seed: int = 0,
    num_workers: int = 1,
    clip_inverse: bool = False,
    show_pbar: bool = False,
    save_csv_every: int = 10,
    empty_cache_each_batch: bool = False,
    seen_age_mean: Optional[float] = None,
    seen_age_std: Optional[float] = None,
    seen_age_num_std: float = 2.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run DDIM inverse only with both models and compare the final x_T noises.

    Writes three CSV files:

      1) noise_refined_metrics.csv
         Per-sample norm/stat info for noise_refined.

      2) noise_unrefined_metrics.csv
         Per-sample norm/stat info for noise_unrefined.

      3) noise_refined_vs_unrefined_pairwise_metrics.csv
         Per-sample comparison metrics:
           split, global_idx, site, age, age_bucket, seen_or_unseen,
           noise_mae, noise_rmse, noise_cosine_similarity,
           noise_cosine_distance, norm_unrefined, norm_refined, norm_ratio

    Important:
      This does not run DDIM forward and does not save NIfTI images.
    """
    export_root = Path(export_root)
    noise_dir = export_root / "noise_inverse_analysis"
    noise_dir.mkdir(parents=True, exist_ok=True)

    refined_csv = noise_dir / "noise_refined_metrics.csv"
    unrefined_csv = noise_dir / "noise_unrefined_metrics.csv"
    pairwise_csv = noise_dir / "noise_refined_vs_unrefined_pairwise_metrics.csv"

    # Infer train age support if not provided.
    if seen_age_mean is None or seen_age_std is None:
        if "train" not in splits:
            raise ValueError(
                "seen_age_mean/std were not provided and splits does not contain 'train'. "
                "Pass seen_age_mean and seen_age_std manually."
            )

        seen_age_mean, seen_age_std = _infer_train_age_mean_std(splits["train"])

    seen_low = float(seen_age_mean - seen_age_num_std * seen_age_std)
    seen_high = float(seen_age_mean + seen_age_num_std * seen_age_std)

    print("[noise analysis] DDPM seen age range:")
    print(f"  mean={seen_age_mean:.4f}, std={seen_age_std:.4f}")
    print(f"  mean ± {seen_age_num_std} std = [{seen_low:.4f}, {seen_high:.4f}]")

    refined_rows = []
    unrefined_rows = []
    pairwise_rows = []

    total_expected = sum(len(split_ds) for split_ds in splits.values())
    total_done = 0

    for split_name, split_ds in splits.items():
        print(f"\n==== noise inverse analysis split: {split_name} ====")

        ds = WithGlobalIndex(split_ds)
        dl = get_eval_loader(
            "standard",
            ds,
            batch_size=batch_size,
            drop_last=False,
            num_workers=num_workers,
        )

        iterator = tqdm(dl, desc=f"noise inverse [{split_name}]", leave=False)

        for batch_i, batch in enumerate(iterator):
            x0_bchwd = batch[0].float()
            y = batch[1]
            metadata = batch[2]
            global_ids = _tensor_to_list_int(batch[3])

            ages = _extract_age_vector(y)
            sites = _extract_site_vector(metadata)

            x0_bchwd_device = x0_bchwd.to(refined_pack.device, non_blocking=True)

            # DDIM inverse only.
            # These are model-specific final noises x_T.
            noise_refined = refined_pack.invert_x0_to_xT(
                x0_bchwd_device,
                steps=steps,
                seed=seed,
                clip_x0=clip_inverse,
                show_pbar=show_pbar,
            )

            noise_unrefined = unrefined_pack.invert_x0_to_xT(
                x0_bchwd_device,
                steps=steps,
                seed=seed,
                clip_x0=clip_inverse,
                show_pbar=show_pbar,
            )

            stats = _per_sample_noise_stats(
                noise_unrefined=noise_unrefined,
                noise_refined=noise_refined,
            )

            # Per-source norms/statistics.
            nr = noise_refined.detach().float().flatten(start_dim=1)
            nu = noise_unrefined.detach().float().flatten(start_dim=1)

            refined_norm = torch.linalg.vector_norm(nr, dim=1).detach().cpu().numpy()
            unrefined_norm = torch.linalg.vector_norm(nu, dim=1).detach().cpu().numpy()

            refined_mean = nr.mean(dim=1).detach().cpu().numpy()
            refined_std = nr.std(dim=1).detach().cpu().numpy()
            unrefined_mean = nu.mean(dim=1).detach().cpu().numpy()
            unrefined_std = nu.std(dim=1).detach().cpu().numpy()

            B = len(global_ids)

            for bi in range(B):
                gidx = int(global_ids[bi])
                age = float(ages[bi])
                site = int(sites[bi])
                age_bucket, seen_or_unseen = _age_bucket_from_seen_range(age, seen_low, seen_high)

                refined_rows.append({
                    "split": split_name,
                    "global_idx": gidx,
                    "site": site,
                    "age": age,
                    "age_bucket": age_bucket,
                    "seen_or_unseen": seen_or_unseen,
                    "noise_source": "refined",
                    "noise_norm": float(refined_norm[bi]),
                    "noise_mean": float(refined_mean[bi]),
                    "noise_std": float(refined_std[bi]),
                })

                unrefined_rows.append({
                    "split": split_name,
                    "global_idx": gidx,
                    "site": site,
                    "age": age,
                    "age_bucket": age_bucket,
                    "seen_or_unseen": seen_or_unseen,
                    "noise_source": "unrefined",
                    "noise_norm": float(unrefined_norm[bi]),
                    "noise_mean": float(unrefined_mean[bi]),
                    "noise_std": float(unrefined_std[bi]),
                })

                pairwise_rows.append({
                    "split": split_name,
                    "global_idx": gidx,
                    "site": site,
                    "age": age,
                    "age_bucket": age_bucket,
                    "seen_or_unseen": seen_or_unseen,
                    "noise_mae": float(stats["noise_mae"][bi]),
                    "noise_rmse": float(stats["noise_rmse"][bi]),
                    "noise_cosine_similarity": float(stats["noise_cosine_similarity"][bi]),
                    "noise_cosine_distance": float(stats["noise_cosine_distance"][bi]),
                    "norm_unrefined": float(stats["norm_unrefined"][bi]),
                    "norm_refined": float(stats["norm_refined"][bi]),
                    "norm_ratio": float(stats["norm_ratio"][bi]),
                })

            total_done += B
            iterator.set_postfix(done=f"{total_done}/{total_expected}")

            if save_csv_every is not None and save_csv_every > 0 and ((batch_i + 1) % save_csv_every == 0):
                pd.DataFrame(refined_rows).to_csv(refined_csv, index=False)
                pd.DataFrame(unrefined_rows).to_csv(unrefined_csv, index=False)
                pd.DataFrame(pairwise_rows).to_csv(pairwise_csv, index=False)

            del x0_bchwd_device, noise_refined, noise_unrefined, nr, nu
            if refined_pack.device.type == "cuda" and empty_cache_each_batch:
                torch.cuda.empty_cache()

    refined_df = pd.DataFrame(refined_rows)
    unrefined_df = pd.DataFrame(unrefined_rows)
    pairwise_df = pd.DataFrame(pairwise_rows)

    refined_df.to_csv(refined_csv, index=False)
    unrefined_df.to_csv(unrefined_csv, index=False)
    pairwise_df.to_csv(pairwise_csv, index=False)

    print("\n[noise analysis] Done.")
    print(f"Saved refined noise CSV   : {refined_csv}")
    print(f"Saved unrefined noise CSV : {unrefined_csv}")
    print(f"Saved pairwise noise CSV  : {pairwise_csv}")

    return refined_df, unrefined_df, pairwise_df


# =========================================================
# Alpha interpolation experiment helpers
# =========================================================
def _extract_split_arrays(split_ds):
    """
    Return arrays aligned with local split indices:
      local_idx, global_idx, age, site
    """
    n = len(split_ds)

    base = getattr(split_ds, "indices", None)
    if base is None:
        base = getattr(split_ds, "_indices", None)

    if base is not None and hasattr(base, "tolist"):
        base = base.tolist()

    if base is None:
        global_idx = np.arange(n, dtype=np.int64)
    else:
        global_idx = np.asarray(base, dtype=np.int64)

    y_arr = getattr(split_ds, "y_array", None)
    meta_arr = getattr(split_ds, "metadata_array", None)

    if y_arr is not None:
        if torch.is_tensor(y_arr):
            age = y_arr.detach().cpu().float().view(n, -1)[:, 0].numpy()
        else:
            age = np.asarray(y_arr).reshape(n, -1)[:, 0].astype(np.float32)
    else:
        ages = []
        for i in range(n):
            _, y, _ = split_ds[i]
            if torch.is_tensor(y):
                ages.append(float(y.detach().cpu().view(-1)[0]))
            else:
                ages.append(float(np.asarray(y).reshape(-1)[0]))
        age = np.asarray(ages, dtype=np.float32)

    if meta_arr is not None:
        if torch.is_tensor(meta_arr):
            site = meta_arr.detach().cpu().long()[:, 0].numpy()
        else:
            site = np.asarray(meta_arr)[:, 0].astype(np.int64)
    else:
        sites = []
        for i in range(n):
            _, _, m = split_ds[i]
            if torch.is_tensor(m):
                sites.append(int(m.detach().cpu().view(-1)[0]))
            else:
                sites.append(int(np.asarray(m).reshape(-1)[0]))
        site = np.asarray(sites, dtype=np.int64)

    local_idx = np.arange(n, dtype=np.int64)
    return local_idx, global_idx, age, site


def select_alpha_experiment_indices(
    split_ds,
    *,
    n_young: int = 8,
    n_old: int = 8,
    young_max_age: float = 25.0,
    old_min_age: float = 40.0,
    selected_global_ids: Optional[Sequence[int]] = None,
    seed: int = 0,
):
    """
    Select a small subset for the alpha interpolation experiment.

    If selected_global_ids is given, use those exact OpenBHB global indices.
    Otherwise, select n_young samples with age <= young_max_age and
    n_old samples with age >= old_min_age.
    """
    rng = np.random.default_rng(seed)

    local_idx, global_idx, age, site = _extract_split_arrays(split_ds)

    if selected_global_ids is not None:
        selected_global_ids = set(int(x) for x in selected_global_ids)
        mask = np.asarray([int(g) in selected_global_ids for g in global_idx], dtype=bool)
        return local_idx[mask].astype(int).tolist()

    young_candidates = local_idx[age <= young_max_age]
    old_candidates = local_idx[age >= old_min_age]

    if len(young_candidates) == 0:
        young_candidates = local_idx[np.argsort(age)[:max(1, n_young)]]

    if len(old_candidates) == 0:
        old_candidates = local_idx[np.argsort(-age)[:max(1, n_old)]]

    rng.shuffle(young_candidates)
    rng.shuffle(old_candidates)

    selected = list(young_candidates[:n_young]) + list(old_candidates[:n_old])

    # remove duplicates while preserving order
    out = []
    seen = set()
    for idx in selected:
        idx = int(idx)
        if idx not in seen:
            out.append(idx)
            seen.add(idx)

    return out


class SelectedWithGlobalIndex(torch.utils.data.Dataset):
    """
    A small selected subset wrapper that preserves global_idx.
    """
    def __init__(self, split_ds, selected_local_indices):
        self.base = WithGlobalIndex(split_ds)
        self.selected_local_indices = [int(i) for i in selected_local_indices]

        if hasattr(self.base, "collate"):
            self.collate = self.base.collate

    def __len__(self):
        return len(self.selected_local_indices)

    def __getitem__(self, i):
        return self.base[self.selected_local_indices[i]]


def _to_hwd_numpy_from_model(x_bcdhw: torch.Tensor) -> np.ndarray:
    """
    Convert model layout (B,C,D,H,W) -> numpy (B,H,W,D), using channel 0.
    """
    x = x_bcdhw.detach().float().cpu()
    x = x.permute(0, 1, 3, 4, 2).contiguous()  # B,C,H,W,D
    return x[:, 0].numpy()


def _psnr_from_mse(mse: float, max_val: float = 1.0) -> float:
    if mse <= 1e-12:
        return float("inf")
    return float(20.0 * np.log10(max_val / np.sqrt(mse)))


def _simple_volume_features(vol_hwd: np.ndarray, real_mask_hwd: np.ndarray) -> Dict[str, float]:
    """
    Fast feature metrics using the same real mask for real and generated images.
    """
    vol = vol_hwd.astype(np.float32, copy=False)
    mask = real_mask_hwd.astype(bool)

    if mask.sum() < 10:
        mask = np.ones_like(vol, dtype=bool)

    fg = vol[mask]

    gx, gy, gz = np.gradient(vol)
    grad_mag = np.sqrt(gx * gx + gy * gy + gz * gz)
    edge_vals = grad_mag[mask]

    lap = (
        np.roll(vol, 1, axis=0) + np.roll(vol, -1, axis=0)
        + np.roll(vol, 1, axis=1) + np.roll(vol, -1, axis=1)
        + np.roll(vol, 1, axis=2) + np.roll(vol, -1, axis=2)
        - 6.0 * vol
    )
    lap_vals = np.abs(lap)[mask]

    masked_vol = np.zeros_like(vol, dtype=np.float32)
    masked_vol[mask] = vol[mask]

    fft = np.fft.fftn(masked_vol)
    power = np.abs(fft) ** 2

    H, W, D = vol.shape
    fx = np.fft.fftfreq(H)
    fy = np.fft.fftfreq(W)
    fz = np.fft.fftfreq(D)

    rr = np.sqrt(
        fx[:, None, None] ** 2
        + fy[None, :, None] ** 2
        + fz[None, None, :] ** 2
    )
    rr = rr / (rr.max() + 1e-12)

    total_power = power.sum() + 1e-12
    low_power = power[rr <= 0.15].sum()
    high_power = power[rr >= 0.35].sum()

    return {
        "fg_mean": float(np.mean(fg)),
        "fg_std": float(np.std(fg)),
        "fg_abs_mean": float(np.mean(np.abs(fg))),
        "edge_mean": float(np.mean(edge_vals)),
        "lap_abs_mean": float(np.mean(lap_vals)),
        "fft_low_frac": float(low_power / total_power),
        "fft_high_to_low": float(high_power / (low_power + 1e-12)),
    }


def _endpoint_position(d_to_cross: float, d_to_refined: float, eps: float = 1e-12) -> tuple[float, float, float]:
    """
    endpoint_position:
      0.0 -> exactly cross-like
      1.0 -> exactly refined-like

    cross_like_score:
      high -> closer to cross endpoint

    refined_like_score:
      high -> closer to refined endpoint
    """
    denom = float(d_to_cross + d_to_refined + eps)

    endpoint_position = float(d_to_cross / denom)
    cross_like_score = float(d_to_refined / denom)
    refined_like_score = float(d_to_cross / denom)

    return endpoint_position, cross_like_score, refined_like_score


def _scalar_feature_position(
    f_alpha: float,
    f_cross: float,
    f_refined: float,
    eps: float = 1e-12,
) -> tuple[float, float, float, float]:
    """
    Position of one scalar feature between cross and refined.

    feature_position:
      0.0 -> same as cross
      1.0 -> same as refined

    It can go below 0 or above 1 if alpha output moves outside the endpoint segment.
    """
    denom = abs(float(f_refined) - float(f_cross)) + eps

    d_to_cross = abs(float(f_alpha) - float(f_cross))
    d_to_refined = abs(float(f_alpha) - float(f_refined))

    feature_position = float((float(f_alpha) - float(f_cross)) / (float(f_refined) - float(f_cross) + eps))

    endpoint_position, cross_like_score, refined_like_score = _endpoint_position(
        d_to_cross,
        d_to_refined,
        eps=eps,
    )

    return feature_position, endpoint_position, cross_like_score, refined_like_score


def _feature_endpoint_distance(
    alpha_feat: dict,
    cross_feat: dict,
    refined_feat: dict,
    feature_keys: Sequence[str],
    eps: float = 1e-12,
) -> tuple[float, float, float, float, float]:
    """
    Multi-feature endpoint distance using normalized feature differences.

    Each feature is normalized by the absolute refined-cross endpoint gap
    for that feature, so no single large-scale feature dominates.
    """
    d_cross_sq = 0.0
    d_ref_sq = 0.0

    for k in feature_keys:
        scale = abs(float(refined_feat[k]) - float(cross_feat[k])) + eps

        da = (float(alpha_feat[k]) - float(cross_feat[k])) / scale
        dr = (float(alpha_feat[k]) - float(refined_feat[k])) / scale

        d_cross_sq += da * da
        d_ref_sq += dr * dr

    d_to_cross = float(np.sqrt(d_cross_sq))
    d_to_refined = float(np.sqrt(d_ref_sq))

    endpoint_position, cross_like_score, refined_like_score = _endpoint_position(
        d_to_cross,
        d_to_refined,
        eps=eps,
    )

    return d_to_cross, d_to_refined, endpoint_position, cross_like_score, refined_like_score

# =========================================================
# export based on alpha, alpha = 1.0 fully refined, alpha = 0.0 fully cross
# =========================================================

@torch.no_grad()
def export_alpha_noise_interpolation_experiment(
    *,
    refined_pack: MRIObjectivePack,
    unrefined_pack: MRIObjectivePack,
    splits: Mapping[str, torch.utils.data.Dataset],
    export_root: str | Path = "/rhome/ssafa013/bigdata/simple_unet_exports",
    split_names: Sequence[str] = ("val", "test"),
    alphas: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    n_young_per_split: int = 8,
    n_old_per_split: int = 8,
    young_max_age: float = 25.0,
    old_min_age: float = 40.0,
    selected_global_ids_by_split: Optional[Mapping[str, Sequence[int]]] = None,
    batch_size: int = 1,
    steps: int = 1000,
    seed: int = 0,
    num_workers: int = 1,
    clip_inverse: bool = False,
    clip_decode: bool = True,
    show_pbar: bool = False,
    save_nifti: bool = True,
    save_csv_every: int = 1,
    empty_cache_each_batch: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Alpha interpolation between unrefined-inverse noise and refined-inverse noise.

      noise_unref = unrefined_pack.invert_x0_to_xT(real)
      noise_ref   = refined_pack.invert_x0_to_xT(real)

      z_alpha = noise_unref + alpha * (noise_ref - noise_unref)
      x_alpha = refined_pack.decode_xT_to_x0(z_alpha)

    alpha = 0.0 gives cross-style output.
    alpha = 1.0 gives refined-style output.

    Saves:
      export_root/alpha_noise_interpolation/alpha_noise_interpolation_metrics.csv
      export_root/alpha_noise_interpolation/alpha_noise_interpolation_selected_samples.csv
      export_root/alpha_noise_interpolation/alpha_noise_interpolation_summary.csv

    If save_nifti=True, saves:
      export_root/alpha_noise_interpolation/nifti/alpha_0.00/recon_XXXXXXX.nii.gz
      ...
    """
    export_root = Path(export_root)
    out_dir = export_root / "alpha_noise_interpolation"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = out_dir / "alpha_noise_interpolation_metrics.csv"
    selected_csv = out_dir / "alpha_noise_interpolation_selected_samples.csv"
    summary_csv = out_dir / "alpha_noise_interpolation_summary.csv"

    rows = []
    selected_rows = []

    for split_name in split_names:
        if split_name not in splits:
            print(f"[alpha-exp] split={split_name} not found. Skipping.")
            continue

        split_ds = splits[split_name]

        manual_ids = None
        if selected_global_ids_by_split is not None:
            manual_ids = selected_global_ids_by_split.get(split_name, None)

        selected_local = select_alpha_experiment_indices(
            split_ds,
            n_young=n_young_per_split,
            n_old=n_old_per_split,
            young_max_age=young_max_age,
            old_min_age=old_min_age,
            selected_global_ids=manual_ids,
            seed=seed,
        )

        print(
            f"[alpha-exp] split={split_name}: selected "
            f"{len(selected_local)} samples"
        )

        selected_ds = SelectedWithGlobalIndex(split_ds, selected_local)

        dl = get_eval_loader(
            "standard",
            selected_ds,
            batch_size=batch_size,
            drop_last=False,
            num_workers=num_workers,
        )

        iterator = tqdm(dl, desc=f"alpha-exp [{split_name}]", leave=False)

        for batch_i, batch in enumerate(iterator):
            x0_bchwd = batch[0].float()
            y = batch[1]
            metadata = batch[2]
            global_ids = _tensor_to_list_int(batch[3])

            ages = _extract_age_vector(y)
            sites = _extract_site_vector(metadata)

            x0_device = x0_bchwd.to(refined_pack.device, non_blocking=True)
            x0_real_model = to_model_layout(x0_device)  # B,C,D,H,W

            # Real image HWD for feature/mask computations.
            real_hwd = _to_hwd_numpy_from_model(x0_real_model)

            # Invert using both models.
            noise_unrefined = unrefined_pack.invert_x0_to_xT(
                x0_device,
                steps=steps,
                seed=seed,
                clip_x0=clip_inverse,
                show_pbar=show_pbar,
            )

            noise_refined = refined_pack.invert_x0_to_xT(
                x0_device,
                steps=steps,
                seed=seed,
                clip_x0=clip_inverse,
                show_pbar=show_pbar,
            )

            delta_noise = noise_refined - noise_unrefined

            # Noise diagnostics.
            u = noise_unrefined.detach().float().flatten(start_dim=1)
            r = noise_refined.detach().float().flatten(start_dim=1)
            d = delta_noise.detach().float().flatten(start_dim=1)

            norm_unrefined = torch.linalg.vector_norm(u, dim=1)
            norm_refined = torch.linalg.vector_norm(r, dim=1)
            norm_delta = torch.linalg.vector_norm(d, dim=1)

            noise_mae = d.abs().mean(dim=1)
            noise_rmse = torch.sqrt((d ** 2).mean(dim=1) + 1e-12)
            noise_cos_sim = torch.sum(u * r, dim=1) / ((norm_unrefined * norm_refined) + 1e-8)
            noise_cos_dist = 1.0 - noise_cos_sim
            norm_ratio = norm_unrefined / (norm_refined + 1e-8)

            B = len(global_ids)

            for bi in range(B):
                age = float(ages[bi])
                if age <= young_max_age:
                    age_group = "young"
                elif age >= old_min_age:
                    age_group = "old"
                else:
                    age_group = "middle"

                selected_rows.append({
                    "split": split_name,
                    "global_idx": int(global_ids[bi]),
                    "site": int(sites[bi]),
                    "age": age,
                    "age_group": age_group,
                    "noise_mae_ref_vs_unref": float(noise_mae[bi].detach().cpu()),
                    "noise_rmse_ref_vs_unref": float(noise_rmse[bi].detach().cpu()),
                    "noise_cosine_similarity": float(noise_cos_sim[bi].detach().cpu()),
                    "noise_cosine_distance": float(noise_cos_dist[bi].detach().cpu()),
                    "norm_unrefined": float(norm_unrefined[bi].detach().cpu()),
                    "norm_refined": float(norm_refined[bi].detach().cpu()),
                    "norm_delta": float(norm_delta[bi].detach().cpu()),
                    "norm_ratio": float(norm_ratio[bi].detach().cpu()),
                })

            # Decode each alpha using refined forward.
            # ---------------------------------------------------------
            # Decode each alpha using refined forward.
            # We first cache all alpha outputs for this batch so we can
            # compare every alpha to alpha=0.0 cross and alpha=1.0 refined.
            # ---------------------------------------------------------
            alpha_outputs_model = {}
            alpha_outputs_hwd = {}
            alpha_feature_cache = {}
            alpha_rows_cache = {}

            feature_keys_for_endpoint = [
                "fg_std",
                "edge_mean",
                "lap_abs_mean",
                "fft_low_frac",
                "fft_high_to_low",
            ]

            alphas = tuple(float(a) for a in alphas)

            if 0.0 not in alphas:
                raise ValueError("alphas must include 0.0 because it is the cross endpoint.")
            if 1.0 not in alphas:
                raise ValueError("alphas must include 1.0 because it is the refined endpoint.")

            for alpha in alphas:
                z_alpha = noise_unrefined + alpha * delta_noise

                x_alpha = refined_pack.decode_xT_to_x0(
                    z_alpha,
                    steps=steps,
                    clip_x0=clip_decode,
                    show_pbar=show_pbar,
                )

                # Keep CPU copy for endpoint metrics.
                alpha_outputs_model[alpha] = x_alpha.detach().float().cpu()
                alpha_hwd = _to_hwd_numpy_from_model(x_alpha)
                alpha_outputs_hwd[alpha] = alpha_hwd

                diff = (x_alpha - x0_real_model).flatten(start_dim=1)
                mae_vec = diff.abs().mean(dim=1).detach().cpu().numpy()
                mse_vec = (diff ** 2).mean(dim=1).detach().cpu().numpy()

                alpha_rows_cache[alpha] = []
                alpha_feature_cache[alpha] = []

                for bi in range(B):
                    gidx = int(global_ids[bi])
                    age = float(ages[bi])
                    site = int(sites[bi])

                    if age <= young_max_age:
                        age_group = "young"
                    elif age >= old_min_age:
                        age_group = "old"
                    else:
                        age_group = "middle"

                    real_mask = np.abs(real_hwd[bi]) > 1e-6

                    real_feat = _simple_volume_features(real_hwd[bi], real_mask)
                    alpha_feat = _simple_volume_features(alpha_hwd[bi], real_mask)

                    alpha_feature_cache[alpha].append(alpha_feat)

                    row = {
                        "split": split_name,
                        "global_idx": gidx,
                        "site": site,
                        "age": age,
                        "age_group": age_group,
                        "alpha": alpha,
                        "alpha_label": (
                            "cross_alpha0"
                            if abs(alpha - 0.0) < 1e-12
                            else "refined_alpha1"
                            if abs(alpha - 1.0) < 1e-12
                            else f"interp_alpha{alpha:.2f}"
                        ),

                        # noise diagnostics
                        "noise_mae_ref_vs_unref": float(noise_mae[bi].detach().cpu()),
                        "noise_rmse_ref_vs_unref": float(noise_rmse[bi].detach().cpu()),
                        "noise_cosine_similarity": float(noise_cos_sim[bi].detach().cpu()),
                        "noise_cosine_distance": float(noise_cos_dist[bi].detach().cpu()),
                        "norm_unrefined": float(norm_unrefined[bi].detach().cpu()),
                        "norm_refined": float(norm_refined[bi].detach().cpu()),
                        "norm_delta": float(norm_delta[bi].detach().cpu()),
                        "norm_ratio": float(norm_ratio[bi].detach().cpu()),

                        # reconstruction metrics vs real
                        "mae_to_real": float(mae_vec[bi]),
                        "mse_to_real": float(mse_vec[bi]),
                        "psnr_to_real": _psnr_from_mse(float(mse_vec[bi]), max_val=1.0),
                    }

                    for k in real_feat.keys():
                        row[f"{k}_real"] = float(real_feat[k])
                        row[f"{k}_alpha"] = float(alpha_feat[k])
                        row[f"{k}_gap_alpha_minus_real"] = float(alpha_feat[k] - real_feat[k])
                        row[f"{k}_abs_gap"] = float(abs(alpha_feat[k] - real_feat[k]))

                    if save_nifti:
                        alpha_dir = out_dir / "nifti" / f"alpha_{alpha:.2f}"
                        alpha_dir.mkdir(parents=True, exist_ok=True)

                        nii_path = alpha_dir / f"{split_name}_recon_{gidx:07d}.nii.gz"
                        save_single_nifti(
                            alpha_hwd[bi],
                            nii_path,
                            affine=np.eye(4, dtype=np.float32),
                        )
                        row["nifti_path"] = str(nii_path)
                    else:
                        row["nifti_path"] = ""

                    alpha_rows_cache[alpha].append(row)

                del z_alpha, x_alpha
                if refined_pack.device.type == "cuda" and empty_cache_each_batch:
                    torch.cuda.empty_cache()

            # ---------------------------------------------------------
            # Add endpoint metrics relative to alpha=0 cross and alpha=1 refined.
            # ---------------------------------------------------------
            x_cross_cpu = alpha_outputs_model[0.0]
            x_refined_cpu = alpha_outputs_model[1.0]

            for alpha in alphas:
                x_alpha_cpu = alpha_outputs_model[alpha]

                for bi in range(B):
                    row = alpha_rows_cache[alpha][bi]

                    # -----------------------------
                    # Voxel endpoint metrics
                    # -----------------------------
                    x_a = x_alpha_cpu[bi:bi+1].flatten()
                    x_c = x_cross_cpu[bi:bi+1].flatten()
                    x_r = x_refined_cpu[bi:bi+1].flatten()

                    voxel_d_to_cross = torch.mean(torch.abs(x_a - x_c)).item()
                    voxel_d_to_refined = torch.mean(torch.abs(x_a - x_r)).item()

                    (
                        voxel_endpoint_position,
                        voxel_cross_like_score,
                        voxel_refined_like_score,
                    ) = _endpoint_position(voxel_d_to_cross, voxel_d_to_refined)

                    row["voxel_d_to_cross"] = float(voxel_d_to_cross)
                    row["voxel_d_to_refined"] = float(voxel_d_to_refined)
                    row["voxel_endpoint_position"] = float(voxel_endpoint_position)
                    row["voxel_cross_like_score"] = float(voxel_cross_like_score)
                    row["voxel_refined_like_score"] = float(voxel_refined_like_score)

                    # -----------------------------
                    # Feature endpoint metrics
                    # -----------------------------
                    alpha_feat = alpha_feature_cache[alpha][bi]
                    cross_feat = alpha_feature_cache[0.0][bi]
                    refined_feat = alpha_feature_cache[1.0][bi]

                    (
                        feature_d_to_cross,
                        feature_d_to_refined,
                        feature_endpoint_position,
                        feature_cross_like_score,
                        feature_refined_like_score,
                    ) = _feature_endpoint_distance(
                        alpha_feat,
                        cross_feat,
                        refined_feat,
                        feature_keys=feature_keys_for_endpoint,
                    )

                    row["feature_d_to_cross"] = float(feature_d_to_cross)
                    row["feature_d_to_refined"] = float(feature_d_to_refined)
                    row["feature_endpoint_position"] = float(feature_endpoint_position)
                    row["feature_cross_like_score"] = float(feature_cross_like_score)
                    row["feature_refined_like_score"] = float(feature_refined_like_score)

                    # Balance score: high only when alpha is not too close to either endpoint.
                    # This is a proxy for keeping some cross-like domain suppression while
                    # recovering refined-like biological fidelity.
                    row["feature_balance_score"] = float(
                        feature_cross_like_score * feature_refined_like_score
                    )
                    row["voxel_balance_score"] = float(
                        voxel_cross_like_score * voxel_refined_like_score
                    )

                    # -----------------------------
                    # Per-feature positions
                    # -----------------------------
                    feature_positions = []
                    feature_endpoint_positions = []

                    for k in feature_keys_for_endpoint:
                        (
                            raw_pos,
                            endpoint_pos,
                            cross_like,
                            refined_like,
                        ) = _scalar_feature_position(
                            alpha_feat[k],
                            cross_feat[k],
                            refined_feat[k],
                        )

                        row[f"{k}_position_raw_cross0_refined1"] = float(raw_pos)
                        row[f"{k}_endpoint_position"] = float(endpoint_pos)
                        row[f"{k}_cross_like_score"] = float(cross_like)
                        row[f"{k}_refined_like_score"] = float(refined_like)

                        feature_positions.append(float(raw_pos))
                        feature_endpoint_positions.append(float(endpoint_pos))

                    row["mean_feature_position_raw_cross0_refined1"] = float(np.mean(feature_positions))
                    row["mean_feature_endpoint_position"] = float(np.mean(feature_endpoint_positions))

                    rows.append(row)

            # Save often because this experiment is expensive.
            if save_csv_every is not None and save_csv_every > 0:
                pd.DataFrame(rows).to_csv(metrics_csv, index=False)
                pd.DataFrame(selected_rows).drop_duplicates(
                    subset=["split", "global_idx"],
                    keep="last",
                ).to_csv(selected_csv, index=False)

            del alpha_outputs_model, alpha_outputs_hwd, alpha_feature_cache, alpha_rows_cache
            if refined_pack.device.type == "cuda" and empty_cache_each_batch:
                torch.cuda.empty_cache()
            del x0_device, x0_real_model, noise_unrefined, noise_refined, delta_noise
            if refined_pack.device.type == "cuda" and empty_cache_each_batch:
                torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    selected_df = pd.DataFrame(selected_rows).drop_duplicates(
        subset=["split", "global_idx"],
        keep="last",
    )

    df.to_csv(metrics_csv, index=False)
    selected_df.to_csv(selected_csv, index=False)

    summary_cols = [
        # real-fidelity metrics
        "mae_to_real",
        "mse_to_real",
        "psnr_to_real",
        "fg_std_abs_gap",
        "edge_mean_abs_gap",
        "lap_abs_mean_abs_gap",
        "fft_low_frac_abs_gap",
        "fft_high_to_low_abs_gap",

        # endpoint metrics: where alpha sits between cross and refined
        "voxel_d_to_cross",
        "voxel_d_to_refined",
        "voxel_endpoint_position",
        "voxel_cross_like_score",
        "voxel_refined_like_score",
        "voxel_balance_score",

        "feature_d_to_cross",
        "feature_d_to_refined",
        "feature_endpoint_position",
        "feature_cross_like_score",
        "feature_refined_like_score",
        "feature_balance_score",

        # interpretable per-feature positions
        "fg_std_endpoint_position",
        "edge_mean_endpoint_position",
        "lap_abs_mean_endpoint_position",
        "fft_low_frac_endpoint_position",
        "fft_high_to_low_endpoint_position",
        "mean_feature_endpoint_position",
    ]

    summary = (
        df.groupby(["split", "age_group", "alpha"], observed=True)[summary_cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    summary.columns = [
        "_".join(str(x) for x in col if str(x) != "")
        for col in summary.columns.to_flat_index()
    ]

    summary.to_csv(summary_csv, index=False)

    print("\n[alpha-exp] Done.")
    print(f"Saved metrics CSV         : {metrics_csv}")
    print(f"Saved selected samples CSV: {selected_csv}")
    print(f"Saved summary CSV         : {summary_csv}")

    return df, selected_df, summary

# =========================================================
# Main export function
# =========================================================
@torch.no_grad()
def reconstruct_and_export_all_splits_two_models(
    *,
    refined_pack: MRIObjectivePack,
    unrefined_pack: MRIObjectivePack,
    splits: Mapping[str, torch.utils.data.Dataset],
    export_root: str | Path = "/rhome/ssafa013/bigdata/simple_unet_exports",
    generate_sets: Sequence[str] = ("refined", "unrefined", "cross"),
    batch_size: int = 2,
    steps: int = 1000,
    seed: int = 0,
    num_workers: int = 1,
    clip_inverse: bool = False,
    clip_decode: bool = True,
    overwrite: bool = False,
    save_csv_every: int = 10,
    empty_cache_each_batch: bool = False,
    show_pbar: bool = False,
    resume_from_outputs: bool = False,
    resume_overlap: int = 2,
) -> pd.DataFrame:
    """
    Reconstruct all samples from all splits with refined/unrefined packs.

    The same global_idx is used in all folders, so one patient/sample can be
    matched across refined/unrefined/cross outputs later.
    """
    generate_sets = _validate_generate_sets(generate_sets)
    export_root = Path(export_root)
    export_root.mkdir(parents=True, exist_ok=True)

    set_to_dir = {}

    for set_name in generate_sets:
        set_to_dir[set_name] = export_root / set_name
        
    for set_name in generate_sets:
        set_to_dir[set_name].mkdir(parents=True, exist_ok=True)


    csv_path = export_root / "reconstruction_index.csv"

    resume_position = 0
    if resume_from_outputs:
        resume_position = _infer_resume_position_by_counts(
            set_to_dir=set_to_dir,
            generate_sets=generate_sets,
            resume_overlap=resume_overlap,
        )

    records = []
    if resume_from_outputs and csv_path.exists():
        old_df = pd.read_csv(csv_path)
        records = old_df.to_dict("records")
        print(f"[resume] loaded {len(records)} old CSV rows from {csv_path}")

    seen_global_idxs = set()
    total_expected = sum(len(split_ds) for split_ds in splits.values())
    total_done = 0
    current_position = 0
    
    for split_name, split_ds in splits.items():
        print(f"\n==== reconstructing split: {split_name} ====")

        ds = WithGlobalIndex(split_ds)
        dl = get_eval_loader(
            "standard",
            ds,
            batch_size=batch_size,
            drop_last=False,
            num_workers=num_workers,
        )

        split_done = 0

        for batch_i, batch in enumerate(dl):
            x0_bchwd = batch[0].float()   # dataset layout: (B, C, H, W, D)
            idxs_batch = batch[3]

            if torch.is_tensor(idxs_batch):
                idxs_batch = idxs_batch.long().cpu().tolist()
            else:
                idxs_batch = [int(x) for x in idxs_batch]

            # Resume mode:
            # Skip samples strictly before resume_start_global_idx.
            # We still keep one/two previous samples by subtracting resume_overlap
            # when computing resume_start_global_idx.
            if resume_from_outputs:
                batch_start = current_position
                batch_end = current_position + len(idxs_batch)  # exclusive

                if batch_end <= resume_position:
                    current_position += len(idxs_batch)
                    split_done += len(idxs_batch)
                    total_done += len(idxs_batch)
                    continue
                
                
            dupes = [g for g in idxs_batch if int(g) in seen_global_idxs]
            if dupes:
                raise RuntimeError(f"Duplicate global indices detected: {dupes}")

            out_paths = {
                set_name: [set_to_dir[set_name] / f"recon_{int(gidx):07d}.nii.gz" for gidx in idxs_batch]
                for set_name in generate_sets
            }

            all_requested_exist = all(
                all(path.exists() for path in out_paths[set_name])
                for set_name in generate_sets
            )

            # If all requested files already exist and overwrite=False, skip the
            # expensive DDIM calls but still preserve paths in the CSV. Metrics
            # are NaN because they were not recomputed in this run.
            if (not overwrite) and all_requested_exist:
                print(f"[{split_name}] batch {batch_i + 1}: all requested files exist, skipping batch")
                for bi, gidx in enumerate(idxs_batch):
                    record = _empty_record(int(gidx), split_name)
                    _fill_paths(record, out_paths, bi)
                    records.append(record)
                    seen_global_idxs.add(int(gidx))
                    split_done += 1
                    total_done += 1

                current_position += len(idxs_batch)
                continue

            # Pack methods expect dataset layout for inversion. They internally
            # call to_model_layout from simple_diffusion_test.py.
            x0_bchwd_device = x0_bchwd.to(refined_pack.device, non_blocking=True)

            outputs_model = reconstruct_requested_sets_with_packs(
                x0_bchwd=x0_bchwd_device,
                refined_pack=refined_pack,
                unrefined_pack=unrefined_pack,
                generate_sets=generate_sets,
                steps=steps,
                seed=seed,
                clip_inverse=clip_inverse,
                clip_decode=clip_decode,
                show_pbar=show_pbar,
            )

            x0_real_np = from_model_layout(to_model_layout(x0_bchwd)).detach().cpu().numpy()  # (B, C, H, W, D)
            outputs_np = {
                set_name: from_model_layout(x_hat).detach().cpu().numpy()
                for set_name, x_hat in outputs_model.items()
            }

            metrics = {}
            for set_name, x_hat_np in outputs_np.items():
                # diff = x_hat_np - x0_real_np
                mae_vec, mse_vec = _compute_per_sample_metrics(
                    x_hat_np,
                    x0_real_np,
                    set_name=set_name,
                )
                metrics[set_name] = {
                    "mae": mae_vec,
                    "mse": mse_vec,
                }

            for bi, gidx in enumerate(idxs_batch):
                gidx = int(gidx)
                record = _empty_record(gidx, split_name)
                _fill_paths(record, out_paths, bi)

                for set_name in generate_sets:
                    out_path = out_paths[set_name][bi]
                    vol_hwd = outputs_np[set_name][bi, 0]  # channel 0: (H, W, D)
                    if overwrite or (not out_path.exists()):
                        save_single_nifti(vol_hwd, out_path, affine=np.eye(4, dtype=np.float32))

                    _set_metric(
                        record,
                        set_name,
                        mae=metrics[set_name]["mae"][bi],
                        mse=metrics[set_name]["mse"][bi],
                    )

                records.append(record)
                seen_global_idxs.add(gidx)
                split_done += 1
                total_done += 1

            print(
                f"[{split_name}] batch {batch_i + 1}: "
                f"saved {split_done}/{len(ds)} | total {total_done}/{total_expected}"
            )

            if save_csv_every is not None and save_csv_every > 0 and ((batch_i + 1) % save_csv_every == 0):
                _write_index_csv(records, csv_path)

            del outputs_model, outputs_np, x0_bchwd_device, x0_bchwd
            if refined_pack.device.type == "cuda" and empty_cache_each_batch:
                torch.cuda.empty_cache()
                
            current_position += len(idxs_batch)

    df = _write_index_csv(records, csv_path)

    print("\nDone.")
    for set_name in generate_sets:
        print(f"Saved {set_name:15s} reconstructions to: {set_to_dir[set_name]}")
    print(f"Saved index CSV to              : {csv_path}")
    print(f"Number of unique saved samples  : {len(df)}")
    return df


# =========================================================
# Usage example
# =========================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = OpenBHBDataset()
    train_dataset = dataset.get_subset("train")
    val_dataset = dataset.get_subset("val")
    test_dataset = dataset.get_subset("test")

    splits = {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }
    all_splits = {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }
    
    target_split = os.environ.get("EXPORT_SPLIT", "all").strip().lower()

    if target_split == "all":
        splits = all_splits
    elif target_split in all_splits:
        splits = {
            target_split: all_splits[target_split],
        }
    else:
        raise ValueError(
            f"Invalid EXPORT_SPLIT={target_split!r}. "
            f"Expected one of: all, train, val, test"
        )

    print(f"Exporting splits: {list(splits.keys())}")

    # Needed only if you want to construct/load the refined discriminator.
    shape_loader = get_eval_loader(
        "standard",
        train_dataset,
        batch_size=1,
        drop_last=False,
        num_workers=1,
    )
    input_shape = infer_model_input_shape_from_loader(shape_loader)
    print(f"Inferred model input shape: {input_shape}")

    '''
    Set to True to run the full reconstruction export with both refined and unrefined packs. 
    This will generate reconstructions for all samples in all splits and save them as NIfTI files, along with a CSV index of paths and metrics.
    '''
    RUN_RECONSTRUCTION_EXPORT = True 
    
    '''
    Set to True to run the noise inversion analysis that compares the refined and unrefined inverse noises.
    This will compute noise metrics and save them to CSV files.
    '''
    RUN_NOISE_INVERSE_ANALYSIS = False
    
    '''
    Set to True to run the alpha interpolation experiment that generates reconstructions at different alpha values between the refined and unrefined inverse noises.
    (refined inverse -> refined image set, unrefined inverse -> cross image set, and in-between alphas give interpolated styles)
    This will save reconstructions and metrics for each alpha to CSV files and NIfTI files.
    '''
    RUN_ALPHA_NOISE_INTERPOLATION = False
    
    '''
    Set to True to reconstructed all the images based on the alpha value between refined and unrefined noise, and export them.
    (alpha = 0.0 gives cross-style reconstructions, alpha = 1.0 gives refined-style reconstructions, and alpha in between gives interpolated styles)
    This will create a folder for the alpha value under simple_unet_exports, and save all the reconstructions there.
    '''
    RUN_FULL_ALPHA_RECONSTRUCTION_EXPORT = False
    
    packs, refined_discriminator = load_pretrained_v_packs(
        refined_model_path="./med-ddpm/results/mri_tests/proposed_split/mri_v_dann_unet3d.pt",
        unrefined_model_path="./med-ddpm/results/mri_tests/proposed_split/mri_vpred_unet3d.pt",
        device=device,
        T=1000,
        in_ch=1,
        base=16,
        t_dim=128,
        objective="v",
        strict_unet=True,
        input_shape=input_shape,
        load_refined_discriminator=False,
        num_domains=15,
        discriminator_use_time_conditioning=False,
    )
    
    
    if RUN_FULL_ALPHA_RECONSTRUCTION_EXPORT:
        alpha_value = 0.5
        alpha_set_name = _alpha_set_name(alpha_value)

        df_alpha_saved = reconstruct_and_export_all_splits_two_models(
            refined_pack=packs["refined"],
            unrefined_pack=packs["unrefined"],
            splits=splits,
            export_root="/rhome/ssafa013/bigdata/simple_unet_exports",

            # Only export alpha reconstruction.
            # This will create:
            #   simple_unet_exports/cross-alpha-0.5/
            generate_sets=(alpha_set_name,),

            batch_size=2,
            steps=1000,
            seed=0,
            num_workers=1,
            clip_inverse=False,
            clip_decode=True,

            overwrite=False,
            save_csv_every=1,
            empty_cache_each_batch=True,
            show_pbar=True,

            resume_from_outputs=True,
            resume_overlap=2,
        )

        print(df_alpha_saved.head())
    
    if RUN_ALPHA_NOISE_INTERPOLATION:
        alpha_df, alpha_selected_df, alpha_summary_df = export_alpha_noise_interpolation_experiment(
            refined_pack=packs["refined"],
            unrefined_pack=packs["unrefined"],
            splits=splits,
            export_root="/rhome/ssafa013/bigdata/simple_unet_exports",

            # Start with only OOD val/test. You can add "train" later if needed.
            split_names=("val", "test"),

            # alpha=0 is cross, alpha=1 is refined.
            alphas=(0.0, 0.25, 0.5, 0.625, 0.75, 0.875, 1.0),

            # Small subset first to keep it fast.
            n_young_per_split=20,
            n_old_per_split=20,
            young_max_age=25.0,
            old_min_age=40.0,

            # Use 1 first because each sample runs:
            # 2 inversions + len(alphas) refined decodes.
            batch_size=1,

            steps=1000,
            seed=0,
            num_workers=1,
            clip_inverse=False,
            clip_decode=True,
            show_pbar=True,

            # Keep this True if you want to run brain-age/domain classifiers later.
            save_nifti=False,

            save_csv_every=1,
            empty_cache_each_batch=True,
        )

        print(alpha_summary_df.head())

    if RUN_RECONSTRUCTION_EXPORT:
        export_root = os.environ.get(
            "EXPORT_ROOT",
            "/rhome/ssafa013/bigdata/simple_unet_exports/proposed_split",
        )

        export_batch_size = int(os.environ.get("EXPORT_BATCH_SIZE", "2"))

        export_resume = os.environ.get(
            "EXPORT_RESUME",
            "1",
        ).strip().lower() in {"1", "true", "yes"}
        
        df_saved = reconstruct_and_export_all_splits_two_models(
            refined_pack=packs["refined"],
            unrefined_pack=packs["unrefined"],
            splits=splits,
            export_root=export_root,
            generate_sets=("refined", "unrefined", "cross"),
            # Examples:
            # generate_sets=("refined",)
            # generate_sets=("unrefined",)
            # generate_sets=("cross",)
            batch_size=export_batch_size,
            steps=1000,
            seed=0,
            num_workers=1,
            clip_inverse=False,
            clip_decode=True,
            overwrite=False,
            save_csv_every=1,
            empty_cache_each_batch=False,
            show_pbar=True,

            # Resume from generated files if the HPCC job stopped.
            resume_from_outputs=export_resume,
            resume_overlap=2,
        )

        print(df_saved.head())
        
    if RUN_NOISE_INVERSE_ANALYSIS:
        refined_noise_df, unrefined_noise_df, pairwise_noise_df = export_refined_unrefined_noise_metrics(
            refined_pack=packs["refined"],
            unrefined_pack=packs["unrefined"],
            splits=splits,
            export_root="/rhome/ssafa013/bigdata/simple_unet_exports",
            batch_size=2,
            steps=1000,
            seed=0,
            num_workers=1,
            clip_inverse=False,
            show_pbar=True,
            save_csv_every=1,
            empty_cache_each_batch=True,
            seen_age_num_std=2.0,
        )

        print(pairwise_noise_df.head())
    # Example: load one sample later by OpenBHB global index.
    # vol, path = load_reconstruction_by_global_idx(
    #     1234,
    #     "/rhome/ssafa013/bigdata/simple_unet_exports",
    #     set_name="cross",
    # )
    # print(vol.shape, path)
