#!/usr/bin/env python3
"""
vbm_metrics_benchmark.py

Voxelwise CAT12-VBM GLM benchmark for real / cyclegan_target12 / combat images.
(Same tests as the original refined/unrefined/cross version; only the source
loading is adapted to the CycleGAN/ComBat per-split-subfolder exports.)

Runs Prof. Paul Thompson-style tests:
  1. age-effect VBM: voxel ~ age + nuisance covariates
  2. site-effect VBM: voxel ~ site + nuisance covariates
  3. FDR voxel counts, cluster extent, peak F/min p, partial R²
  4. split-half reliability for age-effect maps

The script keeps the project conventions: OpenBHBDataset, GeneratedStore,
global_idx pairing. Harmonized sources are per-split-subfolder exports
(<root>/{train,ood_val,ood_test}/recon_{global_idx:07d}.nii.gz + a
reconstruction_index.csv) at DIFFERENT roots, given as name=root pairs.

Important:
  - real images are loaded from OpenBHBDataset tensors, not generated NIfTI files
  - generated sources are loaded from NIfTI files via helper.load_nii
  - site covariates use original_site when available, not remapped domain_site
  - disease is not used
  - GLMs are fit separately within each split because age distributions differ

Example:
  RUN_MODE=all RESULTS_ROOT=./vbm_metrics_results/harmonized \
  VBM_SOURCES="cyclegan_target12=~/bigdata/cyclegan_target12,combat=/rhome/ssafa013/bigdata/data/openBHB_v1.1_harmonized" \
  python vbm_metrics_benchmark.py

Environment variables:
  VBM_SOURCES="name=root,name=root"   # harmonized sources (real is implicit)
  RESULTS_ROOT=./vbm_metrics_results/harmonized
  VBM_SPLITS=id_full,ood_val,ood_test,ood_val_age_le35,ood_test_age_le35
  RUN_MODE=all                 # all | glm | split_half | plots
  VBM_BATCH_SIZE=8
  VBM_CHUNK_SIZE=8192
  VBM_MAX_AGE=35
  VBM_Q_THRESH=0.05
  VBM_TOPK_FRAC=0.05
  VBM_SPLIT_HALF_SEEDS=0,1,2,3,4
  FORCE_REBUILD_CACHE=0
"""

import os
import re
import sys
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

try:
    from scipy import stats, ndimage
except Exception as e:
    raise ImportError("This script requires scipy for F-tests/FDR cluster labeling.") from e

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DataLoader import OpenBHBDataset
from helper import load_nii

SEED = int(os.environ.get("SEED", "1337"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


set_seed(SEED)


class WithGlobalIndex(Dataset):
    def __init__(self, subset):
        self.subset = subset
        if hasattr(subset, "collate"):
            self.collate = subset.collate
        if hasattr(subset, "metadata_array"):
            self.metadata_array = subset.metadata_array
        if hasattr(subset, "y_array"):
            self.y_array = subset.y_array
        base = getattr(subset, "indices", None)
        if base is None:
            base = getattr(subset, "_indices", None)
        if base is not None and hasattr(base, "tolist"):
            base = base.tolist()
        self.global_indices = base if base is not None else list(range(len(subset)))

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, i):
        x, y, m = self.subset[i]
        return x, y, m, int(self.global_indices[i])


@dataclass
class GeneratedStore:
    sources: Dict[str, Dict[int, Path]]

    @property
    def available_sources(self) -> List[str]:
        preferred = ["unrefined", "refined", "cross", "cyclegan_target12", "combat"]
        return [s for s in preferred if s in self.sources] + [
            s for s in sorted(self.sources.keys()) if s not in preferred
        ]

    def has(self, global_idx: int, source: str) -> bool:
        return int(global_idx) in self.sources[source]

    def path(self, global_idx: int, source: str) -> Path:
        return self.sources[source][int(global_idx)]

    def available_ids(self, required_sources: Optional[List[str]] = None) -> set:
        req = required_sources or self.available_sources
        if len(req) == 0:
            return set()
        id_sets = [set(self.sources[s].keys()) for s in req]
        return set.intersection(*id_sets) if id_sets else set()

    def summary(self):
        print("GeneratedStore summary:")
        for src in self.available_sources:
            print(f"  - {src}: {len(self.sources[src])} files")

    @classmethod
    def from_multi_folder_export(cls, export_root: str, sources: Optional[List[str]] = None,
                                 prefix: str = "recon", require_nonempty: bool = True):
        export_root = Path(export_root)
        sources = sources or ["refined", "unrefined", "cross"]
        pat = re.compile(rf"^{re.escape(prefix)}_(\d+)\.nii\.gz$")
        source_maps = {}
        for src in sources:
            img_dir = export_root / src
            mapping = {}
            if not img_dir.exists():
                print(f"[store] WARNING: missing folder: {img_dir}")
                source_maps[src] = mapping
                continue
            for p in img_dir.glob(f"{prefix}_*.nii.gz"):
                m = pat.match(p.name)
                if p.is_file() and m:
                    mapping[int(m.group(1))] = p
            if require_nonempty and len(mapping) == 0:
                raise RuntimeError(f"No files found for source={src} in {img_dir}")
            print(f"[store] loaded {src}: {len(mapping)} files from {img_dir}")
            source_maps[src] = mapping
        return cls(source_maps)

    @classmethod
    def from_split_subdir_export(cls, export_root, source_name, prefix="recon",
                                 splits=("train", "ood_val", "ood_test", "id_test", "id_val")):
        """Load ONE source stored in per-split subfolders (train/, ood_val/, ood_test/)
        keyed by global_idx -- the CycleGAN / ComBat export layout. Prefers
        reconstruction_index.csv (absolute file_path); falls back to globbing
        <root>/<split>/recon_*.nii.gz (and a flat <root>/)."""
        export_root = Path(export_root)
        mapping = {}
        csv_path = export_root / "reconstruction_index.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                gidx = int(row["global_idx"])
                fp = Path(str(row["file_path"]))
                if not fp.is_absolute():
                    fp = export_root / fp
                if fp.exists():
                    mapping[gidx] = fp
        else:
            pat = re.compile(rf"^{re.escape(prefix)}_(\d+)\.nii\.gz$")
            for sp in list(splits) + [""]:
                d = export_root / sp if sp else export_root
                if not d.exists():
                    continue
                for p in d.glob(f"{prefix}_*.nii.gz"):
                    m = pat.match(p.name)
                    if m:
                        mapping.setdefault(int(m.group(1)), p)
        if not mapping:
            raise RuntimeError(f"No files for source={source_name} under {export_root}")
        print(f"[store] loaded {source_name}: {len(mapping)} files from {export_root}")
        return cls({source_name: mapping})

    @classmethod
    def from_source_specs(cls, specs, prefix="recon"):
        """specs: dict {source_name: export_root}. Merge several per-split-subdir
        harmonized sources (e.g. cyclegan_target12 and combat, at different roots)
        into one store. 'real' stays implicit (loaded from OpenBHBDataset tensors)."""
        sources = {}
        for name, root in specs.items():
            s = cls.from_split_subdir_export(root, name, prefix=prefix)
            sources[name] = s.sources[name]
        return cls(sources)


def _real_batch_to_hwd_numpy(x: torch.Tensor) -> np.ndarray:
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    x = x.detach().float().cpu()
    if x.ndim == 6:
        assert x.shape[2] == 1, f"Expected singleton axis 2, got {tuple(x.shape)}"
        x = x.squeeze(2)
    if x.ndim == 4:
        return x.contiguous().numpy()
    assert x.ndim == 5, f"Expected 5D real batch, got {tuple(x.shape)}"
    assert x.shape[1] == 1, f"Expected single channel, got {tuple(x.shape)}"
    return x[:, 0].contiguous().numpy()


def _load_generated_volume_hwd(fp: Path) -> np.ndarray:
    v = load_nii(fp)
    v = torch.from_numpy(v) if isinstance(v, np.ndarray) else torch.as_tensor(v)
    v = v.detach().float().cpu()
    if v.ndim == 4:
        assert v.shape[0] == 1, f"Expected (1,H,W,D), got {tuple(v.shape)}"
        v = v.squeeze(0)
    assert v.ndim == 3, f"Expected (H,W,D), got {tuple(v.shape)}"
    return v.contiguous().numpy()


def _generated_batch_to_hwd_numpy(store: GeneratedStore, global_ids: List[int], source: str) -> np.ndarray:
    return np.stack([_load_generated_volume_hwd(store.path(int(g), source)) for g in global_ids], axis=0)


def _source_batch_to_hwd_numpy(batch, store: GeneratedStore, source: str) -> Tuple[np.ndarray, List[int], np.ndarray]:
    image, _, metadata, global_ids = batch
    global_ids = global_ids.long().cpu().tolist() if torch.is_tensor(global_ids) else [int(x) for x in global_ids]
    metadata_np = metadata.detach().cpu().numpy() if torch.is_tensor(metadata) else np.asarray(metadata)
    vols = _real_batch_to_hwd_numpy(image) if source == "real" else _generated_batch_to_hwd_numpy(store, global_ids, source)
    return vols, global_ids, metadata_np


def build_eligible_local_pool(split_ds, store: GeneratedStore, required_sources: List[str]):
    wrapped = WithGlobalIndex(split_ds)
    available_global_ids = store.available_ids(required_sources)
    sites = split_ds.metadata_array[:, 0].to(torch.int64).cpu().numpy()
    eligible_by_site = defaultdict(list)
    eligible_all_local = []
    for local_idx, global_idx in enumerate(wrapped.global_indices):
        if int(global_idx) not in available_global_ids:
            continue
        site = int(sites[local_idx])
        eligible_by_site[site].append(local_idx)
        eligible_all_local.append(local_idx)
    return wrapped, {k: sorted(v) for k, v in sorted(eligible_by_site.items())}, sorted(eligible_all_local)


def split_indices_by_site(eligible_by_site: Dict[int, List[int]], frac_train: float = 0.9, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx_train, idx_holdout = [], []
    for _, ids in sorted(eligible_by_site.items()):
        ids = list(ids)
        rng.shuffle(ids)
        if len(ids) == 1:
            idx_train.extend(ids)
            continue
        cut = int(round(frac_train * len(ids)))
        cut = min(max(cut, 1), len(ids) - 1)
        idx_train.extend(ids[:cut])
        idx_holdout.extend(ids[cut:])
    return sorted(idx_train), sorted(idx_holdout)


def build_vbm_loaders(dataset, store: GeneratedStore, batch_size: int = 8, train_frac: float = 0.9,
                      num_workers: int = 1, pin_memory: bool = True):
    splits = {"train": dataset.get_subset("train"), "val": dataset.get_subset("val"), "test": dataset.get_subset("test")}
    required_sources = store.available_sources[:]
    print(f"Required generated sources: {required_sources}")
    loaders = {}

    train_wrapped, eligible_by_site_train, eligible_train_local = build_eligible_local_pool(
        splits["train"], store, required_sources
    )
    _, idx_holdout = split_indices_by_site(eligible_by_site_train, frac_train=train_frac, seed=42)
    loaders["id_holdout"] = DataLoader(Subset(train_wrapped, idx_holdout), batch_size=batch_size,
                                        shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=pin_memory)
    loaders["id_full"] = DataLoader(Subset(train_wrapped, eligible_train_local), batch_size=batch_size,
                                     shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=pin_memory)
    print(f"[train/id] eligible={len(eligible_train_local)}, id_holdout={len(idx_holdout)}, sites={len(eligible_by_site_train)}")

    for split_name in ["val", "test"]:
        wrapped, eligible_by_site, eligible_local = build_eligible_local_pool(splits[split_name], store, required_sources)
        print(f"[{split_name}] eligible={len(eligible_local)}, sites={len(eligible_by_site)}")
        if len(eligible_local) > 0:
            loaders[f"ood_{split_name}"] = DataLoader(Subset(wrapped, eligible_local), batch_size=batch_size,
                                                       shuffle=False, drop_last=False, num_workers=num_workers,
                                                       pin_memory=pin_memory)
    return loaders


def add_age_limited_ood_loaders(loaders, dataset, store, batch_size=8, num_workers=1, pin_memory=True,
                                max_age=35.0, min_age=0.0):
    splits = {"val": dataset.get_subset("val"), "test": dataset.get_subset("test")}
    required_sources = store.available_sources[:]
    tag = f"age_le{int(max_age)}"
    info = {"age_filter": f"{min_age} <= age <= {max_age}", "min_age": float(min_age), "max_age": float(max_age)}
    for split_name in ["val", "test"]:
        split_ds = splits[split_name]
        wrapped, _, eligible_local = build_eligible_local_pool(split_ds, store, required_sources)
        eligible_local = np.asarray(eligible_local, dtype=int)
        y = split_ds.y_array[eligible_local]
        y_np = y.detach().cpu().float().view(len(eligible_local), -1)[:, 0].numpy() if torch.is_tensor(y) else np.asarray(y).reshape(len(eligible_local), -1)[:, 0].astype(np.float32)
        keep = (y_np >= min_age) & (y_np <= max_age)
        idx_keep = eligible_local[keep].astype(int).tolist()
        ages_keep = y_np[keep]
        loader_name = f"ood_{split_name}_{tag}"
        if len(idx_keep) == 0:
            print(f"[age-limited OOD] {loader_name}: empty, skipping")
            continue
        site_ids = split_ds.metadata_array[idx_keep, 0].to(torch.int64).cpu().numpy()
        print(f"[age-limited OOD] {loader_name}: n={len(idx_keep)}, sites={len(np.unique(site_ids))}, "
              f"age_mean={np.mean(ages_keep):.3f}, age_std={np.std(ages_keep, ddof=1):.3f}, "
              f"age_min={np.min(ages_keep):.3f}, age_max={np.max(ages_keep):.3f}")
        loaders[loader_name] = DataLoader(Subset(wrapped, idx_keep), batch_size=batch_size, shuffle=False,
                                          drop_last=False, num_workers=num_workers, pin_memory=pin_memory)
        info[loader_name] = {"n": int(len(idx_keep)), "n_sites": int(len(np.unique(site_ids))),
                             "age_mean": float(np.mean(ages_keep)),
                             "age_std": float(np.std(ages_keep, ddof=1)) if len(ages_keep) > 1 else 0.0,
                             "age_min": float(np.min(ages_keep)), "age_max": float(np.max(ages_keep))}
    return loaders, info


def _prepare_age_targets(y) -> np.ndarray:
    y = torch.as_tensor(y).detach().cpu().float()
    if y.ndim == 0:
        y = y.unsqueeze(0)
    elif y.ndim > 1:
        y = y.view(y.shape[0], -1)[:, 0]
    return y.numpy().astype(np.float64)


def _find_metadata_dataframe(dataset) -> Optional[pd.DataFrame]:
    for attr in ["metadata", "metadata_df", "df", "participants_df"]:
        obj = getattr(dataset, attr, None)
        if isinstance(obj, pd.DataFrame):
            return obj
    return None


def _candidate_col(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    lookup = {str(c).lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lookup:
            return lookup[n.lower()]
    for c in df.columns:
        cl = str(c).lower()
        if any(n.lower() in cl for n in names):
            return c
    return None


def _encode_sex_values(vals: Sequence) -> np.ndarray:
    out = []
    for v in vals:
        if pd.isna(v):
            out.append(np.nan)
        elif isinstance(v, str):
            s = v.strip().lower()
            if s in {"m", "male", "man", "1"}:
                out.append(1.0)
            elif s in {"f", "female", "woman", "0"}:
                out.append(0.0)
            else:
                try: out.append(float(s))
                except Exception: out.append(np.nan)
        else:
            try: out.append(float(v))
            except Exception: out.append(np.nan)
    return np.asarray(out, dtype=np.float64)


def extract_sex_from_dataset(dataset, global_ids: List[int]) -> np.ndarray:
    df = _find_metadata_dataframe(dataset)
    if df is None:
        return np.full(len(global_ids), np.nan)
    sex_col = _candidate_col(df, ["sex", "gender", "participant_sex", "sex_at_birth"])
    if sex_col is None:
        return np.full(len(global_ids), np.nan)
    vals = []
    for gidx in global_ids:
        try: vals.append(df.iloc[int(gidx)][sex_col])
        except Exception: vals.append(np.nan)
    return _encode_sex_values(vals)


def original_site_from_metadata(metadata_np: np.ndarray) -> np.ndarray:
    if metadata_np.ndim == 1:
        metadata_np = metadata_np.reshape(1, -1)
    return metadata_np[:, 4].astype(np.float64) if metadata_np.shape[1] >= 5 else metadata_np[:, 0].astype(np.float64)


def domain_site_from_metadata(metadata_np: np.ndarray) -> np.ndarray:
    if metadata_np.ndim == 1:
        metadata_np = metadata_np.reshape(1, -1)
    return metadata_np[:, 0].astype(np.float64)


def collect_sample_table(dataloader, dataset, store: GeneratedStore, split_name: str, out_dir: Optional[Path] = None) -> pd.DataFrame:
    rows = []
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"collect_samples[{split_name}]", leave=False)):
        _, y, metadata, global_ids = batch
        global_ids = global_ids.long().cpu().tolist() if torch.is_tensor(global_ids) else [int(x) for x in global_ids]
        y_np = _prepare_age_targets(y)
        metadata_np = metadata.detach().cpu().numpy() if torch.is_tensor(metadata) else np.asarray(metadata)
        orig_site = original_site_from_metadata(metadata_np)
        domain_site = domain_site_from_metadata(metadata_np)
        sex = extract_sex_from_dataset(dataset, global_ids)
        for i, gidx in enumerate(global_ids):
            rows.append({"split": split_name, "row": len(rows), "batch_idx": int(batch_idx),
                         "global_idx": int(gidx), "age": float(y_np[i]),
                         "original_site": float(orig_site[i]), "domain_site": float(domain_site[i]),
                         "sex": float(sex[i]) if np.isfinite(sex[i]) else np.nan})
    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise RuntimeError(f"No samples found for split={split_name}")
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / f"sample_metadata_{split_name}.csv", index=False)
    return df


def save_mask_middle_slice(mask: np.ndarray, out_path: Path):
    z = mask.shape[-1] // 2
    plt.figure(figsize=(5, 5))
    plt.imshow(mask[:, :, z], cmap="gray")
    plt.title(f"Foreground mask, z={z}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def compute_or_load_foreground_mask(dataloader, store, split_name, out_dir, eps=1e-6, min_subject_frac=0.05,
                                    force_rebuild=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_path = out_dir / f"mask_{split_name}.npy"
    meta_path = out_dir / f"mask_{split_name}.json"
    if mask_path.exists() and meta_path.exists() and not force_rebuild:
        mask = np.load(mask_path).astype(bool)
        print(f"[mask] loaded {split_name}: {mask.shape}, voxels={int(mask.sum())}")
        return mask
    count, n, shape = None, 0, None
    for batch in tqdm(dataloader, desc=f"mask[{split_name}]", leave=False):
        vols, _, _ = _source_batch_to_hwd_numpy(batch, store, "real")
        if shape is None:
            shape = vols.shape[1:]
            count = np.zeros(shape, dtype=np.int32)
        if vols.shape[1:] != shape:
            raise ValueError(f"Shape changed in split={split_name}: {vols.shape[1:]} vs {shape}")
        count += (np.isfinite(vols) & (np.abs(vols) > eps)).sum(axis=0).astype(np.int32)
        n += vols.shape[0]
    min_count = max(1, int(math.ceil(min_subject_frac * n)))
    mask = count >= min_count
    np.save(mask_path, mask.astype(bool))
    with open(meta_path, "w") as f:
        json.dump({"split": split_name, "n_subjects": int(n), "shape": list(shape),
                   "eps": float(eps), "min_subject_frac": float(min_subject_frac),
                   "min_count": int(min_count), "n_mask_voxels": int(mask.sum())}, f, indent=2)
    save_mask_middle_slice(mask, out_dir / f"mask_{split_name}_middle_slice.png")
    print(f"[mask] saved {split_name}: shape={shape}, voxels={int(mask.sum())}")
    return mask.astype(bool)


def materialize_source_matrix(dataloader, store, source, split_name, sample_df, mask, out_dir, force_rebuild=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    n, v = int(len(sample_df)), int(mask.sum())
    dat_path = out_dir / f"matrix_{split_name}_{source}.dat"
    meta_path = out_dir / f"matrix_{split_name}_{source}.json"
    if dat_path.exists() and meta_path.exists() and not force_rebuild:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        if meta.get("shape") == [n, v]:
            print(f"[matrix] reuse {split_name}:{source} shape=({n},{v})")
            return dat_path, (n, v)
    mm = np.memmap(dat_path, dtype="float32", mode="w+", shape=(n, v))
    row_cursor = 0
    mask_flat = mask.reshape(-1)
    for batch in tqdm(dataloader, desc=f"materialize[{split_name}:{source}]", leave=False):
        vols, global_ids, _ = _source_batch_to_hwd_numpy(batch, store, source)
        b = vols.shape[0]
        expected = sample_df.iloc[row_cursor:row_cursor + b]["global_idx"].astype(int).tolist()
        if expected != [int(x) for x in global_ids]:
            raise RuntimeError(f"Global index order mismatch for {split_name}:{source}")
        flat = vols.reshape(b, -1)[:, mask_flat]
        mm[row_cursor:row_cursor + b, :] = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
        row_cursor += b
    if row_cursor != n:
        raise RuntimeError(f"Expected {n} rows but wrote {row_cursor} for {split_name}:{source}")
    mm.flush()
    with open(meta_path, "w") as f:
        json.dump({"split": split_name, "source": source, "shape": [n, v], "dtype": "float32",
                   "global_idx": sample_df["global_idx"].astype(int).tolist()}, f, indent=2)
    print(f"[matrix] saved {split_name}:{source} shape=({n},{v})")
    return dat_path, (n, v)


def _safe_z(x: np.ndarray):
    x = np.asarray(x, dtype=np.float64)
    mu = float(np.nanmean(x))
    sd = float(np.nanstd(x, ddof=1)) if np.sum(np.isfinite(x)) > 1 else 0.0
    if not np.isfinite(sd) or sd < 1e-12:
        sd = 1.0
    z = (x - mu) / sd
    z[~np.isfinite(z)] = 0.0
    return z, mu, sd


def build_design_matrices(sample_df: pd.DataFrame):
    n = len(sample_df)
    cols, arrs = ["intercept"], [np.ones(n, dtype=np.float64)]
    age_z, age_mu, age_sd = _safe_z(sample_df["age"].to_numpy(dtype=np.float64))
    age_col_index = len(cols)
    cols.append("age_z"); arrs.append(age_z)

    sex_used = False
    if "sex" in sample_df.columns:
        sex = sample_df["sex"].to_numpy(dtype=np.float64)
        finite = np.isfinite(sex)
        if finite.sum() >= 3 and len(np.unique(sex[finite])) >= 2:
            sex_filled = sex.copy(); sex_filled[~finite] = np.nanmean(sex[finite])
            sex_z, _, _ = _safe_z(sex_filled)
            cols.append("sex_z"); arrs.append(sex_z); sex_used = True

    site = sample_df["original_site"].to_numpy(dtype=np.float64)
    site_int = np.asarray([int(round(s)) for s in site], dtype=int)
    levels = sorted(np.unique(site_int).tolist())
    site_cols = []
    if len(levels) > 1:
        for lv in levels[1:]:
            site_cols.append(len(cols)); cols.append(f"site_{lv}"); arrs.append((site_int == lv).astype(np.float64))

    X_full = np.stack(arrs, axis=1).astype(np.float64)
    X_age_red = X_full[:, [i for i in range(X_full.shape[1]) if i != age_col_index]]
    site_set = set(site_cols)
    X_site_red = X_full[:, [i for i in range(X_full.shape[1]) if i not in site_set]] if len(site_cols) else None
    rank_full = int(np.linalg.matrix_rank(X_full))
    df2 = int(n - rank_full)
    return {"X_full": X_full, "X_age_red": X_age_red, "X_site_red": X_site_red,
            "columns_full": cols, "age_col_index": age_col_index, "site_col_indices": site_cols,
            "site_levels": levels, "age_mean": age_mu, "age_std": age_sd, "sex_used": bool(sex_used),
            "rank_full": rank_full, "df2": df2, "n": int(n)}


def _rss_from_design(X: np.ndarray, Y: np.ndarray):
    beta = np.linalg.pinv(X) @ Y
    resid = Y - X @ beta
    return beta, np.sum(resid * resid, axis=0)


def fdr_bh(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=np.float64)
    q = np.full_like(p, np.nan, dtype=np.float64)
    finite = np.isfinite(p)
    if finite.sum() == 0:
        return q
    pf = p[finite]
    m = len(pf)
    order = np.argsort(pf)
    ranked = pf[order]
    q_ranked = ranked * m / np.arange(1, m + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)
    qf = np.empty_like(pf); qf[order] = q_ranked
    q[finite] = qf
    return q


def _masked_vector_to_volume(vec: np.ndarray, mask: np.ndarray, fill=0.0) -> np.ndarray:
    vol = np.full(mask.size, fill, dtype=np.float32)
    vol[mask.reshape(-1)] = np.asarray(vec, dtype=np.float32)
    return vol.reshape(mask.shape)


def save_masked_vector_as_volume(vec, mask, out_path: Path):
    vol = np.full(mask.size, np.nan, dtype=np.float32)
    vol[mask.reshape(-1)] = np.asarray(vec, dtype=np.float32)
    np.save(out_path, vol.reshape(mask.shape))


def empty_cluster_summary(effect_name: str):
    return {"effect": effect_name, "n_clusters": 0, "total_fdr_voxels": 0, "largest_cluster_voxels": 0, "cluster_table": []}


def cluster_summary(qvals, stat_vals, mask, q_thresh=0.05, effect_name="age"):
    if qvals is None or np.all(~np.isfinite(qvals)):
        return empty_cluster_summary(effect_name)
    sig_vol = _masked_vector_to_volume((np.asarray(qvals) < q_thresh).astype(np.float32), mask, fill=0.0) > 0.5
    if sig_vol.sum() == 0:
        return empty_cluster_summary(effect_name)
    labels, n_clusters = ndimage.label(sig_vol, structure=np.ones((3, 3, 3), dtype=np.int8))
    sizes = np.asarray(ndimage.sum(sig_vol.astype(np.int32), labels, index=np.arange(1, n_clusters + 1)), dtype=np.int64)
    stat_vol = _masked_vector_to_volume(stat_vals, mask, fill=np.nan)
    table = []
    for cid in range(1, n_clusters + 1):
        m = labels == cid
        coords = np.argwhere(m)
        vals = stat_vol[m]
        if vals.size == 0:
            continue
        peak_i = int(np.nanargmax(vals)); peak_coord = coords[peak_i].tolist()
        table.append({"cluster_id": int(cid), "cluster_voxels": int(m.sum()),
                      "peak_stat": float(np.nanmax(vals)), "mean_stat": float(np.nanmean(vals)),
                      "centroid_h": float(coords[:, 0].mean()), "centroid_w": float(coords[:, 1].mean()),
                      "centroid_d": float(coords[:, 2].mean()), "peak_h": int(peak_coord[0]),
                      "peak_w": int(peak_coord[1]), "peak_d": int(peak_coord[2])})
    table = sorted(table, key=lambda r: r["cluster_voxels"], reverse=True)
    return {"effect": effect_name, "n_clusters": int(n_clusters), "total_fdr_voxels": int(sig_vol.sum()),
            "largest_cluster_voxels": int(sizes.max()) if len(sizes) else 0, "cluster_table": table}


def cluster_to_rows(cluster, split, source, effect):
    base = {"split": split, "source": source, "effect": effect,
            "n_clusters_total": int(cluster.get("n_clusters", 0)),
            "largest_cluster_voxels": int(cluster.get("largest_cluster_voxels", 0)),
            "total_fdr_voxels": int(cluster.get("total_fdr_voxels", 0))}
    if not cluster.get("cluster_table"):
        return [{**base, "cluster_id": np.nan, "cluster_voxels": 0, "peak_stat": np.nan, "mean_stat": np.nan,
                 "centroid_h": np.nan, "centroid_w": np.nan, "centroid_d": np.nan,
                 "peak_h": np.nan, "peak_w": np.nan, "peak_d": np.nan}]
    return [{**base, **r} for r in cluster["cluster_table"]]


def save_quicklook(age_f, age_q, site_f, site_q, mask, out_path, title, q_thresh):
    z = mask.shape[-1] // 2
    ncols = 4 if site_f is not None and site_q is not None else 2
    plt.figure(figsize=(4.2 * ncols, 4.2))
    plt.subplot(1, ncols, 1); plt.imshow(_masked_vector_to_volume(np.log10(age_f + 1.0), mask)[:, :, z], cmap="viridis"); plt.title("log10(age F+1)"); plt.axis("off")
    plt.subplot(1, ncols, 2); plt.imshow(_masked_vector_to_volume((age_q < q_thresh).astype(np.float32), mask)[:, :, z], cmap="gray"); plt.title(f"age q<{q_thresh}"); plt.axis("off")
    if ncols == 4:
        plt.subplot(1, ncols, 3); plt.imshow(_masked_vector_to_volume(np.log10(site_f + 1.0), mask)[:, :, z], cmap="viridis"); plt.title("log10(site F+1)"); plt.axis("off")
        plt.subplot(1, ncols, 4); plt.imshow(_masked_vector_to_volume((site_q < q_thresh).astype(np.float32), mask)[:, :, z], cmap="gray"); plt.title(f"site q<{q_thresh}"); plt.axis("off")
    plt.suptitle(title); plt.tight_layout(); plt.savefig(out_path, dpi=180, bbox_inches="tight"); plt.close()


def fit_vbm_glm_from_matrix(matrix_path, shape, sample_df, mask, split_name, source, out_dir, chunk_size=8192, q_thresh=0.05):
    maps_dir = out_dir / "maps"; maps_dir.mkdir(parents=True, exist_ok=True)
    n, v = shape
    design = build_design_matrices(sample_df)
    X_full, X_age_red, X_site_red = design["X_full"], design["X_age_red"], design["X_site_red"]
    rank_full, df2 = design["rank_full"], design["df2"]
    if df2 <= 0:
        raise RuntimeError(f"Not enough degrees of freedom for {split_name}:{source}: n={n}, rank={rank_full}")
    q_age = rank_full - int(np.linalg.matrix_rank(X_age_red))
    has_site = X_site_red is not None and len(design["site_col_indices"]) > 0
    q_site = rank_full - int(np.linalg.matrix_rank(X_site_red)) if has_site else 0
    has_site = has_site and q_site > 0
    Xmat = np.memmap(matrix_path, dtype="float32", mode="r", shape=(n, v))

    age_beta = np.empty(v, dtype=np.float32); age_t = np.empty(v, dtype=np.float32)
    age_f = np.empty(v, dtype=np.float32); age_p = np.empty(v, dtype=np.float64); age_pr2 = np.empty(v, dtype=np.float32)
    site_f = np.full(v, np.nan, dtype=np.float32); site_p = np.full(v, np.nan, dtype=np.float64); site_pr2 = np.full(v, np.nan, dtype=np.float32)
    age_col = design["age_col_index"]

    for start in tqdm(range(0, v, chunk_size), desc=f"glm[{split_name}:{source}]", leave=False):
        end = min(v, start + chunk_size)
        Y = np.asarray(Xmat[:, start:end], dtype=np.float64)
        beta_full, rss_full = _rss_from_design(X_full, Y)
        _, rss_age_red = _rss_from_design(X_age_red, Y)
        rss_full = np.maximum(rss_full, 1e-18); rss_age_red = np.maximum(rss_age_red, rss_full)
        F_age = np.maximum(((rss_age_red - rss_full) / float(q_age)) / (rss_full / float(df2)), 0.0)
        p_age = stats.f.sf(F_age, q_age, df2)
        beta_age = beta_full[age_col, :]
        age_beta[start:end] = beta_age.astype(np.float32)
        age_t[start:end] = (np.sign(beta_age) * np.sqrt(F_age)).astype(np.float32)
        age_f[start:end] = F_age.astype(np.float32)
        age_p[start:end] = p_age.astype(np.float64)
        age_pr2[start:end] = np.clip((rss_age_red - rss_full) / np.maximum(rss_age_red, 1e-18), 0.0, 1.0).astype(np.float32)
        if has_site:
            _, rss_site_red = _rss_from_design(X_site_red, Y)
            rss_site_red = np.maximum(rss_site_red, rss_full)
            F_site = np.maximum(((rss_site_red - rss_full) / float(q_site)) / (rss_full / float(df2)), 0.0)
            site_f[start:end] = F_site.astype(np.float32)
            site_p[start:end] = stats.f.sf(F_site, q_site, df2).astype(np.float64)
            site_pr2[start:end] = np.clip((rss_site_red - rss_full) / np.maximum(rss_site_red, 1e-18), 0.0, 1.0).astype(np.float32)

    age_q = fdr_bh(age_p).astype(np.float32)
    site_q = fdr_bh(site_p).astype(np.float32) if has_site else np.full(v, np.nan, dtype=np.float32)
    mask_flat_idx = np.flatnonzero(mask.reshape(-1)).astype(np.int64)
    maps_npz = maps_dir / f"maps_{split_name}_{source}.npz"
    np.savez_compressed(maps_npz, split=split_name, source=source,
                        mask_shape=np.asarray(mask.shape, dtype=np.int64), mask_flat_idx=mask_flat_idx,
                        age_beta=age_beta, age_t=age_t, age_f=age_f, age_p=age_p.astype(np.float32),
                        age_q=age_q, age_partial_r2=age_pr2,
                        site_f=site_f, site_p=site_p.astype(np.float32), site_q=site_q, site_partial_r2=site_pr2,
                        design_json=json.dumps({"columns_full": design["columns_full"], "site_levels": design["site_levels"],
                                                "age_mean": design["age_mean"], "age_std": design["age_std"],
                                                "sex_used": design["sex_used"], "rank_full": rank_full, "df2": df2,
                                                "q_age": int(q_age), "q_site": int(q_site), "has_site_test": bool(has_site)}))
    for name, vec in [("age_f", age_f), ("age_q", age_q), ("age_t", age_t), ("age_partial_r2", age_pr2),
                      ("site_f", site_f), ("site_q", site_q), ("site_partial_r2", site_pr2)]:
        if name.startswith("site") and not has_site:
            continue
        save_masked_vector_as_volume(vec, mask, maps_dir / f"{name}_{split_name}_{source}.npy")

    age_cl = cluster_summary(age_q, age_f, mask, q_thresh, "age")
    site_cl = cluster_summary(site_q, site_f, mask, q_thresh, "site") if has_site else empty_cluster_summary("site")
    summary = {"split": split_name, "source": source, "n": int(n), "n_mask_voxels": int(v),
               "n_original_sites": int(sample_df["original_site"].nunique()), "age_mean": float(sample_df["age"].mean()),
               "age_std": float(sample_df["age"].std(ddof=1)) if n > 1 else 0.0,
               "age_min": float(sample_df["age"].min()), "age_max": float(sample_df["age"].max()),
               "sex_used": bool(design["sex_used"]), "rank_full": int(rank_full), "df2": int(df2),
               "q_age": int(q_age), "q_site": int(q_site), "has_site_test": bool(has_site),
               "age_fdr_voxels": int(np.nansum(age_q < q_thresh)),
               "age_fdr_pct_mask": float(np.nanmean(age_q < q_thresh) * 100.0),
               "age_peak_f": float(np.nanmax(age_f)), "age_peak_abs_t": float(np.nanmax(np.abs(age_t))),
               "age_min_p": float(np.nanmin(age_p)), "age_mean_partial_r2": float(np.nanmean(age_pr2)),
               "age_median_partial_r2": float(np.nanmedian(age_pr2)),
               "age_mean_f_in_fdr": float(np.nanmean(age_f[age_q < q_thresh])) if np.any(age_q < q_thresh) else np.nan,
               "age_mean_pr2_in_fdr": float(np.nanmean(age_pr2[age_q < q_thresh])) if np.any(age_q < q_thresh) else np.nan,
               "site_fdr_voxels": int(np.nansum(site_q < q_thresh)) if has_site else 0,
               "site_fdr_pct_mask": float(np.nanmean(site_q < q_thresh) * 100.0) if has_site else np.nan,
               "site_peak_f": float(np.nanmax(site_f)) if has_site else np.nan,
               "site_min_p": float(np.nanmin(site_p)) if has_site else np.nan,
               "site_mean_partial_r2": float(np.nanmean(site_pr2)) if has_site else np.nan,
               "site_median_partial_r2": float(np.nanmedian(site_pr2)) if has_site else np.nan,
               "site_mean_f_in_fdr": float(np.nanmean(site_f[site_q < q_thresh])) if has_site and np.any(site_q < q_thresh) else np.nan,
               "site_mean_pr2_in_fdr": float(np.nanmean(site_pr2[site_q < q_thresh])) if has_site and np.any(site_q < q_thresh) else np.nan,
               "maps_npz": str(maps_npz)}
    clusters = cluster_to_rows(age_cl, split_name, source, "age") + cluster_to_rows(site_cl, split_name, source, "site")
    save_quicklook(age_f, age_q, site_f if has_site else None, site_q if has_site else None, mask,
                   maps_dir / f"quicklook_{split_name}_{source}.png", f"{split_name} | {source}", q_thresh)
    print(f"[glm] {split_name}:{source} | age_FDR={summary['age_fdr_voxels']} ({summary['age_fdr_pct_mask']:.2f}%) | site_FDR={summary['site_fdr_voxels']}")
    return {"summary": summary, "cluster_rows": clusters, "maps_npz": maps_npz}


def stratified_half_split(sample_df: pd.DataFrame, seed=0):
    rng = np.random.default_rng(seed)
    df = sample_df.copy(); n = len(df)
    try:
        df["age_bin_tmp"] = pd.qcut(df["age"], q=min(4, max(2, n // 20)), duplicates="drop")
    except Exception:
        df["age_bin_tmp"] = pd.cut(df["age"], bins=[0, 15, 20, 25, 30, 35, 50, 80, 120], include_lowest=True)
    idx1, idx2 = [], []
    for _, group in df.groupby(["original_site", "age_bin_tmp"], observed=True):
        ids = group.index.to_numpy(); rng.shuffle(ids); cut = len(ids) // 2
        idx1.extend(ids[:cut].tolist()); idx2.extend(ids[cut:].tolist())
    if len(idx1) == 0 or len(idx2) == 0 or abs(len(idx1) - len(idx2)) > 0.25 * n:
        ids = np.arange(n); rng.shuffle(ids); cut = n // 2
        idx1, idx2 = ids[:cut].tolist(), ids[cut:].tolist()
    return np.asarray(sorted(idx1), dtype=int), np.asarray(sorted(idx2), dtype=int)


def fit_age_map_for_rows(Xmat, row_idx, sample_df, chunk_size=8192):
    sub_df = sample_df.iloc[row_idx].reset_index(drop=True)
    design = build_design_matrices(sub_df)
    X_full, X_age_red = design["X_full"], design["X_age_red"]
    rank_full, df2 = design["rank_full"], design["df2"]
    q_age = rank_full - int(np.linalg.matrix_rank(X_age_red))
    if df2 <= 0 or q_age <= 0:
        raise RuntimeError(f"Invalid split-half design: n={len(sub_df)}, rank={rank_full}, df2={df2}, q_age={q_age}")
    _, v = Xmat.shape
    age_beta = np.empty(v, dtype=np.float32); age_t = np.empty(v, dtype=np.float32); age_f = np.empty(v, dtype=np.float32); age_p = np.empty(v, dtype=np.float64)
    for start in range(0, v, chunk_size):
        end = min(v, start + chunk_size)
        Y = np.asarray(Xmat[row_idx, start:end], dtype=np.float64)
        beta_full, rss_full = _rss_from_design(X_full, Y)
        _, rss_age_red = _rss_from_design(X_age_red, Y)
        rss_full = np.maximum(rss_full, 1e-18); rss_age_red = np.maximum(rss_age_red, rss_full)
        F_age = np.maximum(((rss_age_red - rss_full) / float(q_age)) / (rss_full / float(df2)), 0.0)
        beta_age = beta_full[design["age_col_index"], :]
        age_beta[start:end] = beta_age.astype(np.float32)
        age_t[start:end] = (np.sign(beta_age) * np.sqrt(F_age)).astype(np.float32)
        age_f[start:end] = F_age.astype(np.float32)
        age_p[start:end] = stats.f.sf(F_age, q_age, df2).astype(np.float64)
    return {"age_beta": age_beta, "age_t": age_t, "age_f": age_f, "age_p": age_p.astype(np.float32), "age_q": fdr_bh(age_p).astype(np.float32)}


def _corr_finite(a, b):
    a = np.asarray(a).reshape(-1); b = np.asarray(b).reshape(-1)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) < 1e-12 or np.std(b[m]) < 1e-12:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _dice(a, b):
    a = np.asarray(a).astype(bool); b = np.asarray(b).astype(bool)
    denom = int(a.sum() + b.sum())
    return np.nan if denom == 0 else float(2.0 * np.logical_and(a, b).sum() / denom)


def _topk_mask(values, frac=0.05):
    v = np.asarray(values); finite = np.isfinite(v); out = np.zeros_like(v, dtype=bool); n = finite.sum()
    if n == 0: return out
    k = max(1, int(round(frac * n))); finite_idx = np.flatnonzero(finite)
    order = finite_idx[np.argsort(np.abs(v[finite]))[::-1]]; out[order[:k]] = True
    return out


def run_split_half_reliability_for_source(matrix_path, shape, sample_df, split_name, source, seeds, chunk_size=8192,
                                          q_thresh=0.05, topk_frac=0.05):
    n, v = shape
    Xmat = np.memmap(matrix_path, dtype="float32", mode="r", shape=(n, v))
    rows = []
    for seed in seeds:
        idx1, idx2 = stratified_half_split(sample_df, seed=int(seed))
        try:
            m1 = fit_age_map_for_rows(Xmat, idx1, sample_df, chunk_size)
            m2 = fit_age_map_for_rows(Xmat, idx2, sample_df, chunk_size)
            sig1, sig2 = m1["age_q"] < q_thresh, m2["age_q"] < q_thresh
            row = {"split": split_name, "source": source, "seed": int(seed),
                   "n_half1": int(len(idx1)), "n_half2": int(len(idx2)),
                   "age_beta_corr": _corr_finite(m1["age_beta"], m2["age_beta"]),
                   "age_t_corr": _corr_finite(m1["age_t"], m2["age_t"]),
                   "age_f_corr": _corr_finite(m1["age_f"], m2["age_f"]),
                   "fdr_dice": _dice(sig1, sig2),
                   "topk_abs_t_dice": _dice(_topk_mask(m1["age_t"], topk_frac), _topk_mask(m2["age_t"], topk_frac)),
                   "half1_fdr_voxels": int(np.nansum(sig1)), "half2_fdr_voxels": int(np.nansum(sig2)),
                   "topk_frac": float(topk_frac), "error": ""}
        except Exception as e:
            print(f"[split-half] WARNING failed {split_name}:{source}:seed={seed}: {e}")
            row = {"split": split_name, "source": source, "seed": int(seed),
                   "n_half1": int(len(idx1)), "n_half2": int(len(idx2)),
                   "age_beta_corr": np.nan, "age_t_corr": np.nan, "age_f_corr": np.nan,
                   "fdr_dice": np.nan, "topk_abs_t_dice": np.nan,
                   "half1_fdr_voxels": np.nan, "half2_fdr_voxels": np.nan,
                   "topk_frac": float(topk_frac), "error": str(e)}
        rows.append(row)
        print(f"[split-half] {split_name}:{source}:seed={seed} | t_corr={row['age_t_corr']} | topk_dice={row['topk_abs_t_dice']}")
    return rows


def source_sort_key(src):
    return {"real": 0, "unrefined": 1, "refined": 2, "cross": 3, "cyclegan_target12": 4, "combat": 5}.get(src, 100)


def split_sort_key(split):
    return {"id_holdout": 0, "id_full": 1, "ood_val": 2, "ood_test": 3,
            "ood_val_age_le35": 4, "ood_test_age_le35": 5}.get(split, 100)


def plot_grouped_bars(df, value_col, ylabel, title, out_path):
    if len(df) == 0: return
    splits = sorted(df["split"].unique().tolist(), key=split_sort_key)
    sources = sorted(df["source"].unique().tolist(), key=source_sort_key)
    x = np.arange(len(splits)); width = 0.82 / max(1, len(sources))
    plt.figure(figsize=(max(9, 1.6 * len(splits)), 5.4)); ax = plt.gca()
    for j, src in enumerate(sources):
        vals = [float(df[(df["split"] == s) & (df["source"] == src)].iloc[0][value_col]) if len(df[(df["split"] == s) & (df["source"] == src)]) else np.nan for s in splits]
        off = (j - (len(sources) - 1) / 2) * width
        bars = ax.bar(x + off, vals, width, label=src)
        for bar, val in zip(bars, vals):
            if np.isfinite(val):
                ax.text(bar.get_x() + bar.get_width()/2, val, f"{val:.1f}" if abs(val) >= 10 else f"{val:.2f}", ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(splits, rotation=20, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(axis="y", alpha=0.25); ax.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=260, bbox_inches="tight"); plt.close(); print(f"[plot] saved {out_path}")


def plot_tradeoff(df, out_path):
    if len(df) == 0: return
    plt.figure(figsize=(8.5, 6.5)); ax = plt.gca()
    for split in sorted(df["split"].unique().tolist(), key=split_sort_key):
        sub = df[df["split"] == split]
        ax.scatter(sub["site_fdr_pct_mask"], sub["age_fdr_pct_mask"], s=70, label=split)
        for _, r in sub.iterrows():
            ax.text(r["site_fdr_pct_mask"], r["age_fdr_pct_mask"], f" {r['source']}", fontsize=8, va="center")
    ax.set_xlabel("Site-effect FDR voxels (% of mask); lower is better")
    ax.set_ylabel("Age-effect FDR voxels (% of mask); higher can indicate more sensitivity")
    ax.set_title("VBM tradeoff: reduce site effect while preserving/enhancing age effect")
    ax.grid(alpha=0.25); ax.legend(); plt.tight_layout(); plt.savefig(out_path, dpi=260, bbox_inches="tight"); plt.close(); print(f"[plot] saved {out_path}")


def plot_split_half(sh, out_path):
    if len(sh) == 0: return
    summary = sh.groupby(["split", "source"], as_index=False).agg(age_t_corr_mean=("age_t_corr", "mean"), age_t_corr_std=("age_t_corr", "std"))
    splits = sorted(summary["split"].unique().tolist(), key=split_sort_key)
    sources = sorted(summary["source"].unique().tolist(), key=source_sort_key)
    x = np.arange(len(splits)); width = 0.82 / max(1, len(sources))
    plt.figure(figsize=(max(9, 1.6 * len(splits)), 5.4)); ax = plt.gca()
    for j, src in enumerate(sources):
        vals, errs = [], []
        for s in splits:
            row = summary[(summary["split"] == s) & (summary["source"] == src)]
            vals.append(float(row.iloc[0]["age_t_corr_mean"]) if len(row) else np.nan)
            errs.append(float(row.iloc[0]["age_t_corr_std"]) if len(row) and np.isfinite(row.iloc[0]["age_t_corr_std"]) else 0.0)
        off = (j - (len(sources)-1)/2) * width
        ax.bar(x + off, vals, width, yerr=errs, capsize=3, label=src)
    ax.set_xticks(x); ax.set_xticklabels(splits, rotation=20, ha="right")
    ax.set_ylabel("Split-half age t-map correlation"); ax.set_title("Reliability of voxelwise age-effect maps")
    ax.grid(axis="y", alpha=0.25); ax.legend(); plt.tight_layout(); plt.savefig(out_path, dpi=260, bbox_inches="tight"); plt.close(); print(f"[plot] saved {out_path}")


def make_summary_plots(results_root: Path):
    plots_dir = results_root / "plots"; plots_dir.mkdir(parents=True, exist_ok=True)
    sp = results_root / "summaries" / "vbm_glm_summary.csv"
    hp = results_root / "summaries" / "vbm_split_half_reliability.csv"
    if sp.exists():
        df = pd.read_csv(sp)
        plot_grouped_bars(df, "age_fdr_pct_mask", "Age-effect FDR voxels (% of mask)", "VBM age-effect sensitivity by source", plots_dir / "age_fdr_voxels_by_source.png")
        plot_grouped_bars(df, "site_fdr_pct_mask", "Site-effect FDR voxels (% of mask)", "Residual VBM site effect by source", plots_dir / "site_fdr_voxels_by_source.png")
        plot_grouped_bars(df, "age_mean_partial_r2", "Mean partial R² for age", "Mean voxelwise age partial R² by source", plots_dir / "age_mean_partial_r2_by_source.png")
        plot_grouped_bars(df, "site_mean_partial_r2", "Mean partial R² for site", "Mean voxelwise site partial R² by source", plots_dir / "site_mean_partial_r2_by_source.png")
        plot_tradeoff(df, plots_dir / "age_vs_site_fdr_tradeoff.png")
    if hp.exists():
        plot_split_half(pd.read_csv(hp), plots_dir / "split_half_age_t_reliability.png")


def parse_csv_env(name, default):
    return [x.strip() for x in os.environ.get(name, default).split(",") if x.strip()]


def parse_source_specs(spec_str):
    """Parse "name=root,name=root" -> ordered dict {source_name: export_root}."""
    specs = {}
    for part in spec_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Bad source spec '{part}'; expected name=export_root")
        name, root = part.split("=", 1)
        specs[name.strip()] = os.path.expanduser(root.strip())
    if not specs:
        raise ValueError("No sources parsed from VBM_SOURCES.")
    return specs


def parse_int_csv_env(name, default):
    return [int(x.strip()) for x in os.environ.get(name, default).split(",") if x.strip()]


def run_glm_benchmark(loaders, dataset, store, results_root, split_names, sources, chunk_size, q_thresh,
                      mask_min_subject_frac, force_rebuild):
    masks_dir, matrices_dir, summaries_dir = results_root / "masks", results_root / "matrices", results_root / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    all_sample_dfs, summary_rows, cluster_rows = [], [], []
    for split_name in split_names:
        if split_name not in loaders:
            print(f"[glm] skipping missing split: {split_name}"); continue
        print("\n" + "="*80 + f"\n[glm] split={split_name}\n" + "="*80)
        sample_df = collect_sample_table(loaders[split_name], dataset, store, split_name, summaries_dir)
        all_sample_dfs.append(sample_df)
        mask = compute_or_load_foreground_mask(loaders[split_name], store, split_name, masks_dir,
                                               min_subject_frac=mask_min_subject_frac, force_rebuild=force_rebuild)
        for source in sources:
            if source != "real" and source not in store.sources:
                print(f"[glm] skipping missing source={source}"); continue
            matrix_path, shape = materialize_source_matrix(loaders[split_name], store, source, split_name,
                                                           sample_df, mask, matrices_dir, force_rebuild)
            res = fit_vbm_glm_from_matrix(matrix_path, shape, sample_df, mask, split_name, source,
                                          results_root, chunk_size, q_thresh)
            summary_rows.append(res["summary"]); cluster_rows.extend(res["cluster_rows"])
            pd.DataFrame(summary_rows).to_csv(summaries_dir / "vbm_glm_summary.csv", index=False)
            pd.DataFrame(cluster_rows).to_csv(summaries_dir / "vbm_cluster_summary.csv", index=False)
    if all_sample_dfs:
        pd.concat(all_sample_dfs, ignore_index=True).to_csv(summaries_dir / "sample_metadata_all_splits.csv", index=False)
    print(f"[glm] saved summary to {summaries_dir / 'vbm_glm_summary.csv'}")


def run_split_half_benchmark(loaders, dataset, store, results_root, split_names, sources, chunk_size, q_thresh,
                             topk_frac, split_half_seeds, mask_min_subject_frac, force_rebuild):
    masks_dir, matrices_dir, summaries_dir = results_root / "masks", results_root / "matrices", results_root / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for split_name in split_names:
        if split_name not in loaders:
            print(f"[split-half] skipping missing split: {split_name}"); continue
        sample_df = collect_sample_table(loaders[split_name], dataset, store, split_name, summaries_dir)
        mask = compute_or_load_foreground_mask(loaders[split_name], store, split_name, masks_dir,
                                               min_subject_frac=mask_min_subject_frac, force_rebuild=force_rebuild)
        for source in sources:
            if source != "real" and source not in store.sources: continue
            matrix_path, shape = materialize_source_matrix(loaders[split_name], store, source, split_name,
                                                           sample_df, mask, matrices_dir, force_rebuild)
            rows.extend(run_split_half_reliability_for_source(matrix_path, shape, sample_df, split_name, source,
                                                              split_half_seeds, chunk_size, q_thresh, topk_frac))
            pd.DataFrame(rows).to_csv(summaries_dir / "vbm_split_half_reliability.csv", index=False)
    print(f"[split-half] saved to {summaries_dir / 'vbm_split_half_reliability.csv'}")


if __name__ == "__main__":
    RUN_MODE = os.environ.get("RUN_MODE", "all").strip().lower()
    if RUN_MODE not in {"all", "glm", "split_half", "plots"}:
        raise ValueError("RUN_MODE must be one of: all, glm, split_half, plots")

    # Harmonized sources live in per-split-subfolder exports at DIFFERENT roots, so
    # they are given as name=root pairs. 'real' is always implicit (from OpenBHBDataset).
    VBM_SOURCES = os.environ.get(
        "VBM_SOURCES",
        "cyclegan_target12=~/bigdata/cyclegan_target12,"
        "combat=/rhome/ssafa013/bigdata/data/openBHB_v1.1_harmonized")
    SOURCE_SPECS = parse_source_specs(VBM_SOURCES)
    RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", "./vbm_metrics_results/harmonized"))
    DATASET_SOURCES = list(SOURCE_SPECS.keys())
    ANALYSIS_SOURCES = ["real"] + DATASET_SOURCES
    VBM_SPLITS = parse_csv_env("VBM_SPLITS", "id_full,ood_val,ood_test,ood_val_age_le35,ood_test_age_le35")

    BATCH_SIZE = int(os.environ.get("VBM_BATCH_SIZE", "8"))
    NUM_WORKERS = int(os.environ.get("VBM_NUM_WORKERS", "1"))
    CHUNK_SIZE = int(os.environ.get("VBM_CHUNK_SIZE", "8192"))
    Q_THRESH = float(os.environ.get("VBM_Q_THRESH", "0.05"))
    TOPK_FRAC = float(os.environ.get("VBM_TOPK_FRAC", "0.05"))
    MAX_AGE = float(os.environ.get("VBM_MAX_AGE", "35"))
    MASK_MIN_SUBJECT_FRAC = float(os.environ.get("VBM_MASK_MIN_SUBJECT_FRAC", "0.05"))
    FORCE_REBUILD_CACHE = os.environ.get("FORCE_REBUILD_CACHE", "0").strip() == "1"
    SPLIT_HALF_SEEDS = parse_int_csv_env("VBM_SPLIT_HALF_SEEDS", "0,1,2,3,4")

    print("RUN_MODE:", RUN_MODE)
    print("VBM_SOURCES:", SOURCE_SPECS)
    print("RESULTS_ROOT:", RESULTS_ROOT)
    print("DATASET_SOURCES:", DATASET_SOURCES)
    print("ANALYSIS_SOURCES:", ANALYSIS_SOURCES)
    print("VBM_SPLITS:", VBM_SPLITS)
    print("BATCH_SIZE:", BATCH_SIZE)
    print("CHUNK_SIZE:", CHUNK_SIZE)
    print("Q_THRESH:", Q_THRESH)
    print("MASK_MIN_SUBJECT_FRAC:", MASK_MIN_SUBJECT_FRAC)
    print("FORCE_REBUILD_CACHE:", FORCE_REBUILD_CACHE)

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    if RUN_MODE != "plots":
        store = GeneratedStore.from_source_specs(SOURCE_SPECS, prefix="recon")
        store.summary()
        dataset = OpenBHBDataset()
        loaders = build_vbm_loaders(dataset, store, batch_size=BATCH_SIZE, train_frac=0.9,
                                    num_workers=NUM_WORKERS, pin_memory=True)
        loaders, age_info = add_age_limited_ood_loaders(loaders, dataset, store, batch_size=BATCH_SIZE,
                                                        num_workers=NUM_WORKERS, pin_memory=True,
                                                        min_age=0.0, max_age=MAX_AGE)
        pd.DataFrame([{"key": k, **v} if isinstance(v, dict) else {"key": k, "value": v}
                      for k, v in age_info.items()]).to_csv(RESULTS_ROOT / "age_limited_loader_info.csv", index=False)

    if RUN_MODE in {"all", "glm"}:
        run_glm_benchmark(loaders, dataset, store, RESULTS_ROOT, VBM_SPLITS, ANALYSIS_SOURCES,
                          CHUNK_SIZE, Q_THRESH, MASK_MIN_SUBJECT_FRAC, FORCE_REBUILD_CACHE)

    if RUN_MODE in {"all", "split_half"}:
        run_split_half_benchmark(loaders, dataset, store, RESULTS_ROOT, VBM_SPLITS, ANALYSIS_SOURCES,
                                 CHUNK_SIZE, Q_THRESH, TOPK_FRAC, SPLIT_HALF_SEEDS,
                                 MASK_MIN_SUBJECT_FRAC, FORCE_REBUILD_CACHE)

    make_summary_plots(RESULTS_ROOT)
    print("Done.")
