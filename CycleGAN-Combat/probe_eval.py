
import os
import re
import sys
import copy
import random
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sympy import false
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, Subset
from tqdm.auto import tqdm

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from DataLoader import OpenBHBDataset
from site_discriminator import DANN3D, BrainCancerFeaturizer, BrainCancerRegressor
from ukbb_brain_age_model import make_age_model as make_ukbb_age_model
from sklearn.metrics import r2_score
from helper import load_nii


# ============================================================
# Reproducibility
# ============================================================
SEED = 1337
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

set_seed(SEED)


# ============================================================
# Dataset wrapper with GLOBAL index
# ============================================================
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
        gidx = int(self.global_indices[i])
        return x, y, m, gidx


# ============================================================
# Generated data store
# Supports:
#   1) NEW export: one reconstructed set keyed by global_idx
#   2) LEGACY export: refined / unrefined keyed by base_name + idx
# ============================================================
@dataclass
class GeneratedStore:
    sources: Dict[str, Dict[int, Path]]   # source_name -> {global_idx: filepath}

    @property
    def available_sources(self) -> List[str]:
        preferred = ["unrefined", "refined", "cross", "cross-alpha-0.5"]
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
    def from_new_export(
        cls,
        export_root: str,
        alias: str = "unrefined",
        csv_name: str = "reconstruction_index.csv",
        img_subdir: str = "reconstructed_nii",
        prefix: str = "recon",
    ):
        """
        New single-set export from your new code.
        Expected layout:
            export_root/
              reconstruction_index.csv
              reconstructed_nii/
                recon_0000123.nii.gz
        """
        export_root = Path(export_root)
        mapping = {}

        csv_path = export_root / csv_name
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if "global_idx" not in df.columns or "file_path" not in df.columns:
                raise ValueError(f"{csv_path} must contain columns: global_idx, file_path")

            for _, row in df.iterrows():
                gidx = int(row["global_idx"])
                fp = Path(row["file_path"])
                if not fp.is_absolute():
                    fp = export_root / fp
                if fp.exists():
                    mapping[gidx] = fp
        else:
            img_dir = export_root / img_subdir
            pat = re.compile(rf"^{re.escape(prefix)}_(\d+)\.nii\.gz$")
            for p in img_dir.iterdir():
                if not p.is_file():
                    continue
                m = pat.match(p.name)
                if m:
                    gidx = int(m.group(1))
                    mapping[gidx] = p

        if len(mapping) == 0:
            raise RuntimeError(f"No reconstructed files found in new export root: {export_root}")

        return cls(sources={alias: mapping})


    @classmethod
    def from_legacy_export(
        cls,
        img_dir: str,
        base_name: str = "1_sample",
        use_refined: bool = True,
    ):
        """
        Old export naming:
            refined   : {base_name}_{idx}.nii.gz
            unrefined : {base_name}_{idx}_unrefined.nii.gz
        """
        img_dir = Path(img_dir)
        refined_map = {}
        unrefined_map = {}

        pat_ref = re.compile(rf"^{re.escape(base_name)}_(\d+)\.nii\.gz$")
        pat_unref = re.compile(rf"^{re.escape(base_name)}_(\d+)_unrefined\.nii\.gz$")

        for p in img_dir.iterdir():
            if not p.is_file():
                continue

            m = pat_unref.match(p.name)
            if m:
                gidx = int(m.group(1))
                unrefined_map[gidx] = p
                continue

            m = pat_ref.match(p.name)
            if m:
                gidx = int(m.group(1))
                refined_map[gidx] = p

        sources = {"unrefined": unrefined_map}
        if use_refined:
            sources["refined"] = refined_map

        if len(sources["unrefined"]) == 0 and (not use_refined or len(sources.get("refined", {})) == 0):
            raise RuntimeError(f"No generated files found in legacy folder: {img_dir}")

        return cls(sources=sources)

    @classmethod
    def from_multi_folder_export(
        cls,
        export_root: str,
        sources: Optional[List[str]] = None,
        prefix: str = "recon",
        require_nonempty: bool = True,
    ):
        """
        Load generated datasets directly from folders, without using reconstruction_index.csv.

        Expected layout:
            export_root/
              refined/
                recon_0000123.nii.gz
              unrefined/
                recon_0000123.nii.gz
              cross/
                recon_0000123.nii.gz

        This ignores:
            export_root/reconstruction_index.csv
            export_root/reconstructed_nii/
        """
        export_root = Path(export_root)

        if sources is None:
            sources = ["refined", "unrefined", "cross"]

        pat = re.compile(rf"^{re.escape(prefix)}_(\d+)\.nii\.gz$")
        source_maps = {}

        for src in sources:
            img_dir = export_root / src
            mapping = {}

            if not img_dir.exists():
                print(f"[store] WARNING: source folder does not exist: {img_dir}")
                source_maps[src] = mapping
                continue

            for p in img_dir.glob(f"{prefix}_*.nii.gz"):
                if not p.is_file():
                    continue

                m = pat.match(p.name)
                if m:
                    gidx = int(m.group(1))
                    mapping[gidx] = p

            if require_nonempty and len(mapping) == 0:
                raise RuntimeError(f"No files found for source={src} in {img_dir}")

            print(f"[store] loaded {src}: {len(mapping)} files from {img_dir}")
            source_maps[src] = mapping

        return cls(sources=source_maps)

# ============================================================
# Tensor layout helpers
# Real dataset:
#   usually [B,C,1,H,W,D] or [B,C,H,W,D]
# We convert everything to [B,1,D,H,W] for the discriminator.
# Generated NIfTI volumes are loaded as [H,W,D] or [1,H,W,D].
# ============================================================

def _real_batch_to_hwd_numpy(x: torch.Tensor) -> np.ndarray:
    """
    Real dataset batch -> numpy (B,H,W,D)

    Accepts:
      - (B,C,1,H,W,D)
      - (B,C,H,W,D)

    Assumes single-channel MRI.
    """
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)

    x = x.detach().float().cpu()

    if x.ndim == 6:
        assert x.shape[2] == 1, f"Expected singleton dim at axis 2, got {tuple(x.shape)}"
        x = x.squeeze(2)   # -> (B,C,H,W,D)

    assert x.ndim == 5, f"Expected 5D real batch, got {tuple(x.shape)}"
    assert x.shape[1] == 1, f"Expected single channel, got {tuple(x.shape)}"

    # (B,1,H,W,D) -> (B,H,W,D)
    return x[:, 0].contiguous().numpy()


def _generated_batch_to_hwd_numpy(
    store: GeneratedStore,
    global_ids: List[int],
    source: str,
) -> np.ndarray:
    """
    Generated batch -> numpy (B,H,W,D)

    Reads each generated volume by global index through GeneratedStore.
    Supports outputs loaded as:
      - (H,W,D)
      - (1,H,W,D)
    """
    vols = []
    for gidx in global_ids:
        fp = store.path(int(gidx), source)
        v = load_nii(fp)

        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v)
        else:
            v = torch.as_tensor(v)

        v = v.detach().float().cpu()

        if v.ndim == 4:
            assert v.shape[0] == 1, f"Expected generated volume shape (1,H,W,D), got {tuple(v.shape)}"
            v = v.squeeze(0)   # -> (H,W,D)

        assert v.ndim == 3, f"Expected generated volume shape (H,W,D), got {tuple(v.shape)}"
        vols.append(v)

    return torch.stack(vols, dim=0).numpy()   # (B,H,W,D)


def _source_batch_to_hwd_numpy(
    batch,
    store: GeneratedStore,
    source: str,
) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Return volume batch as numpy (B,H,W,D), plus global_ids and site_ids.

    source can be:
      - "real"
      - any generated source in store, e.g. refined/unrefined/cross
    """
    image, _, metadata, global_ids = batch

    if torch.is_tensor(global_ids):
        global_ids = global_ids.long().cpu().tolist()
    else:
        global_ids = [int(x) for x in global_ids]

    site_ids = metadata[:, 0].long().cpu().tolist()

    if source == "real":
        x_bhwd = _real_batch_to_hwd_numpy(image)
    else:
        x_bhwd = _generated_batch_to_hwd_numpy(store, global_ids, source)

    return x_bhwd, global_ids, site_ids

def _volume_basic_stats(vol: np.ndarray) -> Dict[str, float]:
    """
    vol: single 3D volume, shape (H,W,D)
    """
    vol = vol.astype(np.float32, copy=False)
    finite = np.isfinite(vol)

    if not finite.all():
        safe = vol[finite]
    else:
        safe = vol.reshape(-1)

    if safe.size == 0:
        return {
            "finite_pct": 0.0,
            "min": np.nan,
            "p01": np.nan,
            "p05": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "max": np.nan,
            "nonzero_frac": np.nan,
            "abs_mean": np.nan,
        }

    return {
        "finite_pct": float(finite.mean() * 100.0),
        "min": float(np.min(safe)),
        "p01": float(np.percentile(safe, 1)),
        "p05": float(np.percentile(safe, 5)),
        "mean": float(np.mean(safe)),
        "std": float(np.std(safe)),
        "p95": float(np.percentile(safe, 95)),
        "p99": float(np.percentile(safe, 99)),
        "max": float(np.max(safe)),
        "nonzero_frac": float(np.mean(np.abs(safe) > 1e-8)),
        "abs_mean": float(np.mean(np.abs(safe))),
    }


def _pairwise_basic_diff_stats(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """
    a,b: single 3D volumes, same shape.
    """
    if a.shape != b.shape:
        raise ValueError(f"Pairwise shape mismatch: {a.shape} vs {b.shape}")

    diff = a.astype(np.float32, copy=False) - b.astype(np.float32, copy=False)
    abs_diff = np.abs(diff)

    return {
        "mae": float(np.mean(abs_diff)),
        "mse": float(np.mean(diff ** 2)),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "diff_mean": float(np.mean(diff)),
        "diff_std": float(np.std(diff)),
        "diff_p01": float(np.percentile(diff, 1)),
        "diff_p99": float(np.percentile(diff, 99)),
        "absdiff_p95": float(np.percentile(abs_diff, 95)),
        "absdiff_p99": float(np.percentile(abs_diff, 99)),
        "max_absdiff": float(np.max(abs_diff)),
    }


def compute_source_intensity_sanity_for_loader(
    dataloader,
    store: GeneratedStore,
    split_name: str,
    sources: Optional[List[str]] = None,
    pairwise_comparisons: Optional[List[Tuple[str, str]]] = None,
    out_dir: Optional[str] = None,
    max_batches: Optional[int] = None,
):
    """
    Computes simple volume statistics before expensive model-based metrics.

    Saves:
      intensity_stats_{split}.csv
      intensity_summary_{split}.csv
      pairwise_diff_stats_{split}.csv
      pairwise_diff_summary_{split}.csv
    """
    if sources is None:
        sources = ["real"] + store.available_sources

    if pairwise_comparisons is None:
        pairwise_comparisons = [
            ("real", "refined"),
            ("real", "unrefined"),
            ("real", "cross"),
            ("refined", "unrefined"),
            ("refined", "cross"),
            ("unrefined", "cross"),
        ]

    rows = []
    pair_rows = []

    pbar = tqdm(dataloader, desc=f"intensity_sanity[{split_name}]", leave=False)

    for batch_idx, batch in enumerate(pbar):
        if max_batches is not None and batch_idx >= max_batches:
            break

        source_vols = {}
        source_global_ids = None
        source_site_ids = None

        for src in sources:
            vols_bhwd, global_ids, site_ids = _source_batch_to_hwd_numpy(batch, store, src)

            if source_global_ids is None:
                source_global_ids = global_ids
                source_site_ids = site_ids
            elif global_ids != source_global_ids:
                raise RuntimeError(f"global_ids mismatch for source={src}")

            source_vols[src] = vols_bhwd

            for i, gidx in enumerate(global_ids):
                stats = _volume_basic_stats(vols_bhwd[i])
                rows.append({
                    "split": split_name,
                    "batch_idx": int(batch_idx),
                    "global_idx": int(gidx),
                    "site": int(site_ids[i]),
                    "source": src,
                    "shape": str(tuple(vols_bhwd[i].shape)),
                    **stats,
                })

        for src_a, src_b in pairwise_comparisons:
            if src_a not in source_vols or src_b not in source_vols:
                continue

            A = source_vols[src_a]
            B = source_vols[src_b]

            if A.shape != B.shape:
                raise ValueError(
                    f"Shape mismatch for pair {src_a} vs {src_b}: "
                    f"{A.shape} vs {B.shape}"
                )

            for i, gidx in enumerate(source_global_ids):
                dstats = _pairwise_basic_diff_stats(A[i], B[i])
                pair_rows.append({
                    "split": split_name,
                    "batch_idx": int(batch_idx),
                    "global_idx": int(gidx),
                    "site": int(source_site_ids[i]),
                    "source_a": src_a,
                    "source_b": src_b,
                    "pair": f"{src_a}_vs_{src_b}",
                    **dstats,
                })

    stats_df = pd.DataFrame(rows)
    pair_df = pd.DataFrame(pair_rows)

    if len(stats_df) == 0:
        raise RuntimeError(f"No intensity sanity rows for split={split_name}")

    summary_df = (
        stats_df
        .groupby(["split", "source"])[
            ["finite_pct", "min", "p01", "p05", "mean", "std", "p95", "p99", "max", "nonzero_frac", "abs_mean"]
        ]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )

    if len(pair_df) > 0:
        pair_summary_df = (
            pair_df
            .groupby(["split", "pair"])[
                ["mae", "mse", "rmse", "diff_mean", "diff_std", "absdiff_p95", "absdiff_p99", "max_absdiff"]
            ]
            .agg(["mean", "std", "min", "max", "count"])
            .reset_index()
        )
    else:
        pair_summary_df = pd.DataFrame()

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        stats_df.to_csv(out_dir / f"intensity_stats_{split_name}.csv", index=False)
        summary_df.to_csv(out_dir / f"intensity_summary_{split_name}.csv", index=False)

        if len(pair_df) > 0:
            pair_df.to_csv(out_dir / f"pairwise_diff_stats_{split_name}.csv", index=False)
            pair_summary_df.to_csv(out_dir / f"pairwise_diff_summary_{split_name}.csv", index=False)

        print(f"[sanity] saved intensity stats for {split_name} to {out_dir}")

    return stats_df, summary_df, pair_df, pair_summary_df


def run_all_intensity_sanity_checks(
    loaders: Dict[str, DataLoader],
    store: GeneratedStore,
    out_dir: str,
    sources: Optional[List[str]] = None,
    pairwise_comparisons: Optional[List[Tuple[str, str]]] = None,
    split_names: Optional[List[str]] = None,
    max_batches: Optional[int] = None,
):
    """
    Runs intensity sanity checks before all expensive metrics.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if sources is None:
        sources = ["real"] + store.available_sources

    if split_names is None:
        split_names = ["id_full", "ood_val", "ood_test"]

    all_stats = []
    all_summaries = []
    all_pair_stats = []
    all_pair_summaries = []

    for split_name in split_names:
        if split_name not in loaders:
            print(f"[sanity] skipping missing loader: {split_name}")
            continue

        stats_df, summary_df, pair_df, pair_summary_df = compute_source_intensity_sanity_for_loader(
            dataloader=loaders[split_name],
            store=store,
            split_name=split_name,
            sources=sources,
            pairwise_comparisons=pairwise_comparisons,
            out_dir=str(out_dir),
            max_batches=max_batches,
        )

        all_stats.append(stats_df)
        all_summaries.append(summary_df)

        if len(pair_df) > 0:
            all_pair_stats.append(pair_df)
            all_pair_summaries.append(pair_summary_df)

    stats_all = pd.concat(all_stats, ignore_index=True) if all_stats else pd.DataFrame()
    summary_all = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
    pair_stats_all = pd.concat(all_pair_stats, ignore_index=True) if all_pair_stats else pd.DataFrame()
    pair_summary_all = pd.concat(all_pair_summaries, ignore_index=True) if all_pair_summaries else pd.DataFrame()

    stats_all.to_csv(out_dir / "intensity_stats_all_splits.csv", index=False)
    summary_all.to_csv(out_dir / "intensity_summary_all_splits.csv", index=False)

    if len(pair_stats_all) > 0:
        pair_stats_all.to_csv(out_dir / "pairwise_diff_stats_all_splits.csv", index=False)
        pair_summary_all.to_csv(out_dir / "pairwise_diff_summary_all_splits.csv", index=False)

    print(f"[sanity] saved combined sanity CSVs to {out_dir}")

    return {
        "stats": stats_all,
        "summary": summary_all,
        "pair_stats": pair_stats_all,
        "pair_summary": pair_summary_all,
    }



def plot_sanity_summary_figures(
    sanity_dir: str,
    out_dir: Optional[str] = None,
):
    """
    Creates 2 compact plots from intensity sanity CSVs:

    1) real-vs-generated MAE/RMSE by split/source
    2) source intensity summary by split/source

    Uses:
      pairwise_diff_stats_all_splits.csv
      intensity_stats_all_splits.csv
    """
    sanity_dir = Path(sanity_dir)
    if out_dir is None:
        out_dir = sanity_dir / "plots"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_csv = sanity_dir / "pairwise_diff_stats_all_splits.csv"
    stats_csv = sanity_dir / "intensity_stats_all_splits.csv"

    pair_df = pd.read_csv(pair_csv)
    stats_df = pd.read_csv(stats_csv)

    # --------------------------------------------------------
    # Plot 1: Real-vs-generated closeness
    # --------------------------------------------------------
    real_pairs = [
        "real_vs_unrefined",
        "real_vs_refined",
        "real_vs_cross",
        "real_vs_cross-alpha-0.5",
    ]

    df_real = pair_df[pair_df["pair"].isin(real_pairs)].copy()

    # cleaner order
    pair_order = {
        "real_vs_unrefined": 0,
        "real_vs_refined": 1,
        "real_vs_cross": 2,
        "real_vs_cross-alpha-0.5": 3,
    }
    
    split_order = {
        "id_holdout": 0,
        "id_full": 1,
        "ood_val": 2,
        "ood_test": 3,
        "ood_val_age_similar": 4,
        "ood_test_age_similar": 5,
        "ood_val_age_shift": 6,
        "ood_test_age_shift": 7,
    }

    df_real["pair_order"] = df_real["pair"].map(pair_order)
    df_real["split_order"] = df_real["split"].map(split_order)

    summary = (
        df_real
        .groupby(["split", "pair"], as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
        )
    )
    summary["pair_order"] = summary["pair"].map(pair_order)
    summary["split_order"] = summary["split"].map(split_order)
    summary = summary.sort_values(["split_order", "pair_order"])

    fig, ax = plt.subplots(figsize=(10, 4.5))

    labels = []
    x = np.arange(len(summary))
    vals = summary["mae_mean"].values
    errs = summary["mae_std"].fillna(0.0).values

    for _, row in summary.iterrows():
        labels.append(f"{row['split']}\n{row['pair'].replace('real_vs_', '')}")

    ax.bar(x, vals, yerr=errs, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("MAE to real image")
    ax.set_title("Generated image closeness to real MRI\n(lower is closer to real)")
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    out_path = out_dir / "sanity_real_vs_generated_mae.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    summary.to_csv(out_dir / "sanity_real_vs_generated_mae_summary.csv", index=False)

    # --------------------------------------------------------
    # Plot 2: Intensity distribution by source
    # --------------------------------------------------------
    source_order = {
        "real": 0,
        "unrefined": 1,
        "refined": 2,
        "cross": 3,
        "cross-alpha-0.5": 4,
    }

    intensity_summary = (
        stats_df
        .groupby(["split", "source"], as_index=False)
        .agg(
            mean_mean=("mean", "mean"),
            std_mean=("std", "mean"),
            p99_mean=("p99", "mean"),
            abs_mean_mean=("abs_mean", "mean"),
        )
    )
    intensity_summary["split_order"] = intensity_summary["split"].map(split_order)
    intensity_summary["source_order"] = intensity_summary["source"].map(source_order)
    intensity_summary = intensity_summary.sort_values(["split_order", "source_order"])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=False)

    metrics = [
        ("std_mean", "Intensity std"),
        ("p99_mean", "P99 intensity"),
        ("abs_mean_mean", "Mean |intensity|"),
    ]

    for ax, (metric, title) in zip(axes, metrics):
        labels = []
        vals = []

        for _, row in intensity_summary.iterrows():
            labels.append(f"{row['split']}\n{row['source']}")
            vals.append(row[metric])

        ax.bar(np.arange(len(vals)), vals)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Intensity sanity check across real/refined/unrefined/cross")
    plt.tight_layout()

    out_path = out_dir / "sanity_intensity_distribution_by_source.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    intensity_summary.to_csv(out_dir / "sanity_intensity_summary_for_plot.csv", index=False)

    return summary, intensity_summary


# ============================================================
# Image feature checks by age bin
# Goal:
#   Check whether age-related image features are compressed/shifted
#   in real/refined/unrefined/cross.
# ============================================================

def _safe_percentile(x, q):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(np.percentile(x, q))


def _brain_mask_proxy(vol: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Very simple foreground proxy.
    Your images are mostly zero/background, so this works as a first pass.

    vol: (H,W,D)
    """
    v = vol.astype(np.float32, copy=False)
    finite = np.isfinite(v)
    mask = finite & (np.abs(v) > eps)
    return mask


def _edge_features(vol: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Gradient/edge sharpness proxy.
    Higher gradient magnitude can indicate sharper tissue boundaries.
    """
    v = vol.astype(np.float32, copy=False)

    gx, gy, gz = np.gradient(v)
    grad_mag = np.sqrt(gx * gx + gy * gy + gz * gz)

    if mask is not None and mask.any():
        g = grad_mag[mask]
    else:
        g = grad_mag.reshape(-1)

    return {
        "edge_mean": float(np.mean(g)),
        "edge_std": float(np.std(g)),
        "edge_p95": _safe_percentile(g, 95),
        "edge_p99": _safe_percentile(g, 99),
    }


def _laplacian_sharpness(vol: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Simple 3D Laplacian-like sharpness proxy using finite differences.
    No scipy dependency.
    """
    v = vol.astype(np.float32, copy=False)

    lap = (
        -6.0 * v
        + np.roll(v, 1, axis=0) + np.roll(v, -1, axis=0)
        + np.roll(v, 1, axis=1) + np.roll(v, -1, axis=1)
        + np.roll(v, 1, axis=2) + np.roll(v, -1, axis=2)
    )

    if mask is not None and mask.any():
        l = lap[mask]
    else:
        l = lap.reshape(-1)

    abs_l = np.abs(l)
    return {
        "lap_abs_mean": float(np.mean(abs_l)),
        "lap_abs_std": float(np.std(abs_l)),
        "lap_abs_p95": _safe_percentile(abs_l, 95),
        "lap_var": float(np.var(l)),
    }


def _frequency_features(vol: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    FFT energy features.
    These are useful to test whether refined/cross compress high-frequency
    edge/texture information.

    Uses normalized radial frequency radius:
      low  = r < 0.15
      mid  = 0.15 <= r < 0.35
      high = 0.35 <= r < 0.60
      vhigh = r >= 0.60
    """
    v = vol.astype(np.float32, copy=False)

    # Remove global mean over foreground to avoid DC dominating everything.
    if mask is not None and mask.any():
        vv = v.copy()
        vv = vv - float(np.mean(vv[mask]))
    else:
        vv = v - float(np.mean(v))

    Fvol = np.fft.fftn(vv)
    power = np.abs(Fvol) ** 2
    power = np.fft.fftshift(power)

    H, W, D = v.shape
    fy = np.fft.fftshift(np.fft.fftfreq(H))
    fx = np.fft.fftshift(np.fft.fftfreq(W))
    fz = np.fft.fftshift(np.fft.fftfreq(D))

    yy, xx, zz = np.meshgrid(fy, fx, fz, indexing="ij")
    r = np.sqrt(xx * xx + yy * yy + zz * zz)
    r = r / (r.max() + 1e-12)

    total = float(power.sum() + 1e-12)

    low = power[r < 0.15].sum() / total
    mid = power[(r >= 0.15) & (r < 0.35)].sum() / total
    high = power[(r >= 0.35) & (r < 0.60)].sum() / total
    vhigh = power[r >= 0.60].sum() / total

    return {
        "fft_low_frac": float(low),
        "fft_mid_frac": float(mid),
        "fft_high_frac": float(high),
        "fft_vhigh_frac": float(vhigh),
        "fft_high_to_low": float((high + vhigh) / (low + 1e-12)),
    }


def _single_volume_image_features(vol: np.ndarray) -> Dict[str, float]:
    """
    Extract simple age-relevant image proxies from one volume.

    These are not perfect anatomical biomarkers, but they can reveal whether
    refined/cross compress intensity, edge, or frequency patterns by age.
    """
    v = vol.astype(np.float32, copy=False)
    mask = _brain_mask_proxy(v)

    if mask.any():
        fg = v[mask]
    else:
        fg = v.reshape(-1)

    out = {
        "finite_pct": float(np.isfinite(v).mean() * 100.0),

        # Foreground/intensity proxies
        "fg_voxel_frac": float(mask.mean()),
        "fg_mean": float(np.mean(fg)),
        "fg_std": float(np.std(fg)),
        "fg_p01": _safe_percentile(fg, 1),
        "fg_p05": _safe_percentile(fg, 5),
        "fg_p50": _safe_percentile(fg, 50),
        "fg_p95": _safe_percentile(fg, 95),
        "fg_p99": _safe_percentile(fg, 99),
        "fg_abs_mean": float(np.mean(np.abs(fg))),
    }

    out.update(_edge_features(v, mask))
    out.update(_laplacian_sharpness(v, mask))
    out.update(_frequency_features(v, mask))

    return out


def _single_volume_image_features_with_mask(vol: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    v = vol.astype(np.float32, copy=False)

    if mask is not None and mask.any():
        fg = v[mask]
    else:
        fg = v.reshape(-1)

    out = {
        "finite_pct": float(np.isfinite(v).mean() * 100.0),
        "fg_voxel_frac": float(mask.mean()) if mask is not None else np.nan,
        "fg_mean": float(np.mean(fg)),
        "fg_std": float(np.std(fg)),
        "fg_p01": _safe_percentile(fg, 1),
        "fg_p05": _safe_percentile(fg, 5),
        "fg_p50": _safe_percentile(fg, 50),
        "fg_p95": _safe_percentile(fg, 95),
        "fg_p99": _safe_percentile(fg, 99),
        "fg_abs_mean": float(np.mean(np.abs(fg))),
    }

    out.update(_edge_features(v, mask))
    out.update(_laplacian_sharpness(v, mask))
    out.update(_frequency_features(v, mask))

    return out


def compute_image_features_for_loader(
    dataloader,
    store: GeneratedStore,
    split_name: str,
    sources: Optional[List[str]] = None,
    age_bins: Optional[List[float]] = None,
    age_bin_labels: Optional[List[str]] = None,
    out_dir: Optional[str] = None,
    max_batches: Optional[int] = None,
):
    """
    Computes image-level features for real/refined/unrefined/cross.

    Saves per-image rows:
      image_features_{split}.csv

    and age-bin summary:
      image_features_by_age_bin_{split}.csv
    """
    if sources is None:
        sources = ["real"] + store.available_sources

    if age_bins is None:
        age_bins = [0, 15, 20, 25, 30, 40, 60, 100]

    if age_bin_labels is None:
        age_bin_labels = ["<15", "15-20", "20-25", "25-30", "30-40", "40-60", "60+"]

    rows = []

    pbar = tqdm(dataloader, desc=f"image_features[{split_name}]", leave=False)

    for batch_idx, batch in enumerate(pbar):
        if max_batches is not None and batch_idx >= max_batches:
            break

        image, y, metadata, global_ids = batch

        if torch.is_tensor(global_ids):
            global_ids = global_ids.long().cpu().tolist()
        else:
            global_ids = [int(x) for x in global_ids]

        if torch.is_tensor(y):
            y_np = y.detach().cpu().float().view(y.shape[0], -1)[:, 0].numpy()
        else:
            y_np = np.asarray(y).reshape(len(global_ids), -1)[:, 0].astype(np.float32)

        site_ids = metadata[:, 0].long().cpu().tolist()

        # Build one real-image mask per subject in this batch.
        # Use these same masks for real/refined/unrefined/cross so feature comparisons are fair.
        real_bhwd, real_gids, _ = _source_batch_to_hwd_numpy(batch, store, "real")

        if real_gids != global_ids:
            raise RuntimeError("global_idx mismatch while building real masks")

        real_masks = [_brain_mask_proxy(real_bhwd[i]) for i in range(len(global_ids))]

        for src in sources:
            vols_bhwd, gids_check, site_check = _source_batch_to_hwd_numpy(batch, store, src)

            if gids_check != global_ids:
                raise RuntimeError(f"global_idx mismatch for source={src}")

            for i, gidx in enumerate(global_ids):
                feats = _single_volume_image_features_with_mask(vols_bhwd[i], real_masks[i])
                
                
                rows.append({
                    "split": split_name,
                    "batch_idx": int(batch_idx),
                    "global_idx": int(gidx),
                    "site": int(site_ids[i]),
                    "source": src,
                    "age": float(y_np[i]),
                    **feats,
                })

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise RuntimeError(f"No image-feature rows for split={split_name}")

    df["age_bin"] = pd.cut(
        df["age"],
        bins=age_bins,
        labels=age_bin_labels,
        right=False,
        include_lowest=True,
    )

    feature_cols = [
        c for c in df.columns
        if c not in ["split", "batch_idx", "global_idx", "site", "source", "age", "age_bin"]
    ]

    summary_df = (
        df
        .groupby(["split", "source", "age_bin"], observed=True)[feature_cols]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        df.to_csv(out_dir / f"image_features_{split_name}.csv", index=False)
        summary_df.to_csv(out_dir / f"image_features_by_age_bin_{split_name}.csv", index=False)

        print(f"[image features] saved {split_name} to {out_dir}")

    return df, summary_df


def run_all_image_feature_checks(
    loaders: Dict[str, DataLoader],
    store: GeneratedStore,
    out_dir: str,
    sources: Optional[List[str]] = None,
    split_names: Optional[List[str]] = None,
    max_batches: Optional[int] = None,
):
    """
    Runs image-feature extraction for selected splits.

    Outputs:
      image_features_all_splits.csv
      image_features_by_age_bin_all_splits.csv
      plots/
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if sources is None:
        sources = ["real"] + store.available_sources

    if split_names is None:
        split_names = ["id_full", "ood_val", "ood_test"]

    all_rows = []
    all_summaries = []

    for split_name in split_names:
        if split_name not in loaders:
            print(f"[image features] skipping missing loader: {split_name}")
            continue

        df, summary_df = compute_image_features_for_loader(
            dataloader=loaders[split_name],
            store=store,
            split_name=split_name,
            sources=sources,
            out_dir=str(out_dir),
            max_batches=max_batches,
        )

        all_rows.append(df)
        all_summaries.append(summary_df)

    feat_all = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    summary_all = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()

    feat_all.to_csv(out_dir / "image_features_all_splits.csv", index=False)
    summary_all.to_csv(out_dir / "image_features_by_age_bin_all_splits.csv", index=False)

    plot_image_feature_age_trends(
        feature_df=feat_all,
        out_dir=str(out_dir / "plots"),
        sources=sources,
    )

    print(f"[image features] saved combined CSVs to {out_dir}")

    return {
        "features": feat_all,
        "summary": summary_all,
    }


def plot_image_feature_age_trends(
    feature_df: pd.DataFrame,
    out_dir: str,
    split_names: Optional[List[str]] = None,
    sources: Optional[List[str]] = None,
):
    """
    Creates simple plots:
      feature value vs age bin, grouped by source.

    These are the key features to inspect for age-signal compression:
      - fg_std
      - edge_mean
      - lap_abs_mean
      - fft_high_to_low
      - fft_low_frac
      - fg_voxel_frac
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if split_names is None:
        split_names = [
            "ood_val",
            "ood_test",
            "ood_val_age_similar",
            "ood_test_age_similar",
            "ood_val_age_shift",
            "ood_test_age_shift",
        ]
        
    if sources is None:
        sources = ["real", "unrefined", "refined", "cross", "cross-alpha-0.5"]
        
    feature_list = [
        ("fg_std", "Foreground intensity std"),
        ("edge_mean", "Mean gradient magnitude"),
        ("lap_abs_mean", "Mean |Laplacian| sharpness"),
        ("fft_high_to_low", "High-frequency / low-frequency energy"),
        ("fft_low_frac", "Low-frequency energy fraction"),
        ("fg_voxel_frac", "Foreground voxel fraction"),
    ]

    df = feature_df.copy()

    # Make sure age_bin is string for plotting order.
    age_order = ["<15", "15-20", "20-25", "25-30", "30-40", "40-60", "60+"]

    for split_name in split_names:
        sub_split = df[df["split"] == split_name].copy()
        if len(sub_split) == 0:
            continue

        for feat, title in feature_list:
            if feat not in sub_split.columns:
                continue

            summary = (
                sub_split
                .groupby(["age_bin", "source"], observed=True)[feat]
                .agg(["mean", "std", "count"])
                .reset_index()
            )

            age_bins_present = [b for b in age_order if b in summary["age_bin"].astype(str).unique()]
            x = np.arange(len(age_bins_present))
            width = 0.8 / max(1, len(sources))

            plt.figure(figsize=(11, 5))

            for j, src in enumerate(sources):
                ssub = summary[summary["source"] == src].copy()
                ssub["age_bin"] = ssub["age_bin"].astype(str)
                ssub = ssub.set_index("age_bin")

                vals = []
                errs = []

                for b in age_bins_present:
                    if b in ssub.index:
                        vals.append(float(ssub.loc[b, "mean"]))
                        errs.append(float(ssub.loc[b, "std"]) if not pd.isna(ssub.loc[b, "std"]) else 0.0)
                    else:
                        vals.append(np.nan)
                        errs.append(0.0)

                offset = (j - (len(sources) - 1) / 2) * width
                plt.bar(
                    x + offset,
                    vals,
                    width=width,
                    label=src,
                    yerr=errs,
                    capsize=2,
                )

            plt.xticks(x, age_bins_present)
            plt.xlabel("True age bin")
            plt.ylabel(feat)
            plt.title(f"{title} by age range ({split_name})")
            plt.grid(axis="y", alpha=0.25)
            plt.legend()
            plt.tight_layout()

            safe_feat = feat.replace("/", "_")
            out_path = out_dir / f"{split_name}_{safe_feat}_by_age_bin.png"
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close()

    print(f"[image features] saved plots to {out_dir}")



def plot_domain_invariance_from_xmat(
    xmat_csv: str,
    out_dir: str,
    train_source: str = "real",
    n_classes: int = 15
):
    """
    Plot domain prediction performance of a real-trained domain classifier
    across real/refined/unrefined/cross.

    Lower balanced accuracy / macro-F1 on generated images means the source
    carries less real-domain/site signal under this classifier.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(xmat_csv)
    df = df[df["train_source"] == train_source].copy()

    source_order = {
        "real": 0,
        "unrefined": 1,
        "refined": 2,
        "cross": 3,
        "cross-alpha-0.5": 4,
    }
    split_order = {
        "id_holdout": 0,
        "id_full": 1,
        "ood_val": 2,
        "ood_test": 3,
        "ood_val_age_similar": 4,
        "ood_test_age_similar": 5,
        "ood_val_age_shift": 6,
        "ood_test_age_shift": 7,
    }

    df["source_order"] = df["eval_source"].map(source_order)
    df["split_order"] = df["eval_split"].map(split_order)
    df = df.sort_values(["split_order", "source_order"])

    fig, ax = plt.subplots(figsize=(9, 4.5))

    labels = [
        f"{r.eval_split}\n{r.eval_source}"
        for r in df.itertuples()
    ]

    ax.bar(np.arange(len(df)), df["bal_acc"].values)
    ax.axhline(
        1.0 / float(n_classes),
        linestyle="--",
        linewidth=1,
        label=f"random ≈ 1/{n_classes}",
    )
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Balanced domain accuracy")
    ax.set_title("Domain signal measured by real-trained classifier\n(lower means less recoverable site/domain signal)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    plt.tight_layout()
    out_path = out_dir / "domain_invariance_real_trained_bal_acc.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")

def _valid_ssim_win_size(shape3d, preferred=7):
    m = int(min(shape3d))
    if m < 3:
        raise ValueError(f"Volume too small for SSIM: shape={shape3d}")
    win = min(preferred, m)
    if win % 2 == 0:
        win -= 1
    if win < 3:
        win = 3
    return win


def _pair_data_range(real_vol: np.ndarray, recon_vol: np.ndarray, mode="pair", fixed=None) -> float:
    """
    mode:
      - 'pair': range from min/max over both real and recon volume
      - 'real': range from real volume only
      - 'fixed': use `fixed`
    """
    if mode == "fixed":
        if fixed is None:
            raise ValueError("fixed data range requested but fixed=None")
        dr = float(fixed)
    elif mode == "real":
        dr = float(real_vol.max() - real_vol.min())
    elif mode == "pair":
        dr = float(max(real_vol.max(), recon_vol.max()) - min(real_vol.min(), recon_vol.min()))
    else:
        raise ValueError(f"Unknown data_range mode: {mode}")

    if dr <= 1e-12:
        dr = 1.0
    return dr


def compute_psnr_ssim_for_loader(
    dataloader,
    store: GeneratedStore,
    source: str = "unrefined",
    split_name: str = "id_full",
    out_csv: Optional[str] = None,
    max_batches: Optional[int] = None,
    data_range_mode: str = "pair",   # "pair", "real", or "fixed"
    fixed_data_range: Optional[float] = None,
    ssim_win_size: int = 7,
):
    """
    Computes paired PSNR/SSIM between REAL images from the loader and GENERATED images
    loaded from store[source], matched by global index.

    Returns:
      df         : per-sample metrics
      summary    : overall summary dict
      per_site_df: grouped per-site summary
    """
    rows = []

    pbar = tqdm(dataloader, desc=f"PSNR/SSIM[{split_name}:{source}]", leave=False)
    for batch_idx, batch in enumerate(pbar):
        if max_batches is not None and batch_idx >= max_batches:
            break

        image, _, metadata, global_ids = batch

        if torch.is_tensor(global_ids):
            global_ids = global_ids.long().cpu().tolist()
        else:
            global_ids = [int(x) for x in global_ids]

        site_ids = metadata[:, 0].long().cpu().tolist()

        real_bhwd = _real_batch_to_hwd_numpy(image)
        gen_bhwd = _generated_batch_to_hwd_numpy(store, global_ids, source)

        if real_bhwd.shape != gen_bhwd.shape:
            raise ValueError(
                f"Shape mismatch between real and generated volumes: "
                f"real={real_bhwd.shape}, gen={gen_bhwd.shape}, "
                f"batch_idx={batch_idx}, source={source}"
            )

        for i, gidx in enumerate(global_ids):
            r = real_bhwd[i].astype(np.float64)
            g = gen_bhwd[i].astype(np.float64)

            dr = _pair_data_range(r, g, mode=data_range_mode, fixed=fixed_data_range)
            win = _valid_ssim_win_size(r.shape, preferred=ssim_win_size)

            psnr_val = peak_signal_noise_ratio(r, g, data_range=dr)
            ssim_val = structural_similarity(
                r,
                g,
                data_range=dr,
                channel_axis=None,
                win_size=win,
            )

            rows.append({
                "global_idx": int(gidx),
                "split": split_name,
                "source": source,
                "site": int(site_ids[i]),
                "psnr": float(psnr_val),
                "ssim": float(ssim_val),
                "data_range": float(dr),
            })

    df = pd.DataFrame(rows).sort_values("global_idx").reset_index(drop=True)

    if len(df) == 0:
        raise RuntimeError(f"No PSNR/SSIM rows were produced for split={split_name}, source={source}")

    summary = {
        "split": split_name,
        "source": source,
        "n": int(len(df)),
        "psnr_mean": float(df["psnr"].mean()),
        "psnr_std": float(df["psnr"].std(ddof=1)) if len(df) > 1 else 0.0,
        "psnr_min": float(df["psnr"].min()),
        "psnr_max": float(df["psnr"].max()),
        "ssim_mean": float(df["ssim"].mean()),
        "ssim_std": float(df["ssim"].std(ddof=1)) if len(df) > 1 else 0.0,
        "ssim_min": float(df["ssim"].min()),
        "ssim_max": float(df["ssim"].max()),
    }

    per_site_df = (
        df.groupby("site")[["psnr", "ssim"]]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )

    if out_csv is not None:
        out_csv = str(out_csv)
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df.to_csv(out_csv, index=False)

        site_csv = out_csv.replace(".csv", "_per_site.csv")
        per_site_df.to_csv(site_csv, index=False)

    return df, summary, per_site_df


def compute_pairwise_psnr_ssim_for_loader(
    dataloader,
    store: GeneratedStore,
    source_a: str,
    source_b: str,
    split_name: str = "id_full",
    out_csv: Optional[str] = None,
    max_batches: Optional[int] = None,
    data_range_mode: str = "pair",
    fixed_data_range: Optional[float] = None,
    ssim_win_size: int = 7,
):
    """
    Computes paired PSNR/SSIM between any two sources:
      real/refined/unrefined/cross.

    Matched by global_idx through the same dataloader batch.
    """
    rows = []

    pbar = tqdm(
        dataloader,
        desc=f"PSNR/SSIM[{split_name}:{source_a}_vs_{source_b}]",
        leave=False,
    )

    for batch_idx, batch in enumerate(pbar):
        if max_batches is not None and batch_idx >= max_batches:
            break

        xa_bhwd, global_ids, site_ids = _source_batch_to_hwd_numpy(batch, store, source_a)
        xb_bhwd, global_ids_b, site_ids_b = _source_batch_to_hwd_numpy(batch, store, source_b)

        if global_ids != global_ids_b:
            raise RuntimeError("global_ids mismatch between pair sources")

        if xa_bhwd.shape != xb_bhwd.shape:
            raise ValueError(
                f"Shape mismatch for {source_a} vs {source_b}: "
                f"{xa_bhwd.shape} vs {xb_bhwd.shape}, batch_idx={batch_idx}"
            )

        for i, gidx in enumerate(global_ids):
            a = xa_bhwd[i].astype(np.float64)
            b = xb_bhwd[i].astype(np.float64)

            dr = _pair_data_range(a, b, mode=data_range_mode, fixed=fixed_data_range)
            win = _valid_ssim_win_size(a.shape, preferred=ssim_win_size)

            psnr_val = peak_signal_noise_ratio(a, b, data_range=dr)
            ssim_val = structural_similarity(
                a,
                b,
                data_range=dr,
                channel_axis=None,
                win_size=win,
            )

            rows.append({
                "global_idx": int(gidx),
                "split": split_name,
                "source_a": source_a,
                "source_b": source_b,
                "pair": f"{source_a}_vs_{source_b}",
                "site": int(site_ids[i]),
                "psnr": float(psnr_val),
                "ssim": float(ssim_val),
                "data_range": float(dr),
            })

    df = pd.DataFrame(rows).sort_values("global_idx").reset_index(drop=True)

    if len(df) == 0:
        raise RuntimeError(
            f"No pairwise PSNR/SSIM rows for split={split_name}, "
            f"pair={source_a}_vs_{source_b}"
        )

    summary = {
        "split": split_name,
        "source_a": source_a,
        "source_b": source_b,
        "pair": f"{source_a}_vs_{source_b}",
        "n": int(len(df)),
        "psnr_mean": float(df["psnr"].mean()),
        "psnr_std": float(df["psnr"].std(ddof=1)) if len(df) > 1 else 0.0,
        "psnr_min": float(df["psnr"].min()),
        "psnr_max": float(df["psnr"].max()),
        "ssim_mean": float(df["ssim"].mean()),
        "ssim_std": float(df["ssim"].std(ddof=1)) if len(df) > 1 else 0.0,
        "ssim_min": float(df["ssim"].min()),
        "ssim_max": float(df["ssim"].max()),
    }

    per_site_df = (
        df.groupby("site")[["psnr", "ssim"]]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )

    if out_csv is not None:
        out_csv = str(out_csv)
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df.to_csv(out_csv, index=False)

        site_csv = out_csv.replace(".csv", "_per_site.csv")
        per_site_df.to_csv(site_csv, index=False)

    return df, summary, per_site_df



def plot_pairwise_metric_bars(summary_df: pd.DataFrame, out_dir: str, prefix: str):
    """
    Grouped bar plot:
      x-axis: source pair
      bars: split names

    Expected columns:
      split, pair, psnr_mean, ssim_mean
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep a clean order if these values exist
    pair_order = [
        "real_vs_unrefined",
        "real_vs_refined",
        "real_vs_cross",
        "real_vs_cross-alpha-0.5",

        "cross_vs_cross-alpha-0.5",
        "cross-alpha-0.5_vs_refined",

        "refined_vs_unrefined",
        "refined_vs_cross",
        "unrefined_vs_cross",
    ]
    
    split_order = [
        "id_full",
        "ood_val",
        "ood_test",
        "ood_val_age_similar",
        "ood_test_age_similar",
        "ood_val_age_shift",
        "ood_test_age_shift",
    ]
    df = summary_df.copy()

    # Only keep pairs/splits that actually exist
    pairs = [p for p in pair_order if p in df["pair"].unique()]
    pairs += [p for p in sorted(df["pair"].unique()) if p not in pairs]

    splits = [s for s in split_order if s in df["split"].unique()]
    splits += [s for s in sorted(df["split"].unique()) if s not in splits]

    for metric in ["psnr_mean", "ssim_mean"]:
        fig, ax = plt.subplots(figsize=(10, 4.5))

        x = np.arange(len(pairs))
        width = 0.8 / max(1, len(splits))

        for j, split in enumerate(splits):
            vals = []
            for pair in pairs:
                sub = df[(df["split"] == split) & (df["pair"] == pair)]
                if len(sub) == 0:
                    vals.append(np.nan)
                else:
                    vals.append(float(sub.iloc[0][metric]))

            offset = (j - (len(splits) - 1) / 2) * width
            ax.bar(x + offset, vals, width=width, label=split)

        pretty_pairs = [
            p.replace("real_vs_", "real vs ")
             .replace("refined_vs_", "refined vs ")
             .replace("unrefined_vs_", "unrefined vs ")
             .replace("_", " ")
            for p in pairs
        ]

        ax.set_xticks(x)
        ax.set_xticklabels(pretty_pairs, rotation=25, ha="right")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(f"Pairwise {metric.replace('_', ' ').upper()} by split")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(title="Split")

        # Helpful reference ranges
        if metric == "ssim_mean":
            ax.set_ylim(0.0, 1.05)

        plt.tight_layout()

        out_path = out_dir / f"{prefix}_{metric}_grouped_by_split.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved plot: {out_path}")


def prepare_batch_for_3dcnn(x: torch.Tensor) -> torch.Tensor:
    """
    Accepts:
      [B,C,1,H,W,D]
      [B,C,H,W,D]
      [B,H,W,D]
      [B,1,H,W,D]
    Returns:
      [B,1,D,H,W]
    """
    x = x.float()

    if x.dim() == 6:
        assert x.size(2) == 1, f"Expected singleton dim at axis 2, got {tuple(x.shape)}"
        x = x.squeeze(2)   # -> [B,C,H,W,D]

    if x.dim() == 4:
        x = x.unsqueeze(1)  # -> [B,1,H,W,D]

    assert x.dim() == 5, f"Expected 5D tensor before permute, got {tuple(x.shape)}"
    assert x.size(1) == 1, f"Expected channel dim = 1, got {tuple(x.shape)}"

    # [B,1,H,W,D] -> [B,1,D,H,W]
    x = x.permute(0, 1, 4, 2, 3).contiguous()
    return x


def load_generated_batch_by_global_ids(
    store: GeneratedStore,
    global_ids: List[int],
    source: str,
    device: torch.device,
) -> torch.Tensor:
    vols = []
    for gidx in global_ids:
        fp = store.path(gidx, source)
        v = load_nii(fp)

        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v)

        v = v.float()

        if v.dim() == 3:
            v = v.unsqueeze(0)   # [1,H,W,D]

        assert v.dim() == 4, f"Loaded volume must be [1,H,W,D] or [H,W,D], got {tuple(v.shape)}"
        vols.append(v)

    x = torch.stack(vols, dim=0)      # [B,1,H,W,D]
    x = prepare_batch_for_3dcnn(x)    # [B,1,D,H,W]
    return x.to(device, non_blocking=True)


def get_inputs_and_domains(
    batch,
    source: str,
    store: GeneratedStore,
    device: torch.device,
):
    image, _, metadata, global_ids = batch
    domains = metadata[:, 0].to(device).long()

    if torch.is_tensor(global_ids):
        global_ids = global_ids.long().cpu().tolist()
    else:
        global_ids = [int(x) for x in global_ids]

    if source == "real":
        x = prepare_batch_for_3dcnn(image.to(device, dtype=torch.float32, non_blocking=True))
    else:
        x = load_generated_batch_by_global_ids(store, global_ids, source, device)

    return x, domains, global_ids


# ============================================================
# Pool building + uniform-by-site sampling
# Everything here is based on local positions inside each split,
# but eligibility is decided using GLOBAL ids.
# ============================================================
def split_indices(
    indices,
    frac_train=0.9,
    train_bs=16,
    align_train_to_full_batches=True,
):
    N = len(indices)
    cut = int(round(frac_train * N))
    if align_train_to_full_batches and train_bs and train_bs > 0:
        cut = (cut // train_bs) * train_bs
    cut = max(0, min(cut, N))
    return indices[:cut], indices[cut:]


def build_eligible_local_pool(
    split_ds,
    store: GeneratedStore,
    required_sources: List[str],
):
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

    eligible_by_site = {k: sorted(v) for k, v in sorted(eligible_by_site.items())}
    eligible_all_local = sorted(eligible_all_local)
    return wrapped, eligible_by_site, eligible_all_local

def split_indices_by_site(
    eligible_by_site: Dict[int, List[int]],
    frac_train: float = 0.9,
    seed: int = 42,
):
    """
    Stratified split by site using ALL unique eligible indices.
    Keeps both train and holdout populated for a site whenever possible.
    """
    rng = np.random.default_rng(seed)
    idx_train, idx_holdout = [], []

    for site, ids in sorted(eligible_by_site.items()):
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


def compute_site_class_weights_from_local_indices(
    split_ds,
    local_indices: List[int],
    n_classes: int = 15,
    power: float = 1.0,
):
    """
    Inverse-frequency class weights for site/domain labels.
    power=1.0  -> full inverse frequency
    power=0.5  -> milder reweighting
    """
    sites = split_ds.metadata_array[local_indices, 0].to(torch.int64).cpu().numpy()
    counts = np.bincount(sites, minlength=n_classes).astype(np.float64)

    weights = np.zeros(n_classes, dtype=np.float32)
    nz = counts > 0
    weights[nz] = (counts[nz].sum() / counts[nz]) ** power

    # normalize so average active class weight is ~1
    if np.any(nz):
        weights[nz] /= weights[nz].mean()

    return torch.tensor(weights, dtype=torch.float32), counts.tolist()


def uniform_indices_fixed_total_from_pool(
    eligible_by_site: Dict[int, List[int]],
    total_n: Optional[int] = None,
    seed: int = 42,
    allow_replacement: bool = True,
):
    """
    Uniform-ish balanced sampling across sites using local indices.
    If total_n=None, uses all currently eligible items as the requested total.
    """
    rng = np.random.default_rng(seed)
    sites = sorted(eligible_by_site.keys())

    if not sites:
        raise ValueError("No eligible sites found.")

    pool_size = sum(len(v) for v in eligible_by_site.values())

    if total_n is None:
        total_n = pool_size

    if (not allow_replacement) and (total_n > pool_size):
        print(f"[uniform sample] Capping total_n from {total_n} to pool_size={pool_size} (no replacement).")
        total_n = pool_size

    base = total_n // len(sites)
    rem = total_n % len(sites)
    alloc = {s: base + (i < rem) for i, s in enumerate(sites)}

    picked = []
    for s in sites:
        pool = eligible_by_site[s]
        need = alloc[s]

        if need <= len(pool):
            choose = rng.choice(pool, size=need, replace=False).tolist()
        else:
            if not allow_replacement:
                choose = pool[:]
            else:
                base_pick = rng.choice(pool, size=min(need, len(pool)), replace=False).tolist()
                top_up = rng.choice(pool, size=need - len(base_pick), replace=True).tolist()
                choose = base_pick + top_up

        picked.extend(int(i) for i in choose)

    if allow_replacement and len(picked) != total_n:
        union = np.array([i for s in sites for i in eligible_by_site[s]])
        if len(picked) > total_n:
            picked = picked[:total_n]
        else:
            extra = rng.choice(union, size=total_n - len(picked), replace=True).tolist()
            picked.extend(int(i) for i in extra)

    return picked, total_n


def build_probe_loaders(
    dataset,
    store: GeneratedStore,
    batch_size: int = 16,
    train_frac: float = 0.9,
    num_workers: int = 1,
    pin_memory: bool = True,
    site_weight_power: float = 1.0,
    n_site_classes: int = 15,
):
    """
    Uses ALL unique eligible images.

    Builds:
      - probe_train: unique train subset from TRAIN split
      - id_holdout: unique holdout from TRAIN split
      - id_full   : all unique eligible TRAIN images
      - ood_val   : all unique eligible VAL images
      - ood_test  : all unique eligible TEST images

    Returns:
      loaders, info
    """
    splits = {
        "train": dataset.get_subset("train"),
        "val": dataset.get_subset("val"),
        "test": dataset.get_subset("test"),
    }

    required_sources = store.available_sources[:]
    print(f"Required generated sources: {required_sources}")

    # ---------------- TRAIN ----------------
    train_wrapped, eligible_by_site_train, eligible_train_local = build_eligible_local_pool(
        splits["train"], store, required_sources
    )

    print(f"[train pool] eligible unique local positions = {len(eligible_train_local)} "
          f"across {len(eligible_by_site_train)} sites")

    # Use ALL unique eligible train images, split stratified by site
    idx_train, idx_holdout = split_indices_by_site(
        eligible_by_site_train,
        frac_train=train_frac,
        seed=42,
    )

    idx_id_full = eligible_train_local  # <-- all unique eligible train images

    train_site_weights, train_site_counts = compute_site_class_weights_from_local_indices(
        splits["train"],
        idx_train,
        n_classes=n_site_classes,
        power=site_weight_power,
    )

    print(f"[train split] unique_total={len(eligible_train_local)} | "
          f"probe_train={len(idx_train)} | id_holdout={len(idx_holdout)} | id_full={len(idx_id_full)}")
    print(f"[train split] site counts (probe_train): {train_site_counts}")

    loaders = {}

    loaders["probe_train"] = DataLoader(
        Subset(train_wrapped, idx_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,   # use all unique train images
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    loaders["id_holdout"] = DataLoader(
        Subset(train_wrapped, idx_holdout),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    loaders["id_full"] = DataLoader(
        Subset(train_wrapped, idx_id_full),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # ---------------- OOD val/test ----------------
    for split_name in ["val", "test"]:
        wrapped, eligible_by_site, eligible_local = build_eligible_local_pool(
            splits[split_name], store, required_sources
        )

        print(f"[{split_name} pool] eligible unique local positions = {len(eligible_local)} "
              f"across {len(eligible_by_site)} sites")

        if len(eligible_local) == 0:
            print(f"[{split_name}] skipped because no eligible generated files were found.")
            continue

        loaders[f"ood_{split_name}"] = DataLoader(
            Subset(wrapped, eligible_local),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    info = {
        "train_site_weights": train_site_weights,
        "train_site_counts": train_site_counts,
        "idx_train": idx_train,
        "idx_holdout": idx_holdout,
        "idx_id_full": idx_id_full,
        "eligible_train_local": eligible_train_local,
    }

    return loaders, info

def add_age_controlled_ood_loaders(
    loaders: Dict[str, DataLoader],
    dataset,
    store: GeneratedStore,
    batch_size: int = 16,
    num_workers: int = 1,
    pin_memory: bool = True,
    age_mode: str = "meanpm2std",
):
    """
    Adds controlled OOD loaders to the existing loaders dict.

    New loaders:
      ood_val_age_similar
      ood_test_age_similar
      ood_val_age_shift
      ood_test_age_shift

    age_similar:
      subjects whose age is inside the train age support.

    age_shift:
      subjects outside the train age support.

    Default train age support:
      mean ± 2 std
    """
    splits = {
        "train": dataset.get_subset("train"),
        "val": dataset.get_subset("val"),
        "test": dataset.get_subset("test"),
    }

    required_sources = store.available_sources[:]

    # Build eligible train pool, because we only want ages for subjects
    # that have all generated sources available.
    train_wrapped, _, eligible_train_local = build_eligible_local_pool(
        splits["train"],
        store,
        required_sources,
    )

    train_y = splits["train"].y_array[eligible_train_local]
    if torch.is_tensor(train_y):
        train_y = train_y.detach().cpu().float().view(len(eligible_train_local), -1)[:, 0].numpy()
    else:
        train_y = np.asarray(train_y).reshape(len(eligible_train_local), -1)[:, 0].astype(np.float32)

    if age_mode == "meanpm2std":
        age_center = float(np.mean(train_y))
        age_std = float(np.std(train_y, ddof=1))
        age_lo = age_center - 2.0 * age_std
        age_hi = age_center + 2.0 * age_std
        rule_name = "mean±2std"

    elif age_mode == "p5p95":
        age_lo, age_hi = np.percentile(train_y, [5, 95])
        age_lo = float(age_lo)
        age_hi = float(age_hi)
        age_center = float(np.mean(train_y))
        age_std = float(np.std(train_y, ddof=1))
        rule_name = "p5-p95"

    else:
        raise ValueError(f"Unknown age_mode={age_mode}. Use 'meanpm2std' or 'p5p95'.")

    print("\n[age-controlled OOD]")
    print(f"Train eligible age mean={age_center:.3f}, std={age_std:.3f}")
    print(f"Age-similar rule ({rule_name}): {age_lo:.3f} <= age <= {age_hi:.3f}")

    controlled_info = {
        "age_mode": age_mode,
        "age_lo": age_lo,
        "age_hi": age_hi,
        "train_age_mean": age_center,
        "train_age_std": age_std,
    }

    for split_name in ["val", "test"]:
        split_ds = splits[split_name]

        wrapped, eligible_by_site, eligible_local = build_eligible_local_pool(
            split_ds,
            store,
            required_sources,
        )

        y = split_ds.y_array[eligible_local]
        if torch.is_tensor(y):
            y_np = y.detach().cpu().float().view(len(eligible_local), -1)[:, 0].numpy()
        else:
            y_np = np.asarray(y).reshape(len(eligible_local), -1)[:, 0].astype(np.float32)

        eligible_local = np.asarray(eligible_local, dtype=int)

        age_similar_mask = (y_np >= age_lo) & (y_np <= age_hi)
        age_shift_mask = ~age_similar_mask

        idx_age_similar = eligible_local[age_similar_mask].astype(int).tolist()
        idx_age_shift = eligible_local[age_shift_mask].astype(int).tolist()

        for tag, idxs in [
            ("age_similar", idx_age_similar),
            ("age_shift", idx_age_shift),
        ]:
            loader_name = f"ood_{split_name}_{tag}"

            if len(idxs) == 0:
                print(f"[age-controlled OOD] {loader_name}: empty, skipping")
                continue

            ages_this = y_np[age_similar_mask] if tag == "age_similar" else y_np[age_shift_mask]
            site_ids = split_ds.metadata_array[idxs, 0].to(torch.int64).cpu().numpy()

            print(
                f"[age-controlled OOD] {loader_name}: "
                f"n={len(idxs)}, sites={len(np.unique(site_ids))}, "
                f"age_mean={np.mean(ages_this):.3f}, "
                f"age_std={np.std(ages_this, ddof=1):.3f}, "
                f"age_min={np.min(ages_this):.3f}, "
                f"age_max={np.max(ages_this):.3f}"
            )

            loaders[loader_name] = DataLoader(
                Subset(wrapped, idxs),
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

            controlled_info[loader_name] = {
                "n": int(len(idxs)),
                "n_sites": int(len(np.unique(site_ids))),
                "age_mean": float(np.mean(ages_this)),
                "age_std": float(np.std(ages_this, ddof=1)) if len(ages_this) > 1 else 0.0,
                "age_min": float(np.min(ages_this)),
                "age_max": float(np.max(ages_this)),
            }

    return loaders, controlled_info



def add_age_limited_ood_loaders(
    loaders: Dict[str, DataLoader],
    dataset,
    store: GeneratedStore,
    batch_size: int = 16,
    num_workers: int = 1,
    pin_memory: bool = True,
    max_age: float = 35.0,
    min_age: float = 0.0,
):
    """
    Adds age-limited OOD loaders for brain-age evaluation.

    This is intended for the final brain-age preservation test:
      - remove older subjects where real->real brain-age prediction is unreliable
      - keep the original train/val/test/site split unchanged

    New loaders:
      ood_val_age_le35
      ood_test_age_le35
    """
    splits = {
        "val": dataset.get_subset("val"),
        "test": dataset.get_subset("test"),
    }

    required_sources = store.available_sources[:]

    controlled_info = {
        "age_filter": f"{min_age} <= age <= {max_age}",
        "min_age": float(min_age),
        "max_age": float(max_age),
    }

    tag = f"age_le{int(max_age)}"

    for split_name in ["val", "test"]:
        split_ds = splits[split_name]

        wrapped, eligible_by_site, eligible_local = build_eligible_local_pool(
            split_ds,
            store,
            required_sources,
        )

        eligible_local = np.asarray(eligible_local, dtype=int)

        y = split_ds.y_array[eligible_local]
        if torch.is_tensor(y):
            y_np = y.detach().cpu().float().view(len(eligible_local), -1)[:, 0].numpy()
        else:
            y_np = np.asarray(y).reshape(len(eligible_local), -1)[:, 0].astype(np.float32)

        keep_mask = (y_np >= min_age) & (y_np <= max_age)
        idx_keep = eligible_local[keep_mask].astype(int).tolist()
        ages_keep = y_np[keep_mask]

        loader_name = f"ood_{split_name}_{tag}"

        if len(idx_keep) == 0:
            print(f"[age-limited OOD] {loader_name}: empty, skipping")
            continue

        site_ids = split_ds.metadata_array[idx_keep, 0].to(torch.int64).cpu().numpy()

        print(
            f"[age-limited OOD] {loader_name}: "
            f"n={len(idx_keep)}, sites={len(np.unique(site_ids))}, "
            f"age_mean={np.mean(ages_keep):.3f}, "
            f"age_std={np.std(ages_keep, ddof=1):.3f}, "
            f"age_min={np.min(ages_keep):.3f}, "
            f"age_max={np.max(ages_keep):.3f}"
        )

        loaders[loader_name] = DataLoader(
            Subset(wrapped, idx_keep),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        controlled_info[loader_name] = {
            "n": int(len(idx_keep)),
            "n_sites": int(len(np.unique(site_ids))),
            "age_mean": float(np.mean(ages_keep)),
            "age_std": float(np.std(ages_keep, ddof=1)) if len(ages_keep) > 1 else 0.0,
            "age_min": float(np.min(ages_keep)),
            "age_max": float(np.max(ages_keep)),
        }

    return loaders, controlled_info
# ============================================================
# Metrics
# ============================================================
def class_histogram(targets: torch.Tensor, n_classes: int):
    y = targets.detach().cpu().numpy()
    counts = np.bincount(y, minlength=n_classes)
    present = np.nonzero(counts)[0]
    return counts, present


def multiclass_ece_from_logits(logits: torch.Tensor, targets: torch.Tensor, n_bins: int = 15) -> float:
    with torch.no_grad():
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        correct = pred.eq(targets)

        conf = conf.detach().cpu().numpy()
        correct = correct.detach().cpu().numpy().astype(np.float32)

        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (conf > bins[i]) & (conf <= bins[i + 1])
            if not np.any(mask):
                continue
            bin_acc = correct[mask].mean()
            bin_conf = conf[mask].mean()
            ece += abs(bin_conf - bin_acc) * mask.mean()
        return float(ece)


def _ece_binary(confidence: np.ndarray, labels01: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (confidence > bins[i]) & (confidence <= bins[i + 1])
        if not np.any(m):
            continue
        bin_acc = labels01[m].mean()
        bin_conf = confidence[m].mean()
        ece += abs(bin_conf - bin_acc) * m.mean()
    return float(ece)


def normalized_accuracy(micro_acc: float, counts: List[int]):
    p_max = max(c / max(1, sum(counts)) for c in counts)
    return float((micro_acc - p_max) / (1.0 - p_max + 1e-12))


def metrics_multiclass_robust(
    logits: torch.Tensor,
    targets: torch.Tensor,
    n_classes: int,
    n_bins: int = 15,
):
    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
    y = targets.detach().cpu().numpy()
    yhat = probs.argmax(1)

    counts, present = class_histogram(targets, n_classes)

    micro_acc = float((yhat == y).mean())
    ce_micro = float(F.cross_entropy(logits, targets, reduction="mean").item())

    bal_acc = float(recall_score(y, yhat, average="macro", labels=present, zero_division=0))
    macro_f1 = float(f1_score(y, yhat, average="macro", labels=present, zero_division=0))

    ce_vec = F.cross_entropy(logits, targets, reduction="none").detach().cpu().numpy()
    per_class_ce = []
    for k in present:
        m = (y == k)
        if np.any(m):
            per_class_ce.append(ce_vec[m].mean())
    ce_macro = float(np.mean(per_class_ce)) if len(per_class_ce) else float("nan")

    aucs = []
    for k in present:
        yk = (y == k).astype(np.int32)
        if yk.min() == yk.max():
            continue
        aucs.append(roc_auc_score(yk, probs[:, k]))
    auc_macro_present = float(np.mean(aucs)) if len(aucs) else float("nan")

    ece_micro = multiclass_ece_from_logits(logits, targets, n_bins=n_bins)

    eces = []
    for k in present:
        conf = probs[:, k]
        lab = (y == k).astype(np.int32)
        eces.append(_ece_binary(conf, lab, n_bins=n_bins))
    ece_macro = float(np.mean(eces)) if len(eces) else float("nan")

    worst_site_acc = 1.0
    worst_site_f1 = 1.0
    for k in present:
        m = (y == k)
        acc_k = float((yhat[m] == y[m]).mean()) if np.any(m) else np.nan
        y_bin = (y == k).astype(np.int32)
        yhat_bin = (yhat == k).astype(np.int32)
        f1_k = f1_score(y_bin[m], yhat_bin[m], zero_division=0) if np.any(m) else np.nan
        worst_site_acc = min(worst_site_acc, acc_k)
        worst_site_f1 = min(worst_site_f1, f1_k)

    return dict(
        micro_acc=micro_acc,
        norm_acc=normalized_accuracy(micro_acc, counts.tolist()),
        bal_acc=bal_acc,
        macro_f1=macro_f1,
        auc_macro_present=auc_macro_present,
        ece_micro=ece_micro,
        ece_macro=ece_macro,
        ce_micro=ce_micro,
        ce_macro=ce_macro,
        worst_site_acc=float(worst_site_acc),
        worst_site_f1=float(worst_site_f1),
        counts=counts.tolist(),
    )


# ============================================================
# Probe training / evaluation
# ============================================================
def train_on_source(
    model: nn.Module,
    optimizer: optim.Optimizer,
    dataloader,
    store: GeneratedStore,
    source: str,
    epochs: int = 3,
    max_batches: Optional[int] = None,
    class_weights: Optional[torch.Tensor] = None,
):
    device = next(model.parameters()).device

    loss_fn = nn.CrossEntropyLoss(
        label_smoothing=0.01,
        weight=class_weights.to(device) if class_weights is not None else None,
    )

    model.train()
    for ep in range(epochs):
        losses = []

        pbar = tqdm(dataloader, desc=f"train[{source}] ep {ep+1}/{epochs}", leave=False)
        for j, batch in enumerate(pbar):
            if max_batches is not None and j >= max_batches:
                break

            x, domains, _ = get_inputs_and_domains(batch, source, store, device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, domains)
            loss.backward()
            optimizer.step()

            losses.append(float(loss.detach().cpu()))
            pbar.set_postfix(loss=np.mean(losses))

        print(f"[train on {source}] epoch {ep+1}/{epochs} | avg_loss={np.mean(losses):.4f} "
              f"| n_batches={len(losses)}")

@torch.no_grad()
def eval_on_source(
    model: nn.Module,
    dataloader,
    store: GeneratedStore,
    source: str,
    n_classes: int = 15,
    max_batches: Optional[int] = None,
):
    device = next(model.parameters()).device
    model.eval()

    logits_all = []
    targets_all = []

    pbar = tqdm(dataloader, desc=f"eval[{source}]", leave=False)
    for j, batch in enumerate(pbar):
        if max_batches is not None and j >= max_batches:
            break

        x, domains, _ = get_inputs_and_domains(batch, source, store, device)
        logits = model(x)

        logits_all.append(logits)
        targets_all.append(domains)

    logits_all = torch.cat(logits_all, dim=0)
    targets_all = torch.cat(targets_all, dim=0)

    return metrics_multiclass_robust(logits_all, targets_all, n_classes=n_classes)

def cross_training_matrix(
    make_model_fn,
    loaders: Dict[str, DataLoader],
    store: GeneratedStore,
    epochs_probe: int = 3,
    lr: float = 7e-6,
    n_classes: int = 15,
    max_batches_train: Optional[int] = None,
    max_batches_eval: Optional[int] = None,
    class_weights: Optional[torch.Tensor] = None,
    eval_split_names: Optional[Sequence[str]] = None,

):
    train_sources = ["real"] + store.available_sources
    rows = []
    real_trained_model = None

    if eval_split_names is None:
        eval_split_names = ["id_holdout", "id_full"]
        
    for train_source in train_sources:
        model = make_model_fn()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_on_source(
            model=model,
            optimizer=optimizer,
            dataloader=loaders["probe_train"],
            store=store,
            source=train_source,
            epochs=epochs_probe,
            max_batches=max_batches_train,
            class_weights=class_weights,
        )

        if train_source == "real":
            real_trained_model = copy.deepcopy(model).eval()

        for eval_split_name in eval_split_names:
            if eval_split_name not in loaders:
                continue

            for eval_source in train_sources:
                res = eval_on_source(
                    model=model,
                    dataloader=loaders[eval_split_name],
                    store=store,
                    source=eval_source,
                    n_classes=n_classes,
                    max_batches=max_batches_eval,
                )

                rows.append({
                    "train_source": train_source,
                    "eval_split": eval_split_name,
                    "eval_source": eval_source,
                    **res,
                })

                print(
                    f"[train={train_source:10s} | split={eval_split_name:24s} | eval={eval_source:18s}] "
                    f"micro_acc={res['micro_acc']*100:5.1f}  "
                    f"norm_acc={res['norm_acc']:.3f}  "
                    f"bal_acc={res['bal_acc']:.3f}  "
                    f"macro_f1={res['macro_f1']:.3f}  "
                    f"auc_present={res['auc_macro_present']:.3f}  "
                    f"ECE_micro={res['ece_micro']:.3f}  "
                    f"worst_site_acc={res['worst_site_acc']:.3f}"
                )

    df = pd.DataFrame(rows)
    return real_trained_model, df

# ============================================================
# Downstream task: Brain-age prediction
# Uses Y from the dataset loader, not metadata.
# Works with real / unrefined / refined through GeneratedStore.
# ============================================================

# class BrainAge3D(nn.Module):
#     """
#     Simple age-regression wrapper built from your existing
#     BrainCancerFeaturizer + BrainCancerRegressor.
#     """
#     def __init__(self, featurizer: Optional[BrainCancerFeaturizer] = None):
#         super().__init__()
#         self.featurizer = featurizer if featurizer is not None else BrainCancerFeaturizer()
#         self.regressor = BrainCancerRegressor()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         fmap = self.featurizer(x)
#         age_pred = self.regressor(fmap)   # [B]
#         return age_pred


def _prepare_age_targets(y) -> torch.Tensor:
    """
    Convert dataset Y to float tensor of shape [B].
    Handles:
      - [B]
      - [B,1]
      - [B, ...]  -> takes first column
    """
    if not torch.is_tensor(y):
        y = torch.as_tensor(y)

    y = y.float()
    if y.ndim == 0:
        y = y.unsqueeze(0)
    elif y.ndim > 1:
        y = y.view(y.shape[0], -1)[:, 0]

    return y


def get_inputs_and_age_targets(
    batch,
    source: str,
    store: GeneratedStore,
    device: torch.device,
):
    """
    Returns:
      x         : [B,1,D,H,W]
      age       : [B]
      global_ids: list[int]
    """
    image, y, _, global_ids = batch

    if torch.is_tensor(global_ids):
        global_ids = global_ids.long().cpu().tolist()
    else:
        global_ids = [int(v) for v in global_ids]

    if source == "real":
        x = prepare_batch_for_3dcnn(image.to(device, dtype=torch.float32, non_blocking=True))
    else:
        x = load_generated_batch_by_global_ids(store, global_ids, source, device)

    age = _prepare_age_targets(y).to(device, non_blocking=True)
    return x, age, global_ids


def regression_metrics(y_true: torch.Tensor, y_pred: torch.Tensor):
    y_true_np = y_true.detach().cpu().float().view(-1).numpy()
    y_pred_np = y_pred.detach().cpu().float().view(-1).numpy()

    err = y_pred_np - y_true_np
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    bias = float(np.mean(err))

    if len(y_true_np) >= 2 and np.std(y_true_np) > 1e-12 and np.std(y_pred_np) > 1e-12:
        pearson = float(np.corrcoef(y_true_np, y_pred_np)[0, 1])
    else:
        pearson = float("nan")

    try:
        r2 = float(r2_score(y_true_np, y_pred_np))
    except Exception:
        r2 = float("nan")

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "bias": bias,
        "pearson": pearson,
        "r2": r2,
        "n": int(len(y_true_np)),
    }

def weighted_regression_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    site_ids: torch.Tensor,
    site_weights: Optional[torch.Tensor] = None,
    loss_name: str = "mae",
):
    if loss_name.lower() == "mse":
        per_item = (pred - target) ** 2
    else:
        per_item = (pred - target).abs()

    if site_weights is None:
        return per_item.mean()

    w = site_weights.to(pred.device)[site_ids.long()]
    w = w / w.mean().clamp_min(1e-12)
    return (w * per_item).mean()

def train_age_on_source(
    model: nn.Module,
    optimizer: optim.Optimizer,
    dataloader,
    store: GeneratedStore,
    source: str,
    epochs: int = 10,
    max_batches: Optional[int] = None,
    loss_name: str = "mae",
    site_weights: Optional[torch.Tensor] = None,
):
    device = next(model.parameters()).device
    model.train()

    for ep in range(epochs):
        losses = []

        pbar = tqdm(dataloader, desc=f"age-train[{source}] ep {ep+1}/{epochs}", leave=False)
        for j, batch in enumerate(pbar):
            if max_batches is not None and j >= max_batches:
                break

            image, y, metadata, global_ids = batch
            site_ids = metadata[:, 0].to(device).long()

            x, age, _ = get_inputs_and_age_targets(
                (image, y, metadata, global_ids),
                source,
                store,
                device,
            )

            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = weighted_regression_loss(
                pred=pred,
                target=age,
                site_ids=site_ids,
                site_weights=site_weights,
                loss_name=loss_name,
            )
            loss.backward()
            optimizer.step()

            losses.append(float(loss.detach().cpu()))
            pbar.set_postfix(loss=np.mean(losses))

        print(f"[age train on {source}] epoch {ep+1}/{epochs} | "
              f"avg_{loss_name}={np.mean(losses):.4f} | n_batches={len(losses)}")
    
        
def train_age_on_source_group(
    model: nn.Module,
    optimizer: optim.Optimizer,
    dataloader,
    store: GeneratedStore,
    train_sources: List[str],
    epochs: int = 10,
    max_batches: Optional[int] = None,
    loss_name: str = "mae",
    site_weights: Optional[torch.Tensor] = None,
):
    """
    Train one age model using multiple input sources.

    Examples:
      train_sources=["real"]
      train_sources=["refined"]
      train_sources=["real", "refined"]
      train_sources=["real", "cross-alpha-0.5"]

    For a batch with B subjects and K sources, this creates K*B training samples.
    The age label is repeated for each source.
    """
    device = next(model.parameters()).device
    model.train()

    train_sources = list(train_sources)
    group_name = "+".join(train_sources)

    for ep in range(epochs):
        losses = []

        pbar = tqdm(
            dataloader,
            desc=f"age-train[{group_name}] ep {ep+1}/{epochs}",
            leave=False,
        )

        for j, batch in enumerate(pbar):
            if max_batches is not None and j >= max_batches:
                break

            image, y, metadata, global_ids = batch

            # Original age/site labels for this batch.
            age = _prepare_age_targets(y).to(device, non_blocking=True)
            site_ids = metadata[:, 0].to(device).long()

            xs = []
            ages = []
            sites = []

            for src in train_sources:
                x_src, age_src, _ = get_inputs_and_age_targets(
                    (image, y, metadata, global_ids),
                    src,
                    store,
                    device,
                )

                xs.append(x_src)
                ages.append(age_src)
                sites.append(site_ids)

            x_train = torch.cat(xs, dim=0)
            age_train = torch.cat(ages, dim=0)
            site_train = torch.cat(sites, dim=0)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x_train)

            loss = weighted_regression_loss(
                pred=pred,
                target=age_train,
                site_ids=site_train,
                site_weights=site_weights,
                loss_name=loss_name,
            )

            loss.backward()
            optimizer.step()

            losses.append(float(loss.detach().cpu()))
            pbar.set_postfix(loss=np.mean(losses))

        print(
            f"[age train on {group_name}] epoch {ep+1}/{epochs} | "
            f"avg_{loss_name}={np.mean(losses):.4f} | n_batches={len(losses)}"
        )


@torch.no_grad()
def eval_age_on_source(
    model: nn.Module,
    dataloader,
    store: GeneratedStore,
    source: str,
    split_name: str,
    max_batches: Optional[int] = None,
):
    device = next(model.parameters()).device
    model.eval()

    preds_all = []
    targets_all = []
    rows = []

    pbar = tqdm(dataloader, desc=f"age-eval[{split_name}:{source}]", leave=False)
    for j, batch in enumerate(pbar):
        if max_batches is not None and j >= max_batches:
            break

        x, age, global_ids = get_inputs_and_age_targets(batch, source, store, device)
        pred = model(x)

        preds_all.append(pred)
        targets_all.append(age)

        pred_cpu = pred.detach().cpu().float().view(-1).numpy()
        age_cpu = age.detach().cpu().float().view(-1).numpy()

        for i, gidx in enumerate(global_ids):
            rows.append({
                "split": split_name,
                "source": source,
                "global_idx": int(gidx),
                "y_true": float(age_cpu[i]),
                "y_pred": float(pred_cpu[i]),
                "abs_error": float(abs(pred_cpu[i] - age_cpu[i])),
            })

    preds_all = torch.cat(preds_all, dim=0)
    targets_all = torch.cat(targets_all, dim=0)

    metrics = regression_metrics(targets_all, preds_all)
    pred_df = pd.DataFrame(rows).sort_values("global_idx").reset_index(drop=True)
    return metrics, pred_df



def downstream_age_matrix(
    make_age_model_fn,
    loaders: Dict[str, DataLoader],
    store: GeneratedStore,
    epochs_task: int = 10,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    max_batches_train: Optional[int] = None,
    max_batches_eval: Optional[int] = None,
    loss_name: str = "mae",
    train_real_only: bool = True,
    eval_ood: bool = False,
    site_weights: Optional[torch.Tensor] = None,
    eval_split_names: Optional[Sequence[str]] = None,
):
    train_sources = ["real"] if train_real_only else (["real"] + store.available_sources)
    eval_sources = ["real"] + store.available_sources

    rows = []
    pred_tables = []
    real_trained_age_model = None
    if eval_split_names is None:
        eval_split_names = ["ood_val", "ood_test"]

    for train_source in train_sources:
        model = make_age_model_fn()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        train_age_on_source(
            model=model,
            optimizer=optimizer,
            dataloader=loaders["probe_train"],
            store=store,
            source=train_source,
            epochs=epochs_task,
            max_batches=max_batches_train,
            loss_name=loss_name,
            site_weights=site_weights,
        )

        if train_source == "real":
            real_trained_age_model = copy.deepcopy(model).eval()

        for id_split_name in ["id_holdout", "id_full"]:
            if id_split_name not in loaders:
                continue

            for eval_source in eval_sources:
                metrics, pred_df = eval_age_on_source(
                    model=model,
                    dataloader=loaders[id_split_name],
                    store=store,
                    source=eval_source,
                    split_name=id_split_name,
                    max_batches=max_batches_eval,
                )

                rows.append({
                    "train_source": train_source,
                    "eval_split": id_split_name,
                    "eval_source": eval_source,
                    **metrics,
                })

                pred_df["train_source"] = train_source
                pred_tables.append(pred_df)

                print(
                    f"[age train={train_source:10s} | split={id_split_name:10s} | eval={eval_source:10s}] "
                    f"MAE={metrics['mae']:.4f}  "
                    f"RMSE={metrics['rmse']:.4f}  "
                    f"Bias={metrics['bias']:.4f}  "
                    f"Pearson={metrics['pearson']:.4f}  "
                    f"R2={metrics['r2']:.4f}"
                )

        if eval_ood:
            # for split_name in ["ood_val", "ood_test"]:
            for split_name in eval_split_names:
                if split_name not in loaders:
                    continue

                for eval_source in eval_sources:
                    metrics, pred_df = eval_age_on_source(
                        model=model,
                        dataloader=loaders[split_name],
                        store=store,
                        source=eval_source,
                        split_name=split_name,
                        max_batches=max_batches_eval,
                    )

                    rows.append({
                        "train_source": train_source,
                        "eval_split": split_name,
                        "eval_source": eval_source,
                        **metrics,
                    })

                    pred_df["train_source"] = train_source
                    pred_tables.append(pred_df)

                    print(
                        f"[age train={train_source:10s} | split={split_name:8s} | eval={eval_source:10s}] "
                        f"MAE={metrics['mae']:.4f}  "
                        f"RMSE={metrics['rmse']:.4f}  "
                        f"Bias={metrics['bias']:.4f}  "
                        f"Pearson={metrics['pearson']:.4f}  "
                        f"R2={metrics['r2']:.4f}"
                    )

    summary_df = pd.DataFrame(rows)
    pred_df_all = pd.concat(pred_tables, ignore_index=True) if len(pred_tables) else pd.DataFrame()

    return real_trained_age_model, summary_df, pred_df_all


def downstream_age_multiseed_combined_experiments(
    make_age_model_fn,
    loaders: Dict[str, DataLoader],
    store: GeneratedStore,
    out_dir: str,
    seeds: Sequence[int] = (1337, 1338, 1339),
    generated_sources: Optional[List[str]] = None,
    epochs_task: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    max_batches_train: Optional[int] = None,
    max_batches_eval: Optional[int] = None,
    loss_name: str = "mae",
    site_weights: Optional[torch.Tensor] = None,
    eval_splits: Sequence[str] = ("ood_val", "ood_test"),
    also_eval_id: bool = True,
):
    """
    Standalone robust brain-age experiment.

    It runs multiple seeds and tests:
      1) self/source-only training:
           real -> real/generated eval
           refined -> real/generated eval
           cross -> real/generated eval
           cross-alpha-0.5 -> real/generated eval

      2) combined training:
           real + refined
           real + cross
           real + cross-alpha-0.5

    Evaluation:
      - OOD real
      - OOD generated sources
      - optionally id_holdout and id_full

    Saves:
      brain_age_multiseed_predictions.csv
      brain_age_multiseed_summary_raw.csv
      brain_age_multiseed_summary_meanstd.csv
      brain_age_multiseed_self_only_meanstd.csv
      brain_age_multiseed_real_plus_generated_meanstd.csv
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if generated_sources is None:
        generated_sources = store.available_sources[:]

    # Source-only models.
    source_only_train_groups = []
    source_only_train_groups.append(("real", ["real"]))

    for src in generated_sources:
        source_only_train_groups.append((src, [src]))

    # Combined models: real + each generated set.
    combined_train_groups = []
    for src in generated_sources:
        combined_train_groups.append((f"real+{src}", ["real", src]))

    train_groups = source_only_train_groups + combined_train_groups

    # Evaluate on real and all generated sources.
    eval_sources = ["real"] + generated_sources

    split_names = []
    if also_eval_id:
        split_names.extend(["id_holdout", "id_full"])
    split_names.extend(list(eval_splits))

    rows = []
    pred_tables = []

    for seed in seeds:
        print("\n" + "=" * 80)
        print(f"[multi-seed age] seed = {seed}")
        print("=" * 80)

        for train_group_name, train_sources in train_groups:
            print("\n" + "-" * 80)
            print(f"[multi-seed age] train_group={train_group_name} | sources={train_sources}")
            print("-" * 80)

            # Important: reset seed BEFORE model creation.
            set_seed(int(seed))

            model = make_age_model_fn()
            optimizer = optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )

            train_age_on_source_group(
                model=model,
                optimizer=optimizer,
                dataloader=loaders["probe_train"],
                store=store,
                train_sources=train_sources,
                epochs=epochs_task,
                max_batches=max_batches_train,
                loss_name=loss_name,
                site_weights=site_weights,
            )

            for split_name in split_names:
                if split_name not in loaders:
                    continue

                for eval_source in eval_sources:
                    metrics, pred_df = eval_age_on_source(
                        model=model,
                        dataloader=loaders[split_name],
                        store=store,
                        source=eval_source,
                        split_name=split_name,
                        max_batches=max_batches_eval,
                    )

                    row = {
                        "seed": int(seed),
                        "train_group": train_group_name,
                        "train_sources": "+".join(train_sources),
                        "eval_split": split_name,
                        "eval_source": eval_source,
                        **metrics,
                    }
                    rows.append(row)

                    pred_df["seed"] = int(seed)
                    pred_df["train_group"] = train_group_name
                    pred_df["train_sources"] = "+".join(train_sources)
                    pred_df["eval_source"] = eval_source
                    pred_tables.append(pred_df)

                    print(
                        f"[seed={seed} | train={train_group_name:24s} | "
                        f"split={split_name:10s} | eval={eval_source:18s}] "
                        f"MAE={metrics['mae']:.4f}  "
                        f"RMSE={metrics['rmse']:.4f}  "
                        f"Bias={metrics['bias']:.4f}  "
                        f"Pearson={metrics['pearson']:.4f}  "
                        f"R2={metrics['r2']:.4f}"
                    )

            # Avoid GPU memory accumulation across many models.
            del model, optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    raw_df = pd.DataFrame(rows)
    pred_df_all = pd.concat(pred_tables, ignore_index=True) if pred_tables else pd.DataFrame()

    raw_path = out_dir / "brain_age_multiseed_summary_raw.csv"
    pred_path = out_dir / "brain_age_multiseed_predictions.csv"

    raw_df.to_csv(raw_path, index=False)
    pred_df_all.to_csv(pred_path, index=False)

    metric_cols = ["mae", "mse", "rmse", "bias", "pearson", "r2", "n"]

    meanstd_df = (
        raw_df
        .groupby(["train_group", "train_sources", "eval_split", "eval_source"])[metric_cols]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )

    meanstd_path = out_dir / "brain_age_multiseed_summary_meanstd.csv"
    meanstd_df.to_csv(meanstd_path, index=False)

    # Easy-to-read subset 1: source-only/self style.
    self_groups = ["real"] + generated_sources
    self_only = raw_df[raw_df["train_group"].isin(self_groups)].copy()

    self_meanstd = (
        self_only
        .groupby(["train_group", "eval_split", "eval_source"])[metric_cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    self_meanstd.to_csv(
        out_dir / "brain_age_multiseed_self_only_meanstd.csv",
        index=False,
    )

    # Easy-to-read subset 2: real + generated, evaluated on real OOD.
    real_plus = raw_df[raw_df["train_group"].str.startswith("real+")].copy()
    real_plus_real_eval = real_plus[real_plus["eval_source"] == "real"].copy()

    real_plus_meanstd = (
        real_plus_real_eval
        .groupby(["train_group", "eval_split", "eval_source"])[metric_cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    real_plus_meanstd.to_csv(
        out_dir / "brain_age_multiseed_real_plus_generated_eval_real_meanstd.csv",
        index=False,
    )

    print(f"\nSaved raw summary to: {raw_path}")
    print(f"Saved predictions to: {pred_path}")
    print(f"Saved mean/std summary to: {meanstd_path}")

    return {
        "raw": raw_df,
        "predictions": pred_df_all,
        "meanstd": meanstd_df,
        "self_meanstd": self_meanstd,
        "real_plus_eval_real_meanstd": real_plus_meanstd,
    }



def plot_multiseed_combined_age_results(
    summary_raw_csv: str,
    out_dir: str,
):
    """
    Creates compact plots from brain_age_multiseed_summary_raw.csv.

    Main plots:
      1) Real+generated training evaluated on real OOD.
      2) Source-only/self training evaluated on matching source.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_raw_csv)

    # --------------------------------------------------------
    # Plot A: real + generated training, evaluated on real OOD.
    # --------------------------------------------------------
    sub = df[
        (df["train_group"].str.startswith("real+"))
        & (df["eval_source"] == "real")
        & (df["eval_split"].isin(["ood_val", "ood_test"]))
    ].copy()

    if len(sub) > 0:
        summary = (
            sub
            .groupby(["train_group", "eval_split"], as_index=False)
            .agg(
                mae_mean=("mae", "mean"),
                mae_std=("mae", "std"),
                r2_mean=("r2", "mean"),
                r2_std=("r2", "std"),
                bias_mean=("bias", "mean"),
                bias_std=("bias", "std"),
            )
        )

        train_groups = sorted(summary["train_group"].unique().tolist())
        splits = ["ood_val", "ood_test"]

        x = np.arange(len(train_groups))
        width = 0.8 / len(splits)

        plt.figure(figsize=(10, 4.8))
        for j, split in enumerate(splits):
            vals, errs = [], []
            for tg in train_groups:
                tmp = summary[(summary["train_group"] == tg) & (summary["eval_split"] == split)]
                if len(tmp):
                    vals.append(float(tmp["mae_mean"].iloc[0]))
                    errs.append(float(tmp["mae_std"].iloc[0]) if not pd.isna(tmp["mae_std"].iloc[0]) else 0.0)
                else:
                    vals.append(np.nan)
                    errs.append(0.0)

            offset = (j - (len(splits) - 1) / 2) * width
            plt.bar(x + offset, vals, width=width, yerr=errs, capsize=3, label=split)

        plt.xticks(x, train_groups, rotation=25, ha="right")
        plt.ylabel("MAE on real OOD")
        plt.title("Train on real + generated, evaluate on real OOD\nmean ± std over seeds")
        plt.grid(axis="y", alpha=0.25)
        plt.legend()
        plt.tight_layout()

        out_path = out_dir / "real_plus_generated_eval_real_ood_mae.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")

    # --------------------------------------------------------
    # Plot B: source-only/self-style training.
    # For each source, train_group == eval_source.
    # --------------------------------------------------------
    sub = df[
        (df["train_group"] == df["eval_source"])
        & (df["eval_split"].isin(["ood_val", "ood_test"]))
    ].copy()

    if len(sub) > 0:
        source_order = ["real", "unrefined", "refined", "cross", "cross-alpha-0.5"]
        sources = [s for s in source_order if s in sub["train_group"].unique()]
        splits = ["ood_val", "ood_test"]

        summary = (
            sub
            .groupby(["train_group", "eval_split"], as_index=False)
            .agg(
                mae_mean=("mae", "mean"),
                mae_std=("mae", "std"),
                r2_mean=("r2", "mean"),
                bias_mean=("bias", "mean"),
            )
        )

        x = np.arange(len(sources))
        width = 0.8 / len(splits)

        plt.figure(figsize=(10, 4.8))
        for j, split in enumerate(splits):
            vals, errs = [], []
            for src in sources:
                tmp = summary[(summary["train_group"] == src) & (summary["eval_split"] == split)]
                if len(tmp):
                    vals.append(float(tmp["mae_mean"].iloc[0]))
                    errs.append(float(tmp["mae_std"].iloc[0]) if not pd.isna(tmp["mae_std"].iloc[0]) else 0.0)
                else:
                    vals.append(np.nan)
                    errs.append(0.0)

            offset = (j - (len(splits) - 1) / 2) * width
            plt.bar(x + offset, vals, width=width, yerr=errs, capsize=3, label=split)

        plt.xticks(x, sources, rotation=25, ha="right")
        plt.ylabel("MAE")
        plt.title("Source-only/self training OOD MAE\nmean ± std over seeds")
        plt.grid(axis="y", alpha=0.25)
        plt.legend()
        plt.tight_layout()

        out_path = out_dir / "source_only_self_ood_mae.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")

# ============================================================
# Representation diagnostics
# These use ONLY the discriminator trained on REAL images.
# ============================================================
@torch.no_grad()
def collect_source_features(
    model: nn.Module,
    dataloader,
    store: GeneratedStore,
    sources: Optional[List[str]] = None,
):
    device = next(model.parameters()).device
    model.eval()

    featurizer = model.featurizer
    featurizer.eval()

    sources = sources or (["real"] + store.available_sources)

    feats = {src: [] for src in sources}
    doms = {src: [] for src in sources}

    for batch in tqdm(dataloader, desc="collect_features", leave=False):
        for src in sources:
            x, domains, _ = get_inputs_and_domains(batch, src, store, device)
            f = featurizer(x).detach()
            f = torch.flatten(f, start_dim=1)

            feats[src].append(f.cpu())
            doms[src].append(domains.cpu())

    for src in sources:
        feats[src] = torch.cat(feats[src], dim=0).numpy()
        doms[src] = torch.cat(doms[src], dim=0).numpy().astype(int)

    return feats, doms


def hsic_rbf(X, Y, sigma_x=None, sigma_y=None):
    def k_rbf(Z, s):
        Z2 = (Z ** 2).sum(1, keepdim=True)
        d2 = Z2 + Z2.t() - 2 * Z @ Z.t()
        if s is None:
            valid = d2[d2 > 0]
            if valid.numel() == 0:
                s = torch.tensor(1.0, device=Z.device)
            else:
                s = torch.sqrt(0.5 * torch.median(valid) / torch.log(torch.tensor(2.0, device=Z.device)))
        K = torch.exp(-d2 / (2 * s ** 2 + 1e-8))
        return K, s

    Kx, sigma_x = k_rbf(X, sigma_x)
    Ky, sigma_y = k_rbf(Y, sigma_y)
    H = torch.eye(X.size(0), device=X.device) - 1.0 / X.size(0)
    Kc = H @ Kx @ H
    Lc = H @ Ky @ H
    return (Kc * Lc).sum() / ((X.size(0) - 1) ** 2)


@torch.no_grad()
def hsic_perm_test(X, site_ids, n_perm=200):
    S = F.one_hot(site_ids.long(), num_classes=int(site_ids.max().item()) + 1).float().to(X.device)
    stat = hsic_rbf(X, S)
    worse = 0
    for _ in range(n_perm):
        idx = torch.randperm(X.size(0), device=X.device)
        s = hsic_rbf(X, S[idx])
        worse += (s >= stat).item()
    p = (worse + 1) / (n_perm + 1)
    return float(stat.detach().cpu()), float(p)


def coral_distance(A, B):
    A = torch.as_tensor(A, dtype=torch.float32)
    B = torch.as_tensor(B, dtype=torch.float32)
    Ca = torch.cov(A.T)
    Cb = torch.cov(B.T)
    return float(torch.norm(Ca - Cb, p="fro").item())


def mmd_rbf_np(X: np.ndarray, Y: np.ndarray, sigma: float = None) -> float:
    def pdist2(A, B):
        AA = (A ** 2).sum(1, keepdims=True)
        BB = (B ** 2).sum(1, keepdims=True)
        return AA + BB.T - 2 * A @ B.T

    if sigma is None:
        Z = np.concatenate([X, Y], axis=0)
        D = pdist2(Z, Z)
        med = np.median(D[D > 0])
        sigma = np.sqrt(0.5 * med / np.log(2.0 + 1e-12) + 1e-12)

    Kxx = np.exp(-pdist2(X, X) / (2 * sigma ** 2 + 1e-12))
    Kyy = np.exp(-pdist2(Y, Y) / (2 * sigma ** 2 + 1e-12))
    Kxy = np.exp(-pdist2(X, Y) / (2 * sigma ** 2 + 1e-12))

    m = X.shape[0]
    n = Y.shape[0]
    mmd2 = ((Kxx.sum() - np.trace(Kxx)) / (m * (m - 1) + 1e-12)
          + (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1) + 1e-12)
          - 2 * Kxy.mean())
    return float(mmd2)


def fid_from_features(mu1, cov1, mu2, cov2):
    diff = mu1 - mu2
    eps = 1e-6

    cov1_eps = cov1 + np.eye(cov1.shape[0]) * eps
    cov2_eps = cov2 + np.eye(cov2.shape[0]) * eps

    w1, v1 = np.linalg.eigh(cov1_eps)
    sqrt_cov1 = (v1 * np.sqrt(np.clip(w1, 0, None))) @ v1.T

    middle = sqrt_cov1 @ cov2_eps @ sqrt_cov1
    middle = (middle + middle.T) / 2.0

    wm, vm = np.linalg.eigh(middle)
    sqrt_middle = (vm * np.sqrt(np.clip(wm, 0, None))) @ vm.T

    tr = np.trace(cov1_eps) + np.trace(cov2_eps) - 2.0 * np.trace(sqrt_middle)
    return float(np.real(diff.dot(diff) + tr))


def fid_between_sets(X: np.ndarray, Y: np.ndarray) -> float:
    mu1, mu2 = X.mean(0), Y.mean(0)
    cov1, cov2 = np.cov(X, rowvar=False), np.cov(Y, rowvar=False)
    return fid_from_features(mu1, cov1, mu2, cov2)


def intersite_divergence(feats: np.ndarray, domains: np.ndarray) -> Dict[str, float]:
    sites = np.unique(domains)
    corals, mmds = [], []

    for i, si in enumerate(sites):
        Xi = feats[domains == si]
        for sj in sites[i + 1:]:
            Xj = feats[domains == sj]
            if len(Xi) < 2 or len(Xj) < 2:
                continue
            corals.append(coral_distance(Xi, Xj))
            mmds.append(mmd_rbf_np(Xi, Xj))

    return {
        "coral_mean": float(np.mean(corals)) if corals else np.nan,
        "mmd_mean": float(np.mean(mmds)) if mmds else np.nan,
        "pairs": len(corals),
    }


def per_site_fid_summary(real_feats: np.ndarray, gen_feats: np.ndarray, domains: np.ndarray):
    out = {}
    for s in np.unique(domains):
        Xr = real_feats[domains == s]
        Xg = gen_feats[domains == s]
        if len(Xr) < 2 or len(Xg) < 2:
            continue
        out[int(s)] = fid_between_sets(Xr, Xg)
    return out


def summarize_site_metric(d: Dict[int, float]):
    if len(d) == 0:
        return {"mean": np.nan, "std": np.nan, "worst": np.nan, "n_sites": 0}
    vals = np.array(list(d.values()), dtype=np.float64)
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "worst": float(np.max(vals)),
        "n_sites": int(len(vals)),
    }


def plot_pca_tsne(
    feats: Dict[str, np.ndarray],
    doms: Dict[str, np.ndarray],
    save_path: str,
    max_points_per_source: int = 800,
    random_state: int = 0,
):
    """
    Qualitative plot only. Uses consistent colors for domain and markers for source.
    """
    rng = np.random.default_rng(random_state)
    sources = list(feats.keys())

    xs = []
    ds = []
    ss = []

    for src in sources:
        X = feats[src]
        D = doms[src]

        if len(X) > max_points_per_source:
            keep = rng.choice(len(X), size=max_points_per_source, replace=False)
            X = X[keep]
            D = D[keep]

        xs.append(X)
        ds.append(D)
        ss.extend([src] * len(X))

    X = np.concatenate(xs, axis=0)
    D = np.concatenate(ds, axis=0)
    S = np.array(ss)

    n_pca = min(50, X.shape[1], X.shape[0] - 1)
    Xp = PCA(n_components=n_pca, random_state=random_state).fit_transform(X)

    X2 = TSNE(
        n_components=2,
        perplexity=min(30, max(5, (len(Xp) - 1) // 3)),
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    ).fit_transform(Xp)

    unique_domains = sorted(np.unique(D).tolist())
    cmap = plt.cm.get_cmap("tab20", len(unique_domains))
    color_map = {d: cmap(i) for i, d in enumerate(unique_domains)}
    marker_map = {
        "real": "o",
        "unrefined": "^",
        "refined": "s",
        "cross": "D",
        "cross-alpha-0.5": "P",
    }
    
    plt.figure(figsize=(9, 7))
    for src in sources:
        for d in unique_domains:
            m = (S == src) & (D == d)
            if not np.any(m):
                continue
            plt.scatter(
                X2[m, 0],
                X2[m, 1],
                s=18,
                alpha=0.65,
                marker=marker_map.get(src, "o"),
                color=color_map[d],
                edgecolors="k",
                linewidths=0.2,
                label=f"{src}-site{d}",
            )

    plt.title("PCA + t-SNE on discriminator features")
    plt.xticks([])
    plt.yticks([])
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", ncol=2, fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved feature plot to: {save_path}")


def run_representation_diagnostics(
    real_trained_model: nn.Module,
    loader_for_diagnostics,
    store: GeneratedStore,
    out_dir: str,
    run_tSNE: bool = True,
    hsic_n_perm: int = 200,
    max_hsic_samples: int = 800,
):
    """
    IMPORTANT:
      This uses ONLY the model trained on REAL images.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = ["real"] + store.available_sources
    feats, doms = collect_source_features(
        model=real_trained_model,
        dataloader=loader_for_diagnostics,
        store=store,
        sources=sources,
    )

    # ---------- HSIC ----------
    hsic_results = {}
    device = next(real_trained_model.parameters()).device

    for src in sources:
        X = feats[src]
        D = doms[src]

        if len(X) > max_hsic_samples:
            keep = np.random.default_rng(0).choice(len(X), size=max_hsic_samples, replace=False)
            X = X[keep]
            D = D[keep]

        Xt = torch.tensor(X, device=device, dtype=torch.float32)
        Dt = torch.tensor(D, device=device, dtype=torch.long)
        stat, p = hsic_perm_test(Xt, Dt, n_perm=hsic_n_perm)
        hsic_results[src] = {"hsic": stat, "pvalue": p}
        print(f"[HSIC] {src:10s} | stat={stat:.6g} | p={p:.4f}")

    # ---------- inter-site divergence ----------
    divergence_results = {}
    for src in sources:
        divergence_results[src] = intersite_divergence(feats[src], doms[src])
        print(f"[inter-site] {src:10s} | {divergence_results[src]}")

    # ---------- per-site cFID (real vs generated) ----------
    cfid_results = {}
    for src in store.available_sources:
        site_scores = per_site_fid_summary(feats["real"], feats[src], doms["real"])
        cfid_results[src] = {
            "per_site": site_scores,
            "summary": summarize_site_metric(site_scores),
        }
        print(f"[cFID] real vs {src:10s} | {cfid_results[src]['summary']}")

    # ---------- optional plot ----------
    if run_tSNE:
        plot_pca_tsne(
            feats=feats,
            doms=doms,
            save_path=str(out_dir / "pca_tsne_features.png"),
            max_points_per_source=800,
            random_state=0,
        )

    return {
        "hsic": hsic_results,
        "divergence": divergence_results,
        "cfid": cfid_results,
        "feats": feats,
        "domains": doms,
    }





def run_pairwise_feature_distances(
    feats: Dict[str, np.ndarray],
    doms: Dict[str, np.ndarray],
    pairs: List[Tuple[str, str]],
    out_dir: str,
):
    """
    Pairwise feature distances between all requested pairs.

    Saves:
      pairwise_feature_distances.csv
      pairwise_per_site_fid.csv
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    site_rows = []

    for a, b in pairs:
        if a not in feats or b not in feats:
            print(f"[pairwise features] skipping missing pair {a} vs {b}")
            continue

        Xa = feats[a]
        Xb = feats[b]

        # Assumes same dataloader ordering and matched samples.
        global_fid = fid_between_sets(Xa, Xb)
        global_mmd = mmd_rbf_np(Xa, Xb)
        global_coral = coral_distance(Xa, Xb)

        rows.append({
            "source_a": a,
            "source_b": b,
            "pair": f"{a}_vs_{b}",
            "fid": global_fid,
            "mmd": global_mmd,
            "coral": global_coral,
            "n_a": len(Xa),
            "n_b": len(Xb),
        })

        # Per-site FID. Use domains from source a; sources are collected
        # from the same dataloader/order, so domains should match.
        Da = doms[a]
        Db = doms[b]
        if not np.array_equal(Da, Db):
            print(f"[pairwise features] WARNING: domain arrays differ for {a} vs {b}")

        for site in sorted(np.unique(Da).tolist()):
            ma = Da == site
            mb = Db == site

            if ma.sum() < 2 or mb.sum() < 2:
                continue

            site_rows.append({
                "source_a": a,
                "source_b": b,
                "pair": f"{a}_vs_{b}",
                "site": int(site),
                "fid": fid_between_sets(Xa[ma], Xb[mb]),
                "mmd": mmd_rbf_np(Xa[ma], Xb[mb]),
                "coral": coral_distance(Xa[ma], Xb[mb]),
                "n_a": int(ma.sum()),
                "n_b": int(mb.sum()),
            })

    df = pd.DataFrame(rows)
    site_df = pd.DataFrame(site_rows)

    df.to_csv(out_dir / "pairwise_feature_distances.csv", index=False)
    site_df.to_csv(out_dir / "pairwise_per_site_feature_distances.csv", index=False)

    print(f"Saved pairwise feature distances to: {out_dir / 'pairwise_feature_distances.csv'}")
    print(f"Saved per-site feature distances to: {out_dir / 'pairwise_per_site_feature_distances.csv'}")

    return df, site_df

# ============================================================
# Model factory
# No DataParallel here.
# ============================================================
def make_model(device: torch.device, n_domains: int = 15):
    featurizer = BrainCancerFeaturizer()
    model = DANN3D(featurizer=featurizer, n_domains=n_domains)

    for m in model.modules():
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    return model.to(device)

# def make_age_model(device: torch.device):
#     model = BrainAge3D(featurizer=BrainCancerFeaturizer())

#     for m in model.modules():
#         if isinstance(m, (nn.Conv3d, nn.Linear)):
#             nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
#             if m.bias is not None:
#                 nn.init.zeros_(m.bias)

#     return model.to(device)


    # EXPORT_ROOT = "/rhome/ssafa013/bigdata/simple_unet_exports"
    # RESULTS_ROOT = "./probe_results_multi"

    # DATASET_SOURCES = ["refined", "unrefined", "cross"]

    # # Pairwise comparisons requested:
    # PAIRWISE_COMPARISONS = [
    #     ("real", "refined"),
    #     ("real", "unrefined"),
    #     ("real", "cross"),
    #     ("refined", "unrefined"),
    #     ("refined", "cross"),
    #     ("unrefined", "cross"),
    # ]

    # # Flags
    # RUN_CROSS_TRAINING = False
    # RUN_BRAIN_AGE = True
    # RUN_PAIRED_PSNR_SSIM = True
    # RUN_REPRESENTATION_DIAGNOSTICS = False
    # RUN_FEATURE_PAIRWISE_DISTANCES = True
    # RUN_TSNE = True
    
import matplotlib.pyplot as plt
import torch
import numpy as np
from pathlib import Path

# def debug_show_real_generated(loader, store, source_list=("unrefined", "refined", "cross"), max_items=5):
#     batch = next(iter(loader))
#     image, y, metadata, global_ids = batch

#     if torch.is_tensor(global_ids):
#         global_ids = global_ids.long().cpu().tolist()

#     real_bhwd = _real_batch_to_hwd_numpy(image)

#     for i in range(min(max_items, len(global_ids))):
#         gidx = int(global_ids[i])
#         age = float(y[i].view(-1)[0])
#         site = int(metadata[i, 0].item())

#         vols = {"real": real_bhwd[i]}

#         for src in source_list:
#             vols[src] = _generated_batch_to_hwd_numpy(store, [gidx], src)[0]

#         z = vols["real"].shape[-1] // 2

#         plt.figure(figsize=(4 * len(vols), 4))
#         for j, (src, vol) in enumerate(vols.items()):
#             plt.subplot(1, len(vols), j + 1)
#             plt.imshow(vol[:, :, z], cmap="gray")
#             plt.title(f"{src}\ngidx={gidx}, age={age:.1f}, site={site}")
#             plt.axis("off")
#         plt.tight_layout()
#         plt.savefig(save_path, dpi=300, bbox_inches="tight")
#         plt.close()

def save_debug_pairing_grid(
    dataloader,
    store,
    split_name: str,
    out_dir: str,
    sources=("unrefined", "refined", "cross"),
    n_items: int = 8,
    batch_index: int = 0,
):
    """
    Saves visual checks for real/generated pairing.

    For each selected global_idx, saves:
      real | unrefined | absdiff | refined | absdiff | cross | absdiff

    This helps check whether generated files are correctly paired with real images.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get one batch.
    batch = None
    for bidx, b in enumerate(dataloader):
        if bidx == batch_index:
            batch = b
            break

    if batch is None:
        raise RuntimeError(f"Could not get batch_index={batch_index} from {split_name}")

    image, y, metadata, global_ids = batch

    if torch.is_tensor(global_ids):
        global_ids = global_ids.long().cpu().tolist()
    else:
        global_ids = [int(v) for v in global_ids]

    real_bhwd = _real_batch_to_hwd_numpy(image)

    rows = []

    for i in range(min(n_items, len(global_ids))):
        gidx = int(global_ids[i])

        age = float(y[i].view(-1)[0]) if torch.is_tensor(y) else float(np.asarray(y[i]).reshape(-1)[0])
        domain_label = int(metadata[i, 0].item())

        if metadata.shape[1] > 4:
            original_site = float(metadata[i, 4].item())
        else:
            original_site = np.nan

        vols = {"real": real_bhwd[i]}

        for src in sources:
            if src not in store.sources:
                continue
            if not store.has(gidx, src):
                print(f"[debug pairing] missing source={src}, global_idx={gidx}")
                continue
            vols[src] = _generated_batch_to_hwd_numpy(store, [gidx], src)[0]

        real_vol = vols["real"]

        # Pick slice near center of nonzero foreground.
        mask = np.abs(real_vol) > 1e-6
        if mask.any():
            z = int(np.median(np.where(mask)[2]))
        else:
            z = real_vol.shape[-1] // 2

        # Shared display window from real image.
        vmin = float(np.percentile(real_vol, 1))
        vmax = float(np.percentile(real_vol, 99))
        if vmax <= vmin:
            vmin, vmax = float(real_vol.min()), float(real_vol.max())

        # Build columns: real, then each source and absdiff.
        n_cols = 1 + 2 * (len(vols) - 1)
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

        if n_cols == 1:
            axes = [axes]

        col = 0
        axes[col].imshow(real_vol[:, :, z], cmap="gray", vmin=vmin, vmax=vmax)
        axes[col].set_title(f"real\nidx={gidx}\nage={age:.1f}\nsite={original_site}")
        axes[col].axis("off")
        col += 1

        for src in sources:
            if src not in vols:
                continue

            gen = vols[src]
            diff = np.abs(real_vol - gen)

            # Basic foreground correlation.
            if mask.any():
                r_flat = real_vol[mask].reshape(-1)
                g_flat = gen[mask].reshape(-1)
            else:
                r_flat = real_vol.reshape(-1)
                g_flat = gen.reshape(-1)

            if np.std(r_flat) > 1e-8 and np.std(g_flat) > 1e-8:
                corr = float(np.corrcoef(r_flat, g_flat)[0, 1])
            else:
                corr = np.nan

            mae = float(np.mean(np.abs(real_vol - gen)))
            mse = float(np.mean((real_vol - gen) ** 2))

            rows.append({
                "split": split_name,
                "global_idx": gidx,
                "age": age,
                "domain_label": domain_label,
                "original_site": original_site,
                "source": src,
                "mae": mae,
                "mse": mse,
                "foreground_corr": corr,
                "file_path": str(store.path(gidx, src)),
            })

            axes[col].imshow(gen[:, :, z], cmap="gray", vmin=vmin, vmax=vmax)
            axes[col].set_title(f"{src}\nMAE={mae:.4f}\ncorr={corr:.3f}")
            axes[col].axis("off")
            col += 1

            axes[col].imshow(diff[:, :, z], cmap="magma")
            axes[col].set_title(f"|real-{src}|")
            axes[col].axis("off")
            col += 1

        plt.tight_layout()

        out_path = out_dir / f"{split_name}_gidx_{gidx}_z_{z}.png"
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close()

        print(f"[debug pairing] saved {out_path}")

    df = pd.DataFrame(rows)
    csv_path = out_dir / f"{split_name}_debug_pairing_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"[debug pairing] saved {csv_path}")

    return df

# ============================================================
# Main
# ============================================================
# if __name__ == "__main__":
    
#     # --------------------------------------------------------
#     # Settings to Run
#     # --------------------------------------------------------
#     # For debug only
    
#     # RESULTS_ROOT = "./probe_results_debug"

#     # RUN_CROSS_TRAINING = False
#     # RUN_BRAIN_AGE = False
#     # RUN_PAIRED_PSNR_SSIM = True

#     # RUN_REPRESENTATION_DIAGNOSTICS = False
#     # RUN_FEATURE_PAIRWISE_DISTANCES = False
#     # RUN_TSNE = False


#     # PAIRWISE_COMPARISONS = [
#     #     ("real", "refined"),
#     #     ("real", "unrefined"),
#     #     ("real", "cross"),
#     # ]
    
#     # Full run
#     # RESULTS_ROOT = "./probe_results_proposed_split/refined_lambda004"
    
#     # RUN_CROSS_TRAINING = True
#     # RUN_BRAIN_AGE = False
#     # RUN_BRAIN_AGE_MULTI_SEED_COMBINED = False
#     # RUN_PAIRED_PSNR_SSIM = False

#     # RUN_REPRESENTATION_DIAGNOSTICS = False
#     # RUN_FEATURE_PAIRWISE_DISTANCES = False
#     # RUN_TSNE = False
#     # ============================================================
#     # Experiment selection
#     # ============================================================

#     RUN_MODE = os.environ.get(
#         "RUN_MODE",
#         "cross_training",
#     ).strip().lower()

#     RESULTS_ROOT = os.environ.get(
#         "RESULTS_ROOT",
#         "./probe_results_proposed_split/refined_lambda004",
#     )

#     VALID_RUN_MODES = {
#         "cross_training",
#         "brain_age",
#         "brain_age_multiseed",
#         "paired_psnr_ssim",
#         "representation_diagnostics",
#         "feature_pairwise_distances",
#         "intensity_sanity",
#         "image_features",
#     }

#     if RUN_MODE not in VALID_RUN_MODES:
#         raise ValueError(
#             f"Unknown RUN_MODE={RUN_MODE!r}. "
#             f"Expected one of: {sorted(VALID_RUN_MODES)}"
#         )

#     RUN_CROSS_TRAINING = RUN_MODE == "cross_training"
#     RUN_BRAIN_AGE = RUN_MODE == "brain_age"
#     RUN_BRAIN_AGE_MULTI_SEED_COMBINED = RUN_MODE == "brain_age_multiseed"
#     RUN_PAIRED_PSNR_SSIM = RUN_MODE == "paired_psnr_ssim"
#     RUN_REPRESENTATION_DIAGNOSTICS = RUN_MODE == "representation_diagnostics"
#     RUN_FEATURE_PAIRWISE_DISTANCES = RUN_MODE == "feature_pairwise_distances"
#     RUN_INTENSITY_SANITY = RUN_MODE == "intensity_sanity"
#     RUN_IMAGE_FEATURES_BY_AGE = RUN_MODE == "image_features"

#     # t-SNE is part of representation diagnostics.
#     RUN_TSNE = RUN_MODE == "representation_diagnostics"

#     print(f"RUN_MODE: {RUN_MODE}")
#     print(f"RESULTS_ROOT: {RESULTS_ROOT}")

#     PAIRWISE_COMPARISONS = [
#         # compare each generated source to real
#         ("real", "unrefined"),
#         ("real", "refined"),
#         ("real", "cross"),
#         # ("real", "cross-alpha-0.5"),

#         # compare alpha to endpoints
#         # ("cross", "cross-alpha-0.5"),
#         # ("cross-alpha-0.5", "refined"),

#         # optional old endpoint comparisons
#         ("refined", "unrefined"),
#         ("refined", "cross"),
#         ("unrefined", "cross"),
#     ]
    
#     # RUN_INTENSITY_SANITY = False
#     SANITY_MAX_BATCHES = 2
    
#     # RUN_IMAGE_FEATURES_BY_AGE = False
#     IMAGE_FEATURE_MAX_BATCHES = None   # use 2 or 5 for debug
    
#     # --------------------------------------------------------
#     # Choose ONE store config
#     # --------------------------------------------------------

#     EXPORT_ROOT = "/rhome/ssafa013/bigdata/simple_unet_exports/proposed_split"

#     DATASET_SOURCES = [
#         "refined",
#         "unrefined",
#         "cross",
#         # "cross-alpha-0.5",
#     ]

#     store = GeneratedStore.from_multi_folder_export(
#         export_root=EXPORT_ROOT,
#         sources=DATASET_SOURCES,
#         prefix="recon",
#         require_nonempty=True,
#     )

#     store.summary()

#     # --------------------------------------------------------
#     # Dataset + loaders
#     # train_total_n=None => dynamic from what exists on disk
#     # --------------------------------------------------------
#     dataset = OpenBHBDataset()
#     N_SITE_CLASSES = dataset.num_train_domains
#     print("N_SITE_CLASSES:", N_SITE_CLASSES)
#     batch_size = 16
#     epochs_probe = 30

#     loaders, pool_info = build_probe_loaders(
#         dataset=dataset,
#         store=store,
#         batch_size=batch_size,
#         train_frac=0.9,
#         num_workers=1,          # your machine warned max suggested is 1
#         pin_memory=True,
#         site_weight_power=0.5,  # try 0.5 for milder balancing
#         n_site_classes=N_SITE_CLASSES,
#     )
    
#     # loaders, controlled_ood_info = add_age_controlled_ood_loaders(
#     #     loaders=loaders,
#     #     dataset=dataset,
#     #     store=store,
#     #     batch_size=batch_size,
#     #     num_workers=1,
#     #     pin_memory=True,
#     #     age_mode="meanpm2std",   # first main experiment
#     # )
    
#     loaders, controlled_ood_info = add_age_limited_ood_loaders(
#         loaders=loaders,
#         dataset=dataset,
#         store=store,
#         batch_size=batch_size,
#         num_workers=1,
#         pin_memory=True,
#         min_age=0.0,
#         max_age=35.0,
#     )

#     controlled_info_path = Path(RESULTS_ROOT) / "controlled_ood_info.csv"
#     Path(RESULTS_ROOT).mkdir(parents=True, exist_ok=True)

#     pd.DataFrame([
#         {"key": k, **v} if isinstance(v, dict) else {"key": k, "value": v}
#         for k, v in controlled_ood_info.items()
#     ]).to_csv(controlled_info_path, index=False)

#     print(f"Saved controlled OOD info to: {controlled_info_path}")

#     train_site_weights = pool_info["train_site_weights"]
#     print("train_site_weights:", train_site_weights)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    
    
#     DOMAIN_EVAL_SPLITS = [
#         "id_holdout",
#         "id_full",
#     ]

#     AGE_EVAL_SPLITS = [
#         # "ood_val",
#         # "ood_test",
#         # "ood_val_age_similar",
#         # "ood_test_age_similar",
#         # "ood_val_age_shift",
#         # "ood_test_age_shift",
#         "ood_val_age_le35",
#         "ood_test_age_le35",
#     ]

#     IMAGE_EVAL_SPLITS = [
#         "id_full",
#         "ood_val",
#         "ood_test",
#         "ood_val_age_similar",
#         "ood_test_age_similar",
#         "ood_val_age_shift",
#         "ood_test_age_shift",
#     ]
    
    
#     # debug_show_real_generated(loaders["id_full"], store)
#     # debug_show_real_generated(loaders["ood_val"], store)
#     # debug_show_real_generated(loaders["ood_test"], store)
#     # debug_dir = Path(RESULTS_ROOT) / "debug_pairing"

#     # save_debug_pairing_grid(
#     #     dataloader=loaders["id_full"],
#     #     store=store,
#     #     split_name="id_full",
#     #     out_dir=str(debug_dir),
#     #     sources=("unrefined", "refined", "cross"),
#     #     n_items=8,
#     # )

#     # save_debug_pairing_grid(
#     #     dataloader=loaders["ood_val"],
#     #     store=store,
#     #     split_name="ood_val",
#     #     out_dir=str(debug_dir),
#     #     sources=("unrefined", "refined", "cross"),
#     #     n_items=8,
#     # )

#     # save_debug_pairing_grid(
#     #     dataloader=loaders["ood_test"],
#     #     store=store,
#     #     split_name="ood_test",
#     #     out_dir=str(debug_dir),
#     #     sources=("unrefined", "refined", "cross"),
#     #     n_items=8,
#     # )
#     # exit(0)
#     # --------------------------------------------------------
#     # Multi-seed + real/generated combined brain-age experiment
#     # --------------------------------------------------------
#     if RUN_BRAIN_AGE_MULTI_SEED_COMBINED:
#         print("\n== Multi-seed combined brain-age experiments ==")

#         multi_age_dir = Path(RESULTS_ROOT) / "brain_age_multiseed_combined"

#         multi_age_results = downstream_age_multiseed_combined_experiments(
#             # make_age_model_fn=lambda: make_age_model(device),
#             make_age_model_fn=lambda: make_ukbb_age_model(device, normalize_input=False),
#             loaders=loaders,
#             store=store,
#             out_dir=str(multi_age_dir),

#             # Three seeds first. Later you can expand to 5 or 10.
#             seeds=(1337, 1338, 1339),

#             generated_sources=[
#                 # "unrefined",
#                 # "refined",
#                 "cross",
#                 # "cross-alpha-0.5",
#             ],

#             epochs_task=30,
#             lr=1e-3,
#             weight_decay=0.0,
#             max_batches_train=None,
#             max_batches_eval=None,
#             loss_name="mae",
#             site_weights=train_site_weights,

#             # eval_splits=("ood_val", "ood_test"),
#             eval_splits=(
#                 "ood_val",
#                 "ood_test",
#                 # "ood_val_age_similar",
#                 # "ood_test_age_similar",
#                 # "ood_val_age_shift",
#                 # "ood_test_age_shift",
#                 "ood_val_age_le35",
#                 "ood_test_age_le35",
#             ),
#             also_eval_id=True,
#         )

#         plot_multiseed_combined_age_results(
#             summary_raw_csv=str(multi_age_dir / "brain_age_multiseed_summary_raw.csv"),
#             out_dir=str(multi_age_dir / "plots"),
#         )

#         print("\n[multi-seed age] mean/std summary:")
#         print(multi_age_results["meanstd"].head())    

#     # --------------------------------------------------------
#     # Basic intensity sanity checks BEFORE expensive metrics
#     # --------------------------------------------------------
#     if RUN_INTENSITY_SANITY:
#         print("\n== Intensity sanity checks ==")

#         sanity_dir = Path(RESULTS_ROOT) / "intensity_sanity"

#         sanity = run_all_intensity_sanity_checks(
#             loaders=loaders,
#             store=store,
#             out_dir=str(sanity_dir),
#             sources=["real"] + store.available_sources,
#             pairwise_comparisons=PAIRWISE_COMPARISONS,
#             split_names=IMAGE_EVAL_SPLITS, #split_names=["id_full", "ood_val", "ood_test"],
#             max_batches=SANITY_MAX_BATCHES,
#         )

#         plot_sanity_summary_figures(
#             sanity_dir=str(sanity_dir),
#             out_dir=str(sanity_dir / "plots"),
#         )
#         print("\n[sanity] intensity summary:")
#         print(sanity["summary"])

#         print("\n[sanity] pairwise difference summary:")
#         print(sanity["pair_summary"])
        
#     results_dir = Path(RESULTS_ROOT)
#     results_dir.mkdir(parents=True, exist_ok=True)

#     # --------------------------------------------------------
#     # Image feature checks by age bin
#     # --------------------------------------------------------
#     if RUN_IMAGE_FEATURES_BY_AGE:
#         print("\n== Image feature checks by age bin ==")

#         image_feature_dir = Path(RESULTS_ROOT) / "image_features_by_age"

#         image_feature_results = run_all_image_feature_checks(
#             loaders=loaders,
#             store=store,
#             out_dir=str(image_feature_dir),
#             sources=["real"] + store.available_sources,
#             split_names=IMAGE_EVAL_SPLITS, #split_names=["id_full", "ood_val", "ood_test"],
#             max_batches=IMAGE_FEATURE_MAX_BATCHES,
#         )

#         print("\n[image features] summary:")
#         print(image_feature_results["summary"].head())


#     # --------------------------------------------------------
#     # Cross-training matrix
#     # --------------------------------------------------------
#     real_model = None

#     if RUN_CROSS_TRAINING or RUN_REPRESENTATION_DIAGNOSTICS or RUN_FEATURE_PAIRWISE_DISTANCES:
#         print("\n== Cross-training matrix / real-trained domain model ==")

#         real_model, xmat_df = cross_training_matrix(
#             make_model_fn=lambda: make_model(device, N_SITE_CLASSES),
#             loaders=loaders,
#             store=store,
#             epochs_probe=epochs_probe,
#             lr=1e-3,
#             n_classes=N_SITE_CLASSES,
#             max_batches_train=None,
#             max_batches_eval=None,
#             class_weights=train_site_weights,
#             eval_split_names=DOMAIN_EVAL_SPLITS,

#         )

#         if RUN_CROSS_TRAINING:
#             xmat_path = results_dir / "cross_training_matrix.csv"
#             xmat_df.to_csv(xmat_path, index=False)
#             print(f"Saved cross-training results to: {xmat_path}")
        
#         plot_domain_invariance_from_xmat(
#             xmat_csv=str(results_dir / "cross_training_matrix.csv"),
#             out_dir=str(results_dir / "domain_plots"),
#             train_source="real",
#             n_classes=N_SITE_CLASSES
#         )

#     # --------------------------------------------------------
#     # Brain age
#     # --------------------------------------------------------
#     if RUN_BRAIN_AGE:
#         print("\n== Downstream task: brain age prediction ==")

#         real_age_model, age_df, age_pred_df = downstream_age_matrix(
#             # make_age_model_fn=lambda: make_age_model(device),
#             make_age_model_fn=lambda: make_ukbb_age_model(device, normalize_input=False),
#             loaders=loaders,
#             store=store,
#             epochs_task=30,
#             lr=1e-3,
#             weight_decay=0.0,
#             max_batches_train=None,
#             max_batches_eval=None,
#             loss_name="mae",
#             train_real_only=False,
#             eval_ood=True,
#             site_weights=train_site_weights,
#             eval_split_names=AGE_EVAL_SPLITS,
#         )

#         age_df.to_csv(results_dir / "brain_age_downstream_summary.csv", index=False)
#         age_pred_df.to_csv(results_dir / "brain_age_downstream_predictions.csv", index=False)

#         print(f"Saved age downstream summary to: {results_dir / 'brain_age_downstream_summary.csv'}")
#         print(f"Saved age downstream predictions to: {results_dir / 'brain_age_downstream_predictions.csv'}")

#     # --------------------------------------------------------
#     # Pairwise PSNR / SSIM
#     # --------------------------------------------------------
#     if RUN_PAIRED_PSNR_SSIM:
#         print("\n== Pairwise PSNR / SSIM ==")

#         pair_dir = results_dir / "pairwise_psnr_ssim"
#         pair_dir.mkdir(parents=True, exist_ok=True)

#         all_summaries = []

#         # for split_name in ["id_full", "ood_val", "ood_test"]:
#         for split_name in IMAGE_EVAL_SPLITS:
#             if split_name not in loaders:
#                 continue

#             for source_a, source_b in PAIRWISE_COMPARISONS:
#                 csv_name = f"psnr_ssim_{split_name}_{source_a}_vs_{source_b}.csv"
#                 df_pair, summary_pair, site_pair = compute_pairwise_psnr_ssim_for_loader(
#                     dataloader=loaders[split_name],
#                     store=store,
#                     source_a=source_a,
#                     source_b=source_b,
#                     split_name=split_name,
#                     out_csv=str(pair_dir / csv_name),
#                     max_batches=None,
#                     data_range_mode="pair",
#                     fixed_data_range=None,
#                     ssim_win_size=7,
#                 )
#                 all_summaries.append(summary_pair)

#         psnr_summary_df = pd.DataFrame(all_summaries)
#         psnr_summary_df.to_csv(pair_dir / "pairwise_psnr_ssim_summary.csv", index=False)
#         plot_pairwise_metric_bars(
#             psnr_summary_df,
#             out_dir=str(pair_dir),
#             prefix="pairwise_psnr_ssim",
#         )

#         print(f"Saved pairwise PSNR/SSIM summary to: {pair_dir / 'pairwise_psnr_ssim_summary.csv'}")

#     # --------------------------------------------------------
#     # Representation diagnostics and pairwise feature distances
#     # --------------------------------------------------------
#     if RUN_REPRESENTATION_DIAGNOSTICS or RUN_FEATURE_PAIRWISE_DISTANCES:
#         print("\n== Diagnostics using REAL-trained discriminator ==")

#         diag_loader = loaders["id_full"]
#         diag = run_representation_diagnostics(
#             real_trained_model=real_model,
#             loader_for_diagnostics=diag_loader,
#             store=store,
#             out_dir=str(results_dir / "diagnostics_real_model"),
#             run_tSNE=RUN_TSNE,
#             hsic_n_perm=200,
#             max_hsic_samples=800,
#         )

#         if RUN_FEATURE_PAIRWISE_DISTANCES:
#             run_pairwise_feature_distances(
#                 feats=diag["feats"],
#                 doms=diag["domains"],
#                 pairs=PAIRWISE_COMPARISONS,
#                 out_dir=str(results_dir / "diagnostics_real_model"),
#             )
            
#     print("Done.")
    
    
#     # --------------------------------------------------------
#     # Cross-training matrix
#     # --------------------------------------------------------
#     # print("\n== Cross-training matrix ==")

#     # real_model, xmat_df = cross_training_matrix(
#     #     make_model_fn=lambda: make_model(device),
#     #     loaders=loaders,
#     #     store=store,
#     #     epochs_probe=epochs_probe,
#     #     lr=1e-3,
#     #     n_classes=15,
#     #     max_batches_train=None,
#     #     max_batches_eval=None,
#     #     class_weights=train_site_weights,
#     # )

#     # results_dir = Path("./probe_results")
#     # results_dir.mkdir(parents=True, exist_ok=True)
#     # # xmat_df.to_csv(results_dir / "cross_training_matrix.csv", index=False)
#     # # print(f"Saved cross-training results to: {results_dir / 'cross_training_matrix.csv'}")

#     # # --------------------------------------------------------
#     # # Downstream task: brain age prediction
#     # # --------------------------------------------------------
#     # print("\n== Downstream task: brain age prediction ==")

#     # real_age_model, age_df, age_pred_df = downstream_age_matrix(
#     #     make_age_model_fn=lambda: make_age_model(device),
#     #     loaders=loaders,
#     #     store=store,
#     #     epochs_task=30,
#     #     lr=1e-3,
#     #     weight_decay=0.0,
#     #     max_batches_train=None,
#     #     max_batches_eval=None,
#     #     loss_name="mae",
#     #     train_real_only=False,
#     #     eval_ood=True,
#     #     site_weights=train_site_weights,
#     # )

#     # age_df.to_csv(results_dir / "brain_age_downstream_summary.csv", index=False)
#     # age_pred_df.to_csv(results_dir / "brain_age_downstream_predictions.csv", index=False)

#     # print(f"Saved age downstream summary to: {results_dir / 'brain_age_downstream_summary.csv'}")
#     # print(f"Saved age downstream predictions to: {results_dir / 'brain_age_downstream_predictions.csv'}")

#     # # --------------------------------------------------------
#     # # Diagnostics using ONLY the discriminator trained on real
#     # # You can use id_full, ood_val, or ood_test here.
#     # # --------------------------------------------------------
#     # print("\n== Diagnostics using REAL-trained discriminator ==")

#     # diag_loader = loaders["id_full"]   # change to loaders["ood_test"] if you prefer
#     # diag = run_representation_diagnostics(
#     #     real_trained_model=real_model,
#     #     loader_for_diagnostics=diag_loader,
#     #     store=store,
#     #     out_dir=str(results_dir / "diagnostics_real_model"),
#     #     run_tSNE=True,
#     #     hsic_n_perm=200,
#     #     max_hsic_samples=800,
#     # )
    
    
#     # # --------------------------------------------------------
#     # # Paired PSNR/SSIM on ID full (real vs generated)
#     # # Uses the same global_idx pairing for real vs generated.
#     # # data_range_mode="pair" means the PSNR data range is computed per-pair
#     # # --------------------------------------------------------
    
#     # print("\n== Paired PSNR / SSIM on ID full ==")

#     # psnr_df, psnr_summary, psnr_site = compute_psnr_ssim_for_loader(
#     #     dataloader=loaders["id_full"],
#     #     store=store,
#     #     source="unrefined",
#     #     split_name="id_full",
#     #     out_csv=str(results_dir / "psnr_ssim_id_full_unrefined.csv"),
#     #     max_batches=None,
#     #     data_range_mode="pair",   # or "real"
#     #     fixed_data_range=None,
#     #     ssim_win_size=7,
#     # )
    
#     # if "refined" in store.available_sources:
#     #     psnr_df_ref, psnr_summary_ref, psnr_site_ref = compute_psnr_ssim_for_loader(
#     #         dataloader=loaders["id_full"],
#     #         store=store,
#     #         source="refined",
#     #         split_name="id_full",
#     #         out_csv=str(results_dir / "psnr_ssim_id_full_refined.csv"),
#     #     )
#     #     print("Refined PSNR/SSIM summary:", psnr_summary_ref)

#     # print("PSNR/SSIM summary:", psnr_summary)

#     # Optional: save compact summaries
#     # with open(results_dir / "diagnostics_real_model" / "summary.txt", "w") as f:
#     #     f.write("HSIC\n")
#     #     for k, v in diag["hsic"].items():
#     #         f.write(f"{k}: {v}\n")
#     #     f.write("\nINTER-SITE DIVERGENCE\n")
#     #     for k, v in diag["divergence"].items():
#     #         f.write(f"{k}: {v}\n")
#     #     f.write("\nCFID\n")
#     #     for k, v in diag["cfid"].items():
#     #         f.write(f"{k}: {v['summary']}\n")

