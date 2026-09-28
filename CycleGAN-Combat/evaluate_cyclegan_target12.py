#!/usr/bin/env python
"""
evaluate_cyclegan_target12.py
=============================

Evaluation for the CycleGAN target-site harmonization source ``cyclegan_target12``,
whose outputs live in PER-SPLIT subfolders (not one-folder-per-source):

    <export_root>/
      reconstruction_index.csv
      ood_val/   recon_XXXXXXX.nii.gz
      ood_test/  recon_XXXXXXX.nii.gz

It reuses the building blocks from your big evaluation script (imported below).

WHY THIS IS DIFFERENT from the refined/unrefined/cross evaluator
----------------------------------------------------------------
CycleGAN only harmonized the OOD splits, so:
  * there are NO cyclegan_target12 images for the train sites, therefore
    the age/domain probe is TRAINED ON REAL and cyclegan is only used at EVAL
    time on OOD val/test;
  * the site/domain CROSS-TRAINING MATRIX is intentionally NOT run here. That is
    an in-distribution experiment (it classifies TRAIN sites; OOD subjects carry
    domain_site == -1), and it also needs generated versions of train-site images
    to measure residual site signal. To enable it, re-export with
    ``--splits train`` (so ID splits have cyclegan images) and then the existing
    ``cross_training_matrix`` / ``run_representation_diagnostics`` can be used
    the same way as for the DDPM sources.

Experiments included (all real-vs-cyclegan on OOD):
  RUN_MODE=intensity_sanity   -> intensity stats + real-vs-cyclegan diff stats
  RUN_MODE=paired_psnr_ssim   -> paired PSNR/SSIM (real vs cyclegan)
  RUN_MODE=image_features     -> image features by age bin
  RUN_MODE=brain_age          -> train UKBB age model on REAL, eval real vs cyclegan
  RUN_MODE=all                -> all of the above

SETUP: place this file in the SAME directory as your big evaluation script and
change ``probe_eval`` below to that script's filename (without .py).
"""

import os
import re
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # headless-safe; must precede the pyplot import in probe_eval

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

# Make DataLoader.py / ukbb_brain_age_model importable (mirror the shared script).
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==== reuse building blocks from your pasted evaluation script ================
# CHANGE `probe_eval` to the filename (without .py) of that big script.
from probe_eval import (
    GeneratedStore,
    WithGlobalIndex,
    build_eligible_local_pool,
    split_indices_by_site,
    compute_site_class_weights_from_local_indices,
    add_age_limited_ood_loaders,
    run_all_intensity_sanity_checks,
    compute_pairwise_psnr_ssim_for_loader,
    plot_pairwise_metric_bars,
    run_all_image_feature_checks,
    train_age_on_source,
    eval_age_on_source,
    downstream_age_multiseed_combined_experiments,
    plot_multiseed_combined_age_results,
    make_model,
    train_on_source,
    eval_on_source,
    set_seed,
    SEED,
)
from DataLoader import OpenBHBDataset
from ukbb_brain_age_model import make_age_model as make_ukbb_age_model

# Source name is env-driven so this same evaluator works for any harmonized export
# (e.g. SOURCE=combat EXPORT_ROOT=/rhome/ssafa013/bigdata/data/openBHB_v1.1_harmonized).
SOURCE = os.environ.get("SOURCE", "cyclegan_target12")


# --------------------------------------------------------------------------- #
# Store: one source built from per-split subfolders (keyed by global_idx)
# --------------------------------------------------------------------------- #
def build_cyclegan_store(export_root, source=SOURCE,
                         splits=("ood_val", "ood_test"), prefix="recon"):
    """Build a GeneratedStore with a single source (cyclegan_target12).

    Prefers reconstruction_index.csv (absolute ``file_path`` covers whatever
    layout you used); falls back to globbing ``<export_root>/<split>/recon_*.nii.gz``
    (and, if those subfolders don't exist, a flat ``<export_root>/recon_*.nii.gz``).
    global_idx is unique across splits, so combining ood_val + ood_test into one
    {global_idx: path} map is unambiguous.
    """
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
        print(f"[cyclegan store] read {len(mapping)} entries from {csv_path.name}")
    else:
        pat = re.compile(rf"^{re.escape(prefix)}_(\d+)\.nii\.gz$")
        search_dirs = [export_root / sp for sp in splits] + [export_root]
        for d in search_dirs:
            if not d.exists():
                continue
            for p in d.glob(f"{prefix}_*.nii.gz"):
                m = pat.match(p.name)
                if m:
                    mapping[int(m.group(1))] = p

    if not mapping:
        raise RuntimeError(f"No cyclegan recons found under {export_root}")
    print(f"[cyclegan store] {source}: {len(mapping)} files")
    return GeneratedStore(sources={source: mapping})


def parse_source_specs(spec_str):
    """Parse "name=root,name=root" into an ordered dict {source_name: export_root}."""
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
        raise ValueError("No sources parsed from spec string.")
    return specs


def build_multi_source_store(specs, prefix="recon"):
    """Build ONE GeneratedStore holding several harmonized sources at once.

    specs: {source_name: export_root}. Each is loaded via build_cyclegan_store and
    merged, so a single store can serve e.g. real vs cyclegan_target12 vs combat.
    """
    sources = {}
    for name, root in specs.items():
        single = build_cyclegan_store(root, source=name, prefix=prefix)
        sources[name] = single.sources[name]
    return GeneratedStore(sources=sources)


# --------------------------------------------------------------------------- #
# Loaders: REAL train (probe) + OOD filtered to cyclegan availability
# --------------------------------------------------------------------------- #
def build_cyclegan_loaders(dataset, store, source=SOURCE, batch_size=16,
                           train_frac=0.9, num_workers=1, pin_memory=True,
                           site_weight_power=0.5, n_site_classes=15,
                           require_gen_on_train=False, gen_sources=None):
    """
    probe_train / id_holdout / id_full  -> TRAIN images.
        require_gen_on_train=False -> all real train (harmonized not needed on train).
        require_gen_on_train=True  -> only train subjects that ALSO have a harmonized
            image for EVERY source in ``gen_sources`` (intersection). Needed to TRAIN
            on a harmonized set or to EVAL harmonized sources on the id splits.
    ood_val / ood_test                  -> filtered to (all) gen_sources availability.

    gen_sources: list of source names to require; defaults to ``[source]``. Pass the
        full list (e.g. ["cyclegan_target12", "combat"]) for a multi-source matrix.
    """
    gsl = list(gen_sources) if gen_sources else [source]
    splits = {
        "train": dataset.get_subset("train"),
        "val": dataset.get_subset("val"),
        "test": dataset.get_subset("test"),
    }

    def mk(subset, idxs, shuffle):
        return DataLoader(Subset(subset, idxs), batch_size=batch_size,
                          shuffle=shuffle, drop_last=False,
                          num_workers=num_workers, pin_memory=pin_memory)

    loaders = {}

    # ---------- TRAIN pool ----------
    if require_gen_on_train:
        # Only train subjects that have a harmonized image for ALL gen_sources.
        train_wrapped, eligible_by_site_train, eligible_train_local = build_eligible_local_pool(
            splits["train"], store, gsl)
        if len(eligible_train_local) == 0:
            raise RuntimeError(
                f"require_gen_on_train=True but no train subjects covered by ALL of "
                f"{gsl}. Make sure each source has a --splits train export.")
        idx_train, idx_holdout = split_indices_by_site(
            eligible_by_site_train, frac_train=train_frac, seed=42)
        idx_id_full = eligible_train_local
    else:
        # All real train subjects (cyclegan not needed on train).
        train_wrapped = WithGlobalIndex(splits["train"])
        sites = splits["train"].metadata_array[:, 0].to(torch.int64).cpu().numpy()
        eligible_by_site_train = defaultdict(list)
        for local_idx in range(len(train_wrapped)):
            eligible_by_site_train[int(sites[local_idx])].append(local_idx)
        eligible_by_site_train = {k: sorted(v) for k, v in sorted(eligible_by_site_train.items())}
        idx_train, idx_holdout = split_indices_by_site(
            eligible_by_site_train, frac_train=train_frac, seed=42)
        idx_id_full = sorted(range(len(train_wrapped)))

    train_site_weights, train_site_counts = compute_site_class_weights_from_local_indices(
        splits["train"], idx_train, n_classes=n_site_classes, power=site_weight_power)

    print(f"[train] probe_train={len(idx_train)} | id_holdout={len(idx_holdout)} "
          f"| id_full={len(idx_id_full)} | require_gen_on_train={require_gen_on_train}")

    loaders["probe_train"] = mk(train_wrapped, idx_train, True)
    loaders["id_holdout"] = mk(train_wrapped, idx_holdout, False)
    loaders["id_full"] = mk(train_wrapped, idx_id_full, False)

    # ---------- OOD pools filtered to (all) gen_sources availability ----------
    for split_name in ["val", "test"]:
        wrapped, _, eligible_local = build_eligible_local_pool(
            splits[split_name], store, gsl)
        print(f"[ood_{split_name}] eligible {gsl} = {len(eligible_local)} / {len(wrapped)}")
        if eligible_local:
            loaders[f"ood_{split_name}"] = mk(wrapped, eligible_local, False)

    info = {"train_site_weights": train_site_weights,
            "train_site_counts": train_site_counts}
    return loaders, info


# --------------------------------------------------------------------------- #
# Brain-age preservation: train on REAL, eval real vs cyclegan on OOD
# --------------------------------------------------------------------------- #
def run_cyclegan_brain_age(make_age_model_fn, loaders, store, out_dir,
                           source=SOURCE, epochs=30, lr=1e-3, weight_decay=0.0,
                           loss_name="mae", site_weights=None,
                           eval_splits=("ood_val", "ood_test"),
                           also_eval_id=True, seed=SEED):
    """
    Train the UKBB brain-age model on REAL train images, then evaluate:
      - id_holdout / id_full : real only (age-model sanity)
      - ood_val / ood_test   : REAL vs cyclegan_target12

    Headline comparison: under a real-trained model, does harmonizing OOD toward
    site 12 lower the age MAE / shrink the bias vs the raw real OOD image?
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    model = make_age_model_fn()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_age_on_source(
        model=model, optimizer=optimizer, dataloader=loaders["probe_train"],
        store=store, source="real", epochs=epochs,
        loss_name=loss_name, site_weights=site_weights,
    )

    rows, preds = [], []

    for split_name in (["id_holdout", "id_full"] if also_eval_id else []):
        if split_name not in loaders:
            continue
        metrics, pred_df = eval_age_on_source(
            model, loaders[split_name], store, source="real", split_name=split_name)
        rows.append({"train_source": "real", "eval_split": split_name,
                     "eval_source": "real", **metrics})
        pred_df["train_source"] = "real"
        pred_df["eval_source"] = "real"
        preds.append(pred_df)
        print(f"[age id ] {split_name:10s} real  "
              f"MAE={metrics['mae']:.4f} R2={metrics['r2']:.4f} bias={metrics['bias']:.4f}")

    for split_name in eval_splits:
        if split_name not in loaders:
            continue
        for eval_source in ["real", source]:
            metrics, pred_df = eval_age_on_source(
                model, loaders[split_name], store,
                source=eval_source, split_name=split_name)
            rows.append({"train_source": "real", "eval_split": split_name,
                         "eval_source": eval_source, **metrics})
            pred_df["train_source"] = "real"
            pred_df["eval_source"] = eval_source
            preds.append(pred_df)
            print(f"[age ood] {split_name:10s} {eval_source:18s} "
                  f"MAE={metrics['mae']:.4f} R2={metrics['r2']:.4f} bias={metrics['bias']:.4f}")

    summary = pd.DataFrame(rows)
    pred_all = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    summary.to_csv(out_dir / "cyclegan_brain_age_summary.csv", index=False)
    pred_all.to_csv(out_dir / "cyclegan_brain_age_predictions.csv", index=False)
    print(f"[age] saved summary + predictions to {out_dir}")
    return summary, pred_all


# --------------------------------------------------------------------------- #
# Domain cross-training matrix over ALL sources (real + harmonized sources)
# --------------------------------------------------------------------------- #
def run_cyclegan_domain_confusion(make_model_fn, loaders, store, out_dir,
                                  n_classes=15, class_weights=None,
                                  epochs=30, lr=1e-3,
                                  eval_splits=("id_holdout", "id_full"), seed=SEED):
    """
    Full site/domain cross-training matrix. For EACH source in
    ``["real"] + store.available_sources`` (e.g. real, cyclegan_target12, combat):
    train a site classifier on that source's TRAIN images, then evaluate it on
    id_holdout and id_full for EVERY source. Needs the harmonized TRAIN images of
    all sources (require_gen_on_train=True loaders).

    Reading it: a REAL-trained classifier's recoverable-site accuracy should stay
    high on real and COLLAPSE on the harmonized sources (site signal removed) -
    use bal_acc / macro_f1 (micro is inflated by the large, ~identity site-12 class).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_sources = ["real"] + list(store.available_sources)
    print(f"[domain] cross-training over sources: {train_sources}")

    rows = []
    for train_source in train_sources:
        set_seed(seed)  # same init per model for comparability
        model = make_model_fn()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train_on_source(model=model, optimizer=optimizer,
                        dataloader=loaders["probe_train"], store=store,
                        source=train_source, epochs=epochs, class_weights=class_weights)

        for split_name in eval_splits:
            if split_name not in loaders:
                continue
            for eval_source in train_sources:
                res = eval_on_source(model, loaders[split_name], store,
                                     source=eval_source, n_classes=n_classes)
                rows.append({"train_source": train_source, "eval_split": split_name,
                             "eval_source": eval_source, **res})
                print(f"[domain] train={train_source:16s} split={split_name:10s} "
                      f"eval={eval_source:16s} | micro_acc={res['micro_acc']*100:5.1f}% "
                      f"bal_acc={res['bal_acc']:.3f} macro_f1={res['macro_f1']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "domain_cross_training_matrix.csv", index=False)

    # Headline: REAL-trained classifier, bal_acc on real vs each harmonized source.
    chance = 1.0 / max(n_classes, 1)
    for split_name in eval_splits:
        base = df[(df.train_source == "real") & (df.eval_split == split_name)]
        r = base[base.eval_source == "real"]
        if not len(r):
            continue
        ar = float(r.iloc[0]["bal_acc"])
        print(f"\n[domain] real-trained | {split_name}: bal_acc REAL={ar:.3f} "
              f"(chance~{chance:.3f})")
        for es in [s for s in train_sources if s != "real"]:
            h = base[base.eval_source == es]
            if len(h):
                ah = float(h.iloc[0]["bal_acc"])
                print(f"    -> {es:16s} bal_acc={ah:.3f}  drop={ar - ah:.3f}  "
                      f"retained={ah / ar if ar > 1e-9 else float('nan'):.2f}")
    print(f"[domain] saved matrix -> {out_dir / 'domain_cross_training_matrix.csv'}")
    return df


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    RUN_MODE = os.environ.get("RUN_MODE", "brain_age").strip().lower()
    RESULTS_ROOT = os.environ.get("RESULTS_ROOT", f"./probe_results_{SOURCE}")
    # For SOURCE=combat, set EXPORT_ROOT=/rhome/ssafa013/bigdata/data/openBHB_v1.1_harmonized
    default_export = os.path.expanduser("~/bigdata/cyclegan_target12")
    EXPORT_ROOT = os.environ.get("EXPORT_ROOT", default_export)

    VALID = {"intensity_sanity", "paired_psnr_ssim", "image_features", "brain_age",
             "brain_age_combined", "domain_confusion", "all"}
    if RUN_MODE not in VALID:
        raise ValueError(f"Unknown RUN_MODE={RUN_MODE!r}; expected {sorted(VALID)}")

    print(f"RUN_MODE={RUN_MODE} | EXPORT_ROOT={EXPORT_ROOT} | RESULTS_ROOT={RESULTS_ROOT}")
    results_dir = Path(RESULTS_ROOT)
    results_dir.mkdir(parents=True, exist_ok=True)

    store = build_cyclegan_store(EXPORT_ROOT)
    store.summary()

    dataset = OpenBHBDataset()
    N_SITE_CLASSES = dataset.num_train_domains
    print("N_SITE_CLASSES:", N_SITE_CLASSES)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # brain_age_combined / domain_confusion / all TRAIN on (or eval id with) the
    # harmonized train set -> require `export_harmonized.py --splits train --append`.
    NEEDS_GEN_TRAIN = RUN_MODE in ("brain_age_combined", "domain_confusion", "all")
    loaders, pool_info = build_cyclegan_loaders(
        dataset=dataset, store=store, batch_size=16, train_frac=0.9,
        num_workers=1, pin_memory=True, site_weight_power=0.5,
        n_site_classes=N_SITE_CLASSES, require_gen_on_train=NEEDS_GEN_TRAIN,
    )
    train_site_weights = pool_info["train_site_weights"]

    # Age-limited OOD (age <= AGE_MAX, default 35) -> ood_val_age_le35 / ood_test_age_le35.
    # Matches the paper's brain-age eval where older subjects (unreliable real->real
    # prediction) are dropped. These are subsets of the full OOD splits.
    AGE_MAX = float(os.environ.get("AGE_MAX", "35"))
    loaders, age_ood_info = add_age_limited_ood_loaders(
        loaders=loaders, dataset=dataset, store=store,
        batch_size=16, num_workers=1, pin_memory=True,
        min_age=0.0, max_age=AGE_MAX,
    )
    pd.DataFrame([
        {"key": k, **v} if isinstance(v, dict) else {"key": k, "value": v}
        for k, v in age_ood_info.items()
    ]).to_csv(results_dir / "age_limited_ood_info.csv", index=False)

    SOURCES = ["real", SOURCE]
    PAIRWISE = [("real", SOURCE)]
    _tag = f"age_le{int(AGE_MAX)}"
    OOD_SPLITS = [s for s in ["ood_val", "ood_test",
                              f"ood_val_{_tag}", f"ood_test_{_tag}"] if s in loaders]
    print(f"OOD splits available: {OOD_SPLITS}")

    if RUN_MODE in ("intensity_sanity", "all"):
        print("\n== Intensity sanity (real vs cyclegan on OOD) ==")
        run_all_intensity_sanity_checks(
            loaders=loaders, store=store,
            out_dir=str(results_dir / "intensity_sanity"),
            sources=SOURCES, pairwise_comparisons=PAIRWISE,
            split_names=OOD_SPLITS, max_batches=None,
        )

    if RUN_MODE in ("paired_psnr_ssim", "all"):
        print("\n== Pairwise PSNR / SSIM (real vs cyclegan on OOD) ==")
        pair_dir = results_dir / "pairwise_psnr_ssim"
        pair_dir.mkdir(parents=True, exist_ok=True)
        summaries = []
        for split_name in OOD_SPLITS:
            for a, b in PAIRWISE:
                _, s, _ = compute_pairwise_psnr_ssim_for_loader(
                    dataloader=loaders[split_name], store=store,
                    source_a=a, source_b=b, split_name=split_name,
                    out_csv=str(pair_dir / f"psnr_ssim_{split_name}_{a}_vs_{b}.csv"),
                    data_range_mode="pair", ssim_win_size=7,
                )
                summaries.append(s)
        psnr_summary_df = pd.DataFrame(summaries)
        psnr_summary_df.to_csv(pair_dir / "pairwise_psnr_ssim_summary.csv", index=False)
        plot_pairwise_metric_bars(psnr_summary_df, out_dir=str(pair_dir),
                                  prefix="pairwise_psnr_ssim")
        print(f"Saved PSNR/SSIM summary to {pair_dir / 'pairwise_psnr_ssim_summary.csv'}")

    if RUN_MODE in ("image_features", "all"):
        print("\n== Image features by age (real vs cyclegan on OOD) ==")
        run_all_image_feature_checks(
            loaders=loaders, store=store,
            out_dir=str(results_dir / "image_features_by_age"),
            sources=SOURCES, split_names=OOD_SPLITS, max_batches=None,
        )

    if RUN_MODE in ("brain_age", "all"):
        print("\n== Brain-age preservation (train REAL -> eval real vs cyclegan on OOD) ==")
        run_cyclegan_brain_age(
            make_age_model_fn=lambda: make_ukbb_age_model(device, normalize_input=False),
            loaders=loaders, store=store,
            out_dir=str(results_dir / "brain_age"),
            epochs=30, lr=1e-3, weight_decay=0.0, loss_name="mae",
            site_weights=train_site_weights,
            eval_splits=tuple(OOD_SPLITS), also_eval_id=True,
        )

    # Configs 1-4: train on harmonized / real+harmonized, eval OOD real & harmonized
    # (plus the real-trained baseline). Needs the harmonized TRAIN export.
    if RUN_MODE in ("brain_age_combined", "all"):
        print("\n== Brain-age combined (train real / cyclegan / real+cyclegan) ==")
        SEEDS = tuple(int(s) for s in os.environ.get("SEEDS", "1337").split(","))
        multi_dir = results_dir / "brain_age_combined"
        downstream_age_multiseed_combined_experiments(
            make_age_model_fn=lambda: make_ukbb_age_model(device, normalize_input=False),
            loaders=loaders, store=store, out_dir=str(multi_dir),
            seeds=SEEDS, generated_sources=[SOURCE],
            epochs_task=30, lr=1e-3, weight_decay=0.0,
            loss_name="mae", site_weights=train_site_weights,
            eval_splits=tuple(OOD_SPLITS), also_eval_id=True,
        )
        plot_multiseed_combined_age_results(
            summary_raw_csv=str(multi_dir / "brain_age_multiseed_summary_raw.csv"),
            out_dir=str(multi_dir / "plots"))

    # Domain cross-training matrix over real + ALL harmonized sources, on id splits.
    # Set DOMAIN_SOURCES="name=root,name=root" to include several harmonized sources,
    # e.g. real + cyclegan + combat. Requires each source's --splits train export.
    if RUN_MODE in ("domain_confusion", "all"):
        print("\n== Domain cross-training matrix (real + harmonized sources, id splits) ==")
        specs = parse_source_specs(os.environ.get("DOMAIN_SOURCES", f"{SOURCE}={EXPORT_ROOT}"))
        print(f"[domain] harmonized sources: {specs}")
        dstore = build_multi_source_store(specs)
        dstore.summary()
        dloaders, dinfo = build_cyclegan_loaders(
            dataset=dataset, store=dstore, gen_sources=list(specs.keys()),
            batch_size=16, train_frac=0.9, num_workers=1, pin_memory=True,
            site_weight_power=0.5, n_site_classes=N_SITE_CLASSES,
            require_gen_on_train=True,
        )
        run_cyclegan_domain_confusion(
            make_model_fn=lambda: make_model(device, N_SITE_CLASSES),
            loaders=dloaders, store=dstore,
            out_dir=str(results_dir / "domain_confusion"),
            n_classes=N_SITE_CLASSES,
            class_weights=dinfo["train_site_weights"], epochs=30, lr=1e-3,
            eval_splits=("id_holdout", "id_full"),
        )

    print("Done.")
