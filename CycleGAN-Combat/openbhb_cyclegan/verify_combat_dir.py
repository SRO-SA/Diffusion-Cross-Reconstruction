#!/usr/bin/env python
"""
verify_combat_dir.py
====================

Inspect an existing "harmonized" OpenBHB directory and check whether the images
are (a) present / complete and (b) actually DIFFERENT from the real CAT12 VBM
images (i.e. harmonization was applied, not just copied).

    python verify_combat_dir.py --dataloader-dir <dir> \
        --harm-root /rhome/ssafa013/bigdata/data/openBHB_v1.1_harmonized

Needs OpenBHBDataset (torch+wilds) importable + nibabel. No TF / GPU.
"""
import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openbhb_bridge as ob


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataloader-dir", required=True)
    p.add_argument("--openbhb-root", default=None)
    p.add_argument("--harm-root",
                   default="/rhome/ssafa013/bigdata/data/openBHB_v1.1_harmonized")
    p.add_argument("--n-sample", type=int, default=8,
                   help="subjects per split to diff-check against the real image")
    return p.parse_args()


def index_harmonized(harm_root):
    """Map participant_id (parsed from filename `sub-<pid>_...`) -> filepath."""
    pat = re.compile(r"sub-(.+?)_")
    harm_index, per_dir = {}, {}
    subdirs = [p for p in harm_root.iterdir() if p.is_dir()]
    for d in subdirs:
        c = 0
        for p in d.iterdir():
            if not p.is_file():
                continue
            mm = pat.search(p.name)
            if mm:
                # keep the first match per pid; note the subdir
                harm_index.setdefault(str(mm.group(1)), p)
                c += 1
        per_dir[d.name] = (c, sorted(x.suffix for x in d.iterdir() if x.is_file())[:1])
    return harm_index, per_dir, subdirs


def load_any(path):
    if path.suffix == ".npy":
        return np.squeeze(np.load(path)).astype(np.float32)
    import nibabel as nib
    return np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)


def main():
    args = parse_args()
    harm_root = Path(args.harm_root)
    if not harm_root.exists():
        raise SystemExit(f"harm-root does not exist: {harm_root}")

    print(f"=== harmonized dir: {harm_root} ===")
    for p in sorted(harm_root.iterdir()):
        kind = "dir" if p.is_dir() else "file"
        print(f"  [{kind}] {p.name}")

    harm_index, per_dir, subdirs = index_harmonized(harm_root)
    print("\n=== files per subfolder (count, example suffix) ===")
    for name, (c, suf) in per_dir.items():
        print(f"  {name}/: {c} files, example suffix {suf}")
    # show a couple of example names so we can see the naming convention
    for d in subdirs:
        exs = [p.name for p in list(d.iterdir())[:2] if p.is_file()]
        if exs:
            print(f"  {d.name}/ examples: {exs}")
    print(f"\nunique participant_ids found across subfolders: {len(harm_index)}")

    ds = ob.load_openbhb_dataset(args.dataloader_dir, args.openbhb_root)
    m = ds.metadata

    print("\n=== coverage + real-vs-harmonized diff (per split) ===")
    for sp in ["train", "val", "test"]:
        idxs = ob.get_split_indices(ds, sp)
        if len(idxs) == 0:
            continue
        pids = [str(m["participant_id"].iloc[int(i)]) for i in idxs]
        covered = sum(1 for pid in pids if pid in harm_index)

        # diff-check a small evenly spaced sample
        sample = idxs[np.linspace(0, len(idxs) - 1,
                                  min(args.n_sample, len(idxs))).round().astype(int)]
        found = ident = diff = 0
        maes, corrs = [], []
        for i in np.unique(sample):
            i = int(i)
            pid = str(m["participant_id"].iloc[i])
            hp = harm_index.get(pid)
            if hp is None:
                continue
            found += 1
            real = ob.load_raw_volume(ds, i)
            try:
                harm = load_any(hp)
            except Exception as e:
                print(f"  [{sp}] pid={pid}: could not load {hp} ({e})")
                continue
            if harm.shape != real.shape:
                print(f"  [{sp}] pid={pid}: SHAPE mismatch real={real.shape} harm={harm.shape}")
                continue
            mae = float(np.mean(np.abs(real - harm)))
            maes.append(mae)
            b = (real > 0) & (harm > 0)
            if b.sum() > 10 and real[b].std() > 1e-8 and harm[b].std() > 1e-8:
                corrs.append(float(np.corrcoef(real[b], harm[b])[0, 1]))
            if np.array_equal(real, harm):
                ident += 1
            elif mae > 1e-8:
                diff += 1

        mae_mean = float(np.mean(maes)) if maes else float("nan")
        corr_mean = float(np.mean(corrs)) if corrs else float("nan")
        print(f"  [{sp:5s}] coverage={covered}/{len(idxs)} | sampled={found} | "
              f"identical-to-real={ident} | different={diff} | "
              f"mean MAE={mae_mean:.5f} | mean corr={corr_mean:.4f}")

    print("\nHow to read this:")
    print("  * USABLE if, for every split: coverage == split size, different == sampled,")
    print("    identical-to-real == 0, and MAE > 0 (with high corr, harmonization is subtle).")
    print("  * NOT usable if files are missing (coverage short), or identical-to-real > 0")
    print("    (they're copies), or shapes mismatch. Then run combat_harmonize.py.")
    print("  * Note: even if usable, provenance is unknown (which ComBat variant / reference /")
    print("    covariates). For a comparison to cyclegan_target12 (reference = site 12,")
    print("    age preserved), running combat_harmonize.py yourself is the controlled option.")


if __name__ == "__main__":
    main()
