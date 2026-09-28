import os, sys, warnings, torch
from collections import defaultdict
from pathlib import Path
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from datetime import date
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader

import os, warnings, torch, nibabel as nib

# ─── 3 · helpers ───────────────────────────────────────────────────────────────
def load_nii(path: Path) -> torch.Tensor:
    arr = nib.load(str(path)).get_fdata(dtype="float32")
    arr = arr.squeeze()
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    assert arr.ndim == 3, f"Volume dims wrong: {arr.shape}"
    return torch.from_numpy(arr).unsqueeze(0)              # channel dim only


