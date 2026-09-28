from ctypes import sizeof
from logging import config
import math, os, random
from collections.abc import Iterable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import sys
from tqdm.auto import tqdm

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchvision.utils import save_image, make_grid
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from DataLoader import OpenBHBDataset

import os
import numpy as np
import matplotlib.pyplot as plt


def set_requires_grad(module: nn.Module, flag: bool):
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad_(flag)


def extract_domain_labels(metadata, domain_col, device):
    if metadata is None:
        raise ValueError("metadata is None, but domain labels are required.")

    if not torch.is_tensor(metadata):
        metadata = torch.as_tensor(metadata)

    if metadata.dim() == 1:
        domain = metadata.long()
    else:
        domain = metadata[:, domain_col].long()

    return domain.to(device)


def t_to_logsnr(t, sched):
    """
    t : [B] long
    logSNR = log(alpha_bar / (1 - alpha_bar))
    """
    abar = sched.alpha_bar.gather(0, t)
    abar = abar.clamp(1e-8, 1.0 - 1e-8)
    return torch.log(abar) - torch.log(1.0 - abar)



    
def diffusion_target_from_objective(objective, x0, eps, a, b):
    if objective == "v":
        return v_target_from_shared(x0, eps, a, b)
    elif objective == "eps":
        return eps
    elif objective == "x0":
        return x0
    else:
        raise ValueError(f"Unknown objective: {objective}")
    
    
def save_experiment_checkpoint(path, model, discriminator, config):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    ckpt = {
        "objective": config.objective if config is not None else None,
        "config": vars(config) if config is not None else None,   # save all config fields
        "model_state_dict": model.state_dict(),
        "has_discriminator": discriminator is not None,
        "discriminator_state_dict": None if discriminator is None else discriminator.state_dict(),
    }
    torch.save(ckpt, path)


def load_experiment_checkpoint(path, device):
    return torch.load(path, map_location=device)

def save_metric_plots(df, out_dir, prefix, title_prefix=None):
    """
    Save MAE(t) and MSE(t) plots from a DF that has columns:
      - "t" (or "t_cur") as timestep x-axis
      - "mae", "mse" as y-axis

    Works for dfA/dfB/dfC produced by our tests.
    """
    os.makedirs(out_dir, exist_ok=True)
    title_prefix = title_prefix or prefix

    # pick timestep column
    if "t" in df.columns:
        tcol = "t"
    elif "t_cur" in df.columns:
        tcol = "t_cur"
    else:
        raise ValueError(f"{prefix}: couldn't find a timestep column in df. Columns={list(df.columns)}")

    # If multiple rows per timestep, aggregate (mean)
    g = df.groupby(tcol)[["mae", "mse"]].mean().reset_index()

    x = g[tcol].values
    mae = g["mae"].values
    mse = g["mse"].values

    # --- MAE plot ---
    plt.figure()
    plt.plot(x, mae)
    plt.title(f"{title_prefix}: MAE vs t")
    plt.xlabel("t")
    plt.ylabel("MAE")
    plt.grid(True, alpha=0.3)
    # If descending schedule, invert axis for readability
    if len(x) > 1 and x[0] > x[-1]:
        plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_mae_vs_t.png"), dpi=150)
    plt.close()

    # --- MSE plot ---
    plt.figure()
    plt.plot(x, mse)
    plt.title(f"{title_prefix}: MSE vs t")
    plt.xlabel("t")
    plt.ylabel("MSE")
    plt.grid(True, alpha=0.3)
    if len(x) > 1 and x[0] > x[-1]:
        plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_mse_vs_t.png"), dpi=150)
    plt.close()

def _minmax01_per_item(img_b1hw):
    # img_b1hw: (B,1,H,W) float
    flat = img_b1hw.reshape(img_b1hw.shape[0], -1)
    mn = flat.min(dim=1).values.view(-1,1,1,1)
    mx = flat.max(dim=1).values.view(-1,1,1,1)
    out = (img_b1hw - mn) / (mx - mn + 1e-12)
    return out.clamp(0,1)

def save_center_slice_grid(vol_bcdhw, out_path, axis="D", channel=0, nrow=4):
    """
    vol_bcdhw: (B,C,D,H,W)  (MODEL SPACE)
    Saves a grid of central slices as PNG.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    x = vol_bcdhw.detach().float().cpu()
    x = x[:, channel:channel+1]  # (B,1,D,H,W)

    if axis.upper() == "D":
        sl = x.shape[2] // 2
        img = x[:, :, sl, :, :]  # (B,1,H,W)
    elif axis.upper() == "H":
        sl = x.shape[3] // 2
        img = x[:, :, :, sl, :]  # (B,1,D,W)
    else:  # "W"
        sl = x.shape[4] // 2
        img = x[:, :, :, :, sl]  # (B,1,D,H)

    img01 = _minmax01_per_item(img)
    grid = make_grid(img01, nrow=nrow, padding=2)
    save_image(grid, out_path)

def save_tensor_pt(x, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(x.detach().cpu(), path)


import os
import numpy as np
import torch
import nibabel as nib

def model_to_hwd(vol_bcdhw: torch.Tensor) -> np.ndarray:
    """
    Convert model layout (B,C,D,H,W) -> numpy (B,C,H,W,D).
    Then you can take [b,0] to get (H,W,D).
    """
    x = vol_bcdhw.detach().float().cpu()
    x = x.permute(0, 1, 3, 4, 2).contiguous()  # (B,C,H,W,D)
    return x.numpy()

def save_nifti_batch(
    vol_bcdhw: torch.Tensor,
    out_dir: str,
    prefix: str,
    affine: np.ndarray | None = None,
    channel: int = 0,
    clamp: tuple[float, float] | None = None,
):
    """
    Save each item in a batch as NIfTI.
    vol_bcdhw: (B,C,D,H,W) in model space
    Writes: out_dir/prefix_{i:03d}.nii.gz

    affine: if you have a real affine, pass it. Otherwise identity is used.
    clamp: optional (min,max) clamp before saving (for visualization).
           Leave None to save raw outputs.
    """
    os.makedirs(out_dir, exist_ok=True)
    if affine is None:
        affine = np.eye(4, dtype=np.float32)

    x = vol_bcdhw.detach().float().cpu()
    if clamp is not None:
        x = x.clamp(float(clamp[0]), float(clamp[1]))

    x_hwd = model_to_hwd(x)  # (B,C,H,W,D)

    B = x_hwd.shape[0]
    for i in range(B):
        vol = x_hwd[i, channel]  # (H,W,D)
        nii = nib.Nifti1Image(vol.astype(np.float32), affine)
        path = os.path.join(out_dir, f"{prefix}_{i:03d}.nii.gz")
        nib.save(nii, path)

    return [os.path.join(out_dir, f"{prefix}_{i:03d}.nii.gz") for i in range(B)]



from torch.utils.data import Dataset

from dataclasses import dataclass

def compute_domain_counts_from_loader(train_dl, domain_col: int, num_domains: int):
    counts = torch.zeros(num_domains, dtype=torch.float32)

    for batch in train_dl:
        metadata = batch[2]
        labels = metadata[:, domain_col].long()

        if labels.min().item() < 0 or labels.max().item() >= num_domains:
            raise ValueError(
                f"Domain labels out of range. "
                f"min={labels.min().item()}, max={labels.max().item()}, "
                f"num_domains={num_domains}"
            )

        counts += torch.bincount(labels.cpu(), minlength=num_domains).float()

    return counts

def make_domain_class_weights(site_counts, alpha=0.5, mix=0.4, device=None):
    """
    alpha:
      1.0 = strong inverse-frequency style
      0.5 = gentler
      0.25 = even gentler

    mix:
      1.0 = fully weighted
      0.0 = all ones
    """
    counts = torch.as_tensor(site_counts, dtype=torch.float32)
    w = torch.zeros_like(counts)

    mask = counts > 0
    raw = (counts[mask].sum() / counts[mask]).pow(alpha)
    raw = raw / raw.mean()  # normalize active classes to mean ~1

    # blend with uniform weights
    w[mask] = (1.0 - mix) + mix * raw

    if device is not None:
        w = w.to(device)
    return w

@dataclass
class MRITrainConfig:
    objective: str = "v"              # "v" | "eps" | "x0"

    # discriminator / DANN
    use_discriminator: bool = False
    lambda_dann: float = 0.1
    clip_x0_for_disc: bool = False
    freeze_discriminator: bool = False

    # preservation losses on x0_hat
    use_mae: bool = False
    lambda_mae: float = 0.0

    use_ssim: bool = False
    lambda_ssim: float = 0.0

    # extra controls
    clip_x0_for_losses: bool = False

    site_counts = torch.tensor([0, 24, 277, 43, 25, 47, 50, 17, 11, 14, 10, 73, 956, 20, 20], dtype=torch.float32)
    # domain_class_weights = make_domain_class_weights(site_counts, alpha=0.5, mix=0.4)  # for DANN loss
    disc_class_weights = make_domain_class_weights(
        site_counts, alpha=0.8, mix=0.8
    )
    dann_class_weights = make_domain_class_weights(
        site_counts, alpha=0.5, mix=0.4
    )

    use_timestep_loss_weight: bool = True
    tau_loss: float = 0.5 # tested: 0.0
    s_loss: float = 0.7 # tested: 1.0
    min_t_weight: float = 0.05

@dataclass
class MRIExperimentSpec:
    name: str                    # e.g. "v_base", "v_dann", "eps_base"
    objective: str               # "v" | "eps" | "x0"
    ckpt_path: str

    # training mode
    use_discriminator: bool = False
    num_domains: int | None = None
    domain_col: int | None = None

    # optional later
    use_mae: bool = False
    lambda_mae: float = 0.0
    use_ssim: bool = False
    lambda_ssim: float = 0.0

    # DANN
    lambda_dann: float = 0.0
    clip_x0_for_disc: bool = False

    # NEW
    freeze_discriminator: bool = False
    init_discriminator_ckpt: str | None = None
    discriminator_use_time_conditioning: bool = True
    
    
class WithGlobalIndex(Dataset):
    def __init__(self, subset):
        self.subset = subset

        # ---- forward WILDS-specific attributes ----
        # get_eval_loader uses dataset.collate
        if hasattr(subset, "collate"):
            self.collate = subset.collate

        # (optional) forward metadata_array etc if you use them elsewhere
        if hasattr(subset, "metadata_array"):
            self.metadata_array = subset.metadata_array
        if hasattr(subset, "y_array"):
            self.y_array = subset.y_array

        # ---- figure out base indices ----
        base = getattr(subset, "indices", None)
        if base is None:
            base = getattr(subset, "_indices", None)

        if base is not None and hasattr(base, "tolist"):
            base = base.tolist()

        self._base_indices = base
        if(self._base_indices is None):
            print("Warning: couldn't find base indices for dataset. Global indices will be the same as local indices.")
            
    def __len__(self):
        return len(self.subset)

    def __getitem__(self, i):
        x, y, m = self.subset[i]
        gidx = int(self._base_indices[i]) if self._base_indices is not None else int(i)
        return x, y, m, gidx
    
    
def is_plain_base_config(config: MRITrainConfig) -> bool:
    return (
        not config.use_discriminator
        and not config.use_mae
        and not config.use_ssim
    )
    
@torch.no_grad()
def testA_real_inv_fwd_sweep_pack(pack, x0_bchwd, steps=50, clip_inv=False, clip_fwd=True, tag="A"):
    """
    Test A (schedule steps): real x0 -> inverse one step -> forward one step back.
    Uses the pack's DDIM step function.
    x0_bchwd is dataset layout: (B,C,H,W,D).
    """
    # convert once to model space (B,C,D,H,W)
    x0 = to_model_layout(x0_bchwd.float().to(pack.device))
    ts_asc = make_ddim_timesteps(pack.sched.T, steps, descending=False).to(pack.device)

    rows = []
    x_t = x0.clone()

    for i in range(len(ts_asc)-1):
        t = int(ts_asc[i].item())
        t_next = int(ts_asc[i+1].item())

        x_tnext = pack.ddim_step_to_t_target(x_t, t, t_next, clip_x0=clip_inv)
        x_hat_t = pack.ddim_step_to_t_target(x_tnext, t_next, t, clip_x0=clip_fwd)

        rows.append({
            "tag": tag,
            "i": i,
            "t": t,
            "t_next": t_next,
            "mse": float(F.mse_loss(x_hat_t, x_t).item()),
            "mae": float(F.l1_loss(x_hat_t, x_t).item()),
        })

        x_t = x_tnext

    return pd.DataFrame(rows), x_t  # x_t ends near xT


@torch.no_grad()
def testB_noise_fwd_inv_sweep_pack(pack, x0_bchwd, steps=50, seed=0, clip_fwd=True, clip_inv=False, tag="B"):
    """
    Test B (schedule steps): start from Gaussian x_T, go forward one step (t -> t_prev),
    then inverse step back (t_prev -> t). Measure error.
    shape_bchwd: (B,C,H,W,D) dataset layout; internally we sample (B,C,D,H,W).
    """
    set_seed(seed)
    # build correct model-space shape from real data (B,C,D,H,W)
    x0 = to_model_layout(x0_bchwd.float().to(pack.device))
    x_t = torch.randn_like(x0)  # <-- guaranteed valid shape

    ts_desc = make_ddim_timesteps(pack.sched.T, steps, descending=True).to(pack.device)

    rows = []
    for i in range(len(ts_desc)-1):
        t = int(ts_desc[i].item())
        t_prev = int(ts_desc[i+1].item())

        x_prev = pack.ddim_step_to_t_target(x_t, t, t_prev, clip_x0=clip_fwd)
        x_hat  = pack.ddim_step_to_t_target(x_prev, t_prev, t, clip_x0=clip_inv)

        rows.append({
            "tag": tag,
            "i": i,
            "t": t,
            "t_prev": t_prev,
            "mse": float(F.mse_loss(x_hat, x_t).item()),
            "mae": float(F.l1_loss(x_hat, x_t).item()),
        })

        x_t = x_prev

    return pd.DataFrame(rows), x_t  # ends near x0-ish sample


@torch.no_grad()
def testC_true_noise_unit_pack(pack, x0_bchwd, steps=50, seed=0, clip_fwd=True, clip_inv=False, tag="C"):
    """
    Test C (true-noise unit test):
      Build true x_t = a*x0 + b*eps, then do t->t_prev->t using the pack step.
    x0_bchwd is dataset layout (B,C,H,W,D).
    """
    set_seed(seed)
    x0 = to_model_layout(x0_bchwd.float().to(pack.device))  # (B,C,D,H,W)
    B = x0.shape[0]

    ts_desc = make_ddim_timesteps(pack.sched.T, steps, descending=True).to(pack.device)
    rows = []

    for i in range(len(ts_desc)-1):
        t = int(ts_desc[i].item())
        t_prev = int(ts_desc[i+1].item())

        t_b = torch.full((B,), t, device=pack.device, dtype=torch.long)

        eps = torch.randn_like(x0)
        a = pack.sched.gather(pack.sched.sqrt_alpha_bar, t_b, x0.shape)
        b = pack.sched.gather(pack.sched.sqrt_one_minus_alpha_bar, t_b, x0.shape)
        x_t = a * x0 + b * eps

        x_prev = pack.ddim_step_to_t_target(x_t, t, t_prev, clip_x0=clip_fwd)
        x_hat  = pack.ddim_step_to_t_target(x_prev, t_prev, t, clip_x0=clip_inv)

        rows.append({
            "tag": tag,
            "i": i,
            "t": t,
            "t_prev": t_prev,
            "mse": float(F.mse_loss(x_hat, x_t).item()),
            "mae": float(F.l1_loss(x_hat, x_t).item()),
        })

    return pd.DataFrame(rows)
# -------------------------
# Seed
# -------------------------
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Gradient Reversal
# ============================================================
class _GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x):
        return _GradientReversalFn.apply(x, self.alpha)

    def set_alpha(self, alpha: float):
        self.alpha = float(alpha)


# ============================================================
# Stronger 3D feature extractor
# ============================================================
def conv_blk_3d(in_channel: int, out_channel: int, use_instancenorm: bool = True):
    layers = [nn.Conv3d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)]
    if use_instancenorm:
        layers.append(nn.InstanceNorm3d(out_channel, affine=True))
    layers.extend([
        nn.MaxPool3d(kernel_size=2, stride=2),
        nn.ReLU(inplace=True),
    ])
    return nn.Sequential(*layers)


class VolumeDomainFeaturizer(nn.Module):
    """
    Stronger 3D featurizer for domain discrimination.
    Input:  (B, C, D, H, W)
    Output: feature map (B, C', D', H', W')
    """
    def __init__(self, in_ch: int = 1, use_conv5: bool = True, use_instancenorm: bool = True):
        super().__init__()

        self.conv1 = conv_blk_3d(in_ch,   32,  use_instancenorm)
        self.conv2 = conv_blk_3d(32,      64,  use_instancenorm)
        self.conv3 = conv_blk_3d(64,     128,  use_instancenorm)
        self.conv4 = conv_blk_3d(128,    256,  use_instancenorm)

        self.use_conv5 = bool(use_conv5)
        if self.use_conv5:
            self.conv5 = conv_blk_3d(256, 256, use_instancenorm)

        self.d_out = None  # set after shape inference

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        if self.use_conv5:
            x = self.conv5(x)
        return x


# ============================================================
# MLP classifier head
# ============================================================
class DomainDiscriminator(nn.Sequential):
    def __init__(
        self,
        in_feature: int,
        n_domains: int,
        hidden_size: int = 1024,
        batch_norm: bool = False,
        dropout: float = 0.5,
    ):
        if batch_norm:
            super().__init__(
                nn.Linear(in_feature, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, n_domains),
            )
        else:
            super().__init__(
                nn.Linear(in_feature, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, n_domains),
            )


# ============================================================
# Strong diffusion-time image discriminator
# ============================================================
class StrongDiffusionImageDiscriminator3D(nn.Module):
    """
    Stronger domain classifier for diffusion training.

    - works on reconstructed image x0_hat in model space: (B, C, D, H, W)
    - uses a strong 3D featurizer
    - flattens the full feature map
    - optionally conditions on logSNR / timestep
    - GRL can be turned on/off per forward call

    Typical usage:
      # generator step
      logits = disc(x0_hat, logsnr, use_grl=True)

      # discriminator-only step
      logits = disc(x0_hat.detach(), logsnr, use_grl=False)
    """
    def __init__(
        self,
        input_shape,             # tuple: (C, D, H, W)
        in_ch: int = 1,
        n_domains: int = 15,
        hidden_size: int = 1024,
        batch_norm: bool = False,
        dropout: float = 0.5,
        use_conv5: bool = True,
        use_instancenorm: bool = True,
        t_emb_dim: int = 64,
        grl_alpha: float = 1.0,
        use_time_conditioning: bool = True,
        tau_D: float = 0.0,
        s_D: float = 1.0,
    ):
        super().__init__()

        self.use_time_conditioning = bool(use_time_conditioning)

        self.featurizer = VolumeDomainFeaturizer(
            in_ch=in_ch,
            use_conv5=use_conv5,
            use_instancenorm=use_instancenorm,
        )

        # infer flattened feature size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)   # (1, C, D, H, W)
            fmap = self.featurizer(dummy)
            d_out = fmap.flatten(1).shape[1]

        self.featurizer.d_out = d_out
        self.d_out = d_out

        self.GRL = GradientReversal(alpha=grl_alpha)

        # time / logSNR embedding
        self.t_emb_dim = int(t_emb_dim)
        if self.use_time_conditioning:
            self.t_emb = nn.Sequential(
                nn.Linear(1, t_emb_dim),
                nn.SiLU(),
                nn.Linear(t_emb_dim, t_emb_dim),
            )
            in_feature = d_out + t_emb_dim
        else:
            self.t_emb = None
            in_feature = d_out

        self.classifier = DomainDiscriminator(
            in_feature=in_feature,
            n_domains=n_domains,
            hidden_size=hidden_size,
            batch_norm=batch_norm,
            dropout=dropout,
        )

        # same idea as your current discriminator:
        # gate time conditioning strength by logSNR if desired
        self.register_buffer("tau_D", torch.tensor(float(tau_D)))
        self.register_buffer("s_D", torch.tensor(float(s_D)))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_grl_alpha(self, alpha: float):
        self.GRL.set_alpha(alpha)

    def forward(
        self,
        x: torch.Tensor,
        logsnr: torch.Tensor | None = None,
        use_grl: bool = True,
    ) -> torch.Tensor:
        """
        x      : (B, C, D, H, W)
        logsnr : (B,) or None
        """
        fmap = self.featurizer(x)
        feat_vec = fmap.flatten(1)   # keep full spatial information

        if self.use_time_conditioning:
            if logsnr is None:
                raise ValueError("logsnr must be provided when use_time_conditioning=True")

            t_vec = self.t_emb(logsnr.view(-1, 1))

            # optional logSNR gate, same spirit as your old code
            gD = torch.sigmoid((self.tau_D - logsnr) / (self.s_D + 1e-8)).unsqueeze(1).detach()
            t_vec = gD * t_vec

            z = torch.cat([feat_vec, t_vec], dim=1)
        else:
            z = feat_vec

        if use_grl:
            z = self.GRL(z)

        logits = self.classifier(z)
        return logits
        
# -------------------------
# Schedule (same as MNIST)
# -------------------------
class DiffusionSchedule:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=2e-2, device="cpu"):
        self.T = T
        betas = torch.linspace(beta_start, beta_end, T, device=device)  # [T]
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

    def gather(self, vec, t, x_shape):
        return vec.gather(0, t).view(-1, *([1] * (len(x_shape) - 1)))

def make_ddim_timesteps(T, steps, descending=True):
    ts = np.linspace(0, T-1, steps, dtype=np.int64)
    if descending:
        ts = ts[::-1].copy()
    return torch.from_numpy(ts)

# -------------------------
# Time embedding (same as MNIST)
# -------------------------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(0, half, device=t.device).float() / (half - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

# -------------------------
# Minimal "layout adapter" (NOT a transform)
# Your dataset gives: (B,C,121,145,121) == (B,C,H,W,D)
# Conv3d expects:     (B,C,D,H,W)
# -------------------------
def to_model_layout(x):
    """
    Accepts:
      [B, C, 1, H, W, D]
      [B, C, H, W, D]
      [B, H, W, D]

    Returns:
      [B, C, D, H, W]
    """
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)

    x = x.float()

    if x.dim() == 6:
        # [B,C,1,H,W,D] -> [B,C,H,W,D]
        if x.shape[2] != 1:
            raise ValueError(f"Expected singleton dim at axis 2, got shape={tuple(x.shape)}")
        x = x.squeeze(2)

    if x.dim() == 4:
        # [B,H,W,D] -> [B,1,H,W,D]
        x = x.unsqueeze(1)

    if x.dim() != 5:
        raise ValueError(f"Expected 5D after cleanup, got shape={tuple(x.shape)}")

    # [B,C,H,W,D] -> [B,C,D,H,W]
    return x.permute(0, 1, 4, 2, 3).contiguous()

# -------------------------
# UNet3D (MNIST-style)
# - same ResBlock idea, just Conv3d
# - 2-level UNet to fit memory
# -------------------------
def _center_crop_like(x, ref):
    # Crop features only if needed due to odd sizes in transpose conv
    _, _, D, H, W = x.shape
    _, _, Dr, Hr, Wr = ref.shape
    sd = max((D - Dr) // 2, 0)
    sh = max((H - Hr) // 2, 0)
    sw = max((W - Wr) // 2, 0)
    return x[:, :, sd:sd+Dr, sh:sh+Hr, sw:sw+Wr]

def align_to_ref(x, ref):
    """
    Make x have the same (D,H,W) as ref by symmetric pad/crop.
    x, ref: (B,C,D,H,W)
    """
    _, _, D, H, W = x.shape
    _, _, Dr, Hr, Wr = ref.shape

    # --- crop if too big ---
    if D > Dr:
        d0 = (D - Dr) // 2
        x = x[:, :, d0:d0+Dr, :, :]
    if H > Hr:
        h0 = (H - Hr) // 2
        x = x[:, :, :, h0:h0+Hr, :]
    if W > Wr:
        w0 = (W - Wr) // 2
        x = x[:, :, :, :, w0:w0+Wr]

    # --- pad if too small ---
    pd = Dr - x.shape[2]
    ph = Hr - x.shape[3]
    pw = Wr - x.shape[4]

    if pd > 0 or ph > 0 or pw > 0:
        # pad format for F.pad 5D is (W_left, W_right, H_left, H_right, D_left, D_right)
        pad = [0, 0, 0, 0, 0, 0]
        if pw > 0:
            wl = pw // 2
            wr = pw - wl
            pad[0], pad[1] = wl, wr
        if ph > 0:
            hl = ph // 2
            hr = ph - hl
            pad[2], pad[3] = hl, hr
        if pd > 0:
            dl = pd // 2
            dr = pd - dl
            pad[4], pad[5] = dl, dr
        x = F.pad(x, pad)

    return x

class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        # keep same "GroupNorm vibe" as MNIST
        ng = 8 if out_ch % 8 == 0 else 4 if out_ch % 4 == 0 else 1
        self.gn1 = nn.GroupNorm(ng, out_ch)
        self.gn2 = nn.GroupNorm(ng, out_ch)

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.gn1(h)
        h = F.silu(h)

        t_add = self.t_proj(t_emb).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        h = h + t_add

        h = self.conv2(h)
        h = self.gn2(h)
        h = F.silu(h)
        return h + self.skip(x)

class SimpleUNet3D(nn.Module):
    def __init__(self, in_ch=1, base=16, t_dim=128):
        super().__init__()
        self.time = SinusoidalTimeEmbedding(t_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim*2),
            nn.SiLU(),
            nn.Linear(t_dim*2, t_dim),
        )

        self.rb1 = ResBlock3D(in_ch, base, t_dim)
        self.down1 = nn.Conv3d(base, base, 4, stride=2, padding=1)

        self.rb2 = ResBlock3D(base, base*2, t_dim)
        self.down2 = nn.Conv3d(base*2, base*2, 4, stride=2, padding=1)

        self.mid1 = ResBlock3D(base*2, base*2, t_dim)
        self.mid2 = ResBlock3D(base*2, base*2, t_dim)

        self.up2 = nn.ConvTranspose3d(base*2, base*2, 4, stride=2, padding=1, output_padding=1)
        self.urb2 = ResBlock3D(base*2 + base*2, base, t_dim)
        

        self.up1 = nn.ConvTranspose3d(base, base, 4, stride=2, padding=1, output_padding=1)
        self.urb1 = ResBlock3D(base + base, base, t_dim)

        self.out = nn.Conv3d(base, in_ch, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(self.time(t))

        h1 = self.rb1(x, t_emb)
        d1 = self.down1(h1)

        h2 = self.rb2(d1, t_emb)
        d2 = self.down2(h2)

        m = self.mid1(d2, t_emb)
        m = self.mid2(m, t_emb)

        u2 = self.up2(m)
        if u2.shape[2:] != h2.shape[2:]:
            h2 = _center_crop_like(h2, u2)
        h2 = align_to_ref(h2, u2)   # make skip match u2

        u2 = torch.cat([u2, h2], dim=1)
        u2 = self.urb2(u2, t_emb)

        u1 = self.up1(u2)
        if u1.shape[2:] != h1.shape[2:]:
            h1 = _center_crop_like(h1, u1)
        h1 = align_to_ref(h1, u1)
        
        u1 = torch.cat([u1, h1], dim=1)
        u1 = self.urb1(u1, t_emb)

        out = self.out(u1)
        # crop output to match input if needed (rare off-by-one)
        if out.shape[2:] != x.shape[2:]:
            out = _center_crop_like(out, x)
        return out



# -------------------------
# Shared forward noising (same math as MNIST)
# -------------------------
@torch.no_grad()
def sample_xt_shared(x0, t, sched: DiffusionSchedule):
    eps = torch.randn_like(x0)
    a = sched.gather(sched.sqrt_alpha_bar, t, x0.shape)
    b = sched.gather(sched.sqrt_one_minus_alpha_bar, t, x0.shape)
    xt = a * x0 + b * eps
    return xt, eps, a, b

def v_target_from_shared(x0, eps, a, b):
    return a * eps - b * x0

def v_to_x0_eps(x_t, v_hat, a, b):
    x0_hat = a * x_t - b * v_hat
    eps_hat = b * x_t + a * v_hat
    return x0_hat, eps_hat

def eps_to_x0(x_t, eps_hat, a, b):
    return (x_t - b * eps_hat) / (a + 1e-12)

def x0_to_eps(x_t, x0_hat, a, b):
    return (x_t - a * x0_hat) / (b + 1e-12)

def infer_model_input_shape_from_loader(train_dl):
    """
    Infer model-space input shape (C, D, H, W) from one batch in the loader.
    Safe to call before training starts.
    """
    first_batch = next(iter(train_dl))
    x = first_batch[0] if isinstance(first_batch, (tuple, list)) else first_batch
    x = x.float()                  # stay on CPU, no need to move to device
    x = to_model_layout(x)         # -> (B, C, D, H, W)
    return tuple(x.shape[1:])      # (C, D, H, W)


def build_model_and_discriminator(
    *,
    device,
    in_channels,
    base,
    t_dim,
    disc_base,
    spec: MRIExperimentSpec,
    input_shape=None,
):
    model = SimpleUNet3D(in_ch=in_channels, base=base, t_dim=t_dim).to(device)

    discriminator = None
    if spec.use_discriminator:
        if spec.num_domains is None:
            raise ValueError("spec.num_domains is required when use_discriminator=True")

        discriminator = StrongDiffusionImageDiscriminator3D(
            input_shape=input_shape,
            in_ch=input_shape[0],
            n_domains=spec.num_domains,
            hidden_size=1024,
            batch_norm=False,
            dropout=0.1,
            use_conv5=True,
            use_instancenorm=True,
            t_emb_dim=64,
            grl_alpha=1.0,
            use_time_conditioning=spec.discriminator_use_time_conditioning,
            tau_D=0.0,
            s_D=1.0,
        ).to(device)

    return model, discriminator

# ============================================================
# Three objectives: v / eps / x0
# Each has:
#   - predict_*()
#   - pred_x0_eps_at_*()
#   - ddim_step_to_t_target_*()
#   - sample_ddim_*(), invert_ddim_*()
# ============================================================

class MRIObjectiveCore:
    def __init__(self, model: nn.Module, sched: DiffusionSchedule, device: torch.device, objective: str):
        assert objective in ("v", "eps", "x0")
        self.model = model
        self.sched = sched
        self.device = device
        self.objective = objective

    def predict(self, x, t_b):
        return self.model(x, t_b)

    def diffusion_target(self, x0, eps, a, b):
        if self.objective == "v":
            return v_target_from_shared(x0, eps, a, b)
        elif self.objective == "eps":
            return eps
        else:
            return x0
        
    def pred_x0_eps_from_pred(self, x_t, pred, a, b):
        if self.objective == "v":
            x0_hat, eps_hat = v_to_x0_eps(x_t, pred, a, b)
        elif self.objective == "eps":
            eps_hat = pred
            x0_hat = eps_to_x0(x_t, eps_hat, a, b)
        else:
            x0_hat = pred
            eps_hat = x0_to_eps(x_t, x0_hat, a, b)
        return x0_hat, eps_hat

    def pred_x0_eps_at(self, x_t, t_b):
        a = self.sched.gather(self.sched.sqrt_alpha_bar, t_b, x_t.shape)
        b = self.sched.gather(self.sched.sqrt_one_minus_alpha_bar, t_b, x_t.shape)
        out = self.predict(x_t, t_b)

        if self.objective == "v":
            return v_to_x0_eps(x_t, out, a, b)
        if self.objective == "eps":
            eps_hat = out
            x0_hat = eps_to_x0(x_t, eps_hat, a, b)
            return x0_hat, eps_hat
        # x0
        x0_hat = out
        eps_hat = x0_to_eps(x_t, x0_hat, a, b)
        return x0_hat, eps_hat

    def ddim_step_to_t_target(self, x_cur, t_cur, t_target, clip_x0=False):
        B = x_cur.shape[0]
        t_cur_b = torch.full((B,), int(t_cur), device=self.device, dtype=torch.long)
        t_tar_b = torch.full((B,), int(t_target), device=self.device, dtype=torch.long)

        x0_hat, eps_hat = self.pred_x0_eps_at(x_cur, t_cur_b)

        if clip_x0:
            x0_hat = x0_hat.clamp(-1, 1)

        a_tar = self.sched.gather(self.sched.sqrt_alpha_bar, t_tar_b, x_cur.shape)
        b_tar = self.sched.gather(self.sched.sqrt_one_minus_alpha_bar, t_tar_b, x_cur.shape)
        return a_tar * x0_hat + b_tar * eps_hat


class MRIObjectivePack(MRIObjectiveCore):
    def __init__(self, model: nn.Module, sched: DiffusionSchedule, device: torch.device, objective: str):
        assert objective in ("v", "eps", "x0")
        self.model = model
        self.sched = sched
        self.device = device
        self.objective = objective

    @torch.no_grad()
    def sample_ddim(self, shape_bchwd, steps=50, seed=0, clip_x0=True):
        set_seed(seed)
        B, C, H, W, D = shape_bchwd
        x = torch.randn((B, C, D, H, W), device=self.device)
        ts = make_ddim_timesteps(self.sched.T, steps, descending=True).to(self.device)
        for i in range(len(ts) - 1):
            t = int(ts[i].item())
            t_prev = int(ts[i+1].item())
            x = self.ddim_step_to_t_target(x, t, t_prev, clip_x0=clip_x0)
        return x



    @torch.no_grad()
    def invert_x0_to_xT(self, x0_bchwd, steps=50, seed=0, clip_x0=False, show_pbar=True):
        set_seed(seed)
        x0 = x0_bchwd.float().to(self.device)          # no normalization
        x0 = to_model_layout(x0)                       # (B,C,D,H,W)

        ts = make_ddim_timesteps(self.sched.T, steps, descending=False).to(self.device)
        x = x0
        
        iterator = range(len(ts) - 1)
        if show_pbar:
            iterator = tqdm(iterator, desc="Inverse DDIM", leave=False)
        for i in iterator:
            t = int(ts[i].item())
            t_next = int(ts[i+1].item())
            x = self.ddim_step_to_t_target(x, t, t_next, clip_x0=clip_x0)
        return x  # (B,C,D,H,W)

    @torch.no_grad()
    def decode_xT_to_x0(self, xT_bcdhw, steps=50, clip_x0=True, show_pbar=True):
        ts = make_ddim_timesteps(self.sched.T, steps, descending=True).to(self.device)
        x = xT_bcdhw
        iterator = range(len(ts) - 1)
        if show_pbar:
            iterator = tqdm(iterator, desc="Forward DDIM", leave=False)
        for i in iterator:
            t = int(ts[i].item())
            t_prev = int(ts[i+1].item())
            x = self.ddim_step_to_t_target(x, t, t_prev, clip_x0=clip_x0)
        return x

    @torch.no_grad()
    def reconstruct_roundtrip(self, x0_bchwd, steps=50, seed=0, clip_inverse=False, clip_decode=True):
        x0 = x0_bchwd.float().to(self.device)
        x0_dhw = to_model_layout(x0)
        xT = self.invert_x0_to_xT(x0_bchwd, steps=steps, seed=seed, clip_x0=clip_inverse)
        x0_hat = self.decode_xT_to_x0(xT, steps=steps, clip_x0=clip_decode)
        return {"x0": x0_dhw, "xT": xT, "x0_hat": x0_hat}


def predictive_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Mean predictive entropy over a batch.
    Higher => more confusion / uncertainty.
    """
    probs = torch.softmax(logits, dim=1)
    ent = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)
    return ent.mean()


def grad_global_norm_from_loss(loss: torch.Tensor, params) -> torch.Tensor:
    """
    Compute global L2 norm of gradients of `loss` w.r.t. `params`.
    Uses autograd.grad, so call BEFORE backward().
    """
    params = [p for p in params if p.requires_grad]
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )

    sq = loss.new_zeros(())
    for g in grads:
        if g is not None:
            sq = sq + g.detach().float().pow(2).sum()

    return torch.sqrt(sq + 1e-12)

def confusion_to_bal_acc(cm: torch.Tensor):
    row_sum = cm.sum(dim=1).float()
    diag = cm.diag().float()
    mask = row_sum > 0
    recall = torch.zeros_like(row_sum)
    recall[mask] = diag[mask] / row_sum[mask]
    return recall[mask].mean().item()


def pred_fraction(cm: torch.Tensor, cls: int):
    total = cm.sum().item()
    if total == 0:
        return 0.0
    return cm[:, cls].sum().item() / total

class MRIObjectiveFlexiblePack(MRIObjectiveCore):
    def __init__(
        self,
        model: nn.Module,
        sched,
        device: torch.device,
        config: MRITrainConfig,
        discriminator: nn.Module | None = None,
        ssim_loss_fn: nn.Module | None = None,
        # domain_class_weights: torch.Tensor | None = None,   # NEW
        disc_class_weights: torch.Tensor | None = None,   # renamed from domain_class_weights for clarity
        dann_class_weights: torch.Tensor | None = None,   # renamed from domain_class_weights for clarity

    ):
        super().__init__(model, sched, device, config.objective)
        
        self.cfg = config

        self.discriminator = discriminator
        self.ssim_loss_fn = ssim_loss_fn
        # self.domain_class_weights = None if domain_class_weights is None else domain_class_weights.to(device)
        self.disc_class_weights = None if disc_class_weights is None else disc_class_weights.to(device)
        self.dann_class_weights = None if dann_class_weights is None else dann_class_weights.to(device)

        if self.cfg.use_discriminator and self.discriminator is None:
            raise ValueError("use_discriminator=True but discriminator is None.")

        if self.cfg.use_ssim and self.ssim_loss_fn is None:
            raise ValueError("use_ssim=True but ssim_loss_fn is None.")

    # -------------------------------------------------
    # common noisy batch
    # -------------------------------------------------
    def make_noisy_batch(self, x0, t_b=None):
        B = x0.shape[0]
        if t_b is None:
            t_b = torch.randint(0, self.sched.T, (B,), device=self.device, dtype=torch.long)

        xt, eps, a, b = sample_xt_shared(x0, t_b, self.sched)

        return {
            "x0": x0,
            "t": t_b,
            "xt": xt,
            "eps": eps,
            "a": a,
            "b": b,
            "logsnr": t_to_logsnr(t_b, self.sched),
        }

    # -------------------------------------------------
    # optional extra losses on x0_hat
    # -------------------------------------------------
    def _prepare_x0_for_aux_losses(self, x0_hat):
        if self.cfg.clip_x0_for_losses:
            return x0_hat.clamp(-1, 1)
        return x0_hat

    def compute_aux_losses(self, x0_hat, x0):
        x0_hat_aux = self._prepare_x0_for_aux_losses(x0_hat)

        zero = x0_hat.new_zeros(())
        loss_mae = zero
        loss_ssim = zero

        if self.cfg.use_mae:
            loss_mae = F.l1_loss(x0_hat_aux, x0)

        if self.cfg.use_ssim:
            # assumes ssim_loss_fn returns a loss (smaller is better)
            loss_ssim = self.ssim_loss_fn(x0_hat_aux, x0)

        return {
            "loss_mae": loss_mae,
            "loss_ssim": loss_ssim,
        }

    # -------------------------------------------------
    # generator step
    # -------------------------------------------------
    def generator_forward(self, x0, domain_labels=None, t_b=None):
        batch = self.make_noisy_batch(x0, t_b=t_b)

        pred = self.predict(batch["xt"], batch["t"])
        target = self.diffusion_target(batch["x0"], batch["eps"], batch["a"], batch["b"])
        loss_diff = F.mse_loss(pred, target)

        x0_hat, eps_hat = self.pred_x0_eps_from_pred(batch["xt"], pred, batch["a"], batch["b"])

        aux = self.compute_aux_losses(x0_hat, batch["x0"])
        loss_mae = aux["loss_mae"]
        loss_ssim = aux["loss_ssim"]

        zero = x0_hat.new_zeros(())
        loss_dann = zero
        domain_acc = zero

        # NEW diagnostics
        entropy_recon = zero

        # real-at-t0 metrics
        ce_real_t0 = zero
        acc_real_t0 = zero
        entropy_real_t0 = zero

        # optional: real with same timestep context as recon
        ce_real_matcht = zero

        # predicted labels for confusion matrices
        pred_real_t0 = None
        pred_recon = None

        if self.cfg.use_discriminator:
            if domain_labels is None:
                raise ValueError("domain_labels is required when use_discriminator=True")

            x0_hat_disc = x0_hat.clamp(-1, 1) if self.cfg.clip_x0_for_disc else x0_hat

            # ---- adversarial branch on reconstructed image ----
            domain_logits = self.discriminator(x0_hat_disc, batch["logsnr"], use_grl=True)
            ce_per = F.cross_entropy(
                domain_logits,
                domain_labels,
                weight=self.dann_class_weights,
                reduction="none",
            )

            w_t = torch.sigmoid((batch["logsnr"] - self.cfg.tau_loss) / (self.cfg.s_loss + 1e-8)).detach()
            w_t = self.cfg.min_t_weight + (1.0 - self.cfg.min_t_weight) * w_t

            loss_dann = (w_t * ce_per).sum() / w_t.sum().clamp_min(1e-8)
            
            with torch.no_grad():
                domain_acc = (domain_logits.argmax(dim=1) == domain_labels).float().mean()
                entropy_recon = predictive_entropy_from_logits(domain_logits.detach())

            # ---- probe the SAME discriminator on the real x0 at t=0 ----
            with torch.no_grad():
                t0 = torch.zeros_like(batch["t"])
                logsnr_real_t0 = t_to_logsnr(t0, self.sched)

                real_logits_t0 = self.discriminator(batch["x0"], logsnr_real_t0, use_grl=False)
                ce_real_t0 = F.cross_entropy(
                    real_logits_t0,
                    domain_labels,
                    weight=self.disc_class_weights,
                )
                acc_real_t0 = (real_logits_t0.argmax(dim=1) == domain_labels).float().mean()
                entropy_real_t0 = predictive_entropy_from_logits(real_logits_t0)

                # optional: same timestep context as recon, for apples-to-apples weighted CE
                real_logits_matcht = self.discriminator(batch["x0"], batch["logsnr"], use_grl=False)
                ce_real_matcht_per = F.cross_entropy(
                    real_logits_matcht,
                    domain_labels,
                    weight=self.dann_class_weights,
                    reduction="none",
                )
                ce_real_matcht = (w_t * ce_real_matcht_per).sum() / w_t.sum().clamp_min(1e-8)

                pred_real_t0 = real_logits_t0.argmax(dim=1)
                pred_recon = domain_logits.argmax(dim=1)

        loss_total = (
            loss_diff
            + self.cfg.lambda_mae * loss_mae
            + self.cfg.lambda_ssim * loss_ssim
            + self.cfg.lambda_dann * loss_dann
        )

        return {
            "loss": loss_total,
            "loss_diff": loss_diff,
            "loss_mae": loss_mae,
            "loss_ssim": loss_ssim,
            "loss_dann": loss_dann,
            "domain_acc": domain_acc,
            "entropy_recon": entropy_recon,   # NEW
            "ce_real_t0": ce_real_t0,
            "acc_real_t0": acc_real_t0,
            "entropy_real_t0": entropy_real_t0,
            "ce_real_matcht": ce_real_matcht,
            "pred_real_t0": None if pred_real_t0 is None else pred_real_t0.detach(),
            "pred_recon": None if pred_recon is None else pred_recon.detach(),
            "pred": pred,
            "target": target,
            "x0_hat": x0_hat,
            "eps_hat": eps_hat,
            "xt": batch["xt"],
            "t": batch["t"],
            "logsnr": batch["logsnr"],
        }

    # -------------------------------------------------
    # discriminator-only step
    # -------------------------------------------------
    def discriminator_forward(self, x0, domain_labels, t_b=None):
        if not self.cfg.use_discriminator:
            raise RuntimeError("discriminator_forward called while use_discriminator=False")

        batch = self.make_noisy_batch(x0, t_b=t_b)

        with torch.no_grad():
            pred = self.predict(batch["xt"], batch["t"])
            x0_hat, _ = self.pred_x0_eps_from_pred(batch["xt"], pred, batch["a"], batch["b"])

            if self.cfg.clip_x0_for_disc:
                x0_hat = x0_hat.clamp(-1, 1)

        domain_logits = self.discriminator(x0_hat.detach(), batch["logsnr"], use_grl=False)
        ce_per = F.cross_entropy(
            domain_logits,
            domain_labels,
            weight=self.disc_class_weights,
            reduction="none",
        )

        w_t = torch.sigmoid((batch["logsnr"] - self.cfg.tau_loss) / (self.cfg.s_loss + 1e-8)).detach()
        w_t = self.cfg.min_t_weight + (1.0 - self.cfg.min_t_weight) * w_t

        loss_disc = (w_t * ce_per).sum() / w_t.sum().clamp_min(1e-8)
        
        with torch.no_grad():
            domain_acc = (domain_logits.argmax(dim=1) == domain_labels).float().mean()

        return {
            "loss": loss_disc,
            "loss_disc": loss_disc,
            "domain_acc": domain_acc,
            "x0_hat": x0_hat,
            "xt": batch["xt"],
            "t": batch["t"],
            "logsnr": batch["logsnr"],
        }



# -------------------------
# Training functions (same style across objectives)
# -------------------------
def train_objective_baseline(
    *,
    objective: str,                 # "v" | "eps" | "x0"
    train_dl: Iterable,
    device: torch.device,
    sched: DiffusionSchedule,
    in_channels: int,
    base: int = 16,
    t_dim: int = 128,
    epochs: int = 1,
    lr: float = 2e-4,
    weight_decay: float = 1e-4,
    use_amp: bool = True,
    grad_accum: int = 1,
    seed: int = 0,
    log_every: int = 10,
):
    set_seed(seed)
    model = SimpleUNet3D(in_ch=in_channels, base=base, t_dim=t_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model.train()
    step = 0

    for ep in range(epochs):
        for it, batch in enumerate(train_dl):
            x0 = batch[0] if isinstance(batch, (tuple, list)) else batch
            x0 = x0.float().to(device)            # no normalization
            x0 = to_model_layout(x0)              # (B,C,D,H,W)

            B = x0.shape[0]
            t = torch.randint(0, sched.T, (B,), device=device, dtype=torch.long)
            xt, eps, a, b = sample_xt_shared(x0, t, sched)

            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(xt, t)

                if objective == "v":
                    target = v_target_from_shared(x0, eps, a, b)
                elif objective == "eps":
                    target = eps
                else:  # "x0"
                    target = x0

                loss = F.mse_loss(pred, target) / grad_accum

            scaler.scale(loss).backward()

            if (it + 1) % grad_accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            if step % log_every == 0:
                print(f"[{objective}] ep {ep+1}/{epochs} step {step} loss={float(loss.item()*grad_accum):.6f}")
            step += 1

    return model

from collections import defaultdict

# def train_objective_flexible(
#     *,
#     train_dl,
#     device: torch.device,
#     sched,
#     in_channels: int,
#     config: MRITrainConfig,
#     num_domains: int | None = None,
#     domain_col: int | None = None,
#     base: int = 16,
#     t_dim: int = 128,
#     disc_base: int = 16,
#     epochs: int = 1,
#     lr_g: float = 2e-4,
#     lr_d: float = 2e-4,
#     weight_decay: float = 1e-4,
#     use_amp: bool = True,
#     seed: int = 0,
#     log_every: int = 10,
#     ssim_loss_fn: nn.Module | None = None,
#     model: nn.Module | None = None,              # NEW
#     discriminator: nn.Module | None = None,      # NEW
# ):

#     if model is None:
#         if seed is not None:
#             set_seed(seed)
#         model = SimpleUNet3D(in_ch=in_channels, base=base, t_dim=t_dim).to(device)
#     else:
#         model = model.to(device)
        
#     # discriminator = None
#     if config.use_discriminator and discriminator is None:
#         if num_domains is None:
#             raise ValueError("num_domains must be provided when use_discriminator=True")
#         if domain_col is None:
#             raise ValueError("domain_col must be provided when use_discriminator=True")

#         input_shape = infer_model_input_shape_from_loader(train_dl)   # NEW
#         print(f"[disc] inferred input_shape = {input_shape}")

#         discriminator = StrongDiffusionImageDiscriminator3D(
#             input_shape=input_shape,
#             in_ch=input_shape[0],
#             n_domains=num_domains,
#             hidden_size=1024,
#             batch_norm=False,
#             dropout=0.1,
#             use_conv5=True,
#             use_instancenorm=True,
#             t_emb_dim=64,
#             grl_alpha=1.0,
#             use_time_conditioning=False,
#             tau_D=0.0,
#             s_D=1.0,
#         ).to(device)

#     pack = MRIObjectiveFlexiblePack(
#         model=model,
#         discriminator=discriminator,
#         sched=sched,
#         device=device,
#         config=config,
#         ssim_loss_fn=ssim_loss_fn,
#         disc_class_weights=config.disc_class_weights,
#         dann_class_weights=config.dann_class_weights,
#     )

#     opt_g = torch.optim.AdamW(model.parameters(), lr=lr_g, weight_decay=weight_decay)
#     opt_d = None
#     if config.use_discriminator and (not config.freeze_discriminator):
#         opt_d = torch.optim.AdamW(discriminator.parameters(), lr=lr_d, weight_decay=weight_decay)
        
        
#     amp_enabled = bool(use_amp and device.type == "cuda")
#     scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

#     model.train()
#     if discriminator is not None:
#         if config.freeze_discriminator:
#             discriminator.eval()
#             set_requires_grad(discriminator, False)
#         else:
#             discriminator.train()
            
#     step = 0

#     for ep in range(epochs):
        
#         epoch_meter = defaultdict(float)
#         epoch_samples = 0
#         epoch_batches = 0
#         cm_real_t0 = None
#         cm_recon = None
#         if config.use_discriminator:
#             cm_real_t0 = torch.zeros(num_domains, num_domains, dtype=torch.long)
#             cm_recon = torch.zeros(num_domains, num_domains, dtype=torch.long)
            
#         for it, batch in enumerate(train_dl):
#             x = batch[0] if isinstance(batch, (tuple, list)) else batch
#             metadata = batch[2] if isinstance(batch, (tuple, list)) and len(batch) >= 3 else None

#             x0 = x.float().to(device)
#             x0 = to_model_layout(x0)  # [B,C,D,H,W]

#             domain_labels = None
#             if config.use_discriminator:
#                 domain_labels = extract_domain_labels(metadata, domain_col, device)

#             B = x0.shape[0]
#             t_b = torch.randint(0, sched.T, (B,), device=device, dtype=torch.long)

#             # -------------------------------------------------
#             # 1) discriminator update (only if enabled)
#             # -------------------------------------------------
#             d_out = None
#             if config.use_discriminator and (not config.freeze_discriminator):
#                 set_requires_grad(discriminator, True)
#                 opt_d.zero_grad(set_to_none=True)

#                 with torch.cuda.amp.autocast(enabled=amp_enabled):
#                     d_out = pack.discriminator_forward(x0, domain_labels, t_b=t_b)
#                     loss_d = d_out["loss"]

#                 scaler.scale(loss_d).backward()
#                 scaler.step(opt_d)
#                 scaler.update()

#             # -------------------------------------------------
#             # 2) generator / diffusion update
#             # -------------------------------------------------
#             if discriminator is not None:
#                 set_requires_grad(discriminator, False)

#             opt_g.zero_grad(set_to_none=True)

#             with torch.cuda.amp.autocast(enabled=amp_enabled):
#                 g_out = pack.generator_forward(x0, domain_labels=domain_labels, t_b=t_b)
#                 loss_g = g_out["loss"]

#             # ---------------------------------
#             # NEW diagnostics: compute gradient norms and their ratio for adversarial vs diffusion losses
#             # ---------------------------------
#             adv_grad_ratio = None
#             grad_norm_diff = None
#             grad_norm_dann = None

#             if config.use_discriminator:
#                 model_params = [p for p in model.parameters() if p.requires_grad]

#                 # gradient norm from diffusion loss only
#                 grad_norm_diff = grad_global_norm_from_loss(
#                     g_out["loss_diff"],
#                     model_params,
#                 )

#                 # gradient norm from adversarial term only
#                 grad_norm_dann = grad_global_norm_from_loss(
#                     config.lambda_dann * g_out["loss_dann"],
#                     model_params,
#                 )

#                 adv_grad_ratio = (grad_norm_dann / grad_norm_diff.clamp_min(1e-12)).detach()
                
#             # -------------------------------
#             # continue with backward and step
#             # -------------------------------
            
#             scaler.scale(loss_g).backward()
#             scaler.step(opt_g)
#             scaler.update()

#             if discriminator is not None and (not config.freeze_discriminator):
#                 set_requires_grad(discriminator, True)
                
#             # -------------------------------------------------
#             # logging
#             # -------------------------------------------------
#             if config.use_discriminator and d_out is None:
#                 zero = x0.new_zeros(())
#                 d_out = {
#                     "loss_disc": zero,
#                     "domain_acc": zero,
#                 }
                
#             per_batch_print = False
#             if per_batch_print and step % log_every == 0:
#                 msg = (
#                     f"[{config.objective}] ep {ep+1}/{epochs} step {step} "
#                     f"loss_total={g_out['loss'].item():.6f} "
#                     f"loss_diff={g_out['loss_diff'].item():.6f}"
#                 )

#                 if config.use_mae:
#                     msg += f" loss_mae={g_out['loss_mae'].item():.6f}"

#                 if config.use_ssim:
#                     msg += f" loss_ssim={g_out['loss_ssim'].item():.6f}"

#                 if config.use_discriminator:
#                     msg += (
#                         f" loss_dann={g_out['loss_dann'].item():.6f} "
#                         f"loss_disc={d_out['loss_disc'].item():.6f} "
#                         f"D_acc_g={g_out['domain_acc'].item():.4f} "
#                         f"D_acc_d={d_out['domain_acc'].item():.4f} "
#                         f"CE_real={g_out['ce_real'].item():.6f} "
#                         f"Acc_real={g_out['acc_real'].item():.4f} "
#                         f"H_recon={g_out['entropy_recon'].item():.6f} "
#                         f"H_real={g_out['entropy_real'].item():.6f}"
#                     )

#                     if adv_grad_ratio is not None:
#                         msg += (
#                             f" |grad_dann|={grad_norm_dann.item():.6e} "
#                             f"|grad_diff|={grad_norm_diff.item():.6e} "
#                             f"grad_ratio={adv_grad_ratio.item():.6f}"
#                         )

#                 print(msg)
#             bsz = x0.shape[0]
#             epoch_samples += bsz
#             epoch_batches += 1

#             if config.use_discriminator:
#                 y_true = domain_labels.detach().cpu()
#                 y_pred_real = g_out["pred_real_t0"].detach().cpu()
#                 y_pred_recon = g_out["pred_recon"].detach().cpu()

#                 for yt, yp in zip(y_true.tolist(), y_pred_real.tolist()):
#                     cm_real_t0[yt, yp] += 1

#                 for yt, yp in zip(y_true.tolist(), y_pred_recon.tolist()):
#                     cm_recon[yt, yp] += 1
                    
                    
#             epoch_meter["loss_total"] += float(g_out['loss'].item()) * bsz
#             epoch_meter["loss_diff"]  += float(g_out['loss_diff'].item()) * bsz

#             if config.use_discriminator:
#                 epoch_meter["loss_dann"] += float(g_out["loss_dann"].item()) * bsz
#                 epoch_meter["loss_disc"] += float(d_out["loss_disc"].item()) * bsz
#                 epoch_meter["D_acc_g"]   += float(g_out["domain_acc"].item()) * bsz
#                 epoch_meter["D_acc_d"]   += float(d_out["domain_acc"].item()) * bsz

#                 epoch_meter["CE_real_t0"] += float(g_out["ce_real_t0"].item()) * bsz
#                 epoch_meter["Acc_real_t0"] += float(g_out["acc_real_t0"].item()) * bsz
#                 epoch_meter["H_real_t0"] += float(g_out["entropy_real_t0"].item()) * bsz

#                 epoch_meter["CE_real_matcht"] += float(g_out["ce_real_matcht"].item()) * bsz
#                 epoch_meter["H_recon"] += float(g_out["entropy_recon"].item()) * bsz

#                 epoch_meter["grad_dann"] += float(grad_norm_dann.item()) * bsz
#                 epoch_meter["grad_diff"] += float(grad_norm_diff.item()) * bsz
#                 epoch_meter["grad_ratio"] += float(adv_grad_ratio.item()) * bsz
                
#             step += 1
        
#         den = max(epoch_samples, 1)

#         epoch_stats = {k: v / den for k, v in epoch_meter.items()}

#         if config.use_discriminator:
#             bal_acc_real_t0 = confusion_to_bal_acc(cm_real_t0)
#             bal_acc_recon = confusion_to_bal_acc(cm_recon)

#             pred12_frac_real = pred_fraction(cm_real_t0, 12)
#             pred12_frac_recon = pred_fraction(cm_recon, 12)
            
#         if config.use_discriminator:
#             print(
#                 f"[{config.objective}] ep {ep+1}/{epochs} "
#                 f"loss_total={epoch_stats['loss_total']:.6f} "
#                 f"loss_diff={epoch_stats['loss_diff']:.6f} "
#                 f"loss_dann={epoch_stats['loss_dann']:.6f} "
#                 f"loss_disc={epoch_stats['loss_disc']:.6f} "
#                 f"D_acc_g={epoch_stats['D_acc_g']:.4f} "
#                 f"D_acc_d={epoch_stats['D_acc_d']:.4f} "
#                 f"CE_real_t0={epoch_stats['CE_real_t0']:.6f} "
#                 f"Acc_real_t0={epoch_stats['Acc_real_t0']:.4f} "
#                 f"CE_real_matcht={epoch_stats['CE_real_matcht']:.6f} "
#                 f"H_recon={epoch_stats['H_recon']:.6f} "
#                 f"H_real_t0={epoch_stats['H_real_t0']:.6f} "
#                 f"bal_acc_real_t0={bal_acc_real_t0:.4f} "
#                 f"bal_acc_recon={bal_acc_recon:.4f} "
#                 f"pred12_real={pred12_frac_real:.4f} "
#                 f"pred12_recon={pred12_frac_recon:.4f} "
#                 f"|grad_dann|={epoch_stats['grad_dann']:.6e} "
#                 f"|grad_diff|={epoch_stats['grad_diff']:.6e} "
#                 f"grad_ratio={epoch_stats['grad_ratio']:.6f} "
#                 f"n_batches={epoch_batches}"
#             )
#         else:
#             print(
#                 f"[{config.objective}] ep {ep+1}/{epochs} "
#                 f"loss_total={epoch_stats['loss_total']:.6f} "
#                 f"loss_diff={epoch_stats['loss_diff']:.6f} "
#                 f"n_batches={epoch_batches}"
#             )

#     return model, discriminator, pack

def train_objective_flexible(
    *,
    train_dl,
    device: torch.device,
    sched,
    in_channels: int,
    config: MRITrainConfig,
    num_domains: int | None = None,
    domain_col: int | None = None,
    base: int = 16,
    t_dim: int = 128,
    disc_base: int = 16,
    epochs: int = 1,
    lr_g: float = 2e-4,
    lr_d: float = 2e-4,
    weight_decay: float = 1e-4,
    use_amp: bool = True,
    grad_accum: int = 1,
    seed: int | None = None,
    log_every: int = 10,
    ssim_loss_fn: nn.Module | None = None,
    model: nn.Module | None = None,
    discriminator: nn.Module | None = None,
):
    # Seed only if this function itself creates the model.
    # If prepare_experiment already created the model, do not reset RNG here.
    if model is None:
        if seed is not None:
            set_seed(seed)
        model = SimpleUNet3D(in_ch=in_channels, base=base, t_dim=t_dim).to(device)
    else:
        model = model.to(device)

    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    # ============================================================
    # EXACT BASELINE-LIKE PATH
    # No discriminator, no x0_hat, no eps_hat, no aux losses.
    # ============================================================
    if is_plain_base_config(config):
        if discriminator is not None:
            discriminator = None

        opt_g = torch.optim.AdamW(
            model.parameters(),
            lr=lr_g,
            weight_decay=weight_decay,
        )

        model.train()
        step = 0

        for ep in range(epochs):
            epoch_loss_sum = 0.0
            epoch_samples = 0
            epoch_batches = 0

            for it, batch in enumerate(train_dl):
                x0 = batch[0] if isinstance(batch, (tuple, list)) else batch

                x0 = x0.float().to(device)
                x0 = to_model_layout(x0)

                B = x0.shape[0]
                t = torch.randint(
                    0,
                    sched.T,
                    (B,),
                    device=device,
                    dtype=torch.long,
                )

                xt, eps, a, b = sample_xt_shared(x0, t, sched)

                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    pred = model(xt, t)
                    target = diffusion_target_from_objective(
                        config.objective,
                        x0,
                        eps,
                        a,
                        b,
                    )
                    loss_raw = F.mse_loss(pred, target)
                    loss = loss_raw / grad_accum

                scaler.scale(loss).backward()

                if (it + 1) % grad_accum == 0:
                    scaler.step(opt_g)
                    scaler.update()
                    opt_g.zero_grad(set_to_none=True)

                if step % log_every == 0:
                    print(
                        f"[{config.objective}] ep {ep+1}/{epochs} "
                        f"step {step} loss={float(loss_raw.item()):.6f}"
                    )

                bsz = x0.shape[0]
                epoch_loss_sum += float(loss_raw.item()) * bsz
                epoch_samples += bsz
                epoch_batches += 1
                step += 1

            print(
                f"[{config.objective}] ep {ep+1}/{epochs} "
                f"loss_total={epoch_loss_sum / max(epoch_samples, 1):.6f} "
                f"loss_diff={epoch_loss_sum / max(epoch_samples, 1):.6f} "
                f"n_batches={epoch_batches}"
            )

        return model, None, None

    # ============================================================
    # FULL FLEXIBLE PATH
    # Used only when DANN / discriminator / MAE / SSIM is actually enabled.
    # ============================================================

    if config.use_discriminator and discriminator is None:
        if num_domains is None:
            raise ValueError("num_domains must be provided when use_discriminator=True")
        if domain_col is None:
            raise ValueError("domain_col must be provided when use_discriminator=True")

        input_shape = infer_model_input_shape_from_loader(train_dl)
        print(f"[disc] inferred input_shape = {input_shape}")

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
            grl_alpha=1.0,
            use_time_conditioning=False,
            tau_D=0.0,
            s_D=1.0,
        ).to(device)

    pack = MRIObjectiveFlexiblePack(
        model=model,
        discriminator=discriminator,
        sched=sched,
        device=device,
        config=config,
        ssim_loss_fn=ssim_loss_fn,
        disc_class_weights=config.disc_class_weights,
        dann_class_weights=config.dann_class_weights,
    )

    # Keep your existing DANN/flexible training code below this point.
    # For DANN, x0_hat, discriminator loss, entropy, confusion matrices,
    # and grad diagnostics are expected and useful.
    opt_g = torch.optim.AdamW(model.parameters(), lr=lr_g, weight_decay=weight_decay)
    opt_d = None
    if config.use_discriminator and (not config.freeze_discriminator):
        opt_d = torch.optim.AdamW(discriminator.parameters(), lr=lr_d, weight_decay=weight_decay)
        
        
    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    model.train()
    if discriminator is not None:
        if config.freeze_discriminator:
            discriminator.eval()
            set_requires_grad(discriminator, False)
        else:
            discriminator.train()
            
    step = 0

    for ep in range(epochs):
        
        epoch_meter = defaultdict(float)
        epoch_samples = 0
        epoch_batches = 0
        cm_real_t0 = None
        cm_recon = None
        if config.use_discriminator:
            cm_real_t0 = torch.zeros(num_domains, num_domains, dtype=torch.long)
            cm_recon = torch.zeros(num_domains, num_domains, dtype=torch.long)
            
        for it, batch in enumerate(train_dl):
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            metadata = batch[2] if isinstance(batch, (tuple, list)) and len(batch) >= 3 else None

            x0 = x.float().to(device)
            x0 = to_model_layout(x0)  # [B,C,D,H,W]

            domain_labels = None
            if config.use_discriminator:
                domain_labels = extract_domain_labels(metadata, domain_col, device)

            B = x0.shape[0]
            t_b = torch.randint(0, sched.T, (B,), device=device, dtype=torch.long)

            # -------------------------------------------------
            # 1) discriminator update (only if enabled)
            # -------------------------------------------------
            d_out = None
            if config.use_discriminator and (not config.freeze_discriminator):
                discriminator.train()
                set_requires_grad(discriminator, True)
                opt_d.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    d_out = pack.discriminator_forward(x0, domain_labels, t_b=t_b)
                    loss_d = d_out["loss"]

                scaler.scale(loss_d).backward()
                scaler.step(opt_d)
                scaler.update()

            # -------------------------------------------------
            # 2) generator / diffusion update
            # -------------------------------------------------
            if discriminator is not None:
                discriminator.eval()               # important: disable dropout
                set_requires_grad(discriminator, False)

            opt_g.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                g_out = pack.generator_forward(x0, domain_labels=domain_labels, t_b=t_b)
                loss_g = g_out["loss"]
            if config.use_discriminator and step == 0:
                print("[debug] pred.requires_grad:", g_out["pred"].requires_grad)
                print("[debug] x0_hat.requires_grad:", g_out["x0_hat"].requires_grad)
                print("[debug] loss_diff.requires_grad:", g_out["loss_diff"].requires_grad)
                print("[debug] loss_dann.requires_grad:", g_out["loss_dann"].requires_grad)
                print("[debug] loss_total.requires_grad:", g_out["loss"].requires_grad)

                assert g_out["pred"].requires_grad
                assert g_out["x0_hat"].requires_grad
                assert g_out["loss_diff"].requires_grad
                assert g_out["loss_dann"].requires_grad
                assert g_out["loss"].requires_grad
            # ---------------------------------
            # NEW diagnostics: compute gradient norms and their ratio for adversarial vs diffusion losses
            # ---------------------------------
            adv_grad_ratio = None
            grad_norm_diff = None
            grad_norm_dann = None

            if config.use_discriminator:
                model_params = [p for p in model.parameters() if p.requires_grad]

                # gradient norm from diffusion loss only
                grad_norm_diff = grad_global_norm_from_loss(
                    g_out["loss_diff"],
                    model_params,
                )

                # gradient norm from adversarial term only
                grad_norm_dann = grad_global_norm_from_loss(
                    config.lambda_dann * g_out["loss_dann"],
                    model_params,
                )

                adv_grad_ratio = (grad_norm_dann / grad_norm_diff.clamp_min(1e-12)).detach()
                
            # -------------------------------
            # continue with backward and step
            # -------------------------------
            
            scaler.scale(loss_g).backward()
            scaler.step(opt_g)
            scaler.update()

            if discriminator is not None and (not config.freeze_discriminator):
                discriminator.train()
                set_requires_grad(discriminator, True)
                
            # -------------------------------------------------
            # logging
            # -------------------------------------------------
            if config.use_discriminator and d_out is None:
                zero = x0.new_zeros(())
                d_out = {
                    "loss_disc": zero,
                    "domain_acc": zero,
                }
                
            per_batch_print = False
            if per_batch_print and step % log_every == 0:
                msg = (
                    f"[{config.objective}] ep {ep+1}/{epochs} step {step} "
                    f"loss_total={g_out['loss'].item():.6f} "
                    f"loss_diff={g_out['loss_diff'].item():.6f}"
                )

                if config.use_mae:
                    msg += f" loss_mae={g_out['loss_mae'].item():.6f}"

                if config.use_ssim:
                    msg += f" loss_ssim={g_out['loss_ssim'].item():.6f}"

                if config.use_discriminator:
                    msg += (
                        f" loss_dann={g_out['loss_dann'].item():.6f} "
                        f"loss_disc={d_out['loss_disc'].item():.6f} "
                        f"D_acc_g={g_out['domain_acc'].item():.4f} "
                        f"D_acc_d={d_out['domain_acc'].item():.4f} "
                        f"CE_real={g_out['ce_real'].item():.6f} "
                        f"Acc_real={g_out['acc_real'].item():.4f} "
                        f"H_recon={g_out['entropy_recon'].item():.6f} "
                        f"H_real={g_out['entropy_real'].item():.6f}"
                    )

                    if adv_grad_ratio is not None:
                        msg += (
                            f" |grad_dann|={grad_norm_dann.item():.6e} "
                            f"|grad_diff|={grad_norm_diff.item():.6e} "
                            f"grad_ratio={adv_grad_ratio.item():.6f}"
                        )

                print(msg)
            bsz = x0.shape[0]
            epoch_samples += bsz
            epoch_batches += 1

            if config.use_discriminator:
                y_true = domain_labels.detach().cpu()
                y_pred_real = g_out["pred_real_t0"].detach().cpu()
                y_pred_recon = g_out["pred_recon"].detach().cpu()

                for yt, yp in zip(y_true.tolist(), y_pred_real.tolist()):
                    cm_real_t0[yt, yp] += 1

                for yt, yp in zip(y_true.tolist(), y_pred_recon.tolist()):
                    cm_recon[yt, yp] += 1
                    
                    
            epoch_meter["loss_total"] += float(g_out['loss'].item()) * bsz
            epoch_meter["loss_diff"]  += float(g_out['loss_diff'].item()) * bsz

            if config.use_discriminator:
                epoch_meter["loss_dann"] += float(g_out["loss_dann"].item()) * bsz
                epoch_meter["loss_disc"] += float(d_out["loss_disc"].item()) * bsz
                epoch_meter["D_acc_g"]   += float(g_out["domain_acc"].item()) * bsz
                epoch_meter["D_acc_d"]   += float(d_out["domain_acc"].item()) * bsz

                epoch_meter["CE_real_t0"] += float(g_out["ce_real_t0"].item()) * bsz
                epoch_meter["Acc_real_t0"] += float(g_out["acc_real_t0"].item()) * bsz
                epoch_meter["H_real_t0"] += float(g_out["entropy_real_t0"].item()) * bsz

                epoch_meter["CE_real_matcht"] += float(g_out["ce_real_matcht"].item()) * bsz
                epoch_meter["H_recon"] += float(g_out["entropy_recon"].item()) * bsz

                epoch_meter["grad_dann"] += float(grad_norm_dann.item()) * bsz
                epoch_meter["grad_diff"] += float(grad_norm_diff.item()) * bsz
                epoch_meter["grad_ratio"] += float(adv_grad_ratio.item()) * bsz
                
            step += 1
        
        den = max(epoch_samples, 1)

        epoch_stats = {k: v / den for k, v in epoch_meter.items()}

        if config.use_discriminator:
            bal_acc_real_t0 = confusion_to_bal_acc(cm_real_t0)
            bal_acc_recon = confusion_to_bal_acc(cm_recon)

            majority_cls = int(config.site_counts.argmax().item())
            pred_majority_frac_real = pred_fraction(cm_real_t0, majority_cls)
            pred_majority_frac_recon = pred_fraction(cm_recon, majority_cls)
            
        if config.use_discriminator:
            print(
                f"[{config.objective}] ep {ep+1}/{epochs} "
                f"loss_total={epoch_stats['loss_total']:.6f} "
                f"loss_diff={epoch_stats['loss_diff']:.6f} "
                f"loss_dann={epoch_stats['loss_dann']:.6f} "
                f"loss_disc={epoch_stats['loss_disc']:.6f} "
                f"D_acc_g={epoch_stats['D_acc_g']:.4f} "
                f"D_acc_d={epoch_stats['D_acc_d']:.4f} "
                f"CE_real_t0={epoch_stats['CE_real_t0']:.6f} "
                f"Acc_real_t0={epoch_stats['Acc_real_t0']:.4f} "
                f"CE_real_matcht={epoch_stats['CE_real_matcht']:.6f} "
                f"H_recon={epoch_stats['H_recon']:.6f} "
                f"H_real_t0={epoch_stats['H_real_t0']:.6f} "
                f"bal_acc_real_t0={bal_acc_real_t0:.4f} "
                f"bal_acc_recon={bal_acc_recon:.4f} "
                f"pred_majority_real={pred_majority_frac_real:.4f} "
                f"pred_majority_recon={pred_majority_frac_recon:.4f} "
                f"|grad_dann|={epoch_stats['grad_dann']:.6e} "
                f"|grad_diff|={epoch_stats['grad_diff']:.6e} "
                f"grad_ratio={epoch_stats['grad_ratio']:.6f} "
                f"n_batches={epoch_batches}"
            )
        else:
            print(
                f"[{config.objective}] ep {ep+1}/{epochs} "
                f"loss_total={epoch_stats['loss_total']:.6f} "
                f"loss_diff={epoch_stats['loss_diff']:.6f} "
                f"n_batches={epoch_batches}"
            )

    return model, discriminator, pack

def save_discriminator_only_checkpoint(path, discriminator, meta=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "discriminator_state_dict": discriminator.state_dict(),
        "meta": {} if meta is None else meta,
    }
    torch.save(ckpt, path)


def load_discriminator_only_checkpoint(path, device):
    return torch.load(path, map_location=device)


def build_real_only_discriminator(
    *,
    train_dl,
    device,
    num_domains,
    hidden_size=1024,
    dropout=0.1,
    use_conv5=True,
    use_instancenorm=True,
):
    """
    Build the SAME strong discriminator architecture as your DANN setup,
    but WITHOUT timestep embedding / conditioning.
    """
    input_shape = infer_model_input_shape_from_loader(train_dl)   # (C, D, H, W)

    discriminator = StrongDiffusionImageDiscriminator3D(
        input_shape=input_shape,
        in_ch=input_shape[0],
        n_domains=num_domains,
        hidden_size=hidden_size,
        batch_norm=False,
        dropout=dropout,
        use_conv5=use_conv5,
        use_instancenorm=use_instancenorm,
        t_emb_dim=64,                 # ignored when use_time_conditioning=False
        grl_alpha=1.0,                # irrelevant here since use_grl=False
        use_time_conditioning=False,  # IMPORTANT
        tau_D=0.0,
        s_D=1.0,
    ).to(device)

    return discriminator, input_shape


def train_discriminator_real_only(
    *,
    train_dl,
    device,
    num_domains,
    domain_col=0,
    epochs=30,
    lr=1e-4,
    weight_decay=0.0,
    use_amp=False,
    seed=0,
    disc_class_weights=None,
    ckpt_path=None,
    history_csv_path=None,
    pred_majority_class=12,
):
    """
    Train ONLY the discriminator on REAL images (x0), no diffusion update,
    no reconstruction, no timestep embedding.

    Logs epoch-level metrics similar to your current DANN logs:
      - loss_disc
      - CE_real_t0
      - Acc_real_t0
      - H_real_t0
      - bal_acc_real_t0
      - pred_majority_real
      - n_batches

    Returns:
      discriminator, history_df
    """
    set_seed(seed)

    discriminator, input_shape = build_real_only_discriminator(
        train_dl=train_dl,
        device=device,
        num_domains=num_domains,
        hidden_size=1024,
        dropout=0.1,
        use_conv5=True,
        use_instancenorm=True,
    )

    if disc_class_weights is None:
        disc_class_weights = MRITrainConfig.disc_class_weights
    disc_class_weights = disc_class_weights.to(device)

    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=lr, weight_decay=weight_decay)

    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    discriminator.train()
    history_rows = []

    for ep in range(epochs):
        epoch_meter = defaultdict(float)
        epoch_samples = 0
        epoch_batches = 0
        cm_real_t0 = torch.zeros(num_domains, num_domains, dtype=torch.long)

        pbar = tqdm(train_dl, desc=f"disc_real ep {ep+1}/{epochs}", leave=False)

        for batch in pbar:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            metadata = batch[2] if isinstance(batch, (tuple, list)) and len(batch) >= 3 else None

            x0 = x.float().to(device)
            x0 = to_model_layout(x0)   # -> [B, C, D, H, W]

            domain_labels = extract_domain_labels(metadata, domain_col, device)

            opt_d.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = discriminator(x0, logsnr=None, use_grl=False)
                loss_disc = F.cross_entropy(
                    logits,
                    domain_labels,
                    weight=disc_class_weights,
                )

            scaler.scale(loss_disc).backward()
            scaler.step(opt_d)
            scaler.update()

            with torch.no_grad():
                pred = logits.argmax(dim=1)
                acc_real_t0 = (pred == domain_labels).float().mean()
                entropy_real_t0 = predictive_entropy_from_logits(logits)
                ce_real_t0 = F.cross_entropy(
                    logits,
                    domain_labels,
                    weight=disc_class_weights,
                )

            bsz = x0.shape[0]
            epoch_samples += bsz
            epoch_batches += 1

            epoch_meter["loss_disc"] += float(loss_disc.item()) * bsz
            epoch_meter["CE_real_t0"] += float(ce_real_t0.item()) * bsz
            epoch_meter["Acc_real_t0"] += float(acc_real_t0.item()) * bsz
            epoch_meter["H_real_t0"] += float(entropy_real_t0.item()) * bsz

            y_true = domain_labels.detach().cpu()
            y_pred = pred.detach().cpu()
            for yt, yp in zip(y_true.tolist(), y_pred.tolist()):
                cm_real_t0[yt, yp] += 1

            pbar.set_postfix(
                loss=f"{float(loss_disc.item()):.4f}",
                acc=f"{float(acc_real_t0.item()):.3f}",
            )

        den = max(epoch_samples, 1)
        epoch_stats = {k: v / den for k, v in epoch_meter.items()}

        bal_acc_real_t0 = confusion_to_bal_acc(cm_real_t0)
        pred_majority_frac = pred_fraction(cm_real_t0, pred_majority_class)

        row = {
            "epoch": ep + 1,
            "loss_disc": epoch_stats["loss_disc"],
            "CE_real_t0": epoch_stats["CE_real_t0"],
            "Acc_real_t0": epoch_stats["Acc_real_t0"],
            "H_real_t0": epoch_stats["H_real_t0"],
            "bal_acc_real_t0": bal_acc_real_t0,
            f"pred{pred_majority_class}_real": pred_majority_frac,
            "n_batches": epoch_batches,
            "n_samples": epoch_samples,
        }
        history_rows.append(row)

        print(
            f"[disc_real_only] ep {ep+1}/{epochs} "
            f"loss_disc={row['loss_disc']:.6f} "
            f"CE_real_t0={row['CE_real_t0']:.6f} "
            f"Acc_real_t0={row['Acc_real_t0']:.4f} "
            f"H_real_t0={row['H_real_t0']:.6f} "
            f"bal_acc_real_t0={row['bal_acc_real_t0']:.4f} "
            f"pred{pred_majority_class}_real={row[f'pred{pred_majority_class}_real']:.4f} "
            f"n_batches={row['n_batches']}"
        )

    history_df = pd.DataFrame(history_rows)

    if ckpt_path is not None:
        save_discriminator_only_checkpoint(
            ckpt_path,
            discriminator,
            meta={
                "num_domains": num_domains,
                "domain_col": domain_col,
                "input_shape": input_shape,
                "use_time_conditioning": False,
                "hidden_size": 1024,
                "dropout": 0.1,
                "use_conv5": True,
                "use_instancenorm": True,
            },
        )

    if history_csv_path is not None:
        os.makedirs(os.path.dirname(history_csv_path), exist_ok=True)
        history_df.to_csv(history_csv_path, index=False)

    return discriminator, history_df


def prepare_experiment(
    *,
    spec: MRIExperimentSpec,
    train_dl,
    device,
    sched,
    in_channels=1,
    base=16,
    t_dim=128,
    disc_base=16,
    epochs=60,
    lr_g=2e-4,
    lr_d=2e-4,
    weight_decay=0.0,
    use_amp=True,
    seed=0,
    log_every=10,
    load_if_exists=True,
    force_train=False,
    init_model_ckpt=None,         # NEW
    ssim_loss_fn=None,
    grad_accum=1,
):
    # build config for flexible trainer
    cfg = MRITrainConfig(
        objective=spec.objective,
        use_discriminator=spec.use_discriminator,
        lambda_dann=spec.lambda_dann,
        clip_x0_for_disc=spec.clip_x0_for_disc,
        freeze_discriminator=spec.freeze_discriminator,
        use_mae=spec.use_mae,
        lambda_mae=spec.lambda_mae,
        use_ssim=spec.use_ssim,
        lambda_ssim=spec.lambda_ssim,
    )
    print(f"lambda dann = {cfg.lambda_dann}, lambda mae = {cfg.lambda_mae},\
          DDPM learning rate = {lr_g}, discriminator learning rate = {lr_d}, weight decay = {weight_decay},\
          objective = {cfg.objective}, use_discriminator = {cfg.use_discriminator}, use_mae = {cfg.use_mae}, use_ssim = {cfg.use_ssim}\
          batch size = {train_dl.batch_size}, grad_accum = {grad_accum}, epochs = {epochs}, seed = {seed}")
    # -------------------------------------------------
    # LOAD
    # -------------------------------------------------
    if load_if_exists and (not force_train) and os.path.exists(spec.ckpt_path):
        print(f"[{spec.name}] Loading checkpoint from: {spec.ckpt_path}")

        input_shape = None
        if spec.use_discriminator:
            input_shape = infer_model_input_shape_from_loader(train_dl)
        model, discriminator = build_model_and_discriminator(
            device=device,
            in_channels=in_channels,
            base=base,
            t_dim=t_dim,
            disc_base=disc_base,
            spec=spec,
            input_shape=input_shape,
        )

        ckpt = load_experiment_checkpoint(spec.ckpt_path, device=device)

        model.load_state_dict(ckpt["model_state_dict"])

        if spec.use_discriminator:
            if discriminator is None:
                raise RuntimeError("Expected discriminator, but build returned None.")
            disc_state = ckpt.get("discriminator_state_dict", None)
            if disc_state is None:
                raise RuntimeError("Checkpoint does not contain discriminator weights.")
            discriminator.load_state_dict(disc_state)

        infer_pack = MRIObjectivePack(model, sched, device, spec.objective)

        return {
            "spec": spec,
            "config": cfg,
            "model": model,
            "discriminator": discriminator,
            "infer_pack": infer_pack,
            "train_pack": None,
            "loaded": True,
        }

    # -------------------------------------------------
    # TRAIN
    # -------------------------------------------------
    print(f"[{spec.name}] Training from scratch or warm start")

    set_seed(seed)

    input_shape = None
    if spec.use_discriminator:
        input_shape = infer_model_input_shape_from_loader(train_dl)
        
    model, discriminator = build_model_and_discriminator(
        device=device,
        in_channels=in_channels,
        base=base,
        t_dim=t_dim,
        disc_base=disc_base,
        spec=spec,
        input_shape=input_shape,
    )
    if discriminator is not None:
        print("[DANN] GRL alpha =", discriminator.GRL.alpha)
        
    # Optional warm start from plain DDPM checkpoint
    if init_model_ckpt is not None:
        print(f"[{spec.name}] Initializing model weights from: {init_model_ckpt}")
        init_ckpt = load_experiment_checkpoint(init_model_ckpt, device=device)

        if isinstance(init_ckpt, dict) and "model_state_dict" in init_ckpt:
            init_state = init_ckpt["model_state_dict"]
        else:
            init_state = init_ckpt

        missing, unexpected = model.load_state_dict(init_state, strict=False)
        print(f"[{spec.name}] Warm-start load done. missing={missing}, unexpected={unexpected}")
        model.eval()
        with torch.no_grad():
            batch = next(iter(train_dl))   # or train_dl if test_dl is not visible here
            x0 = batch[0] if isinstance(batch, (tuple, list)) else batch
            x0 = x0[:2]

            tmp_pack = MRIObjectivePack(model, sched, device, spec.objective)
            out = tmp_pack.reconstruct_roundtrip(
                x0,
                steps=1000,
                seed=0,
                clip_inverse=False,
                clip_decode=True,
            )
            print(
                "[warm-start unrefined check] MAE:",
                F.l1_loss(out["x0_hat"], out["x0"]).item()
            )
        model.train()

    # Optional warm start for discriminator
    if spec.use_discriminator and spec.init_discriminator_ckpt is not None:
        print(f"[{spec.name}] Initializing discriminator weights from: {spec.init_discriminator_ckpt}")
        disc_ckpt = load_discriminator_only_checkpoint(spec.init_discriminator_ckpt, device=device)

        missing, unexpected = discriminator.load_state_dict(
            disc_ckpt["discriminator_state_dict"],
            strict=False,
        )
        print(f"[{spec.name}] Discriminator warm-start done. missing={missing}, unexpected={unexpected}")
        
        
    model, discriminator, train_pack = train_objective_flexible(
        train_dl=train_dl,
        device=device,
        sched=sched,
        in_channels=in_channels,
        config=cfg,
        num_domains=spec.num_domains,
        domain_col=spec.domain_col,
        base=base,
        t_dim=t_dim,
        disc_base=disc_base,
        epochs=epochs,
        lr_g=lr_g,
        lr_d=lr_d,
        weight_decay=weight_decay,
        use_amp=use_amp,
        seed=None,
        log_every=log_every,
        ssim_loss_fn=ssim_loss_fn,
        model=model,
        discriminator=discriminator,
        grad_accum=grad_accum,
    )

    save_experiment_checkpoint(spec.ckpt_path, model, discriminator, cfg)

    infer_pack = MRIObjectivePack(model, sched, device, spec.objective)

    return {
        "spec": spec,
        "config": cfg,
        "model": model,
        "discriminator": discriminator,
        "infer_pack": infer_pack,
        "train_pack": train_pack,
        "loaded": False,
    }




def run_all_tests_and_save(
    *,
    pack_v = None,
    pack_eps = None,
    pack_x0 = None,
    pack_v_dann = None,
    test_dl,
    out_dir="outputs_mri_tests",
    steps=50,
    seed=0,
    n_vis=4,          # number of volumes to visualize
    clip_inverse=False,
    clip_decode=True,
    axis="D",
):
    os.makedirs(out_dir, exist_ok=True)

    # grab one batch
    batch = next(iter(test_dl))
    x0_bchwd = batch[0] if isinstance(batch, (tuple, list)) else batch
    x0_bchwd = x0_bchwd[:n_vis].to(pack_v_dann.device) if pack_v_dann is not None else x0_bchwd[:n_vis].to(pack_v.device)

    packs = {
        "v": pack_v if pack_v is not None else None,
        "eps": pack_eps if pack_eps is not None else None,
        "x0": pack_x0 if pack_x0 is not None else None,
        "v_dann": pack_v_dann if pack_v_dann is not None else None,
    }

    # ---- Roundtrip recon + save ----
    for name, pack in packs.items():
        if pack is None:
            print(f"Skipping {name} since pack is None")
            continue
        recon = pack.reconstruct_roundtrip(
            x0_bchwd,
            steps=steps,
            seed=seed,
            clip_inverse=clip_inverse,
            clip_decode=clip_decode
        )
        # recon["x0"], ["xT"], ["x0_hat"] are model space (B,C,D,H,W)
        save_center_slice_grid(recon["x0"],     f"{out_dir}/{name}_real_center{axis}.png", axis=axis, nrow=n_vis)
        save_center_slice_grid(recon["xT"],     f"{out_dir}/{name}_xT_center{axis}.png", axis=axis, nrow=n_vis)
        save_center_slice_grid(recon["x0_hat"], f"{out_dir}/{name}_recon_center{axis}.png", axis=axis, nrow=n_vis)
        # Save NIfTI: real / xT / recon
        # Using identity affine by default; you can swap affine if you have it per-volume.
        save_nifti_batch(recon["x0"],     out_dir=f"{out_dir}/nifti_{name}", prefix="real", affine=None, channel=0)
        save_nifti_batch(recon["xT"],     out_dir=f"{out_dir}/nifti_{name}", prefix="xT",   affine=None, channel=0)
        save_nifti_batch(recon["x0_hat"], out_dir=f"{out_dir}/nifti_{name}", prefix="recon",affine=None, channel=0)
        
        # save_tensor_pt(recon["x0"],     f"{out_dir}/{name}_real.pt")
        # save_tensor_pt(recon["xT"],     f"{out_dir}/{name}_xT.pt")
        # save_tensor_pt(recon["x0_hat"], f"{out_dir}/{name}_recon.pt")
        
        # ---- New samples (x_T -> x0_sample) ----
        # sample_ddim expects shape in dataset layout (B,C,H,W,D) *if you used that version*.
        # But in the latest fixes we recommended: sample from x0 shape to avoid axis bugs.
        # So: create x0 template (already have recon["x0"] in model space).

        B = recon["x0"].shape[0]  # same as n_vis
        # Sample in MODEL space directly: start from Gaussian with same shape as x0 (B,C,D,H,W)
        xT_samp = torch.randn_like(recon["x0"])   # model space
        # Decode from xT using the pack's own decode loop (same as reconstruction decode)
        x0_samp = pack.decode_xT_to_x0(xT_samp, steps=steps, clip_x0=True)

        # Save sample middle-slice PNG grid
        save_center_slice_grid(x0_samp, f"{out_dir}/{name}_sample_center{axis}.png", axis=axis, nrow=n_vis)
        save_center_slice_grid(xT_samp, f"{out_dir}/{name}_sample_xT_center{axis}.png", axis=axis, nrow=n_vis)

        # Save sample as NIfTI (each item in batch)
        save_nifti_batch(x0_samp, out_dir=f"{out_dir}/nifti_{name}", prefix="sample", affine=None, channel=0)
        save_nifti_batch(xT_samp, out_dir=f"{out_dir}/nifti_{name}", prefix="sample_xT", affine=None, channel=0)

        # Save raw tensors
        # save_tensor_pt(x0_samp, f"{out_dir}/{name}_sample.pt")
        # save_tensor_pt(xT_samp, f"{out_dir}/{name}_sample_xT.pt")
        
        mae = float(F.l1_loss(recon["x0_hat"], recon["x0"]).item())
        mse = float(F.mse_loss(recon["x0_hat"], recon["x0"]).item())
        print(f"[{name}] recon MAE={mae:.6e} MSE={mse:.6e}")

    # ---- Tests A/B/C ----
    for name, pack in packs.items():
        if pack is None:
            print(f"Skipping tests for {name} since pack is None")
            continue
        dfA, _ = testA_real_inv_fwd_sweep_pack(pack, x0_bchwd, steps=steps, clip_inv=False, clip_fwd=True, tag=f"A_{name}")
        dfB, _ = testB_noise_fwd_inv_sweep_pack(pack, x0_bchwd,
                                                steps=steps, seed=seed, clip_fwd=True, clip_inv=False, tag=f"B_{name}")
        dfC = testC_true_noise_unit_pack(pack, x0_bchwd, steps=steps, seed=seed, clip_fwd=True, clip_inv=False, tag=f"C_{name}")

        dfA.to_csv(f"{out_dir}/{name}_testA.csv", index=False)
        dfB.to_csv(f"{out_dir}/{name}_testB.csv", index=False)
        dfC.to_csv(f"{out_dir}/{name}_testC.csv", index=False)

        print(f"Saved CSVs for {name}: testA/testB/testC")
        
        # ---- inside your loop ----
        # after dfA/dfB/dfC are computed and saved:
        plots_dir = f"{out_dir}/plots"

        save_metric_plots(dfA, plots_dir, prefix=f"{name}_A", title_prefix=f"{name} Test A (real inv->fwd)")
        save_metric_plots(dfB, plots_dir, prefix=f"{name}_B", title_prefix=f"{name} Test B (noise fwd->inv)")
        save_metric_plots(dfC, plots_dir, prefix=f"{name}_C", title_prefix=f"{name} Test C (true-noise unit)")
        print(f"Saved plots for {name} in {plots_dir}")
        





dataset = OpenBHBDataset()
train_dataset = dataset.get_subset('train')
val_dataset = dataset.get_subset('val')
# id_val_dataset = dataset.get_subset('id_val')
test_dataset = dataset.get_subset('test')

splits = {
    "train": train_dataset,
    "val":   val_dataset,
    "test":  test_dataset,
}

if __name__ == "__main__":
    TOTAL_DIFFUSION_TIMESTEPS = 1000   # always fixed for training
    GENERATION_STEPS = 1000            # can be changed independently
    batch_size = 3
    
    ds_train = WithGlobalIndex(splits["train"])
    train_dl = get_eval_loader("standard", ds_train, batch_size=batch_size, drop_last=True, num_workers=1)

    ds_test = WithGlobalIndex(splits["test"])
    test_dl = get_eval_loader("standard", ds_test, batch_size=batch_size, drop_last=True, num_workers=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sched = DiffusionSchedule(T=TOTAL_DIFFUSION_TIMESTEPS, device=device)

    NUM_DOMAINS = dataset.num_train_domains
    DOMAIN_COL = 0

    print("NUM_DOMAINS:", NUM_DOMAINS)
    print("TRAIN SITES:", dataset.train_sites)
    site_counts = compute_domain_counts_from_loader(
        train_dl=train_dl,
        domain_col=DOMAIN_COL,
        num_domains=NUM_DOMAINS,
    )

    print("train domain counts:", site_counts.tolist())

    MRITrainConfig.site_counts = site_counts
    MRITrainConfig.disc_class_weights = make_domain_class_weights(
        site_counts,
        alpha=0.8,
        mix=0.8,
    )

    MRITrainConfig.dann_class_weights = make_domain_class_weights(
        site_counts,
        alpha=0.5,
        mix=0.4,
    )

    TRAIN_DISC_REAL_ONLY = False

    if TRAIN_DISC_REAL_ONLY:
        disc_real_only, disc_hist = train_discriminator_real_only(
            train_dl=train_dl,
            device=device,
            num_domains=NUM_DOMAINS,
            domain_col=DOMAIN_COL,
            epochs=100,
            lr=1e-4,
            weight_decay=0.0,
            use_amp=False,
            seed=0,
            disc_class_weights=MRITrainConfig.disc_class_weights,
            ckpt_path="./med-ddpm/results/mri_tests/proposed_split/disc_real_only_notime.pt",
            history_csv_path="./med-ddpm/results/mri_tests/proposed_split/disc_real_only_notime_history.csv",
            pred_majority_class=12,
        )
        sys.exit(0)


    experiments = [
        # MRIExperimentSpec(
        #     name="v_base",
        #     objective="v",
        #     ckpt_path="./med-ddpm/results/mri_tests/proposed_split/mri_vpred_unet3d.pt",
        #     use_discriminator=False,
        # ),
        # MRIExperimentSpec(
        #     name="eps_base",
        #     objective="eps",
        #     ckpt_path="./med-ddpm/results/mri_tests/mri_eps_unet3d.pt",
        #     use_discriminator=False,
        # ),
        # MRIExperimentSpec(
        #     name="x0_base",
        #     objective="x0",
        #     ckpt_path="./med-ddpm/results/mri_tests/mri_x0_unet3d.pt",
        #     use_discriminator=False,
        # ),
        # Full none frozen experiment:
        # MRIExperimentSpec(
        #     name="v_dann",
        #     objective="v",
        #     ckpt_path="./med-ddpm/results/mri_tests/mri_v_dann_unet3d.pt",
        #     # ckpt_path="./med-ddpm/results/mri_tests/mri_v_unet3d.pt",
        #     use_discriminator=True,
        #     num_domains=NUM_DOMAINS,
        #     domain_col=DOMAIN_COL,
        #     lambda_dann=0.02, # tested: 0.05
        #     clip_x0_for_disc=False,
            
        #     freeze_discriminator=False,
        #     init_discriminator_ckpt="./med-ddpm/results/mri_tests/disc_real_only_notime.pt",
        #     discriminator_use_time_conditioning=False,
            
        # ),
        # Frozen Discriminator Experiment
        MRIExperimentSpec(
            name="v_dann",
            objective="v",
            ckpt_path="./med-ddpm/results/mri_tests/proposed_split/mri_v_dann_unet3d.pt",
            use_discriminator=True,
            num_domains=NUM_DOMAINS,
            domain_col=DOMAIN_COL,
            lambda_dann=0.04, # Increased from 0.003 to 0.004 and then to 0.04
            clip_x0_for_disc=False,

            freeze_discriminator=False,
            init_discriminator_ckpt="./med-ddpm/results/mri_tests/proposed_split/disc_real_only_notime.pt",
            discriminator_use_time_conditioning=False,
        )
    ]

    load_if_exists = False
    force_train = True

    runs = {}
    for spec in experiments:
        # Experiment with old version only
        # model_v = train_objective_baseline(
        #     objective="v",
        #     train_dl=train_dl,
        #     device=device,
        #     sched=sched,
        #     in_channels=1,
        #     base=16,
        #     t_dim=128,
        #     epochs=120,
        #     lr=2e-4,
        #     weight_decay=1e-4,
        #     use_amp=True,
        #     grad_accum=1,
        #     seed=0,
        #     log_every=10,
        # )
        # input_shape = infer_model_input_shape_from_loader(train_dl)
        # model_v = SimpleUNet3D(in_ch=1, base=16, t_dim=128).to(device)
        # ckpt = load_experiment_checkpoint(spec.ckpt_path, device=device)["model_state_dict"]
        # model_v.load_state_dict(ckpt)
        # take one batch
        # batch = next(iter(test_dl))
        # x0 = batch[0] if isinstance(batch, (tuple, list)) else batch
        # x0 = x0[:2]  # small batch
        # save_experiment_checkpoint(spec.ckpt_path, model_v, None, None)
        # infer_pack = MRIObjectivePack(model_v, sched, device, spec.objective)
        # out_v   = infer_pack.reconstruct_roundtrip(x0, steps=GENERATION_STEPS, seed=0)
        # print("MAE v  :", F.l1_loss(out_v["x0_hat"], out_v["x0"]).item())
        # recon_steps = 1000
        # run_all_tests_and_save(
        #     pack_v=infer_pack,
        #     # pack_eps=pack_eps,
        #     # pack_x0=pack_x0,
        #     # pack_v_dann=runs["v_dann"]["infer_pack"],
        #     test_dl=test_dl,
        #     out_dir="./med-ddpm/results/mri_tests/proposed_split/less_epoch/",
        #     steps=recon_steps,
        #     seed=0,
        #     n_vis=4,
        #     clip_inverse=False,
        #     clip_decode=True,
        #     axis="D"
        # )
        # exit(0)
        runs[spec.name] = prepare_experiment(
            spec=spec,
            train_dl=train_dl,
            device=device,
            sched=sched,
            in_channels=1,
            base=16,
            t_dim=128,
            disc_base=16,
            epochs=120,
            lr_g=2e-4, # tested 1e-4, 2e-4, 4e-4
            lr_d=1e-4, # which numbers we tested: 1e-4
            weight_decay=1e-4, #0.0,
            use_amp=True, 
            seed=0,
            log_every=10,
            load_if_exists=load_if_exists,
            force_train=force_train,
            init_model_ckpt="./med-ddpm/results/mri_tests/proposed_split/mri_vpred_unet3d.pt" if spec.name == "v_dann" else None,
            ssim_loss_fn=None,
            grad_accum=1
        )
        

    # take one batch
    batch = next(iter(test_dl))
    x0 = batch[0] if isinstance(batch, (tuple, list)) else batch
    x0 = x0[:2]  # small batch

    # out_v   = runs["v_base"]["infer_pack"].reconstruct_roundtrip(x0, steps=GENERATION_STEPS, seed=0)
    # out_eps = runs["eps_base"]["infer_pack"].reconstruct_roundtrip(x0, steps=GENERATION_STEPS, seed=0)
    # out_x0  = runs["x0_base"]["infer_pack"].reconstruct_roundtrip(x0, steps=GENERATION_STEPS, seed=0)
    out_v_dann = runs["v_dann"]["infer_pack"].reconstruct_roundtrip(x0, steps=GENERATION_STEPS, seed=0)

    # print("MAE v  :", F.l1_loss(out_v["x0_hat"], out_v["x0"]).item())
    # print("MAE eps:", F.l1_loss(out_eps["x0_hat"], out_eps["x0"]).item())
    # print("MAE x0 :", F.l1_loss(out_x0["x0_hat"], out_x0["x0"]).item())
    print("MAE v_dann:", F.l1_loss(out_v_dann["x0_hat"], out_v_dann["x0"]).item())
    
    # Assuming you already have:
    # pack_v = MRIObjectivePack(model_v, sched, device, "v")
    # pack_eps = MRIObjectivePack(model_eps, sched, device, "eps")
    # pack_x0 = MRIObjectivePack(model_x0, sched, device, "x0")

    # pack_v_dann = MRIObjectivePack(run["v_dann"], sched, device, "v")
    recon_steps = 1000
    run_all_tests_and_save(
        # pack_v=runs["v_base"]["infer_pack"],
        # pack_eps=pack_eps,
        # pack_x0=pack_x0,
        pack_v_dann=runs["v_dann"]["infer_pack"],
        test_dl=test_dl,
        out_dir="./med-ddpm/results/mri_tests/proposed_split/",
        steps=recon_steps,
        seed=0,
        n_vis=4,
        clip_inverse=False,
        clip_decode=True,
        axis="D"
    )


