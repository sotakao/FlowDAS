import os
import math
import h5py
import argparse
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

# your local UNet (unchanged)
from unet import Unet


# ------------------------------ Drift model ---------------------------------
class DriftModel(nn.Module):
    """
    UNet input is [zt, cond] -> (B, C + cond_C, H, W) and outputs drift (B, C, H, W).
    """
    def __init__(self, C: int, cond_channels: int, unet_channels: int = 128,
                 dim_mults=(1, 2, 2, 2), resnet_block_groups=8,
                 learned_sinusoidal_cond=True, random_fourier_features=False,
                 learned_sinusoidal_dim=32, attn_dim_head=64, attn_heads=4,
                 use_classes: bool = False):
        super().__init__()
        in_channels = C + cond_channels
        self.use_classes = use_classes
        self.net = Unet(
            num_classes=(C if use_classes else 0),
            in_channels=in_channels,
            out_channels=C,
            dim=unet_channels,
            dim_mults=dim_mults,
            resnet_block_groups=resnet_block_groups,
            learned_sinusoidal_cond=learned_sinusoidal_cond,
            random_fourier_features=random_fourier_features,
            learned_sinusoidal_dim=learned_sinusoidal_dim,
            attn_dim_head=attn_dim_head,
            attn_heads=attn_heads,
            use_classes=use_classes,
        )

    def forward(self, zt: Tensor, t: Tensor, y=None, cond: Optional[Tensor] = None) -> Tensor:
        # make robust if cond accidentally has extra dims (B, T-1, C, H, W)
        if cond is not None and cond.dim() > 4:
            B, *_, H, W = cond.shape
            cond = cond.view(B, -1, H, W).contiguous()
        if cond is not None:
            zt = torch.cat([zt, cond], dim=1)
        if not self.use_classes:
            y = None
        return self.net(zt, t, y)


# ------------------------------ Interpolant ---------------------------------
class Interpolant:
    """
    Minimal interpolant with alpha(t)=1-t, beta(t)=t or t^2, sigma(t)=sigma_coef*(1-t).
    """
    def __init__(self, beta_fn: str = "t^2", sigma_coef: float = 1.0):
        self.beta_fn = beta_fn
        self.sigma_coef = sigma_coef

    @staticmethod
    def _wide(t: Tensor) -> Tensor:
        return t[:, None, None, None]

    def alpha(self, t: Tensor) -> Tensor:
        return self._wide(1.0 - t)

    def beta(self, t: Tensor) -> Tensor:
        if self.beta_fn == "t^2":
            return self._wide(t.pow(2))
        return self._wide(t)

    def sigma(self, t: Tensor) -> Tensor:
        return self.sigma_coef * self._wide(1.0 - t)

    def compute_zt(self, t: Tensor, z0: Tensor, z1: Tensor, noise: Tensor) -> Tensor:
        at = self.alpha(t)
        bt = self.beta(t)
        gamma_t = self._wide(t.sqrt()) * self.sigma(t)
        return at * z0 + bt * z1 + gamma_t * noise


# ------------------------------ Utilities -----------------------------------
def load_h5_first_sample(path: str, window: int) -> Tensor:
    """
    Load first sample from HDF5 file with dataset['x'] of shape (N, T, C, H, W).
    Return (1, T, C, H, W) with possibly cropped window.
    """
    with h5py.File(path, "r") as f:
        x = f["x"][0]  # (T, C, H, W)
    T = x.shape[0]
    w = min(window, T)
    x = torch.from_numpy(x[:w]).unsqueeze(0).float()  # (1, w, C, H, W)
    return x


def build_cond_z0_z1(x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
    x: (B, T, C, H, W)
    cond = x[:, :-1] -> flatten time into channels: (B, (T-1)*C, H, W)
    z0   = last past frame: x[:, -2] -> (B, C, H, W)
    z1   = target frame:    x[:, -1] -> (B, C, H, W)  (optional GT, used for simulating observations)
    """
    assert x.dim() == 5 and x.shape[1] >= 2
    B, T, C, H, W = x.shape
    past = x[:, :-1]                                  # (B, T-1, C, H, W)
    cond = past.reshape(B, (T - 1) * C, H, W).contiguous()
    z0 = x[:, -2]                                     # (B, C, H, W)
    z1 = x[:, -1]                                     # (B, C, H, W)
    return cond, z0, z1


def make_mask_like(z: Tensor, ratio: float, seed: int = 0) -> Tensor:
    g = torch.Generator(device=z.device)
    g.manual_seed(seed)
    return (torch.rand_like(z, generator=g) < ratio).float()


# ------------------------------ Sampling ------------------------------------
@torch.no_grad()
def euler_maruyama_step(interp: Interpolant, model: DriftModel,
                        xt: Tensor, base: Tensor, cond: Tensor, tb: Tensor, dt: float) -> Tensor:
    """
    Unconditioned Euler–Maruyama step.
    """
    bF = model(xt, tb, cond=cond)                     # (B, C, H, W)
    sigma = interp.sigma(tb)                          # (B, 1, 1, 1)
    mu = xt + bF * dt
    return mu + sigma * torch.randn_like(mu) * math.sqrt(dt)


def conditioned_step(interp: Interpolant, model: DriftModel,
                     xt: Tensor, base: Tensor, cond: Tensor, tb: Tensor, dt: float,
                     operator, y_obs: Tensor, guide: float = 0.1,
                     scale: float = 1.0) -> Tensor:
    """
    EM step with simple gradient-based measurement consistency:
      1) predict x1_hat from current xt (first-order extrapolation)
      2) compute measurement MSE vs y_obs
      3) take a small gradient step in xt to reduce that MSE
    """
    xt.requires_grad_(True)
    bF = scale * model(xt / scale, tb, cond=cond / scale)
    sigma = scale * interp.sigma(tb)
    # first-order extrapolation towards t=1 (deterministic part)
    # x1_hat ≈ xt + bF * (1 - t)
    x1_hat = xt + bF * (1.0 - tb[:, None, None, None])

    # measurement loss
    pred_meas = operator(x1_hat)                      # same shape as y_obs
    meas_loss = (pred_meas - y_obs).pow(2).mean()
    grad_xt = torch.autograd.grad(meas_loss, xt, retain_graph=False, create_graph=False)[0]

    # EM update with guidance
    mu = xt + bF * dt
    xt_new = mu + sigma * torch.randn_like(mu) * math.sqrt(dt) - guide * grad_xt
    return xt_new.detach()


def em_sample_conditioned(interp: Interpolant, model: DriftModel,
                          base: Tensor, cond: Tensor, y_obs: Tensor, operator,
                          steps: int = 500, t_min: float = 0.0, t_max: float = 0.999,
                          guide: float = 0.1, scale: float = 1.0) -> Tensor:
    """
    Run EM from base (last past) with measurement guidance towards a sample at t=1.
    """
    device = base.device
    ts = torch.linspace(t_min, t_max, steps, device=device)
    dt = float(ts[1] - ts[0])
    xt = base.clone()

    for t in tqdm(ts, desc="sampling", leave=True):
        tb = t.repeat(xt.shape[0]).to(device)         # (B,)
        xt = conditioned_step(interp, model, xt, base, cond, tb, dt, operator, y_obs, guide=guide, scale=scale)

    return xt  # (B, C, H, W)


# ------------------------------ Main ----------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Path to pretrained checkpoint (.pt with state_dict under 'model' or full state_dict).")
    ap.add_argument("--data_path", type=str, required=True, help="HDF5 with dataset['x'] (N,T,C,H,W)")
    ap.add_argument("--window", type=int, default=7, help="Number of frames to take (>=2)")
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--beta_fn", type=str, default="t^2", choices=["t", "t^2"])
    ap.add_argument("--sigma_coef", type=float, default=1.0)
    ap.add_argument("--em_steps", type=int, default=300)
    ap.add_argument("--obs_ratio", type=float, default=0.1, help="Sparsity ratio for observation mask (demo)")
    ap.add_argument("--noise_std", type=float, default=0.0, help="Additive Gaussian noise on observations (demo)")
    ap.add_argument("--guide", type=float, default=0.1, help="Guidance strength for measurement consistency")
    ap.add_argument("--save", type=str, default="pred.pt", help="Where to save the predicted frame tensor")
    args = ap.parse_args()

    device = torch.device(args.device)

    # 1) Load a sample (B=1, T, C, H, W)
    x = load_h5_first_sample(args.data_path, args.window).to(device)
    B, T, C, H, W = x.shape
    assert T >= 2, "window must be >= 2"

    # 2) Build conditioning and base
    cond, z0, z1_gt = build_cond_z0_z1(x)            # cond: (B,(T-1)*C,H,W), z0/z1: (B,C,H,W)

    # 3) Define observation operator and (optional) noise
    #    Demo: sparse masking of the target frame z1_gt with optional additive noise
    mask = make_mask_like(z1_gt, ratio=args.obs_ratio, seed=0).to(device)
    operator = (lambda x: x * mask)
    y_obs = operator(z1_gt)
    if args.noise_std > 0:
        y_obs = y_obs + args.noise_std * torch.randn_like(y_obs)

    # 4) Recreate the model architecture with correct input channels
    cond_channels = cond.shape[1]
    drift = DriftModel(C=C, cond_channels=cond_channels).to(device)

    # 5) Load weights (both {"model": state_dict} and plain state_dict are supported)
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt["model"] if ("model" in ckpt and isinstance(ckpt["model"], dict)) else ckpt
    drift.load_state_dict(state, strict=True)
    drift.eval()

    # 6) Interpolant
    interp = Interpolant(beta_fn=args.beta_fn, sigma_coef=args.sigma_coef)

    # 7) Run conditioned EM sampling from base=z0 towards t=1
    with torch.no_grad():
        pred = em_sample_conditioned(
            interp=interp,
            model=drift,
            base=z0,                          # last past frame
            cond=cond,                        # history flattened into channels
            y_obs=y_obs,                      # sparse observations of target
            operator=operator,
            steps=args.em_steps,
            t_min=0.0, t_max=0.999,
            guide=args.guide,
        )                                     # (B, C, H, W)

    # 8) Save prediction
    Path(os.path.dirname(args.save) or ".").mkdir(parents=True, exist_ok=True)
    torch.save({"pred": pred.cpu(), "z0": z0.cpu(), "y_obs": y_obs.cpu(), "mask": mask.cpu()}, args.save)
    print(f"Saved prediction to {args.save}")


if __name__ == "__main__":
    main()
