import os
import wandb
import math
import h5py
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from netCDF4 import Dataset as NetCDFDataset

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm
from types import SimpleNamespace

# your local UNet (unchanged)
from models import DriftModel, Interpolant
from utils import OBS_FNS, save_spectrum, save_video
from metrics import rmse, crps_ens, spread_skill_ratio


@torch.no_grad()
def _sigma_broadcast(interp, tb, like: torch.Tensor):
    sig = interp.sigma(tb)
    # make broadcastable to (B,C,H,W)
    if sig.dim() == 1:
        sig = sig[:, None, None, None]
    return sig.expand_as(like) if sig.shape != like.shape else sig


def _roll_cond(cond: torch.Tensor, new_frame: torch.Tensor, C: int) -> torch.Tensor:
    """
    cond: (B,(T-1)C,H,W)
    new_frame: (B,C,H,W)
    returns updated cond with oldest C channels dropped and new_frame appended.
    """
    B, CC, H, W = cond.shape
    assert CC % C == 0, "cond channels must be multiple of C"
    return torch.cat([cond[:, C:], new_frame], dim=1)


def dps_x1_hat(drift, interp, x, t, cond, scale, device):
    tb = t.repeat(x.shape[0]).to(device)
    # denom = interp.beta(tb) * interp.alpha_dot(tb) - interp.alpha(tb) * interp.beta_dot(tb)
    # x1_hat = (scale * interp.beta(tb) * drift(x/scale, tb, cond=cond/scale) \
    #            - interp.beta_dot(tb) * x) / denom
    denom = interp.sigma(tb) * interp.beta_dot(tb) - interp.beta(tb) * interp.sigma_dot(tb)
    x1_hat = (interp.sigma(tb) * drift(x/scale, tb, cond=cond/scale) - interp.sigma_dot(tb) * x) / denom
    return x1_hat


def em_sample_conditioned(
    drift,
    interp,
    xt,            # (B,C,H,W) current state, will be *updated* and returned
    cond,          # (B,(T-1)C,H,W) conditioner
    y_obs,         # (B,C,H,W) observation at this step (normalized same as training)
    operator,      # differentiable measurement operator, e.g., lambda x: x * mask
    *,
    steps=200, t_min=0.0, t_max=0.999,
    guide=0.1, noise_scale=1.0,
    mc_samples=1, second_order=True,
    data_scale=1.0, # scaling between physical and normalized units
    verbose=False,
    guidance_method='MC',
):
    """
    One assimilation window: runs an EM-like sampler from t_min..t_max conditioned on y_obs.
    Returns the final xt (detached).
    """
    device = xt.device
    ts = torch.linspace(t_min, t_max, steps, device=device)
    dt = float(ts[1] - ts[0])

    # Freeze model params during sampling
    req = [p.requires_grad for p in drift.parameters()]
    for p in drift.parameters():
        p.requires_grad_(False)

    if verbose:
        pbar = tqdm(total=len(ts), desc="EM steps")

    for t in ts:
        xt = xt.detach().requires_grad_(True)
        tb = t.repeat(xt.shape[0]).to(device)

        # Drift and sigma (no grads through model params)
        with torch.no_grad():
            bF  = data_scale * drift(xt/data_scale, tb, cond=cond/data_scale)       # (B,C,H,W)
            sig = data_scale * _sigma_broadcast(interp, tb, like=xt)

        # ----- Monte Carlo look-aheads (expectation of measurement loss) -----
        if guidance_method.lower() == 'mc':
            losses = []
            for _ in range(mc_samples):
                eps_k  = torch.randn(xt.shape, device=device)
                # 1st order look-ahead to (approx) x_1
                x1_hat = xt + bF * (1.0 - t) + sig * eps_k * math.sqrt(max(1e-8, 1.0 - float(t)))

                if second_order:
                    with torch.no_grad():
                        t1  = torch.ones_like(tb)      # time ~ 1.0
                        bF2 = data_scale * drift(x1_hat/data_scale, t1, cond=cond/data_scale)
                    x1_hat = xt + 0.5*(bF + bF2)*(1.0 - t) + sig * eps_k * math.sqrt(max(1e-8, 1.0 - float(t)))

                meas = operator(x1_hat)               # same normalization as training!
                losses.append(torch.linalg.vector_norm(meas - y_obs, dim=(1,2,3)))
                # losses.append((meas - y_obs).pow(2).mean())
                # err = (meas - y_obs)
                # losses.append(0.5 * err.pow(2).sum(dim=(1,2,3)).mean())

            loss_meas = torch.stack(losses).mean()
        elif guidance_method.lower() == 'dps':
            x1_hat = dps_x1_hat(drift, interp, xt, t, cond, data_scale, device)
            meas = operator(x1_hat)
            loss_meas = torch.linalg.vector_norm(meas - y_obs)

        grad_xt = torch.autograd.grad(loss_meas, xt, retain_graph=False, create_graph=False)[0]

        # EM update
        mu  = xt + bF * dt
        eps = torch.randn(mu.shape, device=device)
        # xt  = mu + noise_scale * sig * eps * math.sqrt(dt) - guide * grad_xt * dt
        xt  = mu + noise_scale * sig * eps * math.sqrt(dt) - data_scale * guide * grad_xt

        if verbose:
            pbar.update(1)
            pbar.set_postfix({"loss_meas": f"{float(loss_meas):.3f}"})
            
    # rmse = (meas - y_obs).pow(2).mean()
    # print(rmse)
    # import pdb; pdb.set_trace()

    # restore requires_grad flags
    for p, r in zip(drift.parameters(), req):
        p.requires_grad_(r)

    return xt.detach()


def sequential_assimilate(
    drift,
    interp,
    x0: torch.Tensor,         # (B,C,H,W) initial state (e.g., last frame of history)
    cond: torch.Tensor,       # (B,(T-1)C,H,W) time-flattened past
    y_seq: torch.Tensor,      # (B,N,C,H,W) observations for times 1..N (normalized)
    operator,                 # callable on (B,C,H,W) -> (B,C,H,W) (e.g., fixed mask)
    *,
    steps_per_obs=200,
    guide=0.1, noise_scale=1.0,
    mc_samples=1, second_order=True,
    roll_window=True,
    data_scale=1.0, # scaling between physical and normalized units
    log_wandb: bool = False,
    gt_future: Optional[torch.Tensor] = None,  # (N,C,H,W) in PV units
    scalefact: Optional[float] = None,         # scalar
    plot_every: int = 20,
    guidance_method: str = 'MC'
):
    """
    Runs N assimilation windows, one per observation y_seq[:,i], starting from z0.
    Returns tensor of predictions with shape (B, N, C, H, W).
    """
    if y_seq.ndim == 4:
        y_seq = y_seq.unsqueeze(0)  # (1,N,C,H,W)

    B, N, C, H, W = y_seq.shape
    xt = x0.clone()
    preds = []

    for i in tqdm(range(N)):
        y_obs_i = y_seq[:, i]  # (B,C,H,W)

        xt = em_sample_conditioned(
            drift=drift, interp=interp,
            xt=xt, cond=cond, 
            y_obs=y_obs_i, operator=operator,
            steps=steps_per_obs, guide=guide, noise_scale=noise_scale,
            mc_samples=mc_samples, second_order=second_order,
            data_scale=data_scale,
            guidance_method=guidance_method,
            verbose=False
        )
        preds.append(xt.unsqueeze(1))  # (B,1,C,H,W)

        if roll_window:
            cond = _roll_cond(cond, xt, C=C)  # slide window forward

        # ---- stream to wandb ----
        if log_wandb and (gt_future is not None) and (scalefact is not None):
            # Work in physical units (scalefact × PV)
            pred_phys = xt.detach().cpu().float()                          # (B,C,H,W)
            gt_phys   = (scalefact * gt_future[i]).detach().cpu().float()  # (C,H,W)

            # Ensemble mean for RMSE
            pred_mean = pred_phys.mean(dim=0) if B > 1 else pred_phys[0]   # (C,H,W)

            rmse_val = rmse(pred_mean, gt_phys).mean().item()
            crps_val = crps_ens(pred_phys, gt_phys, ens_dim=0).item()
            ssr_val  = spread_skill_ratio(pred_phys, gt_phys, ens_dim=0).item()

            wandb.log({
                "RMSE": rmse_val,
                "CRPS": crps_val,
                "Spread Skill Ratio": ssr_val,
            }, step=i)

            # Spectrum every plot_every
            if (i % plot_every) == 0:
                # save_spectrum expects (ens,C,H,W) and gt as (1,C,H,W)
                fname = save_spectrum(pred_phys, gt_phys.unsqueeze(0), time=i)
                wandb.log({"spectrum": wandb.Image(fname)}, step=i)


    return torch.cat(preds, dim=1)  # (B,N,C,H,W)


# ---------------------------
#  CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser("Run inference for SDA (parity with notebook).")
    # Data
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--train_file", type=str, default="sqg_pv_train.h5")
    p.add_argument("--hrly_freq", type=int, default=3)
    # Model
    p.add_argument("--window", type=int, default=7)
    # Obs + guidance
    p.add_argument("--obs_type", type=str, choices=["grid", "random"], default="random")
    p.add_argument("--obs_stride", type=int, default=4)
    p.add_argument("--obs_pct", type=float, default=0.25)
    p.add_argument("--obs_fn", type=str, default="linear")
    p.add_argument("--obs_sigma", type=float, default=3.0)
    p.add_argument("--fixed_obs", action="store_true")
    p.add_argument("--n_ens", type=int, default=20)
    p.add_argument("--guidance_method", type=str, default="DPS")
    p.add_argument("--guidance_strength", type=float, default=1.0)
    p.add_argument("--mc_samples", type=int, default=1)
    p.add_argument("--em_steps", type=int, default=200)
    # Checkpoint + output
    p.add_argument("--output_dir", type=str, default="./output")
    # Logging
    p.add_argument("--log_wandb", type=int, default=1)
    p.add_argument("--wandb_project", type=str, default="ScoreDA_SQG")
    p.add_argument("--wandb_entity", type=str, default="stima")
    p.add_argument("--plot_every", type=int, default=20)  # match assimilate.py behavior
    # Debug
    p.add_argument("--debug_parity", action="store_true", help="Print first-step invariants and exit.")
    p.add_argument("--true_initial", type=int, default=0, help="Use true initial conditions.")
    # p.add_argument("--no_lookahead", type=int, default=0, help="Use current x for guidance as opposed to x_hat in DPS.")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_dtype(torch.float32)

    # CKPT_PATH = f"./checkpoints/latest_{args.hrly_freq}hrly_window_{args.window}_cosine_normalized.best.pt"
    CKPT_PATH = f"./checkpoints/latest_{args.hrly_freq}hrly_window_{args.window}_cosine_sigma_1.0_label_noise_0.1_normalized.best.pt"

    # (Optional) reproducibility
    torch.manual_seed(0); np.random.seed(0)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # ---- Defaults matching your notebook ----
    pv_mean = torch.as_tensor(0.0, device=device, dtype=torch.float32)
    pv_std  = torch.as_tensor(2672.232, device=device, dtype=torch.float32)

    obs_fn = OBS_FNS[args.obs_fn]
    rng = np.random.RandomState(42)

    # ---- Data ----
    test_file = f"/resnick/groups/astuart/sotakao/score-based-ensemble-filter/EnSFInpainting/data/test/sqg_N64_{args.hrly_freq}hrly_100.nc"
    nc_truth = NetCDFDataset(test_file, 'r')
    pv_truth_nc = nc_truth.variables['pv']  # shape: (T, 2, ny, nx)
    T_all, C, ny, nx = pv_truth_nc.shape

    # grab all truth once
    pv_truth_all = torch.tensor(np.array(pv_truth_nc[:T_all, ...]), dtype=torch.float32, device=device)
    scalefact = nc_truth.f * nc_truth.theta0 / nc_truth.g
    nc_truth.close()

    # ---- Data Scale ----
    scale = pv_std * scalefact

    # ---- Mask (same as before) ----
    def make_obs_mask(ny_, nx_):
        if args.obs_type == "grid":
            stride = args.obs_stride
            mask = torch.zeros(ny_, nx_, device=device)
            mask[::stride, ::stride] = 1.0
        else:
            rng = np.random.RandomState(42)
            nobs = int(ny_ * nx_ * args.obs_pct)
            idx = rng.choice(ny_ * nx_, nobs, replace=False)
            mask = torch.zeros(ny_ * nx_, device=device)
            mask[torch.from_numpy(idx).to(device)] = 1.0
            mask = mask.view(ny_, nx_)
        return mask

    obs_mask = make_obs_mask(ny, nx)
    mask4d   = obs_mask.view(1, 1, ny, nx)   # broadcast over (B,C,H,W)

    # ---- Load initial condition FROM GT history (physical units = scalefact * PV) ----
    W = int(args.window)
    assert W >= 2, "window must be at least 2"

    # history (times 0 .. W-2) and future (times W .. T_all-1)
    hist   = pv_truth_all[:W-1]       # (W-1, C, ny, nx) in PV units
    future = pv_truth_all[W:]         # (N,   C, ny, nx) in PV units
    N = future.shape[0]
    assert N > 0, "no future frames to assimilate"

    # scale to physical units once
    hist_phys   = (scalefact * hist).to(device).float()        # (W-1, C, H, W)
    x0_phys_1   = (scalefact * pv_truth_all[W-1]).to(device).float()  # (C, H, W)

    if args.true_initial:
        # replicate the true initial condition across ensemble
        B_ens = int(args.n_ens)
        C_ch  = hist_phys.shape[1]
        ny, nx = hist_phys.shape[-2:]

        # cond: flatten time into channels and replicate across ensemble
        cond = (hist_phys.unsqueeze(0)                         # (1, W-1, C, H, W)
                        .repeat(B_ens, 1, 1, 1, 1)            # (B, W-1, C, H, W)
                        .reshape(B_ens, (W-1)*C_ch, ny, nx)   # (B, (W-1)*C, H, W)
                        .contiguous())

        # x0: last history frame replicated across ensemble
        x0 = x0_phys_1.unsqueeze(0).repeat(B_ens, 1, 1, 1)     # (B, C, H, W)

    else:
        # Fallback: load SDA ensembles (already in physical units)
        try:
            fname = f"inital_ensembles_linear_3.0_0.25pct"
            x = torch.load(os.path.join("data", fname)).to(device)  # (B, T, C, H, W)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Initial ensembles not found. Put '{fname}' in ./data (see the SDA notebook)."
            )
        B_ens, T_win, C_ch, ny, nx = x.shape
        assert T_win == (W-1), f"Loaded ensembles must have T=W-1 (got T={T_win}, W={W})"

        # flatten time into channels; last frame is x0
        cond = x.reshape(B_ens, T_win*C_ch, ny, nx).contiguous()  # (B, (W-1)*C, H, W)
        x0   = x[:, -1]                                           # (B, C, H, W)

    # keep C consistent for model config
    C = cond.shape[1] // (W-1)

    # ---- Load model from latest checkpoint ----
    data_cfg = SimpleNamespace(C=C)

    # model_cfg: must include all the attributes DriftModel expects
    model_cfg = SimpleNamespace(
        unet_channels=64,
        unet_dim_mults=(1, 2, 2),
        unet_resnet_block_groups=8,
        unet_learned_sinusoidal_dim=32,
        unet_attn_dim_head=64,
        unet_attn_heads=4,
        unet_learned_sinusoidal_cond=True,
        unet_random_fourier_features=False,
        use_classes=False,
    )

    interp_cfg = SimpleNamespace(
        sigma_coef=1.0,
        beta_fn="t",
        t_min_train=0.0,
        t_max_train=1.0,
        t_min_sampling=0.0,
        t_max_sampling=0.999,
        EM_sample_steps=args.em_steps
    )

    cond_channels = cond.shape[1]
    drift = DriftModel(
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        cond_channels=cond_channels
    ).to(device)
    drift.eval()
    ckpt = torch.load(CKPT_PATH, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    drift.load_state_dict(state, strict=True)

    interp = Interpolant(interp_cfg)

    # ---- A(x): PHYSICAL -> OBS (mask applied framewise) ----
    def A_model(x):
        # x: (B,L,C,H,W) in physical units (scalefact × PV)
        return obs_fn(x) * mask4d
    
    # ---- Observations in OBS units (masked) ----
    noise  = torch.randn_like(future) * args.obs_sigma
    y_star = obs_fn(scalefact * future) * mask4d + noise * mask4d 

    if args.log_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"FlowDAS_SQG_win{args.window}_{args.obs_fn}_{args.obs_pct}pct_g{args.guidance_strength}" + ("_GTinit" if args.true_initial else "_SDAinit") + f"_{args.guidance_method}",
            config=vars(args),
        )

    posterior_samples = sequential_assimilate(
        drift, interp,
        x0=x0,                      # (1,C,H,W) @ t = W-1
        cond=cond,                  # (1,(W-1)*C,H,W) = frames 0..W-2
        y_seq=y_star,               # (N, C, H, W) = frames W..T-1
        operator=A_model,
        steps_per_obs=args.em_steps,
        guide=args.guidance_strength,
        noise_scale=0.2,
        second_order=True,
        roll_window=True,
        data_scale=scale,
        log_wandb=bool(args.log_wandb),
        gt_future=future,       # (N,C,H,W) in PV units
        scalefact=scalefact,
        plot_every=args.plot_every,
        # no_lookahead=bool(args.no_lookahead),
        mc_samples=args.mc_samples,
        guidance_method=args.guidance_method,
    )                                # -> (1, N, C, H, W)

    posterior_time_first = torch.swapaxes(posterior_samples, 0, 1)  # (T,B,C,H,W)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    vid_path = save_video(
        posterior_time_first.detach().to(dtype=torch.float32, device='cpu').contiguous(),
        (scalefact * future.clone().detach().float()).cpu(),
        args.obs_sigma,
        args.obs_pct,
        obs_fn=obs_fn,
        level=0,
    )
    print(f"[video] saved to: {vid_path}")

    # Log to W&B only if enabled
    if args.log_wandb:
        wandb.log({"animation": wandb.Video(vid_path, fps=10, format="mp4")})
        wandb.finish()


if __name__ == "__main__":
    main()

