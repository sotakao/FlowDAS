import os
import h5py
import torch
import wandb
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import *
from pathlib import Path
from torch import Tensor
from torch.utils.data import Dataset
from metrics import rmse, crps_ens, spread_skill_ratio


# The specific obs_fns are chosen to have the range match that of x.
OBS_FNS = {
        'linear': lambda x: x,
        'arctan': lambda x: torch.arctan(x),
        'arctan15': lambda x: 15*torch.arctan(x),
        'arctan_scaled': lambda x: 15*torch.arctan(x/7),
        'abs': lambda x: torch.abs(x),
        'square': lambda x: torch.square(x),
        'square_scaled': lambda x: torch.square(x/7),
        'exp': lambda x: torch.exp(x),
        'exp_scaled': lambda x: torch.exp(x/7),
        'log_abs_scaled': lambda x: 6*torch.log(torch.abs(x)+0.1),
        'sin': lambda x: torch.sin(x),
        'sin_scaled': lambda x: 20*torch.sin(x/3),
    }


ACTIVATIONS = {
    "ReLU": torch.nn.ReLU,
    "SiLU": torch.nn.SiLU,
    "Tanh": torch.nn.Tanh,
    "LeakyReLU": torch.nn.LeakyReLU,
    "ELU": torch.nn.ELU,
    # Add other activation functions as needed
}


def to_grid(x, grid_kwargs):
    # nrow = int(np.floor(np.sqrt(x.shape[0])))
    return None


def clip_grad_norm(model, max_norm):
    return torch.nn.utils.clip_grad_norm_(
        model.parameters(), 
        max_norm = max_norm, 
        norm_type= 2.0, 
        error_if_nonfinite = False
    )


def _is_type_for_logging(x):
    if isinstance(x, int):
        return True
    elif isinstance(x, float):
        return True
    elif isinstance(x, bool):
        return True
    elif isinstance(x, str):
        return True
    elif isinstance(x, list):
        return True
    elif isinstance(x, set):
        return True
    else:
        return False
    

def setup_wandb(config):
    from dotenv import load_dotenv

    load_dotenv()

    if not config.train.use_wandb:
        return

    config.wandb_run = wandb.init(
        project = os.getenv("WANDB_PROJECT", config.train.wandb_project),
        entity = os.getenv("WANDB_ENTITY", None),
        name=config.train.run_name,
        resume = None,
        id = None,
    )

    config.wandb_run_id = config.wandb_run.id

    for key in vars(config):
        item = getattr(config, key)
        if _is_type_for_logging(item):
            setattr(wandb.config, key, item)

    print("finished wandb setup")


def make_scheduler(optimizer, *, scheduler: str, total_steps: int,
                   warmup_steps: int = 0, step_size: int = 10000,
                   gamma: float = 0.1, tmax: Optional[int] = None,
                   min_lr: float = 0.0):
    scheduler = scheduler.lower()
    if scheduler == "linear":
        # simple linear decay from 1.0 -> 0.01 across total_steps, with optional flat warmup
        if warmup_steps > 0:
            # two-phase: warmup (constant lr), then linear decay
            from torch.optim.lr_scheduler import SequentialLR, LinearLR, ConstantLR
            warm = ConstantLR(optimizer, factor=1.0, total_iters=warmup_steps)
            decay = LinearLR(optimizer, start_factor=1.0, end_factor=0.01,
                             total_iters=max(1, total_steps - warmup_steps))
            return SequentialLR(optimizer, schedulers=[warm, decay], milestones=[warmup_steps])
        else:
            from torch.optim.lr_scheduler import LinearLR
            return LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=max(1, total_steps))
    elif scheduler == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        T_max = tmax if tmax is not None else max(1, total_steps)
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=min_lr)
    elif scheduler == "step":
        from torch.optim.lr_scheduler import StepLR
        return StepLR(optimizer, step_size=max(1, step_size), gamma=gamma)
    elif scheduler == "plateau":
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        return ReduceLROnPlateau(optimizer, mode="min", factor=gamma, patience=step_size, min_lr=min_lr, verbose=False)
    elif scheduler == "onecycle":
        from torch.optim.lr_scheduler import OneCycleLR
        # max_lr = current lr * 10 is a common heuristic; tweak if needed
        base = optimizer.param_groups[0]["lr"]
        return OneCycleLR(optimizer, max_lr=base*10, total_steps=total_steps,
                          pct_start=(warmup_steps / max(1, total_steps)), anneal_strategy="cos",
                          div_factor=10.0, final_div_factor=1e2)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler}")


# ========================== Dataset =========================================
class TrajectoryDataset(Dataset):
    def __init__(
        self,
        file: Union[str, Path],
        window: int = None,
        flatten: bool = False,
        normalize: bool = True,
        mean: float = None,
        std: float = None
    ):
        super().__init__()
        file = str(file)
        with h5py.File(file, mode='r') as f:
            self.data = f['x'][:]

        self.window = window
        self.flatten = flatten
        self.normalize = normalize

        self.mean = self.data.mean() if mean is None else mean
        self.std = self.data.std() if mean is None else std

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> Tuple[Tensor, Dict]:
        x = torch.from_numpy(self.data[i])

        if self.normalize:
            x = (x - self.mean) / self.std

        if self.window is not None:
            # guard against short trajectories
            T = x.shape[0]
            w = min(self.window, T)
            start_max = max(T - w, 0)
            start = 0 if start_max == 0 else torch.randint(0, start_max + 1, size=()).item()
            x = torch.narrow(x, dim=0, start=start, length=w)

        if self.flatten:
            return x.flatten(0, 1), {}
        else:
            return x, {}
        

# ========================== Logging =========================================
def power_spectrum(x):
    """
    Compute 2D power spectrum for batched input.
    x shape: (B, C, H, W)
    returns: (B, H, W) averaged over channels
    """
    fft2 = torch.fft.fft2(x, norm="ortho")      # (B, C, H, W)
    fftshift = torch.fft.fftshift(fft2, dim=(-2, -1))
    psd2D = torch.abs(fftshift) ** 2
    return psd2D.mean(1)   # average over channels -> (B, H, W)


def radial_average(psd2D):
    """
    Radially average 2D power spectrum.
    psd2D shape: (B, H, W)
    returns: (B, R) radial profiles, where R = max(H,W)//2
    """
    B, H, W = psd2D.shape
    cy, cx = H // 2, W // 2
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    r = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r = r.to(torch.int64)

    R = r.max().item() + 1
    radial_profiles = []
    for b in range(B):
        tbin = torch.bincount(r.flatten(), weights=psd2D[b].flatten(), minlength=R)
        nr = torch.bincount(r.flatten(), minlength=R)
        radial_profiles.append(tbin / torch.clamp(nr, min=1))
    return torch.stack(radial_profiles)  # (B, R)


def save_spectrum(forecast, truth, time, outdir: Union[str, Path, None] = None):
    # Use a non-interactive backend (safe on headless machines)
    try:
        plt.switch_backend("Agg")
    except Exception:
        pass

    # Compute power spectra
    ps_truth = power_spectrum(truth.cpu())
    ps_forecast = power_spectrum(forecast.cpu())
    ps_prior = power_spectrum(torch.mean(forecast, dim=0, keepdim=True).cpu())

    # Radially averaged
    rad_truth = radial_average(ps_truth)       # (B, R)
    rad_forecast = radial_average(ps_forecast)
    rad_prior = radial_average(ps_prior)

    # Plot mean spectra across batch
    plt.figure(figsize=(4,3))
    x = np.arange(1, rad_truth.shape[1]-1)
    plt.loglog(x, rad_truth.mean(0).cpu().numpy()[1:-1], label="Truth")
    plt.loglog(x, rad_forecast.mean(0).cpu().numpy()[1:-1], label="Assimilation")
    plt.loglog(x, rad_prior.mean(0).cpu().numpy()[1:-1], label="Ens. mean")
    plt.xlabel("Wavenumber")
    plt.ylabel("Power")
    plt.legend()
    plt.title("Spectral Power")
    plt.tight_layout()

    # Resolve output directory
    if outdir is None:
        try:
            outdir = Path(wandb.run.dir)  # inside the run folder if wandb is active
        except Exception:
            outdir = Path.cwd()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fname = outdir / f'spectrum_{time}.png'
    plt.savefig(fname, dpi=300)
    plt.close()
    return str(fname)


def save_video(forecast, truth, oberrstdev, obs_p, obs_fn, level=0):
    # Coerce inputs to CPU float tensors first
    forecast_t = torch.as_tensor(forecast, dtype=torch.float32, device='cpu')   # (time, ens, 2, lat, lon)
    truth_t = torch.as_tensor(truth, dtype=torch.float32, device='cpu')   # (time, 2,  lat, lon)

    # Select level
    forecast_t = forecast_t[:, :, level, :, :]   # (time, ens, lat, lon)
    truth_t = truth_t[:, level, :, :]         # (time, lat, lon)

    # Observations (CPU tensor -> NumPy)
    obs = obs_fn(truth_t).numpy() + np.random.randn(*truth_t.shape) * oberrstdev  # (time, lat, lon)

    # RMSEs on CPU tensors, then to NumPy for formatting
    rmse_t = ((forecast_t - truth_t.unsqueeze(1)).pow(2).mean(dim=(2, 3)).sqrt())                # (time, ens)
    rmse_mean_t = ((forecast_t.mean(dim=1) - truth_t).pow(2).mean(dim=(1, 2)).sqrt())            # (time,)
    rmse = rmse_t.numpy()
    rmse_mean = rmse_mean_t.numpy()

    # For plotting, use NumPy arrays
    forecast_np = forecast_t.numpy()
    truth_np = truth_t.numpy()

    ens_mean = forecast_np.mean(axis=1)  # (time, lat, lon)
    ens_std  = forecast_np.std(axis=1)   # (time, lat, lon)
    members  = forecast_np[:, :4]        # (time, 4,   lat, lon)

    vmin = np.min(truth_np); vmax = np.max(truth_np)
    vmin_obs = np.min(obs);  vmax_obs = np.max(obs)

    fig, axs = plt.subplots(2, 4, figsize=(13, 7), constrained_layout=True)
    cmap = 'jet'

    img = []
    titles = ['Truth', 'Obs', 'Mean', 'Std', 'Member 1', 'Member 2', 'Member 3', 'Member 4']

    img.append(axs[0, 0].imshow(truth_np[0], cmap=cmap, vmin=vmin, vmax=vmax))
    img.append(axs[0, 1].imshow(obs[0], cmap=cmap, vmin=vmin_obs, vmax=vmax_obs))
    img.append(axs[0, 2].imshow(ens_mean[0], cmap=cmap, vmin=vmin, vmax=vmax))
    img.append(axs[0, 3].imshow(ens_std[0], cmap=cmap, vmin=0, vmax=ens_std.max()))

    for i in range(4):
        img.append(axs[1, i].imshow(members[0, i], cmap=cmap, vmin=vmin, vmax=vmax))

    for i, ax in enumerate(axs.flatten()):
        ax.set_title(titles[i], fontsize=14)
        ax.axis('off')

    sutitle = f't = {0}'
    fig.suptitle(sutitle, fontsize=14)

    def update(frame):
        img[0].set_array(truth_np[frame])
        img[1].set_array(obs[frame])
        img[2].set_array(ens_mean[frame])
        img[3].set_array(ens_std[frame])
        for i in range(4):
            img[4 + i].set_array(members[frame, i])

        fig.suptitle(f't={frame}', fontsize=14, y=1.1)

        titles = [
            'Truth',
            f'Obs, $\\sigma$={np.round(oberrstdev, 1)}, p={np.round(obs_p*100, 1)}%',
            f'Mean ({rmse_mean[frame]:.2f})',
            'Std',
        ]
        for i in range(4):
            titles.append(f'Member {i+1} ({rmse[frame, i]:.2f})')

        for i, ax in enumerate(axs.flatten()):
            ax.set_title(titles[i], fontsize=14)
            ax.axis('off')

        return img

    fname = 'animation.mp4'
    mpl.rcParams['animation.ffmpeg_path'] = "/groups/astuart/sotakao/miniconda3/envs/sda/bin/ffmpeg" # GPT suggested hack to point to get it working
    ani = animation.FuncAnimation(fig, update, frames=truth_np.shape[0], interval=10, blit=True)
    writer = animation.FFMpegWriter(
        fps=10,
        codec='libx264',
        extra_args=['-pix_fmt','yuv420p','-crf','20','-preset','medium','-movflags','+faststart']
    )
    ani.save(fname, writer=writer, dpi=300)
    # ani.save(fname, writer='ffmpeg', fps=10, dpi=150)
    plt.close(fig)
    return fname


def upload_results(exp_name, args):
    """
    Uploads the results to wandb from a saved .nc file

    exp_name: str, name of the experiment file
    args: arguments used for the experiment (not read from the file)
    """
    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=exp_name,
        config=vars(args)
    )

    # Calculate Metrics and log metrics to wandb
    nc_filename = os.path.join('results', f'{exp_name}.nc')
    nc = Dataset(nc_filename, 'r')
    scalefact = nc.f*nc.theta0/nc.g # 0.003061224412462883 
    forecast = nc['pv_a'] # shape: [time, ens, 2, lat, lon]
    truth =  scalefact * nc['pv_t'] # shape: [time, 2, lat, lon]
    
    obs_fn = OBS_FNS[args.obs_fn]

    for t, (pred, gt) in enumerate(zip(forecast, truth)):   
        # Convert NumPy arrays to PyTorch tensors
        # TODO: Maybe we should save everything in a torch/np format
        pred = torch.tensor(np.array(pred), dtype=torch.float32)  
        gt = torch.tensor(np.array(gt), dtype=torch.float32)      

        # RMSE
        run.log({
            "RMSE": rmse(torch.mean(pred, dim=0), gt),
            "CRPS": crps_ens(pred, gt, ens_dim=0),
            "Spread Skill Ratio": spread_skill_ratio(pred, gt, ens_dim=0),
        }, step=t)

        if t % args.plot_every == 0:
            fname = save_spectrum(pred, gt.unsqueeze(0), time=t)
            run.log({
                "spectrum": wandb.Image(fname)
            }, step=t)

    if hasattr(nc, "obs_sigma"): # check if variable exists
        obs_sigma = nc.obs_sigma
    elif hasattr(nc, "oberrstdev"):  # check if global attribute exists
        obs_sigma = getattr(nc, "oberrstdev")

    fname = save_video(forecast, truth, obs_sigma, args.obs_prob, obs_fn=obs_fn, level=0)
    run.log({
        "animation": wandb.Video(fname, fps=10, format="mp4")
    })
    run.finish()