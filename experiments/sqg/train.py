import io
import os
import math
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, Union

# Optional logging
try:
    import wandb
    WANDB = True
except Exception:
    WANDB = False

# --- Project-local imports you already have ---------------------------------
from models import DriftModel, Interpolant
from utils import to_grid, clip_grad_norm, setup_wandb, make_scheduler, TrajectoryDataset
# ---------------------------------------------------------------------------

# ========================== Configs =========================================
@dataclass
class ModelConfig:
    unet_channels: int = 64
    unet_dim_mults: Tuple[int, ...] = (1, 2, 2)
    unet_resnet_block_groups: int = 8
    unet_learned_sinusoidal_dim: int = 32
    unet_attn_dim_head: int = 64
    unet_attn_heads: int = 4
    unet_learned_sinusoidal_cond: bool = True
    unet_random_fourier_features: bool = False
    use_classes: bool = False  # leave False unless you pass labels


@dataclass
class DataConfig:
    train_path: str = ""          # HDF5 file path
    val_path: str = ""          # HDF5 file path
    hrly_freq: int = 3
    batch_size: int = 32
    num_workers: int = 4
    C: int = 2
    H: int = 64
    W: int = 64
    window: Optional[int] = None
    normalize: bool = True
    shuffle: bool = True
    persistent_workers: bool = True
    pin_memory: bool = True
    grid_kwargs: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.grid_kwargs is None:
            self.grid_kwargs = {"normalize": False}


@dataclass
class TrainingConfig:
    device: str = "cuda"     # "cuda" or "cpu"
    base_lr: float = 2e-4
    epochs: int = 10
    max_grad_norm: float = 1.0
    print_every: int = 100
    sample_every_steps: int = 1000
    save_every_steps: int = 5000
    ckpt_path: str = "./checkpoints/latest.pt"
    run_name: str = ""
    use_wandb: bool = False
    wandb_project: str = "sqg_min"
    wandb_entity: Optional[str] = None
    overfit: bool = False
    scheduler: str = "linear"  # "linear", "cosine", "step", "plateau", "onecycle"
    sample_every_epochs: int = 100  # NEW: log images every N epochs
    label_noise_std: float = 0.03  # try 0.02–0.08


@dataclass
class InterpolantConfig:
    sigma_coef: float = 1.0
    beta_fn: str = "t"    # 't' or 't^2'
    t_min_train: float = 0.0
    t_max_train: float = 1.0
    t_min_sampling: float = 0.0
    t_max_sampling: float = 0.999
    EM_sample_steps: int = 200


@dataclass
class Config:
    model: ModelConfig
    data: DataConfig
    train: TrainingConfig
    interp: InterpolantConfig
    load_path: Optional[str] = None


# ========================== Trainer =========================================
class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

        # Data
        self.trainset = TrajectoryDataset(
            file=cfg.data.train_path,
            window=cfg.data.window,         # MUST be fixed and >= 2
            flatten=False,
            normalize=cfg.data.normalize,
        )

        self.train_dl = DataLoader(
            self.trainset,
            batch_size=cfg.data.batch_size,
            shuffle=cfg.data.shuffle,
            num_workers=cfg.data.num_workers,
            persistent_workers=(cfg.data.num_workers > 0 and cfg.data.persistent_workers),
            pin_memory=cfg.data.pin_memory,
        )

        # Validation loader (optional)
        self.val_dl = None
        if cfg.data.val_path:
            self.valset = TrajectoryDataset(
                file=cfg.data.val_path,
                window=cfg.data.window,
                flatten=False,
                normalize=cfg.data.normalize,
            )
            val_bs = getattr(cfg.data, "val_batch_size", None)
            if val_bs is None:
                val_bs = cfg.data.batch_size  # fallback to train batch size

            self.val_dl = DataLoader(
                self.valset,
                batch_size=val_bs,
                shuffle=False,
                num_workers=cfg.data.num_workers,
                persistent_workers=(cfg.data.num_workers > 0 and cfg.data.persistent_workers),
                pin_memory=cfg.data.pin_memory,
            )
    
        sample_batch = next(iter(self.train_dl))[0]          # (B, T, C, H, W)
        _, T, C, _, _ = sample_batch.shape
        assert T >= 2, "Please set --window >= 2."
        cond_channels = (T - 1) * C
        self.overfit_batch = sample_batch if cfg.train.overfit else None

        # Model & interpolant
        self.model = DriftModel(cfg.data, cfg.model, cond_channels=cond_channels).to(self.device)
        self.interp = Interpolant(cfg.interp)

        # Optim + sched
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.train.base_lr)
        total_steps = cfg.train.epochs * max(1, len(self.train_dl))
        self.sched = make_scheduler(
            self.opt,
            scheduler=getattr(cfg, "scheduler", "linear") if not hasattr(cfg.train, "scheduler") else cfg.train.scheduler,
            total_steps=total_steps,
            warmup_steps=getattr(cfg, "sched_warmup_steps", 0) if not hasattr(cfg.train, "sched_warmup_steps") else cfg.train.sched_warmup_steps,
            step_size=getattr(cfg, "sched_step_size", 10000) if not hasattr(cfg.train, "sched_step_size") else cfg.train.sched_step_size,
            gamma=getattr(cfg, "sched_gamma", 0.1) if not hasattr(cfg.train, "sched_gamma") else cfg.train.sched_gamma,
            tmax=getattr(cfg, "sched_tmax", None) if not hasattr(cfg.train, "sched_tmax") else cfg.train.sched_tmax,
            min_lr=getattr(cfg, "sched_min_lr", 0.0) if not hasattr(cfg.train, "sched_min_lr") else cfg.train.sched_min_lr,
        )
        self._sched_name = (getattr(cfg, "scheduler", "linear") if not hasattr(cfg.train, "scheduler") else cfg.train.scheduler).lower()

        # Time sampling
        self.U = torch.distributions.Uniform(low=cfg.interp.t_min_train, high=cfg.interp.t_max_train)

        # Step counter
        self.global_step = 0

        # Logging
        if cfg.train.use_wandb and WANDB:
            setup_wandb(cfg)

        # Load ckpt
        if cfg.load_path:
            self.load(cfg.load_path)

    def save(self, path: Optional[str] = None):
        path = path or self.cfg.train.ckpt_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {"model": self.model.state_dict(), "opt": self.opt.state_dict(), "step": self.global_step},
            path,
        )

    def load(self, path: str):
        D = torch.load(path, map_location=self.device)
        self.model.load_state_dict(D["model"])
        self.opt.load_state_dict(D["opt"])
        self.global_step = int(D.get("step", 0))

    @torch.no_grad()
    def _prep_batch(self, batch, for_sampling: bool = False) -> Dict[str, torch.Tensor]:
        """
        batch: (B, T, C, H, W)
        cond  = x[:, :-1] reshaped to (B, (T-1)*C, H, W)
        z0    = x[:, -2]  (last observed past frame)     -> used by interpolant
        z1    = x[:, -1]  (target / next frame)
        """
        if self.cfg.train.overfit and self.overfit_batch is not None:
            batch = self.overfit_batch

        x = batch.to(self.device)  # (B, T, C, H, W)
        B, T, C, H, W = x.shape

        past = x[:, :-1]                                  # (B, T-1, C, H, W)
        cond = past.reshape(B, (T-1)*C, H, W).contiguous()# (B, (T-1)*C, H, W)

        z0_base = x[:, -2]                                # (B, C, H, W)
        z1 = x[:, -1]                                     # (B, C, H, W)

        N = B
        D = {"z0": z0_base, "z1": z1, "cond": cond, "label": None, "N": N}
        D["t"] = self.U.sample((N,)).to(self.device)               # (B,)
        # D["noise"] = torch.randn_like(z0_base[:, :1])              # (B, 1, H, W)
        D["noise"] = torch.randn_like(z0_base)              # (B, C, H, W)

        D = self.interp.interpolant_coefs(D)
        D["zt"] = self.interp.compute_zt_new(D)                    # (B, C, H, W)
        D["drift_target"] = self.interp.compute_target_new(D)      # (B, C, H, W)

        # --- quick diversity tweak: label noise on target -----------------
        if (not for_sampling) and getattr(self.cfg.train, "label_noise_std", 0.0) > 0.0:
            ln = float(self.cfg.train.label_noise_std)
            # scale per-sample so noise is magnitude-aware
            scale = D["drift_target"].flatten(1).std(dim=1, keepdim=True).view(-1,1,1,1).clamp_min(1e-8)
            D["drift_target"] = D["drift_target"] + ln * scale * torch.randn_like(D["drift_target"])
        # ------------------------------------------------------------------
        return D

    @staticmethod
    def _img_sq_norm(x: Tensor) -> Tensor:
        return x.pow(2).sum(dim=(-1, -2, -3))  # sum over C,H,W → (B,)

    @torch.no_grad()
    def evaluate(self) -> float:
        if self.val_dl is None:
            return float("nan")
        self.model.eval()
        losses = []
        for batch in self.val_dl:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            D = self._prep_batch(batch, for_sampling=True)
            out = self.model(D["zt"], D["t"], D.get("label"), cond=D["cond"])
            loss = self._img_sq_norm(out - D["drift_target"]).mean()
            losses.append(float(loss.item()))
        return float(np.mean(losses)) if losses else float("nan")
    
    @torch.no_grad()
    def _em_sample_uncond(self, drift, interp, base, cond, steps=300, t_min=0.0, t_max=0.999):
        """
        Euler–Maruyama from base=z0 to t=1, with NO measurement guidance.
        Uses history 'cond' the same way as training (concatenated as channels).
        """
        device = base.device
        ts = torch.linspace(t_min, t_max, steps, device=device)
        dt = float(ts[1] - ts[0])
        xt = base.clone()

        for t in ts:
            tb  = t.repeat(xt.shape[0]).to(device)     # (B,)
            bF  = drift(xt, tb, cond=cond)             # (B,C,H,W)
            sig = interp.sigma(tb)                     # (B,1,1,1)
            mu  = xt + bF * dt
            xt  = mu + sig * torch.randn_like(mu) * math.sqrt(dt)

        return xt  # (B,C,H,W)
    
    @torch.no_grad()
    def _first_batch_from_loader(self, loader: DataLoader) -> torch.Tensor:
        """Return the first batch as a tensor on device: (B, T, C, H, W)."""
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            return batch.to(self.device)
        raise RuntimeError("Empty DataLoader; cannot sample.")

    @torch.no_grad()
    def _get_fixed_z0_and_cond(self, num_copies: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Take one sequence from val set if available (else train set),
        build z0 (= last observed past) and cond (= concatenated history),
        then replicate them 'num_copies' times along the batch dimension.
        """
        if self.val_dl is not None:
            batch = self._first_batch_from_loader(self.val_dl)  # (B, T, C, H, W)
        else:
            batch = self._first_batch_from_loader(self.train_dl)

        # choose the first item to keep samples fixed across epochs
        x = batch[0:1]  # (1, T, C, H, W)
        _, T, C, H, W = x.shape
        assert T >= 2, "Sampling expects window >= 2."

        past = x[:, :-1]                                       # (1, T-1, C, H, W)
        cond = past.reshape(1, (T-1)*C, H, W).contiguous()     # (1, (T-1)*C, H, W)
        z0   = x[:, -2]                                        # (1, C, H, W)

        # replicate to show stochastic variety
        z0  = z0.repeat(num_copies, 1, 1, 1)                   # (B=4, C, H, W)
        cond = cond.repeat(num_copies, 1, 1, 1)                # (B=4, (T-1)C, H, W)
        return z0, cond
    
    @torch.no_grad()
    def sample_and_log(self, epoch: int):
        if not (self.cfg.train.use_wandb and WANDB):
            return

        self.model.eval()

        # fixed inputs (4 copies to visualize stochastic variety)
        z0, cond = self._get_fixed_z0_and_cond(num_copies=4)

        # run unconditional EM sampling
        pred_uncond = self._em_sample_uncond(
            drift=self.model,
            interp=self.interp,
            base=z0,
            cond=cond,
            steps=self.cfg.interp.EM_sample_steps,
            t_min=self.cfg.interp.t_min_sampling,
            t_max=self.cfg.interp.t_max_sampling,
        )  # (4, C, H, W)

        # build a simple 2x2 figure with the first channel
        imgs = pred_uncond.detach().cpu()
        B, C, H, W = imgs.shape
        k = min(4, B)

        fig, axs = plt.subplots(2, 2, figsize=(6, 6), constrained_layout=True)
        axs = axs.ravel()
        for i in range(4):
            ax = axs[i]
            if i < k:
                ax.imshow(imgs[i, 0], origin="upper")  # show channel 0
                ax.set_title(f"sample {i}")
            ax.set_xticks([]); ax.set_yticks([])
        # if less than 4 samples, hide unused axes
        for i in range(k, 4):
            axs[i].axis("off")

        wandb.log(
            {"samples/uncond_grid": wandb.Image(fig),
            "epoch": epoch,
            "step": self.global_step},
            step=self.global_step
        )
        plt.close(fig)

        self.model.train()

    def train_one_epoch(self, epoch: int):
        self.model.train()
        pbar = tqdm(self.train_dl, desc=f"epoch {epoch}", leave=True)
        running = 0.0
        last_lr = self.opt.param_groups[0]["lr"]

        for _, batch in enumerate(pbar):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            D = self._prep_batch(batch, for_sampling=False)
            out = self.model(D["zt"], D["t"], D.get("label"), cond=D["cond"])
            loss = self._img_sq_norm(out - D["drift_target"]).mean()

            loss.backward()
            _ = clip_grad_norm(self.model, max_norm=self.cfg.train.max_grad_norm)
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)

            # step scheduler per batch unless it's ReduceLROnPlateau (handled after validation)
            if getattr(self, "_sched_name", "linear") != "plateau":
                self.sched.step()

            self.global_step += 1
            running += float(loss.item())

            last_lr = self.opt.param_groups[0]["lr"]
            pbar.set_postfix(train_loss=f"{loss.item():.6f}", lr=f"{last_lr:.2e}")

            # optional logging
            if self.cfg.train.use_wandb and WANDB and (self.global_step % self.cfg.train.print_every == 0):
                wandb.log({
                    "train/loss": float(loss.item()),
                    "train/lr": float(last_lr),
                    "epoch": epoch,
                    "step": self.global_step
                }, step=self.global_step)

            # if self.cfg.train.use_wandb and WANDB and (self.global_step % self.cfg.train.sample_every_steps == 0):
            #     self.sample_and_log()

            if self.global_step % self.cfg.train.save_every_steps == 0:
                self.save()

        # end-of-epoch metrics
        avg_train = running / max(1, len(self.train_dl))
        val_loss = self.evaluate() if getattr(self, "val_dl", None) is not None else float("nan")

        # one final postfix update so the kept bar shows epoch stats
        pbar.set_postfix(
            train_loss=f"{avg_train:.6f}",
            val_loss=("n/a" if (val_loss != val_loss) else f"{val_loss:.6f}"),  # NaN-safe
            lr=f"{last_lr:.2e}",
        )
        pbar.refresh()

        # W&B epoch-level logging
        if self.cfg.train.use_wandb and WANDB:
            log = {"train/epoch_avg_loss": avg_train, "train/lr": float(last_lr), "epoch": epoch, "step": self.global_step}
            if val_loss == val_loss:  # not NaN
                log["val/loss"] = float(val_loss)
            wandb.log(log, step=self.global_step)

        return avg_train, val_loss

    def fit(self):

        best_val = float("inf")
        best_path = os.path.splitext(self.cfg.train.ckpt_path)[0] + ".best.pt"

        for epoch in range(1, self.cfg.train.epochs + 1):
            avg_train, val_loss = self.train_one_epoch(epoch)

            # Step ReduceLROnPlateau with validation signal (if chosen)
            if getattr(self, "_sched_name", "linear") == "plateau" and val_loss == val_loss:
                self.sched.step(val_loss)

            # Log image samples every N epochs
            if (self.cfg.train.use_wandb and WANDB
                and self.cfg.train.sample_every_epochs > 0
                and (epoch % self.cfg.train.sample_every_epochs == 0)):
                self.sample_and_log(epoch)

            # Save best-by-val
            if val_loss == val_loss and val_loss < best_val:
                best_val = val_loss
                os.makedirs(os.path.dirname(best_path), exist_ok=True)
                torch.save({
                    "model": self.model.state_dict(),
                    "opt": self.opt.state_dict(),
                    "step": self.global_step,
                    "val_loss": best_val
                }, best_path)

        # final save
        self.save()


# ========================== CLI =============================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_data_path", type=str, required=True, help="HDF5 file with dataset['x'] of shape (N,T,C,H,W)")
    p.add_argument("--val_data_path", type=str, required=True, help="Path to validation HDF5 file")
    p.add_argument("--hrly_freq", type=int, required=True, help="Hourly frequency of the data (e.g., 3 for 3-hourly data)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--window", type=int, default=7)
    p.add_argument("--normalize", type=int, default=1)

    p.add_argument("--C", type=int, default=2)
    p.add_argument("--H", type=int, default=64)
    p.add_argument("--W", type=int, default=64)
    p.add_argument("--sigma_coef", type=float, default=1.0)
    p.add_argument("--label_noise_std", type=float, default=0.0)

    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ckpt_path", type=str, default="./checkpoints/latest.pt")
    p.add_argument("--sample_every_epochs", type=int, default=100)

    p.add_argument("--print_every", type=int, default=100)
    p.add_argument("--sample_every_steps", type=int, default=1000)
    p.add_argument("--save_every_steps", type=int, default=5000)

    p.add_argument("--use_wandb", type=int, default=0)
    p.add_argument("--wandb_project", type=str, default="sqg_min")
    p.add_argument("--wandb_entity", type=str, default=None)

    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--load_path", type=str, default=None)
    p.add_argument("--overfit", type=int, default=0)
    p.add_argument("--em_steps", type=int, default=500)

    # scheduler options
    p.add_argument("--scheduler", type=str, default="linear",
                   choices=["linear", "cosine", "step", "plateau", "onecycle"])
    p.add_argument("--sched_warmup_steps", type=int, default=0)   # linear/onecycle warmup
    p.add_argument("--sched_step_size", type=int, default=10000)  # for step
    p.add_argument("--sched_gamma", type=float, default=0.1)      # for step/plateau
    p.add_argument("--sched_tmax", type=int, default=None)        # for cosine
    p.add_argument("--sched_patience", type=int, default=5)       # for plateau
    p.add_argument("--sched_min_lr", type=float, default=0.0)     # for plateau/cosine

    args = p.parse_args()

    data_dir = f"/central/scratch/sotakao/sqg_train_data/{args.hrly_freq}hrly/"

    data = DataConfig(
        train_path=os.path.join(data_dir, args.train_data_path),
        val_path=os.path.join(data_dir, args.val_data_path),
        hrly_freq=args.hrly_freq,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        C=args.C, H=args.H, W=args.W,
        window=(None if args.window in [None, 0] else args.window),
        normalize=bool(args.normalize),
        grid_kwargs={"normalize": False},
    )

    if args.normalize:
        ckpt_path = f"./checkpoints/latest_{args.hrly_freq}hrly_window_{args.window}_{args.scheduler}_sigma_{args.sigma_coef}_label_noise_{args.label_noise_std}_normalized.pt"
        run_name = f"{args.hrly_freq}hrly_window_{args.window}_{args.scheduler}_sigma_{args.sigma_coef}_label_noise_{args.label_noise_std}_normalized"
    else:
        ckpt_path = f"./checkpoints/latest_{args.hrly_freq}hrly_window_{args.window}_sigma_{args.sigma_coef}_{args.scheduler}_label_noise_{args.label_noise_std}.pt"
        run_name = f"{args.hrly_freq}hrly_window_{args.window}_{args.scheduler}_sigma_{args.sigma_coef}_label_noise_{args.label_noise_std}"
    
    train = TrainingConfig(
        device=args.device,
        base_lr=args.lr,
        epochs=args.epochs,
        print_every=args.print_every,
        sample_every_steps=args.sample_every_steps,
        save_every_steps=args.save_every_steps,
        ckpt_path=ckpt_path,
        run_name=run_name,
        use_wandb=bool(args.use_wandb),
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        overfit=bool(args.overfit),
        scheduler=args.scheduler,
        sample_every_epochs=args.sample_every_epochs,
        label_noise_std=args.label_noise_std,
    )
    interp = InterpolantConfig(EM_sample_steps=args.em_steps,
                               sigma_coef=args.sigma_coef)
    model = ModelConfig()

    cfg = Config(model=model, data=data, train=train, interp=interp,
                 load_path=args.load_path)

    trainer = Trainer(cfg)
    trainer.fit()

if __name__ == "__main__":
    main()
