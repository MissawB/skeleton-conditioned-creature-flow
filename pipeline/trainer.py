import math
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional
from models.flow_matching import OptimalTransportFlowMatching
from pipeline.ema import EMA

class Trainer:
    """Training manager for Flow Matching Point-DiT with automatic device selection."""
    
    def __init__(
        self,
        flow_matching: OptimalTransportFlowMatching,
        dataset,
        batch_size: int = 16,
        lr: float = 2e-4,
        weight_decay: float = 1e-5,
        checkpoint_dir: str = "checkpoints",
        device: str = "cuda",
        ema_decay: float = 0.995,
        lr_schedule: str = "constant",
        num_epochs: int = 1,
        warmup_steps: int = 0,
        min_lr_ratio: float = 0.0,
        mixed_precision: str = "no",
        sampler=None,
        collate_fn=None,
        num_workers: int = 0
    ):
        self.fm = flow_matching
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.fm.model.to(self.device)
        
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=sampler is None, sampler=sampler, drop_last=True,
                                     collate_fn=collate_fn, num_workers=num_workers, persistent_workers=num_workers > 0)
        self.optimizer = torch.optim.AdamW(self.fm.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.ema = EMA(self.fm.model, decay=ema_decay)
        # Linear warmup, then cosine decay from lr to min_lr_ratio * lr over the whole run
        total_steps = num_epochs * len(self.dataloader)
        def lr_factor(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            if lr_schedule == "constant":
                return 1.0
            progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_factor)
        # bf16 autocast keeps fp32 weights and needs no loss scaling
        self.amp_dtype = torch.bfloat16 if mixed_precision == "bf16" else None
        self.checkpoint_dir = checkpoint_dir
        self.skipped_steps = 0
        os.makedirs(checkpoint_dir, exist_ok=True)

    def train_epoch(self, epoch: int) -> float:
        self.fm.model.train()
        total_loss = 0.0
        finite_steps = 0

        for batch in self.dataloader:
            self.optimizer.zero_grad()
            with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
                if hasattr(self.fm, "loss_from_batch"):
                    # Flows with richer batches (free skeletons, masks) read what they need themselves
                    loss = self.fm.loss_from_batch(batch)
                else:
                    pts = batch["points"].to(self.device)
                    jts = batch["joints"].to(self.device)
                    noise = batch["noise"].to(self.device) if "noise" in batch else None
                    loss = self.fm.compute_loss(pts, jts, x_0=noise)
            if not torch.isfinite(loss):
                # One non-finite step would spread NaN through every weight via the optimizer and the EMA
                self.skipped_steps += 1
                self.scheduler.step()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.fm.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.ema.update(self.fm.model)

            total_loss += loss.item()
            finite_steps += 1

        return total_loss / finite_steps if finite_steps else float("nan")

    def save_checkpoint(self, epoch: int) -> str:
        path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch:03d}.pt")
        torch.save({
            "epoch": epoch,
            "model_state": self.fm.model.state_dict(),
            "ema_state": self.ema.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "ema_updates": self.ema.num_updates
        }, path)
        return path

    def load_checkpoint(self, path: str) -> int:
        """Restores model, EMA, optimizer and schedule to continue a run; returns the epoch the checkpoint ended."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.fm.model.load_state_dict(state["model_state"])
        self.ema.model.load_state_dict(state["ema_state"])
        self.ema.num_updates = state.get("ema_updates", 10 ** 6)  # older checkpoints: past the EMA warmup
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.scheduler.load_state_dict(state["scheduler_state"])
        return state["epoch"]
