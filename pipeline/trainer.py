import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional
from models.flow_matching import OptimalTransportFlowMatching

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
        device: str = "cuda"
    ):
        self.fm = flow_matching
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.fm.model.to(self.device)
        
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.optimizer = torch.optim.AdamW(self.fm.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def train_epoch(self, epoch: int) -> float:
        self.fm.model.train()
        total_loss = 0.0
        
        for batch in self.dataloader:
            pts = batch["points"].to(self.device)
            jts = batch["joints"].to(self.device)
            noise = batch["noise"].to(self.device) if "noise" in batch else None
            
            self.optimizer.zero_grad()
            loss = self.fm.compute_loss(pts, jts, x_0=noise)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.fm.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(self.dataloader)
        return avg_loss

    def save_checkpoint(self, epoch: int) -> str:
        path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch:03d}.pt")
        torch.save({
            "epoch": epoch,
            "model_state": self.fm.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict()
        }, path)
        return path
