import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import yaml
import torch
from data.dataset import CreatureDataset
from models.point_dit import PointDiT
from models.flow_matching import OptimalTransportFlowMatching
from pipeline.trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description="Train 3D Creature Flow Matching model.")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    args = parser.parse_args()
    
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    dataset = CreatureDataset(
        num_samples_per_family=cfg["data"]["num_samples_per_family"],
        num_points=cfg["data"]["num_points"],
        augment=True,
        cache_dir=cfg["data"]["cache_dir"],
        ot_noise_per_sample=cfg["data"].get("ot_noise_per_sample", 0),
        independent_noise_prob=cfg["data"].get("independent_noise_prob", 0.0),
        ot_noise_beta=cfg["data"].get("ot_noise_beta", 0.0)
    )
    online_ot = cfg["data"].get("online_ot", False)
    if online_ot and cfg["data"].get("ot_noise_per_sample", 0) > 0:
        raise ValueError("data.online_ot matches noise on the GPU: set data.ot_noise_per_sample to 0")
    
    model = PointDiT(
        point_dim=cfg["model"]["point_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        num_heads=cfg["model"]["num_heads"],
        num_joints=cfg["model"]["num_joints"],
        # Distances grow with data_scale, so the bias scale shrinks by its square to keep the same attention
        spatial_distance_scale=cfg["model"]["spatial_distance_scale"] / cfg["model"].get("data_scale", 1.0) ** 2,
        data_scale=cfg["model"].get("data_scale", 1.0),
        fourier_octaves=cfg["model"].get("fourier_octaves", 0),
        time_scale=cfg["model"].get("time_scale", 1.0),
        local_k=cfg["model"].get("local_k", 0)
    )
    fm = OptimalTransportFlowMatching(model, cond_drop_prob=cfg["training"].get("cond_drop_prob", 0.0),
                                      late_t_prob=cfg["training"].get("late_t_prob", 0.0),
                                      late_t_min=cfg["training"].get("late_t_min", 0.9),
                                      online_ot=online_ot)
    
    trainer = Trainer(
        flow_matching=fm,
        dataset=dataset,
        batch_size=cfg["training"]["batch_size"],
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        checkpoint_dir=cfg["training"]["checkpoint_dir"],
        device=cfg["training"]["device"],
        ema_decay=cfg["training"].get("ema_decay", 0.995),
        lr_schedule=cfg["training"].get("lr_schedule", "constant"),
        num_epochs=cfg["training"]["num_epochs"],
        warmup_steps=cfg["training"].get("warmup_steps", 0),
        min_lr_ratio=cfg["training"].get("min_lr_ratio", 0.0),
        mixed_precision=cfg["training"].get("mixed_precision", "no")
    )
    
    epochs = cfg["training"]["num_epochs"]
    print(f"Starting training on device: {trainer.device} for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        loss = trainer.train_epoch(epoch)
        print(f"Epoch {epoch:03d}/{epochs:03d} | Loss: {loss:.5f}")
        if epoch % cfg["training"]["save_every"] == 0 or epoch == epochs:
            path = trainer.save_checkpoint(epoch)
            print(f"--> Saved checkpoint to {path}")

if __name__ == "__main__":
    main()
