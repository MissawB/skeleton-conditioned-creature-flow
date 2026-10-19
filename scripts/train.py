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
        cache_dir=cfg["data"]["cache_dir"]
    )
    
    model = PointDiT(
        point_dim=cfg["model"]["point_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        num_heads=cfg["model"]["num_heads"],
        num_joints=cfg["model"]["num_joints"]
    )
    fm = OptimalTransportFlowMatching(model)
    
    trainer = Trainer(
        flow_matching=fm,
        dataset=dataset,
        batch_size=cfg["training"]["batch_size"],
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        checkpoint_dir=cfg["training"]["checkpoint_dir"],
        device=cfg["training"]["device"]
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
