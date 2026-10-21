import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import torch
from data.dataset import CreatureDataset
from models.point_dit import PointDiT
from models.flow_matching import OptimalTransportFlowMatching
from pipeline.metrics import compute_chamfer_distance, compute_skeleton_coverage

def main():
    with open("configs/default_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    dataset = CreatureDataset(num_samples_per_family=5, num_points=1024, augment=False, cache_dir=None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = PointDiT(
        point_dim=cfg["model"]["point_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        num_heads=cfg["model"]["num_heads"],
        num_joints=cfg["model"]["num_joints"],
        spatial_distance_scale=cfg["model"]["spatial_distance_scale"]
    )
    ckpt_path = os.path.join(cfg["training"]["checkpoint_dir"], f"model_epoch_{cfg['training']['num_epochs']:03d}.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt.get("ema_state", ckpt["model_state"]))
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        print(f"Warning: no checkpoint at {ckpt_path}, evaluating an untrained model.")
    model.to(device)
    fm = OptimalTransportFlowMatching(model)
    
    print("Evaluating test metrics on generated creatures...")
    all_cd, all_cov = [], []
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        joints = sample["joints"].unsqueeze(0).to(device)
        target_pts = sample["points"].unsqueeze(0).to(device)
        
        _, gen_pts = fm.sample(joints, num_points=1024, num_steps=15)
        cd = compute_chamfer_distance(gen_pts, target_pts).item()
        cov = compute_skeleton_coverage(joints, gen_pts).item()
        all_cd.append(cd)
        all_cov.append(cov)
        
    print(f"Mean Chamfer Distance: {sum(all_cd)/len(all_cd):.5f}")
    print(f"Mean Skeleton Coverage: {sum(all_cov)/len(all_cov):.5f}")

if __name__ == "__main__":
    main()
