import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import yaml
import torch
import numpy as np
from data.smal_quadrupeds import SMALQuadrupedGenerator, PARENT_INDICES
from pipeline.hybrid_splicer import KinematicHybridSplicer
from models.point_dit import PointDiT
from models.flow_matching import OptimalTransportFlowMatching

def main():
    parser = argparse.ArgumentParser(description="Sample creatures and export flow trajectories for the viewer.")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointDiT(
        point_dim=cfg["model"]["point_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        num_heads=cfg["model"]["num_heads"],
        num_joints=cfg["model"]["num_joints"],
        spatial_distance_scale=cfg["model"]["spatial_distance_scale"]
    )
    
    ckpt_path = args.checkpoint or os.path.join(cfg["training"]["checkpoint_dir"],
                                                f"model_epoch_{cfg['training']['num_epochs']:03d}.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt.get("ema_state", ckpt["model_state"]))
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        print("Note: No trained checkpoint found yet. Using freshly initialized model for demonstration.")
        
    model.to(device)
    fm = OptimalTransportFlowMatching(model)
    generator = SMALQuadrupedGenerator(seed=42)
    splicer = KinematicHybridSplicer()
    
    specimens = [
        {"name": "Lion (Felidae)", "type": "pure", "family": "felidae"},
        {"name": "Horse (Equidae)", "type": "pure", "family": "equidae"},
        {"name": "Wolf (Canidae)", "type": "pure", "family": "canidae"},
        {"name": "Bison (Bovidae)", "type": "pure", "family": "bovidae"},
        {"name": "Chimera (Lion Body + Horse Neck/Head)", "type": "hybrid", "host": "felidae", "donor": "equidae", "junction": "neck_base"}
    ]
    
    exported = []
    num_points = cfg["data"]["num_points"]
    steps = cfg["sampling"]["num_steps"]
    solver = cfg["sampling"]["solver"]
    
    for spec in specimens:
        print(f"Generating {spec['name']}...")
        if spec["type"] == "pure":
            joints = generator.generate_instance(spec["family"])["joints"]
        else:
            skel_h = generator.generate_instance(spec["host"])["joints"]
            skel_d = generator.generate_instance(spec["donor"])["joints"]
            joints = splicer.splice(skel_h, skel_d, junction_joint=spec["junction"])
            
        j_tensor = torch.from_numpy(joints).unsqueeze(0).to(device)
        traj, _ = fm.sample(j_tensor, num_points=num_points, num_steps=steps, solver=solver, record_trajectory=True,
                            guidance_scale=cfg["sampling"].get("guidance_scale", 1.0))
        
        exported.append({
            "name": spec["name"],
            "type": spec["type"],
            "joints": joints.tolist(),
            "joint_parents": PARENT_INDICES,
            "trajectory": [step.squeeze(0).numpy().tolist() for step in traj]
        })
        
    out_dir = cfg["sampling"]["export_dir"]
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "creatures.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"creatures": exported}, f)
    print(f"Exported {len(exported)} creature trajectories to {out_file}")

if __name__ == "__main__":
    main()
