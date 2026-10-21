import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import yaml
import torch
import numpy as np
from data.smal_quadrupeds import SMALQuadrupedGenerator, FAMILIES
from models.point_dit import PointDiT
from models.flow_matching import OptimalTransportFlowMatching
from pipeline.metrics import compute_chamfer_distance

def main():
    """
    Family recognition: generate creatures from unseen skeletons and check that each one is
    closest (mean Chamfer Distance) to real shapes of its own family rather than another one.
    """
    parser = argparse.ArgumentParser(description="Evaluate skeleton conditioning via family recognition.")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--samples-per-family", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=None, help="ODE steps (default: config)")
    parser.add_argument("--solver", type=str, default=None, choices=["euler", "heun"], help="ODE solver (default: config)")
    parser.add_argument("--guidance-scale", type=float, default=None, help="classifier-free guidance scale (default: config)")
    parser.add_argument("--raw-weights", action="store_true", help="use the last optimizer weights instead of the EMA")
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
    ckpt_path = args.checkpoint or os.path.join(cfg["training"]["checkpoint_dir"], f"model_epoch_{cfg['training']['num_epochs']:03d}.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"] if args.raw_weights else ckpt.get("ema_state", ckpt["model_state"]))
    model.to(device)
    print(f"Loaded checkpoint from {ckpt_path}")
    fm = OptimalTransportFlowMatching(model)

    # Unseen seed: skeletons and reference shapes are disjoint from the training set (seed 42)
    torch.manual_seed(0)
    generator = SMALQuadrupedGenerator(seed=123)
    n = args.samples_per_family
    to_tensor = lambda a: torch.from_numpy(a).unsqueeze(0).to(device)
    refs = {f: [to_tensor(generator.generate_instance(f)["points"]) for _ in range(n)] for f in FAMILIES}

    confusion = np.zeros((len(FAMILIES), len(FAMILIES)), dtype=int)
    own_cd = []
    for fi, family in enumerate(FAMILIES):
        for _ in range(n):
            joints = to_tensor(generator.generate_instance(family)["joints"])
            _, gen_pts = fm.sample(
                joints,
                num_points=cfg["data"]["num_points"],
                num_steps=args.num_steps or cfg["sampling"]["num_steps"],
                solver=args.solver or cfg["sampling"]["solver"],
                guidance_scale=args.guidance_scale or cfg["sampling"].get("guidance_scale", 1.0)
            )
            scores = [np.mean([compute_chamfer_distance(gen_pts, r).item() for r in refs[o]]) for o in FAMILIES]
            confusion[fi, int(np.argmin(scores))] += 1
            own_cd.append(scores[fi])

    hits = int(np.trace(confusion))
    total = confusion.sum()
    print(f"Family accuracy: {hits}/{total} ({hits / total:.0%})")
    print(f"Mean Chamfer Distance to own family: {np.mean(own_cd):.4f}")
    print("Confusion matrix (rows = true family, cols = predicted):")
    print(" " * 16 + " ".join(f"{f[:4]:>4s}" for f in FAMILIES))
    for fi, family in enumerate(FAMILIES):
        print(f"{family:16s}" + " ".join(f"{c:4d}" for c in confusion[fi]))

if __name__ == "__main__":
    main()
