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

def load_model(cfg, ckpt_path, device, raw_weights=False):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = ckpt["model_state"] if raw_weights else ckpt.get("ema_state", ckpt["model_state"])
    # Size comes from the weights, not the config: an autoguidance guide is usually a smaller model
    model = PointDiT(
        point_dim=cfg["model"]["point_dim"],
        hidden_dim=state["final_norm.weight"].shape[0],
        num_layers=sum(1 for k in state if k.startswith("pos_heads.") and k.endswith(".weight")),
        num_heads=state["blocks.0.cross_attn.log_spatial_scale"].shape[0],
        num_joints=cfg["model"]["num_joints"],
        spatial_distance_scale=cfg["model"]["spatial_distance_scale"],
        fourier_octaves=(state["point_proj.weight"].shape[1] // cfg["model"]["point_dim"] - 1) // 2,
        local_k=int(state.get("local_k", 0))
    )
    model.load_state_dict(state)
    print(f"Loaded checkpoint from {ckpt_path}")
    return model.to(device)

def add_sampling_args(parser):
    parser.add_argument("--num-steps", type=int, default=None, help="ODE steps (default: config)")
    parser.add_argument("--solver", type=str, default=None, choices=["euler", "heun"], help="ODE solver (default: config)")
    parser.add_argument("--guidance-scale", type=float, default=None, help="guidance scale (default: config)")
    parser.add_argument("--guidance-interval", type=float, nargs=2, default=[0.0, 1.0], metavar=("T_LO", "T_HI"),
                        help="apply guidance only for t in [T_LO, T_HI] (t = 0 is noise)")
    parser.add_argument("--guide-checkpoint", type=str, default=None,
                        help="autoguidance: weaker model (e.g. early checkpoint) to guide with instead of CFG")
    parser.add_argument("--raw-weights", action="store_true", help="use the last optimizer weights instead of the EMA")

def sampling_kwargs(args, cfg, device):
    guide = load_model(cfg, args.guide_checkpoint, device, args.raw_weights) if args.guide_checkpoint else None
    return dict(
        num_points=cfg["data"]["num_points"],
        num_steps=args.num_steps or cfg["sampling"]["num_steps"],
        solver=args.solver or cfg["sampling"]["solver"],
        guidance_scale=args.guidance_scale or cfg["sampling"].get("guidance_scale", 1.0),
        guidance_interval=tuple(args.guidance_interval),
        guide_model=guide
    )

def main():
    """
    Family recognition: generate creatures from unseen skeletons and check that each one is
    closest (mean Chamfer Distance) to real shapes of its own family rather than another one.
    """
    parser = argparse.ArgumentParser(description="Evaluate skeleton conditioning via family recognition.")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--samples-per-family", type=int, default=8)
    add_sampling_args(parser)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.checkpoint or os.path.join(cfg["training"]["checkpoint_dir"], f"model_epoch_{cfg['training']['num_epochs']:03d}.pt")
    fm = OptimalTransportFlowMatching(load_model(cfg, ckpt_path, device, args.raw_weights))
    sample_kwargs = sampling_kwargs(args, cfg, device)

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
            _, gen_pts = fm.sample(joints, **sample_kwargs)
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
