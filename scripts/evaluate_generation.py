import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import yaml
import torch
from data.smal_quadrupeds import SMALQuadrupedGenerator, FAMILIES
from models.flow_matching import OptimalTransportFlowMatching
from pipeline.metrics import compute_generation_metrics
from scripts.evaluate_families import load_model, add_sampling_args, sampling_kwargs

def main():
    """
    Standard generation metrics (MMD, COV, 1-NNA with Chamfer Distance): creatures generated from unseen
    skeletons against as many real creatures drawn independently from the same families.
    """
    parser = argparse.ArgumentParser(description="Evaluate generation quality with MMD, COV and 1-NNA.")
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

    # Unseen seed, disjoint from the training set (seed 42); references are separate draws of the same families
    torch.manual_seed(0)
    generator = SMALQuadrupedGenerator(seed=123)
    to_tensor = lambda a: torch.from_numpy(a).to(device)
    gen, ref = [], []
    for family in FAMILIES:
        for _ in range(args.samples_per_family):
            joints = to_tensor(generator.generate_instance(family)["joints"]).unsqueeze(0)
            gen.append(fm.sample(joints, **sample_kwargs)[1][0])
            ref.append(to_tensor(generator.generate_instance(family)["points"]))

    metrics = compute_generation_metrics(torch.stack(gen), torch.stack(ref))
    print(f"MMD-CD: {metrics['mmd']:.5f} | COV-CD: {metrics['cov']:.0%} | 1-NNA-CD: {metrics['1nna']:.1%} (50% is ideal)")

if __name__ == "__main__":
    main()
