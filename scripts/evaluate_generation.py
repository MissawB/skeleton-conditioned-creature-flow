import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import yaml
import torch
from data.smal_quadrupeds import SMALQuadrupedGenerator, FAMILIES
from models.flow_matching import OptimalTransportFlowMatching
from pipeline.metrics import compute_generation_metrics, surface_distance
from scripts.evaluate_families import load_model, add_sampling_args, sampling_kwargs

def main():
    """
    Standard generation metrics (MMD, COV, 1-NNA with Chamfer Distance): creatures generated from unseen
    skeletons against as many real creatures drawn independently from the same families.
    """
    parser = argparse.ArgumentParser(description="Evaluate generation quality with MMD, COV and 1-NNA.")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--samples-per-family", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=20, help="skeletons sampled together")
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
    gen, ref, gaps = [], [], []
    for family in FAMILIES:
        skeletons = []
        for _ in range(args.samples_per_family):
            skeletons.append(to_tensor(generator.generate_instance(family)["joints"]))
            ref.append(to_tensor(generator.generate_instance(family)["points"]))
        radii = generator.bone_radii(family)
        for batch in torch.stack(skeletons).split(args.batch_size):
            points = fm.sample(batch, **sample_kwargs)[1]
            gen.extend(points)
            gaps.extend(surface_distance(p, j, radii) for p, j in zip(points, batch))

    metrics = compute_generation_metrics(torch.stack(gen), torch.stack(ref))
    print(f"MMD-CD: {metrics['mmd']:.5f} | COV-CD: {metrics['cov']:.0%} | 1-NNA-CD: {metrics['1nna']:.1%} (50% is ideal)")
    # Distance to the procedural surface the skeleton defines (real shapes: median about 0.0004)
    gap = torch.cat(gaps)
    print(f"Surface distance: median {gap.abs().median():.4f} | p90 {gap.abs().quantile(0.9):.4f} | "
          f"mean signed {gap.mean():+.4f} (positive = outside) | {len(gen)} vs {len(ref)} shapes")

if __name__ == "__main__":
    main()
