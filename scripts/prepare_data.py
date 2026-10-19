import argparse
import os
import yaml
from data.dataset import CreatureDataset

def main():
    parser = argparse.ArgumentParser(description="Pre-generate synthetic SMAL quadruped dataset.")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    args = parser.parse_args()
    
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    cache_dir = cfg["data"]["cache_dir"]
    samples_per_family = cfg["data"]["num_samples_per_family"]
    num_points = cfg["data"]["num_points"]
    
    print(f"Generating dataset: {samples_per_family} samples per family, {num_points} points...")
    dataset = CreatureDataset(
        num_samples_per_family=samples_per_family,
        num_points=num_points,
        augment=False,
        cache_dir=cache_dir
    )
    print(f"Dataset successfully created and cached at {cache_dir}/dataset.pt (Total: {len(dataset)} samples).")

if __name__ == "__main__":
    main()
