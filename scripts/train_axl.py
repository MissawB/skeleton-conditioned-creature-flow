"""Trains the free-skeleton creature model on the Articulation-XL cache; --resume continues from the last checkpoint."""
import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import yaml
from torch.utils.data import WeightedRandomSampler
from data.axl.dataset import AXLDataset, collate_skeletons, sampling_weights
from models.creature_dit import CreatureDiT
from models.creature_flow import CreatureFlowMatching
from models.shape_encoder import ShapeEncoder
from pipeline.trainer import Trainer

def build_flow(cfg: dict) -> CreatureFlowMatching:
    m, t = cfg["model"], cfg["training"]
    dit = CreatureDiT(hidden_dim=m["hidden_dim"], num_layers=m["num_layers"], num_heads=m["num_heads"],
                      z_dim=m["z_dim"], data_scale=m["data_scale"])
    return CreatureFlowMatching(dit, ShapeEncoder(z_dim=m["z_dim"]), kl_weight=t["kl_weight"],
                                z_drop_prob=t["z_drop_prob"], ot_iters=t["ot_iters"], stochastic_z=t.get("stochastic_z", True))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/axl_config.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-shards", type=int, default=None, help="use only the first shards (quick trials)")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    d, t = cfg["data"], cfg["training"]

    dataset = AXLDataset(d["cache_dir"], "train", num_points=d["num_points"], augment=True, max_shards=args.max_shards)
    sampler = WeightedRandomSampler(torch.as_tensor(sampling_weights(dataset.groups, d["animal_ratio"])), num_samples=len(dataset))
    fm = build_flow(cfg)
    trainer = Trainer(fm, dataset, batch_size=t["batch_size"], lr=t["learning_rate"], weight_decay=t["weight_decay"],
                      checkpoint_dir=t["checkpoint_dir"], device=t["device"], ema_decay=t["ema_decay"],
                      num_epochs=t["num_epochs"], mixed_precision=t["mixed_precision"],
                      lr_schedule=t.get("lr_schedule", "constant"), warmup_steps=t.get("warmup_steps", 0),
                      min_lr_ratio=t.get("min_lr_ratio", 0.0), sampler=sampler, collate_fn=collate_skeletons, num_workers=d["num_workers"])
    fm.model.to(trainer.device)

    start = 1
    if args.resume:
        checkpoints = sorted(glob.glob(os.path.join(t["checkpoint_dir"], "model_epoch_*.pt")))
        if checkpoints:
            start = trainer.load_checkpoint(checkpoints[-1]) + 1
            print(f"Resumed from {checkpoints[-1]}")

    params = sum(p.numel() for p in fm.model.parameters()) / 1e6
    print(f"{len(dataset)} creatures ({dataset.excluded} broken rigs left out), {params:.1f} M parameters, "
          f"{len(trainer.dataloader)} steps per epoch", flush=True)
    for epoch in range(start, t["num_epochs"] + 1):
        began = time.time()
        loss = trainer.train_epoch(epoch)
        elapsed = time.time() - began
        print(f"Epoch {epoch:03d}/{t['num_epochs']:03d} | Loss: {loss:.5f} | fm {fm.last_stats['fm']:.4f} kl {fm.last_stats['kl']:.2f} "
              f"| lr {trainer.optimizer.param_groups[0]['lr']:.1e}, {trainer.skipped_steps} non-finite steps skipped "
              f"| {len(trainer.dataloader) * t['batch_size'] / elapsed:.0f} samples/s, {elapsed / 60:.1f} min", flush=True)
        if epoch % t["save_every"] == 0 or epoch == t["num_epochs"]:
            print(f"--> Saved checkpoint to {trainer.save_checkpoint(epoch)}", flush=True)

if __name__ == "__main__":
    main()
