"""Evaluates a free-skeleton creature checkpoint on Articulation-XL test creatures and draws a fixed plate."""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from data.axl.dataset import AXLDataset, collate_skeletons, sampling_weights
from pipeline.creature_eval import head_joint, skeleton_sensitivity, surface_gap
from pipeline.metrics import compute_generation_metrics
from scripts.train_axl import build_flow

BATCH = 16
HEAD_RADIUS = 0.25

def load_flow(cfg, checkpoint, device):
    fm = build_flow(cfg)
    fm.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["ema_state"])
    fm.model.to(device).eval()
    return fm

def generate(fm, items, device, num_points, use_z, seed=0):
    out = []
    for start in range(0, len(items), BATCH):
        batch = collate_skeletons(items[start:start + BATCH])
        args = [batch[k].to(device) for k in ("joints", "parents", "presence", "joint_mask")]
        z = fm.encode(batch["points"].to(device)) if use_z else None
        g = torch.Generator(device=device).manual_seed(seed + start)
        out.append(fm.sample(*args, z=z, num_points=num_points, generator=g)[1].cpu())
    return torch.cat(out)

def draw_plate(ds, indices, items, gen_skel, gen_z, path):
    fig, axes = plt.subplots(len(indices), 4, figsize=(9, 2.2 * len(indices)), squeeze=False)
    for r, (i, item) in enumerate(zip(indices, items)):
        names = ds.joint_names(i)
        joints, parents = item["joints"].numpy(), item["parents"].numpy()
        head = joints[head_joint(names, joints, parents)]
        for c, (cloud, title) in enumerate([(item["points"], "réel"), (gen_skel[r], "squelette seul"), (gen_z[r], "squelette + z")]):
            ax = axes[r, c]
            ax.scatter(cloud[:, 0], cloud[:, 1], s=0.3, linewidths=0)
            ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{ds.groups[i]} {ds.uuids[i][:8]} · {title}" if c == 0 else title, fontsize=7)
        ax = axes[r, 3]
        near = gen_z[r][(gen_z[r] - torch.from_numpy(head)).norm(dim=1) < HEAD_RADIUS]
        ref = item["points"][(item["points"] - torch.from_numpy(head)).norm(dim=1) < HEAD_RADIUS]
        ax.scatter(ref[:, 0], ref[:, 1], s=1, c="lightgray", linewidths=0)
        ax.scatter(near[:, 0], near[:, 1], s=1, linewidths=0)
        ax.set_xlim(head[0] - HEAD_RADIUS, head[0] + HEAD_RADIUS); ax.set_ylim(head[1] - HEAD_RADIUS, head[1] + HEAD_RADIUS)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title("tête (gris : réel)", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/axl_config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num", type=int, default=200)
    parser.add_argument("--plate", default=None, help="output PNG of the fixed plate")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_points = cfg["data"]["num_points"]
    fm = load_flow(cfg, args.checkpoint, device)
    ds = AXLDataset(cfg["data"]["cache_dir"], "test", num_points=n_points, augment=False)
    rng = np.random.default_rng(0)
    np.random.seed(0)

    weights = sampling_weights(ds.groups, cfg["data"]["animal_ratio"])
    chosen = rng.choice(len(ds), size=min(args.num, len(ds)), replace=False, p=weights / weights.sum())
    items = [ds[i] for i in chosen]
    real = torch.stack([it["points"] for it in items])

    began = time.time()
    results = {}
    for label, use_z in (("squelette seul", False), ("squelette + z (proche d'une reconstruction)", True)):
        gen = generate(fm, items, device, n_points, use_z)
        m = compute_generation_metrics(gen, real)
        gap = np.median([surface_gap(g, r) for g, r in zip(gen, real)])
        results[label] = dict(m, surface_gap=float(gap))
        print(f"{label}: MMD-CD {m['mmd']:.5f} | COV-CD {m['cov']:.0%} | 1-NNA {m['1nna']:.1%} "
              f"(générées {m['1nna_gen']:.0%}, réelles {m['1nna_ref']:.0%}) | écart médian au nuage réel {gap:.4f}", flush=True)
    seconds = (time.time() - began) / (2 * len(items))

    pairs = rng.choice(len(items), size=(20, 2), replace=False) if len(items) >= 40 else [(0, 1)]
    sens = [skeleton_sensitivity(fm, _to(collate_skeletons([items[a]]), device), _to(collate_skeletons([items[b]]), device),
                                 num_points=n_points) for a, b in pairs]
    print(f"Sensibilité au squelette (z fixé) : médiane {np.median(sens):.2f} (0 = z contourne le squelette) | {seconds * 1000:.0f} ms par créature", flush=True)
    results["skeleton_sensitivity"] = float(np.median(sens))

    if args.plate:
        animals = [i for i in range(len(ds)) if ds.groups[i] == "animal"]
        humans = [i for i in range(len(ds)) if ds.groups[i] == "humanoid"]
        plate_rng = np.random.default_rng(42)
        idx = list(plate_rng.choice(animals, 8, replace=False)) + list(plate_rng.choice(humans, 4, replace=False))
        plate_items = [ds[i] for i in idx]
        draw_plate(ds, idx, plate_items, generate(fm, plate_items, device, n_points, False, seed=7),
                   generate(fm, plate_items, device, n_points, True, seed=7), args.plate)
        print(f"Planche : {args.plate}")
    print(json.dumps(results))

def _to(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

if __name__ == "__main__":
    main()
