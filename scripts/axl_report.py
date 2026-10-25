"""Statistics, contact sheets and an orientation audit for the Articulation-XL creature cache."""
import argparse
import collections
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from data.axl.dataset import AXLDataset
from data.axl.remote_parquet import open_parquet
from data.axl.selection import load_tags

SHOWN_POINTS = 2048

def _draw(ax, points, joints, parents, plane, title=None):
    a, b = plane
    ax.scatter(points[:, a], points[:, b], s=0.3, c="tab:blue", linewidths=0)
    for j, p in enumerate(parents):
        if p >= 0:
            ax.plot(joints[[p, j], a], joints[[p, j], b], c="red", lw=0.6)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=7)

def _item(ds, i):
    item = ds[i]
    return item["points"].numpy(), item["joints"].numpy(), item["parents"].numpy()

def contact_sheet(ds, indices, path, rows_per_item=True, augment_copies=0):
    """Side (x, y) and top (x, z) views per creature; with augment_copies, extra columns of deformed copies."""
    cols = 1 + augment_copies if augment_copies else 2
    fig, axes = plt.subplots(len(indices), cols, figsize=(2.2 * cols, 2.2 * len(indices)), squeeze=False)
    for r, i in enumerate(indices):
        ds.augment = False
        points, joints, parents = _item(ds, i)
        _draw(axes[r, 0], points, joints, parents, (0, 1), f"{ds.groups[i]} {ds.uuids[i][:8]}")
        if augment_copies:
            ds.augment = True
            for c in range(1, cols):
                p, j, par = _item(ds, i)
                _draw(axes[r, c], p, j, par, (0, 1), "deformed")
        else:
            _draw(axes[r, 1], points, joints, parents, (0, 2), "top")
    ds.augment = False
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)

def audit_pages(ds, indices, out_dir):
    """Pages of 20 creatures, side (x, y) and top (x, z) views, to check by eye that up is +Y and the gaze is +X."""
    pages = []
    for start in range(0, len(indices), 20):
        chunk = indices[start:start + 20]
        fig, axes = plt.subplots(5, 8, figsize=(16, 10.5), squeeze=False)
        for n in range(20):
            side, top = axes.flat[2 * n], axes.flat[2 * n + 1]
            if n < len(chunk):
                points, joints, parents = _item(ds, chunk[n])
                _draw(side, points, joints, parents, (0, 1), f"#{start + n} {ds.groups[chunk[n]]}")
                _draw(top, points, joints, parents, (0, 2), "top")
            else:
                side.axis("off")
                top.axis("off")
        fig.tight_layout()
        path = out_dir / f"audit_{start // 20:02d}.png"
        fig.savefig(path, dpi=80)
        plt.close(fig)
        pages.append(path.name)
    return pages

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="data/cache/axl")
    parser.add_argument("--out", default="docs/axl")
    parser.add_argument("--audit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    lines = ["# Données Articulation-XL : rapport de construction", ""]
    tags = load_tags(open_parquet("meta_categories.parquet").read(columns=["uuid", "category"]))
    datasets = {}
    for split in ("train", "test"):
        ds = AXLDataset(args.cache, split, num_points=SHOWN_POINTS, augment=False, seed=args.seed)
        datasets[split] = ds
        shard_dir = Path(args.cache) / split
        rejects = collections.Counter()
        for path in shard_dir.glob("rg*.rejects.json"):
            rejects.update(json.loads(path.read_text()).values())
        orient = collections.Counter()
        joint_counts = []
        for path in ds.shards:
            with np.load(path) as shard:
                orient.update(shard["orient"].tolist())
                joint_counts += np.diff(shard["joint_offsets"]).tolist()
        tag_counts = collections.Counter(t for u in ds.uuids for t in tags.get(u, ()))
        size_gb = sum(p.stat().st_size for p in ds.shards) / 1e9

        lines += [f"## {split}", "",
                  f"- Fragments : {len(ds.shards)}, taille {size_gb:.2f} Go",
                  f"- Modèles retenus : {len(ds)} ({', '.join(f'{g} {n}' for g, n in collections.Counter(ds.groups.tolist()).most_common())})",
                  f"- Rejets : {sum(rejects.values())} ({', '.join(f'{r} {n}' for r, n in rejects.most_common())})",
                  f"- Orientation décidée par : {', '.join(f'{m} {n}' for m, n in orient.most_common())}",
                  "", "| Tag | Modèles retenus |", "|---|---|"]
        lines += [f"| {t} | {n} |" for t, n in tag_counts.most_common()]
        if joint_counts:
            edges = [2, 10, 20, 30, 40, 60, 80, 120, 256]
            hist = np.histogram(joint_counts, bins=edges)[0]
            lines += ["", "| Articulations | Modèles |", "|---|---|"]
            lines += [f"| {a}–{b - 1} | {n} |" for a, b, n in zip(edges[:-1], edges[1:], hist)]
        lines.append("")

    ds = datasets["train"]
    animals = np.where(ds.groups == "animal")[0]
    humanoids = np.where(ds.groups == "humanoid")[0]
    contact_sheet(ds, rng.choice(animals, 12, replace=False), out / "sheet_animals.png")
    contact_sheet(ds, rng.choice(humanoids, 8, replace=False), out / "sheet_humanoids.png")
    contact_sheet(ds, rng.choice(animals, 4, replace=False), out / "sheet_augment.png", augment_copies=4)
    audit_idx = rng.choice(len(ds), min(args.audit, len(ds)), replace=False)
    pages = audit_pages(ds, audit_idx, out)
    (out / "audit_index.json").write_text(json.dumps([str(ds.uuids[i]) for i in audit_idx]))

    lines += ["## Planches", "", "![Animaux](sheet_animals.png)", "", "![Humanoïdes](sheet_humanoids.png)", "",
              "![Déformations](sheet_augment.png)", "",
              f"Audit d'orientation : {len(audit_idx)} créatures sur {len(pages)} pages (`audit_*.png`, non commitées).", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:40]))

if __name__ == "__main__":
    main()
