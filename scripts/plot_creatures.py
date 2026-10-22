import os
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    """
    Side and top views of the exported creatures (final step of each trajectory) with their skeleton,
    so the README image always shows the same specimens as the viewer.
    """
    parser = argparse.ArgumentParser(description="Plot exported creatures for the README.")
    parser.add_argument("--input", type=str, default="viewer/data/creatures.json")
    parser.add_argument("--output", type=str, default="docs/images/creatures.png")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        creatures = json.load(f)["creatures"]

    fig, axes = plt.subplots(2, len(creatures), figsize=(4.4 * len(creatures), 8.4), squeeze=False)
    for col, c in enumerate(creatures):
        points = np.array(c["trajectory"][-1])
        joints = np.array(c["joints"])
        bones = [(p, k) for k, p in enumerate(c["joint_parents"]) if p >= 0]
        # Row 0: side view (x, y); row 1: top view (x, z)
        for row, axis in enumerate([1, 2]):
            ax = axes[row, col]
            ax.scatter(points[:, 0], points[:, axis], s=1.5, alpha=0.6)
            for p, k in bones:
                ax.plot(joints[[p, k], 0], joints[[p, k], axis], color="red", linewidth=1.2)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_aspect("equal")
        axes[0, col].set_title(c["name"], fontsize=10)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=80)
    print(f"Saved {len(creatures)} creatures to {args.output}")

if __name__ == "__main__":
    main()
