"""Evaluation helpers for free-skeleton creatures (stage 2a)."""
import numpy as np
import torch
from data.axl.normalize import HEAD_WORDS
from pipeline.metrics import compute_chamfer_distance

def head_joint(names, joints: np.ndarray, parents: np.ndarray) -> int:
    """A joint named like a head (shallowest one), otherwise the highest leaf."""
    named = [i for i, n in enumerate(names) if any(w in n.lower() for w in HEAD_WORDS) and "forehead" not in n.lower()]
    if named:
        depth = np.zeros(len(parents), int)
        for i in range(len(parents)):
            k = i
            while parents[k] >= 0:
                k, depth[i] = parents[k], depth[i] + 1
        return min(named, key=lambda i: depth[i])
    leaves = [i for i in range(len(parents)) if i not in set(parents.tolist())]
    return max(leaves, key=lambda i: joints[i][1])

def surface_gap(generated: torch.Tensor, reference: torch.Tensor) -> float:
    """Median distance from each generated point to the nearest reference point (dense cloud as a stand-in for the mesh)."""
    return torch.cdist(generated.float(), reference.float()).min(dim=1).values.median().item()

def skeleton_sensitivity(flow, a: dict, b: dict, num_points: int = 2048, seed: int = 0) -> float:
    """
    CD(gen(skeleton A, z_B), gen(skeleton B, z_B)) / CD(A, B): with the shape code fixed, does the skeleton still
    change the creature? Near 0 means z bypasses the skeleton.
    """
    z_b = flow.encode(b["points"])
    clouds = []
    for creature in (a, b):
        g = torch.Generator(device=creature["joints"].device).manual_seed(seed)
        _, pts = flow.sample(creature["joints"], creature["parents"], creature["presence"], creature["joint_mask"],
                             z=z_b, num_points=num_points, generator=g)
        clouds.append(pts[0])
    return (compute_chamfer_distance(clouds[0], clouds[1]) / compute_chamfer_distance(a["points"][0], b["points"][0])).item()
