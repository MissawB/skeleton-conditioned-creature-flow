import torch
from typing import Dict, List

def compute_chamfer_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes symmetric squared Chamfer Distance between two point clouds.
    x: (B, N, 3) or (N, 3)
    y: (B, M, 3) or (M, 3)
    """
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if y.dim() == 2:
        y = y.unsqueeze(0)
        
    # Pairwise squared distances: (B, N, M)
    dist_sq = torch.cdist(x, y, p=2.0) ** 2
    
    min_x_to_y = torch.min(dist_sq, dim=2)[0].mean(dim=1)
    min_y_to_x = torch.min(dist_sq, dim=1)[0].mean(dim=1)
    
    return torch.mean(min_x_to_y + min_y_to_x)

def compute_skeleton_coverage(joints: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """
    Average Euclidean distance from each skeleton joint to its closest point in the point cloud.
    joints: (B, K, 3)
    points: (B, N, 3)
    """
    if joints.dim() == 2:
        joints = joints.unsqueeze(0)
    if points.dim() == 2:
        points = points.unsqueeze(0)
        
    dist = torch.cdist(joints, points, p=2.0)
    min_dist = torch.min(dist, dim=2)[0]
    return torch.mean(min_dist)

def compute_novelty_score(gen_points: torch.Tensor, dataset_points: List[torch.Tensor]) -> float:
    """
    Finds the minimum Chamfer Distance to any sample in the training set.
    """
    min_cd = float("inf")
    for sample in dataset_points:
        cd = compute_chamfer_distance(gen_points, sample).item()
        if cd < min_cd:
            min_cd = cd
    return min_cd

def pairwise_chamfer_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Chamfer Distance between every pair of clouds of two sets.
    a: (n, N, 3), b: (m, M, 3) -> (n, m)
    """
    rows = []
    for x in a:
        dist_sq = torch.cdist(x.unsqueeze(0).expand(len(b), -1, -1), b) ** 2
        rows.append(dist_sq.min(dim=2)[0].mean(dim=1) + dist_sq.min(dim=1)[0].mean(dim=1))
    return torch.stack(rows)

def compute_generation_metrics(gen: torch.Tensor, ref: torch.Tensor) -> Dict[str, float]:
    """
    Standard point cloud generation metrics with Chamfer Distance (PointFlow, Yang et al., 2019).
    gen: (n, N, 3) generated clouds, ref: (m, M, 3) reference clouds.
    - mmd: mean distance from each reference to its closest generated cloud (fidelity, lower is better)
    - cov: share of references that are the nearest neighbour of some generated cloud (diversity, higher is better)
    - 1nna: leave-one-out 1-NN accuracy of telling generated from reference clouds (50% is ideal)
    """
    d_gr = pairwise_chamfer_distance(gen, ref)
    d_gg = pairwise_chamfer_distance(gen, gen)
    d_rr = pairwise_chamfer_distance(ref, ref)

    mmd = d_gr.min(dim=0)[0].mean().item()
    cov = d_gr.argmin(dim=1).unique().numel() / ref.shape[0]

    # Leave-one-out: a cloud is classified correctly when its nearest neighbour comes from its own set.
    # Ties (e.g. exact copies) go to the other set, so memorised samples count as indistinguishable.
    d_gg, d_rr = d_gg.fill_diagonal_(float("inf")), d_rr.fill_diagonal_(float("inf"))
    gen_correct = d_gg.min(dim=1)[0] < d_gr.min(dim=1)[0]
    ref_correct = d_rr.min(dim=1)[0] < d_gr.min(dim=0)[0]
    one_nna = torch.cat([gen_correct, ref_correct]).float().mean().item()
    return {"mmd": mmd, "cov": cov, "1nna": one_nna}
