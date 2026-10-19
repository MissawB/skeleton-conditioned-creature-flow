import torch
from typing import List

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
