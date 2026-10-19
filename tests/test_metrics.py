# tests/test_metrics.py
import torch
from pipeline.metrics import compute_chamfer_distance, compute_skeleton_coverage, compute_novelty_score

def test_metrics():
    x = torch.randn(2, 500, 3)
    y = x.clone() # Identical
    
    # Chamfer distance between identical point clouds must be 0
    cd = compute_chamfer_distance(x, y)
    assert torch.isclose(cd, torch.tensor(0.0), atol=1e-5)
    
    # Skeleton coverage
    joints = torch.randn(2, 35, 3)
    cov = compute_skeleton_coverage(joints, x)
    assert cov.item() >= 0.0
    
    # Novelty score
    dataset = [torch.randn(500, 3) for _ in range(3)]
    novelty = compute_novelty_score(x[0], dataset)
    assert novelty > 0.0
