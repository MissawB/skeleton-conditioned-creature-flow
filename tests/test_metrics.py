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

def test_generation_metrics_mmd_cov_1nna():
    from pipeline.metrics import compute_generation_metrics
    torch.manual_seed(0)
    ref = torch.randn(12, 64, 3)

    same = compute_generation_metrics(ref.clone(), ref)
    assert same["mmd"] < 1e-6 and same["cov"] == 1.0
    assert same["1nna"] == 0.0  # exact copies: every sample's nearest neighbour is its twin in the other set

    far = compute_generation_metrics(torch.randn(12, 64, 3) + 10.0, ref)
    assert far["1nna"] == 1.0 and far["mmd"] > 100.0

    # Two independent draws of the same distribution are hard to tell apart
    fair = compute_generation_metrics(torch.randn(12, 64, 3), ref)
    assert 0.2 <= fair["1nna"] <= 0.8 and fair["cov"] > 0.3
