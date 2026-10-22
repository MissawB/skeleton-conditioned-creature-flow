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

def test_surface_distance_is_near_zero_on_real_shapes_and_tracks_offsets():
    import numpy as np
    from data.smal_quadrupeds import SMALQuadrupedGenerator
    from pipeline.metrics import surface_distance
    gen = SMALQuadrupedGenerator(seed=0)
    inst = gen.generate_instance("equidae")
    radii = gen.bone_radii("equidae")
    assert len(radii) == 34
    points = torch.from_numpy(inst["points"]).float()
    joints = torch.from_numpy(inst["joints"]).float()
    gap = surface_distance(points, joints, radii)
    assert gap.shape == (len(points),)
    assert gap.abs().median() < 0.002
    # Far away points are outside every surface; a slightly shifted shape drifts off its surface
    assert (surface_distance(points + 10.0, joints, radii) > 9.0).all()
    shifted = surface_distance(points + torch.tensor([0.0, 0.0, 0.03]), joints, radii)
    assert shifted.abs().median() > 5 * gap.abs().median()

def test_pairwise_chamfer_distance_is_independent_of_chunking():
    from pipeline.metrics import pairwise_chamfer_distance
    a, b = torch.randn(5, 64, 3), torch.randn(7, 64, 3)
    torch.testing.assert_close(pairwise_chamfer_distance(a, b, chunk_size=2), pairwise_chamfer_distance(a, b, chunk_size=100))
