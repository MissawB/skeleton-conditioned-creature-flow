# tests/test_ot_coupling.py
import numpy as np
import torch
from data.ot_coupling import ot_match_noise
from data.dataset import CreatureDataset
from models.point_dit import PointDiT
from models.flow_matching import OptimalTransportFlowMatching

def _pair_cost(x0, x1):
    return float(np.mean(np.sum((np.asarray(x0) - np.asarray(x1)) ** 2, axis=1)))

def test_ot_match_noise_is_a_cheaper_permutation_of_gaussian_noise():
    rng = np.random.default_rng(0)
    points = rng.uniform(-0.5, 0.5, size=(256, 3)).astype(np.float32)
    noise = ot_match_noise(points, np.random.default_rng(1))

    assert noise.shape == points.shape and noise.dtype == np.float32
    # Same multiset of noise rows as a fresh draw with the same seed, only reordered
    raw = np.random.default_rng(1).standard_normal(points.shape).astype(np.float32)
    np.testing.assert_allclose(np.sort(noise, axis=0), np.sort(raw, axis=0))
    assert _pair_cost(noise, points) < 0.8 * _pair_cost(raw, points)

def test_dataset_serves_ot_noise_rotated_with_the_shape():
    dataset = CreatureDataset(num_samples_per_family=1, num_points=128, augment=True, cache_dir=None,
                              ot_noise_per_sample=2, independent_noise_prob=0.0)
    sample = dataset[0]
    assert sample["noise"].shape == (128, 3)

    # The augmentation must rotate noise and points together, so the pairing stays cheap
    shuffled = sample["noise"][torch.randperm(128)]
    assert _pair_cost(sample["noise"], sample["points"]) < 0.8 * _pair_cost(shuffled, sample["points"])

def test_dataset_without_ot_serves_no_noise():
    dataset = CreatureDataset(num_samples_per_family=1, num_points=64, augment=False, cache_dir=None)
    assert "noise" not in dataset[0]

def test_compute_loss_uses_provided_noise():
    model = PointDiT(hidden_dim=32, num_layers=1, num_heads=2)
    fm = OptimalTransportFlowMatching(model)
    x_1, joints = torch.randn(2, 64, 3), torch.randn(2, 35, 3)
    # out_proj is zero-initialised (prediction 0); with x_0 = x_1 the target x_1 - x_0 is 0 too
    loss = fm.compute_loss(x_1, joints, x_0=x_1.clone())
    assert torch.isclose(loss, torch.tensor(0.0))
