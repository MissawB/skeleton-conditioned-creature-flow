# tests/test_dataset.py
import torch
import numpy as np
from data.dataset import CreatureDataset

def test_creature_dataset_synthesis_and_augmentation():
    dataset = CreatureDataset(num_samples_per_family=5, num_points=1024, augment=True, cache_dir=None)
    assert len(dataset) == 25  # 5 families * 5 samples
    
    sample = dataset[0]
    assert "points" in sample
    assert "joints" in sample
    assert "family_id" in sample
    assert isinstance(sample["points"], torch.Tensor)
    assert isinstance(sample["joints"], torch.Tensor)
    
    assert sample["points"].shape == (1024, 3)
    assert sample["joints"].shape == (35, 3)
    assert sample["points"].dtype == torch.float32
    assert sample["joints"].dtype == torch.float32

def test_cache_is_invalidated_when_parameters_change(tmp_path):
    cache_dir = str(tmp_path)
    CreatureDataset(num_samples_per_family=2, num_points=256, augment=False, cache_dir=cache_dir)

    dataset = CreatureDataset(num_samples_per_family=3, num_points=128, augment=False, cache_dir=cache_dir)
    assert len(dataset) == 15
    assert dataset[0]["points"].shape == (128, 3)

def _ot_dataset(tmp_path, beta):
    return CreatureDataset(num_samples_per_family=1, num_points=2048, augment=False, cache_dir=str(tmp_path),
                           ot_noise_per_sample=1, ot_workers=1, ot_noise_beta=beta)

def test_ot_noise_beta_zero_returns_the_matched_draw(tmp_path):
    ds = _ot_dataset(tmp_path, 0.0)
    assert torch.equal(ds[0]["noise"], ds.ot_noise[0][0])

def test_ot_noise_beta_mixes_fresh_gaussian_noise(tmp_path):
    ds = _ot_dataset(tmp_path, 0.2)
    np.random.seed(0)
    matched = ds.ot_noise[0][0].flatten()
    noise = ds[0]["noise"].flatten()
    assert abs(torch.corrcoef(torch.stack([matched, noise]))[0, 1] - 0.8 ** 0.5) < 0.02
    assert abs(noise.std() - 1.0) < 0.03
