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
