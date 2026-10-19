# tests/test_smal_quadrupeds.py
import numpy as np
from data.smal_quadrupeds import SMALQuadrupedGenerator, JOINT_NAMES, PARENT_INDICES

def test_smal_quadruped_generator():
    generator = SMALQuadrupedGenerator(seed=42)
    assert len(JOINT_NAMES) == 35
    assert len(PARENT_INDICES) == 35
    
    # Generate a feline creature
    sample = generator.generate_instance(family="felidae", num_points=2048)
    
    assert "points" in sample
    assert "joints" in sample
    assert "joint_parents" in sample
    assert "family_id" in sample
    
    assert sample["points"].shape == (2048, 3)
    assert sample["joints"].shape == (35, 3)
    assert sample["joint_parents"].shape == (35,)
    
    # Check bounding box normalization within [-1.1, 1.1]
    assert np.all(sample["points"] >= -1.1) and np.all(sample["points"] <= 1.1)
    assert np.all(sample["joints"] >= -1.1) and np.all(sample["joints"] <= 1.1)
    
    # Check parent indices: root must be -1, others must point to valid parent indices < child
    assert PARENT_INDICES[0] == -1
    for i in range(1, 35):
        assert 0 <= PARENT_INDICES[i] < i
