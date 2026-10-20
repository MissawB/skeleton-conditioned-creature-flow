# tests/test_smal_quadrupeds.py
import numpy as np
from data.smal_quadrupeds import SMALQuadrupedGenerator, JOINT_NAMES, PARENT_INDICES, FAMILIES

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

def test_surface_density_is_uniform_between_torso_and_legs():
    from scipy.spatial import cKDTree
    generator = SMALQuadrupedGenerator(seed=0)
    for family in FAMILIES:
        sample = generator.generate_instance(family=family, num_points=4096)
        points, joints = sample["points"], sample["joints"]

        # Mean distance to the 8th nearest neighbour is a proxy for local sampling sparsity
        knn_dist = cKDTree(points).query(points, k=9)[0][:, -1]
        upper = points[:, 1] > joints[8, 1]  # above the shoulder: torso, neck, head
        lower = points[:, 1] < joints[9, 1]  # below the elbow: lower legs

        ratio = knn_dist[upper].mean() / knn_dist[lower].mean()
        assert ratio < 1.35, f"{family}: upper body is {ratio:.2f}x sparser than the legs"
