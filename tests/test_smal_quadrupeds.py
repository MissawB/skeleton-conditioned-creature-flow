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

def test_limbs_point_downward_to_level_ground():
    generator = SMALQuadrupedGenerator(seed=0)
    for family in FAMILIES:
        for _ in range(5):
            j = generator.generate_instance(family=family, num_points=256)["joints"]
            for upper, lower in [(10, 11), (17, 18), (23, 24), (29, 30)]:  # wrist/ankle -> paw
                assert j[lower, 1] < j[upper, 1], f"{family}: paw {lower} is above joint {upper}"
            for paw, toes in [(11, (12, 13)), (18, (19, 20)), (24, (25, 26)), (30, (31, 32))]:
                assert all(j[t, 1] <= j[paw, 1] for t in toes), f"{family}: toes above paw {paw}"
            toe_heights = j[[12, 13, 19, 20, 25, 26, 31, 32], 1]
            assert np.ptp(toe_heights) < 0.03, f"{family}: front and back feet are not on the same ground"

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

def _neck_angle_deg(j):
    d = j[5] - j[4]  # neck_base -> head
    return np.degrees(np.arctan2(d[1], d[0]))

def _stance_ratio(j):
    # shoulder width relative to pelvis -> neck_base length (scale-invariant)
    return np.linalg.norm(j[8] - j[15]) / np.linalg.norm(j[4] - j[0])

def test_bovidae_skeleton_differs_from_equidae():
    """Body thickness and head carriage must be visible in the skeleton, the model's only input."""
    generator = SMALQuadrupedGenerator(seed=0)
    stats = {f: [] for f in ["equidae", "bovidae"]}
    for family in stats:
        for _ in range(10):
            j = generator.generate_instance(family=family, num_points=256)["joints"]
            stats[family].append((_neck_angle_deg(j), _stance_ratio(j)))
    eq, bov = np.array(stats["equidae"]), np.array(stats["bovidae"])
    assert bov[:, 0].max() < eq[:, 0].min() - 15, "bovidae should carry the head much lower than equidae"
    assert bov[:, 1].min() > eq[:, 1].max(), "bovidae should stand wider (thicker body) than equidae"

def test_stance_widens_with_body_thickness():
    generator = SMALQuadrupedGenerator(seed=0)
    ratio = {f: np.mean([_stance_ratio(generator.generate_instance(family=f, num_points=256)["joints"]) for _ in range(10)])
             for f in ["equidae", "bovidae", "hippopotamidae"]}
    assert ratio["equidae"] < ratio["bovidae"] < ratio["hippopotamidae"]
