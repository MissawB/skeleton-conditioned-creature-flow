import numpy as np
from data.axl.augment import AugmentConfig, DeformParams, deform, keep_mask, random_params, resample

JOINTS = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], np.float32)
PARENTS = np.array([-1, 0, 1])


def _rigid(points, joint):
    idx = np.zeros((len(points), 4), np.uint8)
    idx[:, 0] = joint
    w = np.zeros((len(points), 4), np.float32)
    w[:, 0] = 1
    return np.asarray(points, np.float32), idx, w


def test_identity_params_leave_points_unchanged():
    rng = np.random.default_rng(0)
    points = rng.normal(size=(50, 3)).astype(np.float32)
    idx = rng.integers(0, 3, size=(50, 4)).astype(np.uint8)
    w = rng.random((50, 4)).astype(np.float32)
    w /= w.sum(1, keepdims=True)
    out, joints, area = deform(points, idx, w, JOINTS, PARENTS, DeformParams.identity(3))
    assert np.allclose(out, points, atol=1e-5) and np.allclose(joints, JOINTS) and np.allclose(area, 1)


def test_bone_scale_moves_the_child_joint_and_its_rigid_points():
    params = DeformParams.identity(3)
    params.bone_scale[1] = 1.5
    p1, i1, w1 = _rigid([[1.5, 0.1, 0]], 1)
    out1, joints, _ = deform(p1, i1, w1, JOINTS, PARENTS, params)
    assert np.allclose(joints, [[0, 0, 0], [1.5, 0, 0], [2.5, 0, 0]])
    assert np.allclose(out1, [[2.0, 0.1, 0]])
    p0, i0, w0 = _rigid([[0.5, 0.1, 0]], 0)  # joint 0 owns segment 0->1, stretched by 1.5
    out0, _, area0 = deform(p0, i0, w0, JOINTS, PARENTS, params)
    assert np.allclose(out0, [[0.75, 0.1, 0]]) and np.allclose(area0, 1.5)


def test_girth_scales_distance_to_the_bone_axis():
    params = DeformParams.identity(3)
    params.girth[1] = 1.5
    p, i, w = _rigid([[1.5, 0.2, 0]], 1)
    out, _, area = deform(p, i, w, JOINTS, PARENTS, params)
    assert np.allclose(out, [[1.5, 0.3, 0]]) and np.allclose(area, 1.5)


def test_rotation_preserves_distances_within_a_rigid_part():
    params = DeformParams.identity(3)
    params.rotations[1] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], np.float32)  # 90 deg about z
    p, i, w = _rigid([[1.5, 0.1, 0], [1.8, -0.1, 0.05]], 1)
    out, joints, _ = deform(p, i, w, JOINTS, PARENTS, params)
    assert np.allclose(joints[2], [1, 1, 0], atol=1e-5)
    assert np.allclose(out[0], [0.9, 0.5, 0], atol=1e-5)
    assert np.isclose(np.linalg.norm(out[0] - out[1]), np.linalg.norm(p[0] - p[1]), atol=1e-5)


def test_presence_zero_removes_the_subtree_points():
    params = DeformParams.identity(3)
    params.presence[2] = 0.0
    points = np.array([[0.5, 0.1, 0], [2.2, 0.1, 0]], np.float32)
    idx = np.array([[0, 0, 0, 0], [2, 0, 0, 0]], np.uint8)
    w = np.array([[1, 0, 0, 0], [1, 0, 0, 0]], np.float32)
    _, joints, _ = deform(points, idx, w, JOINTS, PARENTS, params)
    assert np.allclose(joints[2], joints[1])
    assert keep_mask(points, idx, w, JOINTS, PARENTS, params.presence).tolist() == [True, False]


def test_partial_presence_shrinks_the_subtree_towards_its_attachment():
    params = DeformParams.identity(3)
    params.presence[2] = 0.5
    p, i, w = _rigid([[2.2, 0.1, 0]], 2)
    out, joints, _ = deform(p, i, w, JOINTS, PARENTS, params)
    assert np.allclose(joints[2], [1.5, 0, 0]) and np.allclose(out, [[1.6, 0.05, 0]])


def test_resample_follows_area_scale():
    points = np.array([[0, 0, 0], [1, 0, 0]], np.float32)
    out = resample(points, np.array([1.0, 3.0]), np.array([True, True]), 20000, np.random.default_rng(0))
    assert abs((out[:, 0] == 1).mean() - 0.75) < 0.02
    out = resample(points, np.array([1.0, 3.0]), np.array([True, False]), 100, np.random.default_rng(0))
    assert (out[:, 0] == 0).all()


def test_random_params_respect_the_config():
    parents = np.array([-1, 0, 1, 2, 1, 4, 0, 6, 7])
    cfg = AugmentConfig(subtree_prob=1.0, pose_prob=1.0)
    for seed in range(20):
        p = random_params(parents, np.random.default_rng(seed), cfg)
        assert ((p.bone_scale >= 0.8) & (p.bone_scale <= 1.25)).all()
        assert ((p.girth >= 0.85) & (p.girth <= 1.2)).all()
        assert p.presence[0] == 1 and (p.presence <= 1).all() and (p.presence < 1).sum() <= len(parents) / 3
        assert np.allclose(np.einsum("jab,jcb->jac", p.rotations, p.rotations), np.eye(3), atol=1e-5)
        angles = np.degrees(np.arccos(np.clip((np.trace(p.rotations, axis1=1, axis2=2) - 1) / 2, -1, 1)))
        assert (angles <= 20.0 + 1e-3).all()


def test_presence_zero_on_a_leaf_removes_the_bone_that_leads_to_it():
    # points of bone 1->2 are weighted on joint 1 (outside the faded subtree {2})
    params = DeformParams.identity(3)
    params.presence[2] = 0.0
    points = np.array([[0.5, 0.1, 0], [1.5, 0.1, 0]], np.float32)
    idx = np.array([[0, 0, 0, 0], [1, 0, 0, 0]], np.uint8)
    w = np.array([[1, 0, 0, 0], [1, 0, 0, 0]], np.float32)
    assert keep_mask(points, idx, w, JOINTS, PARENTS, params.presence).tolist() == [True, False]


def test_partial_presence_shrinks_the_last_bone_and_its_surface():
    params = DeformParams.identity(3)
    params.presence[2] = 0.5
    p, i, w = _rigid([[1.5, 0.1, 0]], 1)
    out, joints, area = deform(p, i, w, JOINTS, PARENTS, params)
    assert np.allclose(joints[2], [1.5, 0, 0])
    assert np.allclose(out, [[1.25, 0.05, 0]]) and np.allclose(area, 0.25)


def test_fading_subtree_points_lose_surface_weight():
    params = DeformParams.identity(3)
    params.presence[2] = 0.5
    p, i, w = _rigid([[2.2, 0.1, 0]], 2)
    _, _, area = deform(p, i, w, JOINTS, PARENTS, params)
    assert np.allclose(area, 0.25)


def test_random_params_never_rotate_the_root():
    parents = np.array([-1, 0, 1, 2, 1, 4, 0, 6, 7])
    cfg = AugmentConfig(pose_prob=1.0, pose_joint_frac=1.0)
    for seed in range(10):
        assert np.allclose(random_params(parents, np.random.default_rng(seed), cfg).rotations[0], np.eye(3))
