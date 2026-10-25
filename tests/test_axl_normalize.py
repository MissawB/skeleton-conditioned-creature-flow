import numpy as np
from data.axl.normalize import canonical_rotation, normalize
from data.smal_quadrupeds import SMALQuadrupedGenerator, JOINT_NAMES, PARENT_INDICES

PARENTS = np.array(PARENT_INDICES)


def rotation_about_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def rotation_about_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)


def angle_between(u, v):
    u, v = np.asarray(u, float), np.asarray(v, float)
    return np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))


def _creature():
    return SMALQuadrupedGenerator(seed=0).generate_instance("felidae")


def test_canonical_rotation_recovers_yaw_of_a_named_quadruped():
    inst = _creature()
    head = JOINT_NAMES.index("head")
    for yaw in (0.0, 1.0, 2.5, -2.0):
        Ry = rotation_about_y(yaw)
        R, how = canonical_rotation(inst["points"] @ Ry, inst["joints"] @ Ry, PARENTS, JOINT_NAMES)
        h = inst["joints"] @ Ry @ R - (inst["joints"] @ Ry @ R)[0]
        assert how == "head"
        assert angle_between(h[head] * [1, 0, 1], [1, 0, 0]) < np.radians(10)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-5) and np.linalg.det(R) > 0


def test_canonical_rotation_fixes_a_z_up_file():
    inst = _creature()
    Rx = rotation_about_x(np.pi / 2)  # file up becomes +Z
    R, _ = canonical_rotation(inst["points"] @ Rx, inst["joints"] @ Rx, PARENTS, JOINT_NAMES)
    j = inst["joints"] @ Rx @ R
    paws = [JOINT_NAMES.index(n) for n in ("l_front_paw", "r_front_paw", "l_back_paw", "r_back_paw")]
    assert j[JOINT_NAMES.index("head"), 1] > j[paws, 1].max() + 0.3
    assert angle_between(j[JOINT_NAMES.index("head")] * [1, 0, 1] - j[0] * [1, 0, 1], [1, 0, 0]) < np.radians(10)


def test_canonical_rotation_without_names_uses_geometry():
    inst = _creature()
    Ry = rotation_about_y(2.0)
    names = [f"j{i}" for i in range(len(JOINT_NAMES))]
    R, how = canonical_rotation(inst["points"] @ Ry, inst["joints"] @ Ry, PARENTS, names)
    j = inst["joints"] @ Ry @ R
    assert how == "geometry"
    assert angle_between(j[JOINT_NAMES.index("head")] * [1, 0, 1] - j[0] * [1, 0, 1], [1, 0, 0]) < np.radians(30)


def test_canonical_rotation_uses_left_right_names_when_the_head_is_above_the_root():
    # upright figure facing +X: anatomical left = up x forward = -Z
    joints = np.array([[0, 0, 0], [0, 0.5, 0], [0, 0.9, 0], [0, 0.6, -0.4], [0, 0.6, 0.4],
                       [0, -0.8, -0.15], [0, -0.8, 0.15], [0.12, -0.9, -0.15], [0.12, -0.9, 0.15]], np.float32)
    parents = np.array([-1, 0, 1, 1, 1, 0, 0, 5, 6])
    names = ["hips", "spine", "head", "LeftHand", "RightHand", "LeftFoot", "RightFoot", "LeftToe", "RightToe"]
    points = np.concatenate([joints + np.random.default_rng(i).normal(0, 0.02, joints.shape) for i in range(20)])
    Ry = rotation_about_y(-1.3)
    R, how = canonical_rotation(points @ Ry, joints @ Ry, parents, names)
    j = joints @ Ry @ R
    assert how == "lateral"
    assert angle_between(j[3] - j[4], [0, 0, -1]) < np.radians(10)


def test_normalize_centres_and_scales_to_unit_radius():
    inst = _creature()
    points, joints, scale = normalize(inst["points"] * 3 + 5, inst["joints"] * 3 + 5, np.eye(3, dtype=np.float32))
    assert abs(np.linalg.norm(points, axis=1).max() - 1) < 1e-5
    assert np.allclose((points.min(0) + points.max(0)) / 2, 0, atol=1e-5)
    assert np.allclose(joints, (inst["joints"] * 3 + 5 - (inst["points"] * 3 + 5).min(0) / 2 - (inst["points"] * 3 + 5).max(0) / 2) * scale, atol=1e-4)


def test_left_right_names_in_prefixed_camel_case_are_recognised():
    joints = np.array([[0, 0, 0], [0, 0.5, 0], [0, 0.9, 0], [0, 0.6, -0.4], [0, 0.6, 0.4],
                       [0, -0.8, -0.15], [0, -0.8, 0.15], [0.12, -0.9, -0.15], [0.12, -0.9, 0.15]], np.float32)
    parents = np.array([-1, 0, 1, 1, 1, 0, 0, 5, 6])
    points = np.concatenate([joints + np.random.default_rng(i).normal(0, 0.02, joints.shape) for i in range(20)])
    Ry = rotation_about_y(2.2)
    for names in (
        ["mixamorig_Hips", "mixamorig_Spine", "mixamorig_Head", "mixamorig_LeftHand", "mixamorig_RightHand",
         "mixamorig_LeftFoot", "mixamorig_RightFoot", "mixamorig_LeftToeBase", "mixamorig_RightToeBase"],
        ["LowManHips", "LowManSpine", "LowManHead", "LowManLeftHand", "LowManRightHand",
         "LowManLeftFoot", "LowManRightFoot", "LowManLeftToeBase", "LowManRightToeBase"],
        ["Bip01 Pelvis", "Bip01 Spine", "Bip01 Head", "Bip01 L Hand", "Bip01 R Hand",
         "Bip01 L Foot", "Bip01 R Foot", "Bip01 L Toe0", "Bip01 R Toe0"],
    ):
        R, how = canonical_rotation(points @ Ry, joints @ Ry, parents, names)
        j = joints @ Ry @ R
        assert how == "lateral", names[0]
        assert angle_between(j[3] - j[4], [0, 0, -1]) < np.radians(10), names[0]
