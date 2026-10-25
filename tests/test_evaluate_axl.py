import numpy as np
import torch
from pipeline.creature_eval import head_joint, skeleton_sensitivity, surface_gap


def test_head_joint_prefers_named_head_then_highest_leaf():
    joints = np.array([[0, 0, 0], [0, 1, 0], [1, 0.5, 0], [0, 2, 0]], float)
    parents = np.array([-1, 0, 0, 1])
    assert head_joint(["hips", "spine", "Head", "hair_tip"], joints, parents) == 2
    assert head_joint(["a", "b", "c", "d"], joints, parents) == 3


def test_surface_gap_is_zero_on_identical_clouds_and_grows_with_noise():
    ref = torch.rand(500, 3)
    assert surface_gap(ref, ref) == 0
    assert 0 < surface_gap(ref + 0.01 * torch.randn(500, 3), ref) < surface_gap(ref + 0.05 * torch.randn(500, 3), ref)


class ToyFlow:
    """Draws a cloud from the skeleton (use_skeleton) or from the code z."""

    def __init__(self, use_skeleton):
        self.use_skeleton = use_skeleton

    def encode(self, points):
        return points.mean(1)

    def sample(self, joints, parents, presence, joint_mask, z=None, num_points=64, generator=None):
        base = joints[:, :1] if self.use_skeleton else z[:, None, :]
        return [], base + 0.1 * torch.randn(joints.shape[0], num_points, 3, generator=generator)


def _creature(offset):
    return {"points": offset + 0.1 * torch.randn(1, 64, 3), "joints": torch.full((1, 2, 3), offset),
            "parents": torch.tensor([[-1, 0]]), "presence": torch.ones(1, 2), "joint_mask": torch.ones(1, 2, dtype=torch.bool)}


def test_skeleton_sensitivity_on_a_toy_flow():
    torch.manual_seed(0)
    a, b = _creature(0.0), _creature(1.0)
    assert abs(skeleton_sensitivity(ToyFlow(True), a, b, num_points=64) - 1) < 0.3
    assert skeleton_sensitivity(ToyFlow(False), a, b, num_points=64) < 0.1
