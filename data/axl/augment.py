"""
Skeleton-driven augmentation of skinned point clouds: bone length, girth, small pose changes and
subtrees that shrink or vanish (so the model learns parts that fade, which chimera blending needs).

A joint's skin weights cover the segment from that joint to its children (measured on Articulation-XL),
so a point weighted on joint j is stretched along that segment and turns with joint j's frame.
"""
from dataclasses import dataclass, field
from typing import Tuple
import numpy as np

@dataclass
class AugmentConfig:
    bone_scale: Tuple[float, float] = (0.8, 1.25)
    girth: Tuple[float, float] = (0.85, 1.2)
    pose_prob: float = 0.5
    pose_joint_frac: float = 0.3
    pose_max_deg: float = 20.0
    subtree_prob: float = 0.2
    subtree_max_frac: float = 1 / 3
    subtree_remove_prob: float = 0.3

@dataclass
class DeformParams:
    bone_scale: np.ndarray
    girth: np.ndarray
    rotations: np.ndarray
    presence: np.ndarray = field(default=None)

    @staticmethod
    def identity(num_joints: int) -> "DeformParams":
        return DeformParams(
            bone_scale=np.ones(num_joints, np.float32),
            girth=np.ones(num_joints, np.float32),
            rotations=np.tile(np.eye(3, dtype=np.float32), (num_joints, 1, 1)),
            presence=np.ones(num_joints, np.float32),
        )

def _children(parents: np.ndarray):
    children = [[] for _ in parents]
    for j, p in enumerate(parents):
        if p >= 0:
            children[p].append(j)
    return children

def _topological_order(parents: np.ndarray) -> list:
    children = _children(parents)
    order = [int(np.where(parents == -1)[0][0])]
    for j in order:
        order.extend(children[j])
    return order

def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K

def _subtree(children, root: int) -> list:
    nodes = [root]
    for j in nodes:
        nodes.extend(children[j])
    return nodes

def random_params(parents: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> DeformParams:
    num_joints = len(parents)
    params = DeformParams.identity(num_joints)
    params.bone_scale[:] = rng.uniform(*cfg.bone_scale, num_joints)
    params.girth[:] = rng.uniform(*cfg.girth, num_joints)

    if rng.random() < cfg.pose_prob:
        count = int(round(cfg.pose_joint_frac * num_joints))
        for j in rng.choice(num_joints, size=count, replace=False):
            angle = np.radians(rng.uniform(-cfg.pose_max_deg, cfg.pose_max_deg))
            params.rotations[j] = _rotation(rng.normal(size=3), angle)

    if rng.random() < cfg.subtree_prob:
        children = _children(parents)
        limit = cfg.subtree_max_frac * num_joints
        roots = [j for j in range(num_joints) if parents[j] >= 0 and len(_subtree(children, j)) <= limit]
        if roots:
            root = roots[rng.integers(len(roots))]
            level = 0.0 if rng.random() < cfg.subtree_remove_prob else rng.uniform(0.0, 1.0)
            params.presence[_subtree(children, root)] = level
    return params

def deform(points, skin_idx, skin_w, joints, parents, params: DeformParams):
    """Returns (points (N,3), joints (J,3), point_area_scale (N,)) after skinning by per-joint affine maps."""
    joints = np.asarray(joints, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    children = _children(parents)
    order = _topological_order(parents)
    num_joints = len(joints)

    # Segment owned by each joint: towards its children, or continuing its parent bone for a leaf
    axes = np.zeros((num_joints, 3))
    stretch = np.ones(num_joints)
    for j in range(num_joints):
        if children[j]:
            direction = joints[children[j]].mean(0) - joints[j]
            stretch[j] = params.bone_scale[children[j]].mean()
        elif parents[j] >= 0:
            direction = joints[j] - joints[parents[j]]
        else:
            direction = np.array([1.0, 0.0, 0.0])
        norm = np.linalg.norm(direction)
        axes[j] = direction / norm if norm > 1e-12 else np.array([1.0, 0.0, 0.0])

    new_joints = joints.copy()
    frames = np.zeros((num_joints, 3, 3))
    for j in order:
        p = parents[j]
        if p < 0:
            frames[j] = params.rotations[j]
        else:
            new_joints[j] = new_joints[p] + frames[p] @ (joints[j] - joints[p]) * params.bone_scale[j]
            frames[j] = frames[p] @ params.rotations[j]

    # Shrink fading subtrees towards the joint they hang from; points follow the unshrunk joints first
    posed_joints = new_joints.copy()
    anchor = np.full(num_joints, -1)
    for j in order:
        p = parents[j]
        if p >= 0 and params.presence[j] < 1:
            anchor[j] = anchor[p] if params.presence[p] < 1 else p
    for j in order:
        if anchor[j] >= 0:
            new_joints[j] = new_joints[anchor[j]] + params.presence[j] * (new_joints[j] - new_joints[anchor[j]])

    out = np.zeros_like(points)
    area = np.zeros(len(points))
    skin_idx = np.asarray(skin_idx, dtype=np.int64)
    for k in range(skin_idx.shape[1]):
        j = skin_idx[:, k]
        weight = np.asarray(skin_w[:, k], dtype=np.float64)
        local = points - joints[j]
        axial = (local * axes[j]).sum(1, keepdims=True) * axes[j]
        local = axial * stretch[j, None] + (local - axial) * params.girth[j, None]
        moved = posed_joints[j] + np.einsum("nab,nb->na", frames[j], local)
        fading = anchor[j] >= 0
        if fading.any():
            base = new_joints[anchor[j[fading]]]
            moved[fading] = base + params.presence[j[fading], None] * (moved[fading] - base)
        out += weight[:, None] * moved
        area += weight * stretch[j] * params.girth[j]
    return out.astype(np.float32), new_joints.astype(np.float32), area.astype(np.float32)

def keep_mask(skin_idx: np.ndarray, skin_w: np.ndarray, presence: np.ndarray) -> np.ndarray:
    """Points whose dominant joint is still present."""
    dominant = skin_idx[np.arange(len(skin_idx)), np.argmax(skin_w, axis=1)]
    return presence[dominant] > 0

def resample(points: np.ndarray, point_area_scale: np.ndarray, keep: np.ndarray, num_points: int, rng: np.random.Generator) -> np.ndarray:
    """Draws num_points with replacement among kept points, in proportion to the surface each one now stands for."""
    candidates = np.where(keep)[0]
    weights = np.asarray(point_area_scale, dtype=np.float64)[candidates]
    chosen = rng.choice(candidates, size=num_points, p=weights / weights.sum())
    return points[chosen]
