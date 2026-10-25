"""
Canonical pose of a rigged model: up = +Y (vertical of the original pose), forward = +X (gaze direction),
centred on the bounding box and scaled to unit radius.
"""
import re
from typing import List, Optional, Tuple
import numpy as np
from data.axl.skeleton import tree_depth

HEAD_WORDS = ("head", "skull", "face", "snout", "beak", "jaw")
FOOT_WORDS = ("foot", "toe", "paw", "hoof", "ankle", "heel")
UP_PRIOR = 0.05          # files are usually Y-up (glTF); another axis must win by more than this
MIN_HEAD_REACH = 0.15    # horizontal root->head offset, as a share of the extent, needed to trust it as forward

def _clean(name: str) -> str:
    """Lower case without a namespace prefix such as 'mixamorig:' or 'Armature|'."""
    return re.split(r"[:|]", name)[-1].lower()

def _find_head(names: List[str], depth: np.ndarray) -> Optional[int]:
    cleaned = [_clean(n) for n in names]
    for word in HEAD_WORDS:
        hits = [i for i, n in enumerate(cleaned) if word in n and "forehead" not in n]
        if hits:
            return min(hits, key=lambda i: depth[i])
    return None

def _side(name: str) -> int:
    """+1 for a left joint, -1 for a right joint, 0 otherwise."""
    n = _clean(name)
    tokens = [t for t in re.split(r"[^a-z]+", n) if t]
    if n.startswith("left") or "left" in tokens or (tokens and (tokens[0] == "l" or tokens[-1] == "l")):
        return 1
    if n.startswith("right") or "right" in tokens or (tokens and (tokens[0] == "r" or tokens[-1] == "r")):
        return -1
    return 0

def _up_score(up: np.ndarray, joints: np.ndarray, parents: np.ndarray, feet: List[int], head: Optional[int], extent: float) -> float:
    height = joints @ up
    root = int(np.where(parents == -1)[0][0])
    if feet:
        score = (height.mean() - height[feet].mean()) / extent
    else:
        leaves = ~np.isin(np.arange(len(joints)), parents)
        k = max(2, len(joints) // 10)
        score = leaves[np.argsort(height)[:k]].mean() - 0.5
    if head is not None:
        score += 0.5 * (height[head] - height[root]) / extent
    return score

def _horizontal(v: np.ndarray, up: np.ndarray) -> np.ndarray:
    return v - (v @ up)[..., None] * up

def canonical_rotation(points: np.ndarray, joints: np.ndarray, parents: np.ndarray, names: List[str]) -> Tuple[np.ndarray, str]:
    """Rotation R (applied as x @ R) bringing the model to up = +Y, forward = +X, and the cue that set the forward."""
    points = np.asarray(points, dtype=np.float64)
    joints = np.asarray(joints, dtype=np.float64)
    parents = np.asarray(parents)
    extent = max(np.ptp(points, axis=0).max(), 1e-9)
    depth = tree_depth(parents)
    head = _find_head(names, depth)
    feet = [i for i, n in enumerate(names) if any(w in _clean(n) for w in FOOT_WORDS)]
    root = int(np.where(parents == -1)[0][0])

    candidates = [s * np.eye(3)[a] for a in range(3) for s in (1.0, -1.0)]
    scores = [_up_score(u, joints, parents, feet, head, extent) + (UP_PRIOR if u[1] == 1 else 0.0) for u in candidates]
    up = candidates[int(np.argmax(scores))]

    forward, how = None, "geometry"
    sides = np.array([_side(n) for n in names])
    if (sides == 1).any() and (sides == -1).any():
        left = _horizontal(joints[sides == 1].mean(0) - joints[sides == -1].mean(0), up)
        if np.linalg.norm(left) > 1e-6:
            forward, how = np.cross(left, up), "lateral"
    if head is not None:
        reach = _horizontal(joints[head] - joints[root], up)
        if np.linalg.norm(reach) > MIN_HEAD_REACH * extent and (forward is None or reach @ forward < 0):
            forward, how = reach, "head"
    if forward is None:
        flat = _horizontal(points - points.mean(0), up)
        axis = np.linalg.svd(flat, full_matrices=False)[2][0]
        axis = _horizontal(axis, up)
        leaves = np.where(~np.isin(np.arange(len(joints)), parents))[0]
        highest_leaf = leaves[np.argmax(joints[leaves] @ up)] if len(leaves) else root
        if (joints[highest_leaf] - points.mean(0)) @ axis < 0:
            axis = -axis
        forward = axis

    forward = forward / np.linalg.norm(forward)
    R = np.stack([forward, up, np.cross(forward, up)], axis=1)
    return R.astype(np.float32), how

def normalize(points: np.ndarray, joints: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Applies R, centres on the points' bounding box and scales the farthest point to distance 1."""
    points = np.asarray(points, dtype=np.float64) @ R
    joints = np.asarray(joints, dtype=np.float64) @ R
    centre = (points.min(0) + points.max(0)) / 2
    scale = 1.0 / max(np.linalg.norm(points - centre, axis=1).max(), 1e-12)
    return ((points - centre) * scale).astype(np.float32), ((joints - centre) * scale).astype(np.float32), float(scale)
