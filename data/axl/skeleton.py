"""Kinematic tree helpers for skeletons given as unordered bone pairs."""
from collections import deque
import numpy as np

def parents_from_bones(bones: np.ndarray, root_index: int, num_joints: int) -> np.ndarray:
    """Orients unordered (a, b) bone pairs away from the root; -1 marks the root. Raises ValueError on a non-tree."""
    if not 0 <= root_index < num_joints:
        raise ValueError("root out of range")
    bones = np.asarray(bones, dtype=np.int64).reshape(-1, 2)
    if len(bones) != num_joints - 1:
        raise ValueError("not a tree: wrong bone count")
    if bones.size and (bones.min() < 0 or bones.max() >= num_joints):
        raise ValueError("bone index out of range")

    neighbours = [[] for _ in range(num_joints)]
    for a, b in bones:
        neighbours[a].append(b)
        neighbours[b].append(a)

    parents = np.full(num_joints, -2, dtype=np.int64)
    parents[root_index] = -1
    queue = deque([root_index])
    while queue:
        j = queue.popleft()
        for k in neighbours[j]:
            if k == parents[j]:
                continue
            if parents[k] != -2:
                raise ValueError("not a tree: cycle")
            parents[k] = j
            queue.append(k)
    if (parents == -2).any():
        raise ValueError("not a tree: disconnected joints")
    return parents

def tree_depth(parents: np.ndarray) -> np.ndarray:
    """Number of bones between each joint and the root."""
    depth = np.full(len(parents), -1, dtype=np.int64)
    for j in range(len(parents)):
        chain = []
        k = j
        while k != -1 and depth[k] < 0:
            chain.append(k)
            k = parents[k]
        base = -1 if k == -1 else depth[k]
        for offset, c in enumerate(reversed(chain), start=1):
            depth[c] = base + offset
    return depth
