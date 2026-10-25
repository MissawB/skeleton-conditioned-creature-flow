import numpy as np
import pytest
from data.axl.skeleton import parents_from_bones, tree_depth


def test_parents_from_bones_orients_unordered_pairs_from_the_root():
    bones = np.array([[1, 0], [1, 2], [3, 2]])
    assert parents_from_bones(bones, root_index=0, num_joints=4).tolist() == [-1, 0, 1, 2]
    assert parents_from_bones(bones, root_index=2, num_joints=4).tolist() == [1, 2, -1, 2]


@pytest.mark.parametrize("bones,root,n", [
    (np.array([[0, 1], [1, 2], [2, 0]]), 0, 3),
    (np.array([[0, 1]]), 0, 3),
    (np.array([[0, 1]]), 5, 2),
])
def test_parents_from_bones_rejects_invalid_trees(bones, root, n):
    with pytest.raises(ValueError):
        parents_from_bones(bones, root, n)


def test_tree_depth():
    assert tree_depth(np.array([-1, 0, 1, 0])).tolist() == [0, 1, 2, 1]
