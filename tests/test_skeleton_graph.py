import torch
from models.skeleton_graph import (MAX_TREE_DIST, REL_CHILD, REL_OTHER, REL_PARENT, REL_SIBLING, SkeletonGraphEncoder,
                                   tree_relations)


def _batch(parents_list):
    J = max(len(p) for p in parents_list)
    parents = torch.full((len(parents_list), J), -2, dtype=torch.long)
    mask = torch.zeros(len(parents_list), J, dtype=torch.bool)
    for b, p in enumerate(parents_list):
        parents[b, :len(p)] = torch.tensor(p)
        mask[b, :len(p)] = True
    return parents, mask


def test_tree_relations_on_a_small_tree():
    parents, mask = _batch([[-1, 0, 0, 1]])
    dist, rel = tree_relations(parents, mask)
    d, r = dist[0], rel[0]
    assert d[3, 2] == 3 and d[1, 2] == 2 and all(d[i, i] == 0 for i in range(4))
    assert r[1, 0] == REL_PARENT and r[0, 1] == REL_CHILD and r[1, 2] == REL_SIBLING and r[3, 2] == REL_OTHER


def test_tree_distance_is_capped_and_padding_is_marked():
    parents, mask = _batch([list(range(-1, 11)), [-1, 0]])
    dist, _ = tree_relations(parents, mask)
    assert dist[0, 0, 11] == MAX_TREE_DIST
    assert (dist[1, :, 2:] == MAX_TREE_DIST + 1).all() and (dist[1, 2:, :] == MAX_TREE_DIST + 1).all()


def _inputs(parents_list, seed=0):
    parents, mask = _batch(parents_list)
    g = torch.Generator().manual_seed(seed)
    joints = torch.randn(*parents.shape, 3, generator=g) * mask[..., None]
    presence = mask.float()
    return joints, parents, presence, mask


def test_encoder_is_equivariant_to_joint_relabelling():
    torch.manual_seed(0)
    enc = SkeletonGraphEncoder(hidden_dim=32, num_layers=2, num_heads=4).eval()
    joints, parents, presence, mask = _inputs([[-1, 0, 1, 1, 0]])
    order = torch.tensor([3, 0, 4, 1, 2])            # new joint i is old joint order[i]
    inverse = torch.argsort(order)
    new_parents = torch.where(parents[0, order] >= 0, inverse[parents[0, order].clamp_min(0)], parents[0, order])[None]
    tokens, glob = enc(joints, parents, presence, mask)
    tokens2, glob2 = enc(joints[:, order], new_parents, presence[:, order], mask[:, order])
    assert torch.allclose(tokens[:, order], tokens2, atol=1e-5) and torch.allclose(glob, glob2, atol=1e-5)


def test_padding_does_not_change_real_joint_outputs():
    torch.manual_seed(0)
    enc = SkeletonGraphEncoder(hidden_dim=32, num_layers=2, num_heads=4).eval()
    joints, parents, presence, mask = _inputs([[-1, 0]])
    tokens, glob = enc(joints, parents, presence, mask)
    big_parents, big_mask = _batch([[-1, 0], list(range(-1, 254))])
    big_joints = torch.zeros(2, 255, 3)
    big_joints[0, :2] = joints[0]
    big_joints[1] = torch.randn(255, 3)
    t2, g2 = enc(big_joints, big_parents, big_mask.float(), big_mask)
    assert torch.allclose(t2[0, :2], tokens[0], atol=1e-5) and torch.allclose(g2[0], glob[0], atol=1e-5)
