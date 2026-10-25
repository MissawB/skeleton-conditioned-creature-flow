import torch
from models.creature_dit import CreatureDiT
from models.point_dit import PointDiT


def _model():
    torch.manual_seed(0)
    return CreatureDiT(hidden_dim=32, num_layers=2, num_heads=4, z_dim=8, data_scale=4.0).eval()


def _creature(J=5, pad_to=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    P = pad_to or J
    joints = torch.zeros(1, P, 3)
    joints[0, :J] = torch.randn(J, 3, generator=g)
    parents = torch.full((1, P), -2, dtype=torch.long)
    parents[0, :J] = torch.tensor([-1] + [i // 2 for i in range(J - 1)])
    mask = torch.zeros(1, P, dtype=torch.bool)
    mask[0, :J] = True
    return joints, parents, mask.float(), mask


def _x(seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 64, 3, generator=g), torch.tensor([0.4])


def test_padded_joints_do_not_change_the_velocity():
    m, (x, t) = _model(), _x()
    small = m(x, t, *_creature())
    joints, parents, presence, mask = _creature(pad_to=40)
    joints[0, 5:] = torch.randn(35, 3)  # garbage in the padding
    assert torch.allclose(small, m(x, t, joints, parents, presence, mask), atol=1e-5)


def test_zero_presence_is_finite_and_ignored():
    m, (x, t) = _model(), _x()
    joints, parents, presence, mask = _creature()
    presence[0, 3:] = 0.0  # joints 3 and 4 are leaves under joint 1
    v = m(x, t, joints, parents, presence, mask)
    moved = joints.clone()
    moved[0, 3] += 5.0
    assert torch.isfinite(v).all()
    assert torch.allclose(v, m(x, t, moved, parents, presence, mask), atol=1e-5)


def test_partial_presence_changes_the_velocity():
    m, (x, t) = _model(), _x()
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.1)
    joints, parents, presence, mask = _creature()
    half = presence.clone()
    half[0, 3] = 0.5
    assert not torch.allclose(m(x, t, joints, parents, presence, mask), m(x, t, joints, parents, half, mask))


def test_missing_z_equals_the_null_code():
    m, (x, t) = _model(), _x()
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.1)
    c = _creature()
    assert torch.allclose(m(x, t, *c), m(x, t, *c, z=torch.zeros(1, 8)))
    assert not torch.allclose(m(x, t, *c), m(x, t, *c, z=torch.ones(1, 8)))


def test_dropped_skeleton_ignores_joint_positions():
    m, (x, t) = _model(), _x()
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.1)
    joints, parents, presence, mask = _creature()
    drop = torch.tensor([False])
    a = m(x, t, joints, parents, presence, mask, cond_mask=drop)
    b = m(x, t, joints + 3.0, parents, presence, mask, cond_mask=drop)
    assert torch.allclose(a, b, atol=1e-5)


def test_output_shape_and_old_point_dit_unchanged():
    m, (x, t) = _model(), _x()
    assert m(x, t, *_creature()).shape == (1, 64, 3)
    torch.manual_seed(0)
    old = PointDiT(hidden_dim=32, num_layers=2, num_heads=4).eval()
    for p in old.parameters():
        torch.nn.init.normal_(p, std=0.1)
    joints = torch.randn(1, 35, 3)
    h = old.point_proj(x)
    block = old.blocks[0]
    t_emb = old.time_embedder(t)
    mem = old.skeleton_encoder(joints)
    assert torch.equal(block(h, t_emb, mem, x, joints), block(h, t_emb, mem, x, joints, key_bias=None))
