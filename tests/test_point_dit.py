# tests/test_point_dit.py
import torch
from models.point_dit import PointDiT

def test_point_dit_forward():
    batch_size = 2
    num_points = 512
    num_joints = 35
    hidden_dim = 128
    
    model = PointDiT(
        point_dim=3,
        hidden_dim=hidden_dim,
        num_layers=2,
        num_heads=4,
        num_joints=num_joints
    )
    
    x_t = torch.randn(batch_size, num_points, 3)
    t = torch.rand(batch_size)
    joints = torch.randn(batch_size, num_joints, 3)
    
    v_pred = model(x_t, t, joints)
    
    assert v_pred.shape == (batch_size, num_points, 3)
    assert not torch.isnan(v_pred).any()

def test_spatial_bias_scales_are_multiscale_learnable_and_config_driven():
    model = PointDiT(hidden_dim=64, num_layers=2, num_heads=4, spatial_distance_scale=3.0)
    scales = model.blocks[0].cross_attn.log_spatial_scale

    assert isinstance(scales, torch.nn.Parameter)
    torch.testing.assert_close(scales.exp(), torch.tensor([3.0, 6.0, 12.0, 24.0]))

    # out_proj is zero-initialised, so give the output a signal to back-propagate through
    torch.nn.init.normal_(model.out_proj.weight, std=0.1)
    loss = model(torch.randn(2, 64, 3), torch.rand(2), torch.randn(2, 35, 3)).pow(2).mean()
    loss.backward()
    assert scales.grad is not None and scales.grad.abs().sum() > 0

def test_shape_estimate_refinement_starts_at_identity():
    model = PointDiT(hidden_dim=64, num_layers=3, num_heads=4)
    # Zero-initialised refinement heads: every block initially biases attention with x_t itself
    for head in model.pos_heads:
        assert torch.count_nonzero(head.weight) == 0 and torch.count_nonzero(head.bias) == 0

def test_time_scale_multiplies_t_before_embedding_and_old_checkpoints_default_to_one():
    torch.manual_seed(0)
    scaled = PointDiT(point_dim=3, hidden_dim=32, num_layers=1, num_heads=2, num_joints=35, time_scale=1000.0)
    plain = PointDiT(point_dim=3, hidden_dim=32, num_layers=1, num_heads=2, num_joints=35)
    plain.load_state_dict({k: v for k, v in scaled.state_dict().items() if k != "time_scale"})
    assert plain.time_scale.item() == 1.0

    seen = []
    scaled.time_embedder.register_forward_hook(lambda m, inp, out: seen.append(inp[0]))
    with torch.no_grad():
        scaled(torch.randn(1, 16, 3), torch.tensor([0.95]), torch.randn(1, 35, 3))
        assert torch.allclose(seen[0], torch.tensor([950.0]))
        # Near t = 1 the unscaled embedding barely moves, the scaled one does
        drift = lambda m: (m.time_embedder(torch.tensor([0.9]) * m.time_scale) - m.time_embedder(torch.tensor([1.0]) * m.time_scale)).norm()
        assert drift(scaled) > 5 * drift(plain)

def test_local_attention_reads_only_the_k_nearest_neighbours():
    from models.point_dit import LocalPointAttention
    torch.manual_seed(0)
    layer = LocalPointAttention(hidden_dim=16, num_heads=2, k=4)
    torch.nn.init.normal_(layer.out_proj.weight, std=0.5)  # zero-initialised output would hide the dependence
    pos = torch.cat([torch.randn(1, 8, 3) * 0.1, torch.randn(1, 8, 3) * 0.1 + 5.0], dim=1)  # two far-apart clusters
    h = torch.randn(1, 16, 16)
    out = layer(h, pos)
    h2 = h.clone()
    h2[:, 8:] = torch.randn(1, 8, 16)  # change only the far cluster
    out2 = layer(h2, pos)
    assert torch.allclose(out[:, :8], out2[:, :8], atol=1e-6)
    assert not torch.allclose(out[:, 8:], out2[:, 8:])

def test_local_attention_starts_as_identity_and_checkpoints_keep_k():
    torch.manual_seed(0)
    x, t, j = torch.randn(2, 64, 3), torch.rand(2), torch.randn(2, 35, 3)
    base = PointDiT(point_dim=3, hidden_dim=32, num_layers=2, num_heads=2, num_joints=35)
    local = PointDiT(point_dim=3, hidden_dim=32, num_layers=2, num_heads=2, num_joints=35, local_k=8)
    assert local.state_dict()["local_k"].item() == 8
    assert base.state_dict()["local_k"].item() == 0
    # Shared weights, local layers left at their zero-initialised output
    local_layers = {n: w for n, w in local.state_dict().items() if ".local_attn." in n}
    local.load_state_dict({**base.state_dict(), **local_layers, "local_k": torch.tensor(8)})
    with torch.no_grad():
        assert torch.allclose(base(x, t, j), local(x, t, j), atol=1e-6)

def test_load_model_rebuilds_local_attention(tmp_path):
    import yaml
    from scripts.evaluate_families import load_model
    cfg = yaml.safe_load(open("configs/default_config.yaml"))
    model = PointDiT(point_dim=3, hidden_dim=32, num_layers=2, num_heads=2, num_joints=35, local_k=8)
    torch.save({"model_state": model.state_dict()}, tmp_path / "ckpt.pt")
    loaded = load_model(cfg, str(tmp_path / "ckpt.pt"), "cpu")
    assert loaded.local_k == 8
