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
