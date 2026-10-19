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
