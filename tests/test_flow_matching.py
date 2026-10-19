# tests/test_flow_matching.py
import torch
from models.point_dit import PointDiT
from models.flow_matching import OptimalTransportFlowMatching

def test_flow_matching_loss_and_sample():
    model = PointDiT(point_dim=3, hidden_dim=64, num_layers=2, num_heads=2, num_joints=35)
    fm = OptimalTransportFlowMatching(model)
    
    batch_size = 2
    num_points = 256
    x_1 = torch.randn(batch_size, num_points, 3)
    joints = torch.randn(batch_size, 35, 3)
    
    loss = fm.compute_loss(x_1, joints)
    assert loss.dim() == 0
    assert loss.item() > 0.0
    
    # Test Euler solver sampling
    traj, x_final = fm.sample(joints[:1], num_points=num_points, num_steps=5, solver="euler", record_trajectory=True)
    assert x_final.shape == (1, num_points, 3)
    assert len(traj) == 6 # t=0 plus 5 steps
    
    # Test Heun solver sampling
    _, x_heun = fm.sample(joints[:1], num_points=num_points, num_steps=5, solver="heun", record_trajectory=False)
    assert x_heun.shape == (1, num_points, 3)
