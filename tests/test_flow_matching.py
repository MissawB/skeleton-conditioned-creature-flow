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

def test_data_scale_round_trip():
    """Training and the ODE run in coordinates multiplied by data_scale; samples come back in data units."""
    model = PointDiT(point_dim=3, hidden_dim=32, num_layers=1, num_heads=2, num_joints=35, data_scale=4.0)
    fm = OptimalTransportFlowMatching(model)
    seen = []
    original = model.forward
    def recording_forward(x_t, t, joints, cond_mask=None):
        seen.append(joints.clone())
        return torch.zeros_like(x_t)  # zero velocity: the sample is the initial noise, rescaled
    model.forward = recording_forward

    joints = torch.randn(1, 35, 3)
    torch.manual_seed(0)
    _, x = fm.sample(joints, num_points=4096, num_steps=2, solver="euler")
    torch.testing.assert_close(seen[0], joints * 4.0)
    assert abs(x.std().item() - 0.25) < 0.01

    seen.clear()
    fm.compute_loss(torch.randn(2, 64, 3), joints.expand(2, -1, -1))
    torch.testing.assert_close(seen[0], joints.expand(2, -1, -1) * 4.0)
    model.forward = original

def test_checkpoints_without_data_scale_load_with_unit_scale():
    state = {k: v for k, v in PointDiT(hidden_dim=32, num_layers=1, num_heads=2).state_dict().items() if k != "data_scale"}
    model = PointDiT(hidden_dim=32, num_layers=1, num_heads=2, data_scale=4.0)
    model.load_state_dict(state)
    assert model.data_scale.item() == 1.0
