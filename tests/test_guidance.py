# tests/test_guidance.py
import torch
from models.point_dit import PointDiT
from models.flow_matching import OptimalTransportFlowMatching

def _model():
    torch.manual_seed(0)
    model = PointDiT(hidden_dim=32, num_layers=2, num_heads=2)
    # out_proj is zero-initialised, so give the output a signal that depends on the input
    torch.nn.init.normal_(model.out_proj.weight, std=0.1)
    return model

def test_dropped_condition_ignores_the_skeleton():
    model = _model().eval()
    x_t, t = torch.randn(2, 64, 3), torch.rand(2)
    drop = torch.zeros(2, dtype=torch.bool)
    out_a = model(x_t, t, torch.randn(2, 35, 3), cond_mask=drop)
    out_b = model(x_t, t, torch.randn(2, 35, 3), cond_mask=drop)
    # Neither the joint tokens nor the joint positions (spatial bias) may leak through
    torch.testing.assert_close(out_a, out_b)

    keep = torch.ones(2, dtype=torch.bool)
    joints = torch.randn(2, 35, 3)
    torch.testing.assert_close(model(x_t, t, joints, cond_mask=keep), model(x_t, t, joints))

def test_guidance_scale_one_matches_plain_conditional_sampling():
    fm = OptimalTransportFlowMatching(_model())
    joints = torch.randn(1, 35, 3)
    torch.manual_seed(1)
    _, plain = fm.sample(joints, num_points=64, num_steps=3, solver="heun")
    torch.manual_seed(1)
    _, guided = fm.sample(joints, num_points=64, num_steps=3, solver="heun", guidance_scale=1.0)
    torch.testing.assert_close(plain, guided)

    torch.manual_seed(1)
    _, strong = fm.sample(joints, num_points=64, num_steps=3, solver="heun", guidance_scale=3.0)
    assert not torch.allclose(plain, strong)

def test_condition_dropout_in_training_loss():
    fm = OptimalTransportFlowMatching(_model(), cond_drop_prob=1.0)
    loss = fm.compute_loss(torch.randn(2, 64, 3), torch.randn(2, 35, 3))
    loss.backward()
    # With every condition dropped, the null tokens are what the cross-attention reads
    assert fm.model.null_joint_tokens.grad is not None and fm.model.null_joint_tokens.grad.abs().sum() > 0

def test_checkpoints_without_null_tokens_still_load():
    state = {k: v for k, v in _model().state_dict().items() if k != "null_joint_tokens"}
    PointDiT(hidden_dim=32, num_layers=2, num_heads=2).load_state_dict(state)
