# tests/test_overfit.py
import torch
from data.smal_quadrupeds import SMALQuadrupedGenerator
from models.point_dit import PointDiT
from models.flow_matching import OptimalTransportFlowMatching
from pipeline.metrics import compute_chamfer_distance

def test_overfit_single_creature():
    """The model must be able to learn a single creature: loss drops and samples approach the target.

    The FM loss does not go to zero here: the model is permutation-equivariant, so it can only learn the
    marginal velocity field, whose variance is irreducible. Chamfer Distance is the meaningful criterion.
    """
    torch.manual_seed(0)
    instance = SMALQuadrupedGenerator(seed=0).generate_instance("felidae", num_points=256)
    x_1 = torch.from_numpy(instance["points"]).unsqueeze(0).repeat(8, 1, 1)
    joints = torch.from_numpy(instance["joints"]).unsqueeze(0).repeat(8, 1, 1)

    model = PointDiT(point_dim=3, hidden_dim=64, num_layers=2, num_heads=4, num_joints=35)
    fm = OptimalTransportFlowMatching(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)

    _, x_init = fm.sample(joints[:1], num_points=256, num_steps=10, solver="euler")
    cd_before = compute_chamfer_distance(x_init, x_1[:1]).item()

    model.train()
    losses = []
    for _ in range(150):
        optimizer.zero_grad()
        loss = fm.compute_loss(x_1, joints)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    first, last = sum(losses[:20]) / 20, sum(losses[-20:]) / 20
    assert last < 0.8 * first

    _, x_gen = fm.sample(joints[:1], num_points=256, num_steps=10, solver="euler")
    cd_after = compute_chamfer_distance(x_gen, x_1[:1]).item()
    assert cd_after < 0.05 * cd_before
