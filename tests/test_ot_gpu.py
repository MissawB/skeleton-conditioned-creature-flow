import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from data.smal_quadrupeds import SMALQuadrupedGenerator
from models.flow_matching import OptimalTransportFlowMatching
from models.ot_gpu import ot_match, sinkhorn_permutation


def test_sinkhorn_returns_a_permutation():
    torch.manual_seed(0)
    x1 = torch.randn(3, 257, 3)
    x1[1, :10] = x1[1, 0]  # duplicated points
    for blocks in (1, 4):
        perm = sinkhorn_permutation(torch.randn(3, 257, 3), x1, num_blocks=blocks)
        for b in range(3):
            assert sorted(perm[b].tolist()) == list(range(257))
    perm = sinkhorn_permutation(torch.randn(2, 256, 3), x1[:2, :256], num_blocks=4)
    assert all(sorted(perm[b].tolist()) == list(range(256)) for b in range(2))


def test_sinkhorn_cost_is_close_to_exact_matching():
    gen = SMALQuadrupedGenerator(seed=0)
    x1 = torch.stack([torch.from_numpy(gen.generate_instance(f, num_points=2048)["points"]) for f in ("felidae", "equidae", "bovidae", "canidae")]) * 4
    torch.manual_seed(0)
    x0 = torch.randn_like(x1)
    matched = ot_match(x0, x1, num_blocks=1, iters=50)
    cost = ((matched - x1) ** 2).sum(-1).mean().item()
    fast = ((ot_match(x0, x1) - x1) ** 2).sum(-1).mean().item()
    exact = np.mean([cdist(x0[b].numpy(), x1[b].numpy(), "sqeuclidean")[linear_sum_assignment(cdist(x0[b].numpy(), x1[b].numpy(), "sqeuclidean"))].mean() for b in range(4)])
    independent = ((x0 - x1) ** 2).sum(-1).mean().item()
    assert cost <= 1.03 * exact
    assert fast <= 1.12 * exact  # default: 4 blocks, 20 iterations, about 8x faster on the GPU
    assert cost <= 0.6 * independent


class ZeroModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("data_scale", torch.tensor(4.0))
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x_t, t, joints, cond_mask=None):
        return torch.zeros_like(x_t) + 0 * self.p


def test_flow_matching_online_ot_pairs_noise_with_the_shape():
    x1 = torch.randn(2, 64, 3) * 0.25
    joints = torch.zeros(2, 35, 3)
    fm = OptimalTransportFlowMatching(ZeroModel(), online_ot=True)
    torch.manual_seed(1)
    online = fm.compute_loss(x1, joints)
    torch.manual_seed(1)
    noise = torch.randn_like(x1)
    explicit = OptimalTransportFlowMatching(ZeroModel()).compute_loss(x1, joints, x_0=ot_match(noise, x1 * 4))
    assert torch.allclose(online, explicit)
