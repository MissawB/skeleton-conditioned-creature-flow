import torch
from models.shape_encoder import ShapeEncoder, kl_divergence, reparameterize


def test_encoding_is_invariant_to_point_order():
    torch.manual_seed(0)
    enc = ShapeEncoder(z_dim=8, hidden_dim=32, num_points=512).eval()
    pts = torch.randn(2, 300, 3)
    mu, logvar = enc(pts)
    mu2, logvar2 = enc(pts[:, torch.randperm(300)])
    assert mu.shape == (2, 8) and logvar.shape == (2, 8)
    assert torch.allclose(mu, mu2, atol=1e-5) and torch.allclose(logvar, logvar2, atol=1e-5)


def test_large_clouds_are_subsampled():
    enc = ShapeEncoder(z_dim=8, hidden_dim=32, num_points=64)
    mu, _ = enc(torch.randn(3, 2048, 3))
    assert mu.shape == (3, 8) and torch.isfinite(mu).all()


def test_kl_is_zero_for_the_prior():
    assert kl_divergence(torch.zeros(4, 8), torch.zeros(4, 8)).item() == 0.0
    assert kl_divergence(torch.ones(4, 8), torch.zeros(4, 8)).item() > 0


def test_reparameterize_has_the_right_moments():
    torch.manual_seed(0)
    z = reparameterize(torch.full((20000, 1), 2.0), torch.full((20000, 1), 2 * torch.log(torch.tensor(0.5))))
    assert abs(z.mean() - 2.0) < 0.02 and abs(z.std() - 0.5) < 0.02
