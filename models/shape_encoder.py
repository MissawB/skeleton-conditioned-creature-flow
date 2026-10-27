"""Encodes a creature's point cloud into a small Gaussian shape code z (the traits the skeleton alone does not fix)."""
import torch
import torch.nn as nn

class ShapeEncoder(nn.Module):
    def __init__(self, z_dim: int = 64, hidden_dim: int = 128, num_points: int = 512):
        super().__init__()
        self.num_points = num_points
        self.point_mlp = nn.Sequential(nn.Linear(3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2 * hidden_dim))
        self.head = nn.Sequential(nn.Linear(4 * hidden_dim, 2 * hidden_dim), nn.GELU(), nn.Linear(2 * hidden_dim, 2 * z_dim))

    def forward(self, points: torch.Tensor):
        """(mu, logvar), each (B, z_dim); clouds larger than num_points are subsampled, at random in training
        and with a fixed stride in eval so that a cloud always gets the same code."""
        B, N, _ = points.shape
        if N > self.num_points:
            if self.training:
                idx = torch.rand(B, N, device=points.device).argsort(dim=1)[:, :self.num_points]
            else:
                idx = torch.linspace(0, N - 1, self.num_points, device=points.device).long().expand(B, -1)
            points = points.gather(1, idx[..., None].expand(-1, -1, 3))
        feat = self.point_mlp(points)
        pooled = torch.cat([feat.max(dim=1).values, feat.mean(dim=1)], dim=-1)
        mu, logvar = self.head(pooled).chunk(2, dim=-1)
        return mu, logvar.clamp(-10.0, 10.0)

def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return mu + torch.randn_like(mu) * (0.5 * logvar).exp()

def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(N(mu, sigma) || N(0, I)), summed over dimensions, averaged over the batch."""
    return (0.5 * (mu ** 2 + logvar.exp() - 1.0 - logvar)).sum(-1).mean()
