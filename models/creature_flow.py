"""Flow matching for creatures: free skeletons, online per-shape OT and a KL-regularised shape code z."""
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from models.creature_dit import CreatureDiT
from models.ot_gpu import ot_match
from models.shape_encoder import ShapeEncoder, kl_divergence, reparameterize

BATCH_TENSORS = ("points", "joints", "parents", "presence", "joint_mask")

class CreatureModel(nn.Module):
    """Velocity network and shape encoder trained together (one optimizer, one EMA, one checkpoint)."""

    def __init__(self, dit: CreatureDiT, shape_encoder: ShapeEncoder):
        super().__init__()
        self.dit = dit
        self.shape_encoder = shape_encoder

    @property
    def data_scale(self) -> torch.Tensor:
        return self.dit.data_scale

class CreatureFlowMatching:
    def __init__(self, model: CreatureDiT, shape_encoder: ShapeEncoder, kl_weight: float = 1e-3,
                 z_drop_prob: float = 0.3, cond_drop_prob: float = 0.0, ot_iters: int = 20, online_ot: bool = True):
        self.model = CreatureModel(model, shape_encoder)
        self.kl_weight = kl_weight
        self.z_drop_prob = z_drop_prob
        self.cond_drop_prob = cond_drop_prob
        self.ot_iters = ot_iters
        self.online_ot = online_ot
        self.last_stats = {}

    def encode(self, points: torch.Tensor) -> torch.Tensor:
        """Shape code (mu) of clouds in data units."""
        return self.model.shape_encoder(points * self.model.data_scale)[0]

    def loss_from_batch(self, batch: dict) -> torch.Tensor:
        device = self.model.data_scale.device
        points, joints, parents, presence, joint_mask = (batch[k].to(device) for k in BATCH_TENSORS)
        s = self.model.data_scale
        x_1, joints = points * s, joints * s
        B = x_1.shape[0]

        mu, logvar = self.model.shape_encoder(x_1)
        z = reparameterize(mu, logvar)
        # Without z the model must still grow a creature from the skeleton alone
        z = z * (torch.rand(B, 1, device=device) >= self.z_drop_prob)

        x_0 = torch.randn_like(x_1)
        if self.online_ot:
            with torch.autocast(device.type, enabled=False):
                x_0 = ot_match(x_0.float(), x_1.float(), iters=self.ot_iters).to(x_1.dtype)
        t = torch.rand(B, device=device)
        x_t = (1 - t.view(B, 1, 1)) * x_0 + t.view(B, 1, 1) * x_1
        cond_mask = torch.rand(B, device=device) >= self.cond_drop_prob if self.cond_drop_prob > 0 else None

        pred = self.model.dit(x_t, t, joints, parents, presence, joint_mask, z=z, cond_mask=cond_mask)
        fm = torch.mean((pred.float() - (x_1 - x_0).float()) ** 2)
        kl = kl_divergence(mu.float(), logvar.float())
        self.last_stats = {"fm": fm.item(), "kl": kl.item()}
        return fm + self.kl_weight * kl

    @torch.no_grad()
    def sample(self, joints: torch.Tensor, parents: torch.Tensor, presence: torch.Tensor, joint_mask: torch.Tensor,
               z: Optional[torch.Tensor] = None, num_points: int = 2048, num_steps: int = 10,
               record_trajectory: bool = False, generator: Optional[torch.Generator] = None) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Euler integration from noise to a cloud in data units; z None generates from the skeleton alone."""
        self.model.eval()
        s = self.model.data_scale
        joints = joints * s
        B = joints.shape[0]
        x = torch.randn(B, num_points, 3, device=joints.device, generator=generator)
        trajectory = [(x / s).cpu()] if record_trajectory else []
        for i in range(num_steps):
            t = torch.full((B,), i / num_steps, device=joints.device)
            x = x + self.model.dit(x, t, joints, parents, presence, joint_mask, z=z) / num_steps
            if record_trajectory:
                trajectory.append((x / s).cpu())
        return trajectory, x / s
