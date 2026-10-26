"""
Point-DiT for creatures of any skeleton topology: the fixed 35-joint encoder becomes a graph encoder,
joints are padded and masked, fading parts weigh in by their presence, and an optional shape code z
joins the time embedding in the AdaLN conditioning.
"""
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from models.point_dit import PointDiTBlock, TimestepEmbedder
from models.skeleton_graph import SkeletonGraphEncoder

MIN_PRESENCE = 1e-4

class CreatureDiT(nn.Module):
    def __init__(self, hidden_dim: int = 256, num_layers: int = 6, num_heads: int = 8, z_dim: int = 64,
                 spatial_distance_scale: Optional[float] = None, data_scale: float = 1.0, skeleton_layers: int = 2,
                 checkpoint_above_joints: int = 128):
        super().__init__()
        # Points and joints arrive multiplied by data_scale (done by the flow); the default bias keeps the
        # quadruped model's per-head reach in data units
        self.register_buffer("data_scale", torch.tensor(float(data_scale)))
        self.z_dim = z_dim
        # Point-to-joint attention grows with the joint count: recompute blocks in backward for very large rigs
        self.checkpoint_above_joints = checkpoint_above_joints
        if spatial_distance_scale is None:
            spatial_distance_scale = 2.0 / data_scale ** 2
        self.point_proj = nn.Linear(3, hidden_dim)
        self.time_embedder = TimestepEmbedder(hidden_dim)
        self.skeleton_encoder = SkeletonGraphEncoder(hidden_dim, num_layers=skeleton_layers, num_heads=4)
        self.null_joint_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.z_proj = nn.Linear(z_dim, hidden_dim)
        nn.init.zeros_(self.z_proj.weight)
        nn.init.zeros_(self.z_proj.bias)
        # Without it the time and skeleton embeddings grew tenfold around epoch 5 on Articulation-XL and the
        # AdaLN modulations (0.3 -> 78) wrecked the residual stream: both 10-epoch runs collapsed in epoch 6
        self.cond_norm = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList([PointDiTBlock(hidden_dim, num_heads, spatial_distance_scale) for _ in range(num_layers)])
        self.pos_heads = nn.ModuleList([nn.Linear(hidden_dim, 3) for _ in range(num_layers)])
        for head in self.pos_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, 3)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def uses_checkpointing(self, num_joints: int) -> bool:
        return self.training and num_joints > self.checkpoint_above_joints

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, joints: torch.Tensor, parents: torch.Tensor,
                presence: torch.Tensor, joint_mask: torch.Tensor, z: Optional[torch.Tensor] = None,
                cond_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Velocity (B, N, 3). joint_mask marks real joints; z None is the null code; cond_mask False drops the skeleton."""
        # A joint with presence 0 is gone: masked like padding, so neither the encoder nor the points see it
        active = joint_mask & (presence > 0)
        joint_mem, skeleton_global = self.skeleton_encoder(joints, parents, presence, active)
        key_bias = torch.where(active, presence.clamp_min(MIN_PRESENCE).log(), torch.full_like(presence, float("-inf")))
        bias_weight = None
        if cond_mask is not None:
            keep = cond_mask.to(joint_mem.dtype)
            joint_mem = keep.view(-1, 1, 1) * joint_mem + (1 - keep.view(-1, 1, 1)) * self.null_joint_token
            skeleton_global = keep.view(-1, 1) * skeleton_global
            bias_weight = keep.view(-1, 1, 1, 1)
            # A dropped skeleton keeps only its padding mask: presence and positions must not leak
            key_bias = torch.where(cond_mask[:, None], key_bias, torch.where(active, torch.zeros_like(key_bias), key_bias))
        if z is None:
            z = joints.new_zeros(joints.shape[0], self.z_dim)
        cond = self.cond_norm(self.time_embedder(t) + skeleton_global + self.z_proj(z))
        key_bias = key_bias[:, None, None, :]

        h = self.point_proj(x_t)
        pos = x_t
        remaining = (1.0 - t).view(-1, 1, 1)
        recompute = self.uses_checkpointing(joints.shape[1])
        for block, pos_head in zip(self.blocks, self.pos_heads):
            if recompute:
                h = checkpoint(block, h, cond, joint_mem, pos, joints, bias_weight, key_bias, use_reentrant=False)
            else:
                h = block(h, cond, joint_mem, pts_pos=pos, joint_pos=joints, bias_weight=bias_weight, key_bias=key_bias)
            pos = x_t + remaining * pos_head(h)
        return self.out_proj(self.final_norm(h))
