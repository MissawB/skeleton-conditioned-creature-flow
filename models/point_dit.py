import math
import torch
import torch.nn as nn
from models.skeleton_encoder import SkeletonEncoder

class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedder followed by 2-layer MLP."""
    def __init__(self, hidden_dim: int, frequency_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.frequency_dim = frequency_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.frequency_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=t.device) / half_dim
        )
        args = t.view(-1, 1) * freqs.view(1, -1)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(embedding)


class AdaLNModulation(nn.Module):
    """Adaptive Layer Norm generating scale, shift, and gate for AdaLN-Zero."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim)
        )
        # Initialize final projection with zeros for stable start
        nn.init.zeros_(self.mlp[1].weight)
        nn.init.zeros_(self.mlp[1].bias)

    def forward(self, t_emb: torch.Tensor):
        return self.mlp(t_emb).chunk(6, dim=-1)


class SpatialCrossAttention(nn.Module):
    """Cross-Attention from 3D points to skeleton joints with spatial distance bias."""
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, pts_feat: torch.Tensor, joint_mem: torch.Tensor, pts_pos: torch.Tensor, joint_pos: torch.Tensor) -> torch.Tensor:
        B, N, D = pts_feat.shape
        _, K, _ = joint_mem.shape
        
        q = self.q_proj(pts_feat).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(joint_mem).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(joint_mem).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Standard attention scores: (B, heads, N, K)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Spatial distance bias: penalize distant joints
        dist_sq = torch.cdist(pts_pos, joint_pos, p=2.0) ** 2 # (B, N, K)
        spatial_bias = -1.5 * dist_sq.unsqueeze(1) # (B, 1, N, K)
        
        attn = attn + spatial_bias
        attn = torch.softmax(attn, dim=-1)
        
        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)


class PointDiTBlock(nn.Module):
    """Point-DiT block with Self-Attention, Spatial Cross-Attention, and AdaLN modulation."""
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = SpatialCrossAttention(hidden_dim, num_heads)
        
        self.norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.adaln = AdaLNModulation(hidden_dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, joint_mem: torch.Tensor, pts_pos: torch.Tensor, joint_pos: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln(t_emb)
        
        # Self-Attention on points
        normed = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        sa_out, _ = self.self_attn(normed, normed, normed)
        x = x + gate_msa.unsqueeze(1) * sa_out
        
        # Spatial Cross-Attention to skeleton joints
        ca_out = self.cross_attn(self.norm2(x), joint_mem, pts_pos, joint_pos)
        x = x + ca_out
        
        # Feed-Forward Network
        normed_ffn = self.norm3(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        ffn_out = self.ffn(normed_ffn)
        x = x + gate_mlp.unsqueeze(1) * ffn_out
        
        return x


class PointDiT(nn.Module):
    """Complete Point Diffusion/Flow Transformer conditioned on continuous time and 3D skeleton."""
    def __init__(
        self,
        point_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        num_joints: int = 35
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.point_proj = nn.Linear(point_dim, hidden_dim)
        self.time_embedder = TimestepEmbedder(hidden_dim)
        self.skeleton_encoder = SkeletonEncoder(num_joints=num_joints, hidden_dim=hidden_dim)
        
        self.blocks = nn.ModuleList([
            PointDiTBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, point_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, joints: torch.Tensor) -> torch.Tensor:
        """
        Predicts velocity field v_t at point cloud x_t.
        Args:
            x_t: (B, N, 3) 3D points
            t: (B,) continuous timestep in [0, 1]
            joints: (B, K, 3) 3D joint positions
        Returns:
            (B, N, 3) predicted velocities
        """
        t_emb = self.time_embedder(t)
        joint_mem = self.skeleton_encoder(joints)
        
        h = self.point_proj(x_t)
        for block in self.blocks:
            h = block(h, t_emb, joint_mem, pts_pos=x_t, joint_pos=joints)
            
        return self.out_proj(self.final_norm(h))
