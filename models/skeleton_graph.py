"""
Encoder for kinematic skeletons of any topology (2 to 255 joints), in the spirit of AnyTop and UniMate:
one token per joint built from geometry only (no per-joint identity), and self-attention biased by the
tree distance and the relation between joints. Padded joints are masked out everywhere.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

REL_SELF, REL_PARENT, REL_CHILD, REL_SIBLING, REL_OTHER = 0, 1, 2, 3, 4
MAX_TREE_DIST = 5

def _ancestors(parents: torch.Tensor, joint_mask: torch.Tensor) -> torch.Tensor:
    """(B, J, J) float, [b, i, a] = 1 if a is i itself or one of its ancestors: transitive closure of
    'self or parent' by repeated squaring, log2(J) batched matmuls instead of one step per tree level."""
    B, J = parents.shape
    step = torch.eye(J, device=parents.device).expand(B, J, J).clone()
    step.scatter_(2, parents.clamp_min(0)[..., None], 1.0)
    step = step * (parents >= 0)[..., None] + torch.eye(J, device=parents.device) * (parents < 0)[..., None]
    step = step * joint_mask[:, :, None] * joint_mask[:, None, :]
    anc = step
    for _ in range(max(1, (J - 1).bit_length())):
        anc = (anc @ anc).clamp_max(1.0)
    return anc

def tree_depth_batch(parents: torch.Tensor, joint_mask: torch.Tensor) -> torch.Tensor:
    return (_ancestors(parents, joint_mask).sum(-1) - 1).clamp_min(0).long()

def tree_relations(parents: torch.Tensor, joint_mask: torch.Tensor):
    """Tree distance (capped at MAX_TREE_DIST, MAX_TREE_DIST + 1 for padding) and relation of i to j, both (B, J, J)."""
    anc = _ancestors(parents, joint_mask)
    depth = anc.sum(-1) - 1
    common = anc @ anc.transpose(1, 2)                  # common ancestors form a chain: their count is depth(lca) + 1
    dist = (depth[:, :, None] + depth[:, None, :] - 2 * (common - 1)).long().clamp(0, MAX_TREE_DIST)
    pair_mask = joint_mask[:, :, None] & joint_mask[:, None, :]
    dist = torch.where(pair_mask, dist, torch.full_like(dist, MAX_TREE_DIST + 1))

    J = parents.shape[1]
    idx = torch.arange(J, device=parents.device)
    is_parent = parents[:, :, None] == idx[None, None, :]          # j is the parent of i
    is_child = parents[:, None, :] == idx[None, :, None]           # j is a child of i
    sibling = (parents[:, :, None] == parents[:, None, :]) & (parents[:, :, None] >= 0)
    rel = torch.full_like(dist, REL_OTHER)
    rel = torch.where(sibling, torch.full_like(rel, REL_SIBLING), rel)
    rel = torch.where(is_child, torch.full_like(rel, REL_CHILD), rel)
    rel = torch.where(is_parent, torch.full_like(rel, REL_PARENT), rel)
    rel = torch.where(torch.eye(J, dtype=torch.bool, device=parents.device)[None], torch.full_like(rel, REL_SELF), rel)
    rel = torch.where(pair_mask, rel, torch.full_like(rel, REL_OTHER))
    return dist, rel

class _GraphLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, 4 * hidden_dim), nn.GELU(), nn.Linear(4 * hidden_dim, hidden_dim))

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        B, J, D = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(B, J, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        h = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        x = x + self.out(h.transpose(1, 2).reshape(B, J, D))
        return x + self.ffn(self.norm2(x))

class SkeletonGraphEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 256, num_layers: int = 2, num_heads: int = 4, max_depth: int = 31):
        super().__init__()
        self.num_heads = num_heads
        self.max_depth = max_depth
        self.joint_mlp = nn.Sequential(nn.Linear(7, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.depth_embed = nn.Embedding(max_depth + 1, hidden_dim)
        self.dist_bias = nn.Embedding(MAX_TREE_DIST + 2, num_heads)
        self.rel_bias = nn.Embedding(5, num_heads)
        self.layers = nn.ModuleList([_GraphLayer(hidden_dim, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.global_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, joints: torch.Tensor, parents: torch.Tensor, presence: torch.Tensor, joint_mask: torch.Tensor):
        """Joint tokens (B, J, D) and a global skeleton token (B, D); padded tokens are zero."""
        parent_pos = joints.gather(1, parents.clamp_min(0)[..., None].expand_as(joints))
        offset = torch.where((parents >= 0)[..., None], joints - parent_pos, torch.zeros_like(joints))
        depth = tree_depth_batch(parents, joint_mask).clamp_max(self.max_depth)
        x = self.joint_mlp(torch.cat([joints, offset, presence[..., None]], dim=-1)) + self.depth_embed(depth)

        dist, rel = tree_relations(parents, joint_mask)
        bias = (self.dist_bias(dist) + self.rel_bias(rel)).permute(0, 3, 1, 2)
        bias = bias.masked_fill(~joint_mask[:, None, None, :], float("-inf"))
        for layer in self.layers:
            x = layer(x, bias)
        x = self.norm(x) * joint_mask[..., None]
        pooled = x.sum(1) / joint_mask.sum(1, keepdim=True).clamp_min(1)
        return x, self.global_mlp(pooled)
