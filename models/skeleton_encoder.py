import torch
import torch.nn as nn

class SkeletonEncoder(nn.Module):
    """
    Kinematic Tree Encoder for 3D Skeletons.
    Processes K joint coordinates and anatomical learned embeddings via Multi-Head Attention.
    """
    def __init__(self, num_joints: int = 35, hidden_dim: int = 256, num_layers: int = 2, num_heads: int = 4):
        super().__init__()
        self.num_joints = num_joints
        self.hidden_dim = hidden_dim
        
        # 3D coordinate projection + learned anatomical ID embedding
        self.joint_proj = nn.Linear(3, hidden_dim)
        self.anatomical_embed = nn.Embedding(num_joints, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, joints: torch.Tensor) -> torch.Tensor:
        """
        Args:
            joints: (B, K, 3) 3D coordinates
        Returns:
            (B, K, hidden_dim) enriched joint memory tokens
        """
        B, K, _ = joints.shape
        joint_indices = torch.arange(K, device=joints.device).unsqueeze(0).expand(B, -1)
        
        # Combine geometric 3D coordinate projection with anatomical identity
        tokens = self.joint_proj(joints) + self.anatomical_embed(joint_indices)
        tokens = self.transformer(tokens)
        return self.norm(tokens)
