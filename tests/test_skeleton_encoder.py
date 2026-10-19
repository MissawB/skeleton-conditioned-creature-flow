# tests/test_skeleton_encoder.py
import torch
from models.skeleton_encoder import SkeletonEncoder

def test_skeleton_encoder_forward():
    batch_size = 4
    num_joints = 35
    hidden_dim = 256
    
    encoder = SkeletonEncoder(num_joints=num_joints, hidden_dim=hidden_dim, num_layers=2)
    joints_3d = torch.randn(batch_size, num_joints, 3)
    
    encoded_joints = encoder(joints_3d)
    
    assert encoded_joints.shape == (batch_size, num_joints, hidden_dim)
    assert not torch.isnan(encoded_joints).any()
