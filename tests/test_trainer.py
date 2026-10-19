# tests/test_trainer.py
import os
import torch
from data.dataset import CreatureDataset
from models.point_dit import PointDiT
from models.flow_matching import OptimalTransportFlowMatching
from pipeline.trainer import Trainer

def test_trainer_single_epoch(tmp_path):
    dataset = CreatureDataset(num_samples_per_family=2, num_points=256, augment=False, cache_dir=None)
    model = PointDiT(point_dim=3, hidden_dim=64, num_layers=2, num_heads=2, num_joints=35)
    fm = OptimalTransportFlowMatching(model)
    
    trainer = Trainer(
        flow_matching=fm,
        dataset=dataset,
        batch_size=2,
        lr=1e-3,
        checkpoint_dir=str(tmp_path),
        device="cpu"
    )
    
    loss = trainer.train_epoch(epoch=0)
    assert loss > 0.0
    
    save_path = trainer.save_checkpoint(epoch=0)
    assert os.path.exists(save_path)
