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

def test_cosine_schedule_warms_up_then_decays_to_min_lr(tmp_path):
    dataset = CreatureDataset(num_samples_per_family=2, num_points=64, augment=False, cache_dir=None)
    model = PointDiT(point_dim=3, hidden_dim=32, num_layers=1, num_heads=2, num_joints=35)
    trainer = Trainer(OptimalTransportFlowMatching(model), dataset, batch_size=2, lr=1e-3,
                      checkpoint_dir=str(tmp_path), device="cpu",
                      lr_schedule="cosine", num_epochs=2, warmup_steps=2, min_lr_ratio=0.1)
    lrs = []
    for epoch in range(2):
        trainer.train_epoch(epoch)
        lrs.append(trainer.optimizer.param_groups[0]["lr"])
    total = 2 * len(trainer.dataloader)
    assert trainer.scheduler.last_epoch == total
    # Halfway through the decay the rate sits between the peak and the floor; at the end it reaches the floor
    assert 1e-4 < lrs[0] < 1e-3
    assert abs(lrs[1] - 1e-4) < 1e-9
