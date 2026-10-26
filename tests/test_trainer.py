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

def test_bf16_mixed_precision_trains_with_fp32_weights(tmp_path):
    dataset = CreatureDataset(num_samples_per_family=2, num_points=64, augment=False, cache_dir=None)
    model = PointDiT(point_dim=3, hidden_dim=32, num_layers=1, num_heads=2, num_joints=35)
    trainer = Trainer(OptimalTransportFlowMatching(model), dataset, batch_size=2, lr=1e-3,
                      checkpoint_dir=str(tmp_path), device="cpu", mixed_precision="bf16")
    before = model.point_proj.weight.detach().clone()
    loss = trainer.train_epoch(0)
    assert torch.isfinite(torch.tensor(loss))
    assert model.point_proj.weight.dtype == torch.float32
    assert not torch.equal(before, model.point_proj.weight)


def test_non_finite_steps_are_skipped_without_touching_the_weights(tmp_path):
    class EveryOtherNaN(OptimalTransportFlowMatching):
        calls = 0
        def compute_loss(self, *args, **kwargs):
            EveryOtherNaN.calls += 1
            loss = super().compute_loss(*args, **kwargs)
            return loss * float("nan") if EveryOtherNaN.calls % 2 else loss
    dataset = CreatureDataset(num_samples_per_family=2, num_points=64, augment=False, cache_dir=None)
    model = PointDiT(point_dim=3, hidden_dim=32, num_layers=1, num_heads=2, num_joints=35)
    trainer = Trainer(EveryOtherNaN(model), dataset, batch_size=2, lr=1e-3, checkpoint_dir=str(tmp_path), device="cpu")
    loss = trainer.train_epoch(0)
    assert trainer.skipped_steps == len(trainer.dataloader) // 2 + len(trainer.dataloader) % 2
    assert torch.isfinite(torch.tensor(loss))
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert all(torch.isfinite(p).all() for p in trainer.ema.model.parameters())


def test_resume_restores_model_ema_optimizer_and_epoch(tmp_path):
    from data.dataset import CreatureDataset
    from models.flow_matching import OptimalTransportFlowMatching
    from models.point_dit import PointDiT
    from pipeline.trainer import Trainer
    ds = CreatureDataset(num_samples_per_family=1, num_points=64, augment=False, cache_dir=None)
    make = lambda: Trainer(OptimalTransportFlowMatching(PointDiT(hidden_dim=32, num_layers=1, num_heads=4)), ds,
                           batch_size=5, checkpoint_dir=str(tmp_path), device="cpu")
    first = make()
    first.train_epoch(1)
    path = first.save_checkpoint(1)
    second = make()
    assert second.load_checkpoint(path) == 1
    for a, b in zip(first.fm.model.parameters(), second.fm.model.parameters()):
        assert torch.equal(a, b)
    for a, b in zip(first.ema.model.parameters(), second.ema.model.parameters()):
        assert torch.equal(a, b)
    assert second.ema.num_updates == first.ema.num_updates
    assert second.optimizer.state_dict()["state"].keys() == first.optimizer.state_dict()["state"].keys()
