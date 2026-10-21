# tests/test_ema.py
import torch
from torch import nn
from pipeline.ema import EMA

def test_ema_tracks_exponential_average():
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.0)
    ema = EMA(model, decay=0.5, warmup=False)

    expected = 0.0
    for value in [1.0, 2.0, 3.0]:
        with torch.no_grad():
            model.weight.fill_(value)
        ema.update(model)
        expected = 0.5 * expected + 0.5 * value

    assert torch.allclose(ema.model.weight, torch.full_like(model.weight, expected))
    # The live model is untouched
    assert torch.allclose(model.weight, torch.full_like(model.weight, 3.0))

def test_ema_warmup_follows_early_weights():
    model = nn.Linear(2, 1, bias=False)
    initial = model.weight.detach().clone()
    ema = EMA(model, decay=0.999)
    with torch.no_grad():
        model.weight.fill_(5.0)
    ema.update(model)
    # First update uses decay min(0.999, 1/10) = 0.1: the average moves 90% of the way
    assert torch.allclose(ema.model.weight, 0.1 * initial + 0.9 * 5.0)

def test_trainer_saves_ema_state(tmp_path):
    from data.dataset import CreatureDataset
    from models.point_dit import PointDiT
    from models.flow_matching import OptimalTransportFlowMatching
    from pipeline.trainer import Trainer

    dataset = CreatureDataset(num_samples_per_family=2, num_points=128, augment=False, cache_dir=None)
    model = PointDiT(point_dim=3, hidden_dim=32, num_layers=1, num_heads=2, num_joints=35)
    trainer = Trainer(OptimalTransportFlowMatching(model), dataset, batch_size=2,
                      checkpoint_dir=str(tmp_path), device="cpu", ema_decay=0.9)
    trainer.train_epoch(epoch=1)
    ckpt = torch.load(trainer.save_checkpoint(epoch=1), weights_only=True)

    assert set(ckpt["ema_state"]) == set(ckpt["model_state"])
    differs = any(not torch.equal(ckpt["ema_state"][k], ckpt["model_state"][k]) for k in ckpt["model_state"])
    assert differs
