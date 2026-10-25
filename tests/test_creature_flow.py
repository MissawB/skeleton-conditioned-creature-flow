import pickle
import numpy as np
import pyarrow.parquet as pq
import torch
from data.axl.build import build_split
from data.axl.dataset import AXLDataset, collate_skeletons
from data.smal_quadrupeds import PARENT_INDICES, SMALQuadrupedGenerator
from models.creature_dit import CreatureDiT
from models.creature_flow import CreatureFlowMatching
from models.shape_encoder import ShapeEncoder
from tests.axl_fixtures import prism_model, write_mirror


def _cache(tmp_path):
    joints = SMALQuadrupedGenerator(seed=0).generate_instance("felidae")["joints"]
    models = [prism_model(joints, np.array(PARENT_INDICES), "quad"),
              prism_model(np.array([[0, 0, 0], [0.5, 0.2, 0]]), np.array([-1, 0]), "stick")]
    write_mirror(tmp_path / "mirror", "train", models, {"quad": ["animal"], "stick": ["character"]})
    build_split("train", tmp_path / "cache", num_points=512, open_fn=lambda p: pq.ParquetFile(tmp_path / "mirror" / p))
    return str(tmp_path / "cache")


def _flow(hidden=32, layers=2):
    torch.manual_seed(0)
    dit = CreatureDiT(hidden_dim=hidden, num_layers=layers, num_heads=4, z_dim=8, data_scale=2.0)
    return CreatureFlowMatching(dit, ShapeEncoder(z_dim=8, hidden_dim=32, num_points=128), z_drop_prob=0.3)


def test_dataset_pickles_without_open_memmaps(tmp_path):
    ds = AXLDataset(_cache(tmp_path), "train", num_points=64, augment=False)
    ds[0]
    assert ds._arrays
    assert pickle.loads(pickle.dumps(ds))._arrays == {}


def test_loss_from_batch_reports_fm_and_kl(tmp_path):
    ds = AXLDataset(_cache(tmp_path), "train", num_points=128, augment=True)
    batch = collate_skeletons([ds[0], ds[1], ds[0]])
    fm = _flow()
    loss = fm.loss_from_batch(batch)
    assert torch.isfinite(loss) and {"fm", "kl"} <= set(fm.last_stats)


def test_creature_flow_overfits_two_creatures(tmp_path):
    ds = AXLDataset(_cache(tmp_path), "train", num_points=256, augment=False)
    batch = collate_skeletons([ds[0], ds[1]])
    fm = _flow(hidden=64, layers=2)
    opt = torch.optim.Adam(fm.model.parameters(), lr=1e-3)
    first = None
    for step in range(300):
        opt.zero_grad()
        loss = fm.loss_from_batch(batch)
        loss.backward()
        opt.step()
        if first is None:
            first = fm.last_stats["fm"]
    assert fm.last_stats["fm"] < 0.5 * first
    _, points = fm.sample(batch["joints"], batch["parents"], batch["presence"], batch["joint_mask"], num_points=256)
    assert points.shape == (2, 256, 3) and torch.isfinite(points).all()
