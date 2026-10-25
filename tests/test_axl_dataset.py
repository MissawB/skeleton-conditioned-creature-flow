import numpy as np
import torch
from data.axl.build import build_split
from data.axl.dataset import AXLDataset, collate_skeletons, sampling_weights
from data.smal_quadrupeds import SMALQuadrupedGenerator, PARENT_INDICES
from tests.axl_fixtures import prism_model, write_mirror
import pyarrow.parquet as pq


def _cache(tmp_path):
    joints = SMALQuadrupedGenerator(seed=0).generate_instance("felidae")["joints"]
    models = [
        prism_model(joints, np.array(PARENT_INDICES), "quad"),
        prism_model(np.array([[0, 0, 0], [0.5, 0.2, 0]]), np.array([-1, 0]), "stick"),
        prism_model(joints, np.array(PARENT_INDICES), "hero"),
    ]
    write_mirror(tmp_path / "mirror", "train", models, {"quad": ["animal"], "stick": ["animal"], "hero": ["character"]})
    build_split("train", tmp_path / "cache", num_points=512, open_fn=lambda p: pq.ParquetFile(tmp_path / "mirror" / p))
    return tmp_path / "cache"


def test_sampling_weights_give_the_requested_animal_ratio():
    groups = np.array(["animal"] * 10 + ["humanoid"] * 90)
    w = sampling_weights(groups, 0.7)
    assert abs(w[groups == "animal"].sum() / w.sum() - 0.7) < 1e-9


def test_collate_pads_skeletons_and_masks():
    def item(j):
        return {"points": torch.zeros(4, 3), "joints": torch.ones(j, 3), "parents": torch.arange(j) - 1,
                "presence": torch.ones(j), "group": torch.tensor(0), "uuid": f"u{j}"}
    batch = collate_skeletons([item(3), item(5)])
    assert batch["joints"].shape == (2, 5, 3) and batch["points"].shape == (2, 4, 3)
    assert batch["joint_mask"][0].tolist() == [True, True, True, False, False]
    assert batch["parents"][0, 3:].tolist() == [-2, -2] and batch["presence"][0, 3:].tolist() == [0, 0]
    assert batch["joints"][0, 3:].abs().sum() == 0 and batch["uuid"] == ["u3", "u5"]


def test_two_joint_skeleton_round_trip(tmp_path):
    ds = AXLDataset(str(_cache(tmp_path)), "train", num_points=2048, augment=True)
    stick = list(ds.uuids).index("stick")
    for _ in range(10):
        item = ds[stick]
        assert item["points"].shape == (2048, 3) and torch.isfinite(item["points"]).all()
        assert item["joints"].shape == (2, 3)


def test_dataset_reads_shards_lazily_and_returns_unit_radius_points(tmp_path):
    ds = AXLDataset(str(_cache(tmp_path)), "train", num_points=256, augment=False)
    assert len(ds) == 3 and sorted(ds.groups.tolist()) == ["animal", "animal", "humanoid"]
    item = ds[list(ds.uuids).index("quad")]
    assert item["points"].dtype == torch.float32 and item["points"].shape == (256, 3)
    assert abs(item["points"].norm(dim=1).max().item() - 1) < 0.05
    assert item["presence"].tolist() == [1.0] * 35 and item["parents"][0] == -1
    batch = collate_skeletons([ds[0], ds[1], ds[2]])
    assert batch["joints"].shape == (3, 35, 3)


def test_max_shards_limits_the_dataset(tmp_path):
    cache = _cache(tmp_path)
    assert len(AXLDataset(str(cache), "train", augment=False, max_shards=1)) < len(AXLDataset(str(cache), "train", augment=False))


def test_joint_names_are_read_from_the_shard(tmp_path):
    ds = AXLDataset(str(_cache(tmp_path)), "train", augment=False)
    i = list(ds.uuids).index("stick")
    assert ds.joint_names(i) == ["j0", "j1"]
