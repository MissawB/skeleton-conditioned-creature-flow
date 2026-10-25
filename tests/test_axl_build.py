import json
import numpy as np
import pyarrow.parquet as pq
import pytest
from data.axl.build import build_split, process_model
from data.smal_quadrupeds import SMALQuadrupedGenerator, PARENT_INDICES
from tests.axl_fixtures import prism_model, write_mirror


def _quadruped(uuid, seed=0):
    joints = SMALQuadrupedGenerator(seed=seed).generate_instance("felidae")["joints"]
    return prism_model(joints, np.array(PARENT_INDICES), uuid)


def _invalid(uuid):
    model = _quadruped(uuid)
    model["bones"] = model["bones"][:-1]  # one joint left disconnected
    return model


class CountingOpener:
    def __init__(self, root):
        self.root, self.reads = root, 0

    def __call__(self, path):
        opener = self
        pf = pq.ParquetFile(self.root / path)

        class Counted:
            metadata = pf.metadata

            def read_row_group(self, i, columns=None):
                opener.reads += 1
                return pf.read_row_group(i, columns=columns)

            def read(self, columns=None):
                return pf.read(columns=columns)
        return Counted()


def test_rejects_more_than_255_joints():
    joints = np.stack([np.arange(300) * 0.01, np.zeros(300), np.zeros(300)], 1)
    row = prism_model(joints, np.arange(300) - 1, "long")
    with pytest.raises(ValueError, match="too many joints"):
        process_model(row, 64, np.random.default_rng(0))


def test_process_model_returns_a_normalised_quantised_creature():
    out = process_model(_quadruped("q"), 512, np.random.default_rng(0))
    assert out["points"].dtype == np.float16 and out["points"].shape == (512, 3)
    assert out["skin_idx"].dtype == np.uint8 and out["skin_w"].dtype == np.uint8
    assert np.abs(np.linalg.norm(out["points"].astype(np.float32), axis=1).max() - 1) < 2e-3
    assert out["joints"].shape == (35, 3) and out["parents"][0] == -1


def test_build_skips_existing_shards_and_writes_atomically(tmp_path):
    mirror = tmp_path / "mirror"
    models = [_quadruped("a", 0), _invalid("b"), _quadruped("c", 1), _quadruped("d", 2)]
    tags = {"a": ["animal"], "b": ["animal"], "c": ["character"], "d": ["weapon"]}
    write_mirror(mirror, "train", models, tags)
    out = tmp_path / "cache"
    opener = CountingOpener(mirror)

    build_split("train", out, num_points=256, open_fn=opener)
    with np.load(out / "train" / "rg0000.npz") as shard0:
        assert list(shard0["uuids"]) == ["a"] and list(shard0["groups"]) == ["animal"]
    assert json.loads((out / "train" / "rg0000.rejects.json").read_text())["b"].startswith("not a tree")
    with np.load(out / "train" / "rg0001.npz") as shard1:
        assert list(shard1["uuids"]) == ["c"]
        assert shard1["points"].dtype == np.float16 and shard1["point_offsets"].tolist() == [0, 256]
        assert np.abs(shard1["points"].astype(np.float32)).max() <= 1 + 1e-3
    assert json.loads((out / "train" / "rg0001.rejects.json").read_text())["d"] == "excluded category"

    (out / "train" / "rg0001.npz").unlink()
    (out / "train" / "rg0001.tmp.npz").write_bytes(b"partial")
    opener.reads = 0
    build_split("train", out, num_points=256, open_fn=opener)
    assert opener.reads == 4  # only row group 1, once per table
    with np.load(out / "train" / "rg0001.npz") as shard1:
        assert list(shard1["uuids"]) == ["c"]
    assert not (out / "train" / "rg0001.tmp.npz").exists()


def test_row_groups_are_read_as_numpy_geometry(tmp_path):
    from data.axl.build import _read_row_group
    write_mirror(tmp_path, "train", [_quadruped("a")], {"a": ["animal"]})
    row = _read_row_group("train", 0, lambda p: pq.ParquetFile(tmp_path / p))[0]
    assert isinstance(row["vertices"], np.ndarray) and row["vertices"].shape[1] == 3
    assert isinstance(row["faces"], np.ndarray) and row["faces"].shape[1] == 3
    assert isinstance(row["value"], np.ndarray) and isinstance(row["joints"], np.ndarray)
