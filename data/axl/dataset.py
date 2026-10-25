"""Training dataset over the Articulation-XL shards: skeleton-driven augmentation, 70/30 sampling, padded skeletons."""
import struct
import zipfile
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from data.axl.augment import AugmentConfig, deform, keep_mask, random_params, resample

GROUP_IDS = {"animal": 0, "humanoid": 1}
PAD_PARENT = -2

def _npz_memmap(path: Path, key: str) -> np.ndarray:
    """Memory-maps one array stored uncompressed inside an .npz (np.load ignores mmap_mode for archives)."""
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(key + ".npy")
    if info.compress_type != zipfile.ZIP_STORED:
        raise ValueError(f"{path}: {key} is compressed")
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        name_len, extra_len = struct.unpack("<HH", f.read(30)[26:30])
        f.seek(info.header_offset + 30 + name_len + extra_len)
        version = np.lib.format.read_magic(f)
        read_header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = read_header(f)
        offset = f.tell()
    if 0 in shape:
        return np.zeros(shape, dtype)
    return np.memmap(path, dtype=dtype, mode="r", shape=shape, offset=offset, order="F" if fortran else "C")

def sampling_weights(groups: np.ndarray, animal_ratio: float = 0.7) -> np.ndarray:
    """Per-example weights for a WeightedRandomSampler so that animals make up animal_ratio of the draws."""
    groups = np.asarray(groups)
    animals = groups == "animal"
    if animals.all() or not animals.any():
        return np.ones(len(groups))
    return np.where(animals, animal_ratio / animals.sum(), (1 - animal_ratio) / (~animals).sum())

def collate_skeletons(batch: list) -> dict:
    """Stacks points and pads skeletons to the largest one in the batch (parents -2, presence 0, masked out)."""
    size, max_joints = len(batch), max(len(item["joints"]) for item in batch)
    joints = torch.zeros(size, max_joints, 3)
    parents = torch.full((size, max_joints), PAD_PARENT, dtype=torch.long)
    presence = torch.zeros(size, max_joints)
    mask = torch.zeros(size, max_joints, dtype=torch.bool)
    for b, item in enumerate(batch):
        j = len(item["joints"])
        joints[b, :j], parents[b, :j], presence[b, :j], mask[b, :j] = item["joints"], item["parents"], item["presence"], True
    return {
        "points": torch.stack([item["points"] for item in batch]),
        "joints": joints, "parents": parents, "presence": presence, "joint_mask": mask,
        "group": torch.stack([item["group"] for item in batch]),
        "uuid": [item["uuid"] for item in batch],
    }

class AXLDataset(Dataset):
    """Creatures from the shard cache built by scripts/build_axl_cache.py."""

    def __init__(self, cache_dir: str, split: str, num_points: int = 2048, augment: bool = True,
                 augment_cfg: Optional[AugmentConfig] = None, seed: int = 0, max_shards: Optional[int] = None):
        super().__init__()
        self.num_points = num_points
        self.augment = augment
        self.augment_cfg = augment_cfg or AugmentConfig()
        self.seed = seed
        self.shards = sorted(p for p in (Path(cache_dir) / split).glob("rg*.npz") if not p.name.endswith(".tmp.npz"))[:max_shards]
        self._arrays = {}

        shard_ids, positions, uuids, groups, self._meta = [], [], [], [], []
        for s, path in enumerate(self.shards):
            with np.load(path) as shard:
                meta = {key: shard[key] for key in ("point_offsets", "joint_offsets", "joints", "parents")}
                shard_uuids, shard_groups = shard["uuids"], shard["groups"]
            self._meta.append(meta)
            shard_ids += [s] * len(shard_uuids)
            positions += list(range(len(shard_uuids)))
            uuids += shard_uuids.tolist()
            groups += shard_groups.tolist()
        self._index = np.stack([shard_ids, positions], axis=1) if shard_ids else np.zeros((0, 2), np.int64)
        self.uuids = np.array(uuids)
        self.groups = np.array(groups)

    def __len__(self) -> int:
        return len(self._index)

    def __getstate__(self):
        # Memory maps are reopened in each DataLoader worker instead of being pickled with the dataset
        state = self.__dict__.copy()
        state["_arrays"] = {}
        return state

    def _shard_arrays(self, s: int) -> dict:
        # Opened lazily, once per worker process
        if s not in self._arrays:
            self._arrays[s] = {key: _npz_memmap(self.shards[s], key) for key in ("points", "skin_idx", "skin_w")}
        return self._arrays[s]

    def __getitem__(self, i: int) -> dict:
        s, k = self._index[i]
        meta, arrays = self._meta[s], self._shard_arrays(s)
        p0, p1 = meta["point_offsets"][k], meta["point_offsets"][k + 1]
        j0, j1 = meta["joint_offsets"][k], meta["joint_offsets"][k + 1]
        points = np.asarray(arrays["points"][p0:p1], dtype=np.float32)
        joints = meta["joints"][j0:j1].astype(np.float32)
        parents = meta["parents"][j0:j1].astype(np.int64)
        presence = np.ones(len(joints), np.float32)
        rng = np.random.default_rng(np.random.randint(2 ** 31))

        if self.augment:
            skin_idx = np.asarray(arrays["skin_idx"][p0:p1])
            skin_w = np.asarray(arrays["skin_w"][p0:p1], dtype=np.float32)
            skin_w /= np.maximum(skin_w.sum(1, keepdims=True), 1e-6)
            params = random_params(parents, rng, self.augment_cfg)
            keep = keep_mask(points, skin_idx, skin_w, joints, parents, params.presence)
            if keep.any():
                points, joints, area = deform(points, skin_idx, skin_w, joints, parents, params)
                points = resample(points, area, keep, self.num_points, rng)
                presence = params.presence.astype(np.float32)
            else:
                points = resample(points, np.ones(len(points)), np.ones(len(points), bool), self.num_points, rng)
        else:
            points = points[rng.choice(len(points), self.num_points, replace=self.num_points > len(points))]

        return {
            "points": torch.from_numpy(np.ascontiguousarray(points, dtype=np.float32)),
            "joints": torch.from_numpy(joints),
            "parents": torch.from_numpy(parents),
            "presence": torch.from_numpy(presence),
            "group": torch.tensor(GROUP_IDS[self.groups[i]]),
            "uuid": str(self.uuids[i]),
        }
