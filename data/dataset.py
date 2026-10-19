import os
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Tuple
from data.smal_quadrupeds import SMALQuadrupedGenerator, FAMILIES, PARENT_INDICES

class CreatureDataset(Dataset):
    """PyTorch Dataset for 3D creatures conditioned on kinematic skeletons."""
    
    def __init__(
        self,
        num_samples_per_family: int = 50,
        num_points: int = 2048,
        augment: bool = True,
        cache_dir: Optional[str] = "data/cache",
        seed: int = 42
    ):
        super().__init__()
        self.num_points = num_points
        self.augment = augment
        self.cache_dir = cache_dir
        self.samples = []
        
        generator = SMALQuadrupedGenerator(seed=seed)
        
        if cache_dir and os.path.exists(os.path.join(cache_dir, "dataset.pt")):
            cached = torch.load(os.path.join(cache_dir, "dataset.pt"), weights_only=False)
            self.samples = cached["samples"]
        else:
            for family in FAMILIES:
                for _ in range(num_samples_per_family):
                    instance = generator.generate_instance(family=family, num_points=num_points)
                    self.samples.append(instance)
                    
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                torch.save({"samples": self.samples}, os.path.join(cache_dir, "dataset.pt"))

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_augmentation(self, points: np.ndarray, joints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Coherent rotation around vertical Y axis
        theta = np.random.uniform(-np.pi, np.pi)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([
            [cos_t, 0.0, sin_t],
            [0.0,   1.0, 0.0  ],
            [-sin_t, 0.0, cos_t]
        ], dtype=np.float32)
        
        points = points @ R
        joints = joints @ R
        
        # Scaling jitter
        scale = np.random.uniform(0.92, 1.08)
        points *= scale
        joints *= scale
        
        return points, joints

    def __getitem__(self, idx: int):
        raw = self.samples[idx]
        pts = raw["points"].copy()
        jts = raw["joints"].copy()
        
        if self.augment:
            pts, jts = self._apply_augmentation(pts, jts)
            
        return {
            "points": torch.from_numpy(pts).float(),
            "joints": torch.from_numpy(jts).float(),
            "joint_parents": torch.from_numpy(raw["joint_parents"]).long(),
            "family_id": torch.tensor(raw["family_id"]).long()
        }
