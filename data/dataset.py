import os
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional
from data.smal_quadrupeds import SMALQuadrupedGenerator, FAMILIES, PARENT_INDICES, GENERATOR_VERSION
from data.ot_coupling import precompute_ot_noise

class CreatureDataset(Dataset):
    """PyTorch Dataset for 3D creatures conditioned on kinematic skeletons."""
    
    def __init__(
        self,
        num_samples_per_family: int = 50,
        num_points: int = 2048,
        augment: bool = True,
        cache_dir: Optional[str] = "data/cache",
        seed: int = 42,
        ot_noise_per_sample: int = 0,
        independent_noise_prob: float = 0.0,
        ot_workers: Optional[int] = None
    ):
        super().__init__()
        self.num_points = num_points
        self.augment = augment
        self.cache_dir = cache_dir
        self.independent_noise_prob = independent_noise_prob
        self.samples = []
        self.ot_noise = None
        
        generator = SMALQuadrupedGenerator(seed=seed)
        # Cache file is keyed on generation parameters so a config change never reuses stale samples
        cache_file = os.path.join(cache_dir, f"dataset_v{GENERATOR_VERSION}_n{num_samples_per_family}_p{num_points}_s{seed}.pt") if cache_dir else None

        if cache_file and os.path.exists(cache_file):
            cached = torch.load(cache_file, weights_only=False)
            self.samples = cached["samples"]
        else:
            for family in FAMILIES:
                for _ in range(num_samples_per_family):
                    instance = generator.generate_instance(family=family, num_points=num_points)
                    self.samples.append(instance)
                    
            if cache_file:
                os.makedirs(cache_dir, exist_ok=True)
                torch.save({"samples": self.samples}, cache_file)
        
        if ot_noise_per_sample > 0:
            self.ot_noise = self._load_or_compute_ot_noise(cache_file, ot_noise_per_sample, seed, ot_workers)

    def _load_or_compute_ot_noise(self, cache_file: Optional[str], num_noise: int, seed: int, workers: Optional[int]):
        noise_file = cache_file.replace(".pt", f"_ot{num_noise}.pt") if cache_file else None
        if noise_file and os.path.exists(noise_file):
            return torch.load(noise_file, weights_only=True)["noise"]
        
        noise = precompute_ot_noise([s["points"] for s in self.samples], num_noise, seed=seed, num_workers=workers)
        noise = [torch.from_numpy(n) for n in noise]
        if noise_file:
            torch.save({"noise": noise}, noise_file)
        return noise

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_augmentation(self, points: np.ndarray, joints: np.ndarray, noise: Optional[np.ndarray] = None):
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
        # N(0, I) is rotation invariant: rotating the matched noise keeps it Gaussian and keeps the OT pairing
        if noise is not None:
            noise = noise @ R
        
        # Scaling jitter
        scale = np.random.uniform(0.92, 1.08)
        points *= scale
        joints *= scale
        
        return points, joints, noise

    def __getitem__(self, idx: int):
        raw = self.samples[idx]
        pts = raw["points"].copy()
        jts = raw["joints"].copy()
        noise = None
        if self.ot_noise is not None:
            if np.random.rand() < self.independent_noise_prob:
                # Hybrid coupling: some independent noise keeps the field easy to learn near t = 0
                noise = np.random.randn(*pts.shape).astype(np.float32)
            else:
                stack = self.ot_noise[idx]
                noise = stack[np.random.randint(len(stack))].numpy().copy()
        
        if self.augment:
            pts, jts, noise = self._apply_augmentation(pts, jts, noise)
            
        item = {
            "points": torch.from_numpy(pts).float(),
            "joints": torch.from_numpy(jts).float(),
            "joint_parents": torch.from_numpy(raw["joint_parents"]).long(),
            "family_id": torch.tensor(raw["family_id"]).long()
        }
        if noise is not None:
            item["noise"] = torch.from_numpy(noise).float()
        return item
