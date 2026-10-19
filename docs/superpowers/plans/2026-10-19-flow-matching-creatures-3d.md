# 3D Creature Flow Matching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an end-to-end 3D creature generative system using Optimal Transport Flow Matching conditioned on kinematic skeletons, featuring hybrid composition and an interactive Three.js web viewer.

**Architecture:** Optimal Transport Flow Matching (OT-FM) with straight-line velocity interpolation drives point clouds ($N=2048$) from Gaussian noise to 3D creature surfaces. The generative core is a Point-DiT Transformer modulating point features with continuous time (AdaLN) and conditioning on a 35-joint kinematic skeleton via spatial cross-attention. Skeletons can be grafted to synthesize chimera creatures, and generated trajectories are exported for inspection in a zero-dependency Three.js web viewer.

**Tech Stack:** Python 3.12, PyTorch, NumPy, SciPy, PyYAML, Tqdm, HTML5 / Three.js (r128 via CDN).

---

### Task 1: Environment Setup & Project Configuration

**Files:**
- Create: `requirements.txt`
- Create: `configs/default_config.yaml`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test for configuration loading**

```python
# tests/test_config.py
import os
import yaml

def test_default_config_exists_and_valid():
    config_path = os.path.join("configs", "default_config.yaml")
    assert os.path.exists(config_path), f"Config file not found at {config_path}"
    
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    assert "data" in cfg
    assert "model" in cfg
    assert "training" in cfg
    assert "sampling" in cfg
    
    assert cfg["data"]["num_points"] == 2048
    assert cfg["data"]["num_joints"] == 35
    assert cfg["model"]["hidden_dim"] == 256
    assert cfg["model"]["num_layers"] == 6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL (FileNotFoundError or config missing)

- [ ] **Step 3: Create `requirements.txt` and `configs/default_config.yaml`**

```text
# requirements.txt
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
pyyaml>=6.0
tqdm>=4.65.0
pytest>=8.0.0
```

```yaml
# configs/default_config.yaml
data:
  num_points: 2048
  num_joints: 35
  num_samples_per_family: 100
  families:
    - felidae
    - canidae
    - equidae
    - bovidae
    - hippopotamidae
  cache_dir: "data/cache"

model:
  point_dim: 3
  hidden_dim: 256
  num_layers: 6
  num_heads: 8
  dropout: 0.05
  num_joints: 35
  spatial_distance_scale: 2.0

training:
  batch_size: 16
  learning_rate: 0.0002
  weight_decay: 0.00001
  num_epochs: 30
  save_every: 5
  device: "cuda" # falls back to cpu automatically if cuda unavailable
  checkpoint_dir: "checkpoints"

sampling:
  num_steps: 20
  solver: "heun" # "euler" or "heun"
  export_dir: "viewer/data"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add requirements.txt configs/default_config.yaml tests/test_config.py
git commit -m "chore: setup configuration and project dependencies"
```

---

### Task 2: Parametric SMAL Quadruped Generator & Kinematic Hierarchy

**Files:**
- Create: `data/smal_quadrupeds.py`
- Test: `tests/test_smal_quadrupeds.py`

- [ ] **Step 1: Write the failing test for SMAL kinematics and morphology generation**

```python
# tests/test_smal_quadrupeds.py
import numpy as np
from data.smal_quadrupeds import SMALQuadrupedGenerator, JOINT_NAMES, PARENT_INDICES

def test_smal_quadruped_generator():
    generator = SMALQuadrupedGenerator(seed=42)
    assert len(JOINT_NAMES) == 35
    assert len(PARENT_INDICES) == 35
    
    # Generate a feline creature
    sample = generator.generate_instance(family="felidae", num_points=2048)
    
    assert "points" in sample
    assert "joints" in sample
    assert "joint_parents" in sample
    assert "family_id" in sample
    
    assert sample["points"].shape == (2048, 3)
    assert sample["joints"].shape == (35, 3)
    assert sample["joint_parents"].shape == (35,)
    
    # Check bounding box normalization within [-1, 1]
    assert np.all(sample["points"] >= -1.05) and np.all(sample["points"] <= 1.05)
    assert np.all(sample["joints"] >= -1.05) and np.all(sample["joints"] <= 1.05)
    
    # Check parent indices: root must be -1, others must point to valid parent indices < child
    assert PARENT_INDICES[0] == -1
    for i in range(1, 35):
        assert 0 <= PARENT_INDICES[i] < i
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_smal_quadrupeds.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'data.smal_quadrupeds')

- [ ] **Step 3: Implement `data/smal_quadrupeds.py`**

```python
# data/smal_quadrupeds.py
"""
Parametric SMAL-inspired quadruped kinematic generator.
Provides standard 35-joint kinematic tree and procedural surface point sampling
for 5 distinct mammal families (Felidae, Canidae, Equidae, Bovidae, Hippopotamidae).
"""
import numpy as np
from typing import Dict, List, Tuple

# Standardized 35 kinematic joint names
JOINT_NAMES = [
    # 0: Root / Pelvis
    "pelvis",
    # 1-4: Spine & Neck
    "spine_1", "spine_2", "spine_3", "neck_base",
    # 5-6: Head
    "head", "jaw",
    # 7-13: Left Front Leg (Clavicle, Shoulder, Elbow, Wrist, Paw) + Claws
    "l_collar", "l_shoulder", "l_elbow", "l_wrist", "l_front_paw", "l_front_toe_1", "l_front_toe_2",
    # 14-20: Right Front Leg
    "r_collar", "r_shoulder", "r_elbow", "r_wrist", "r_front_paw", "r_front_toe_1", "r_front_toe_2",
    # 21-26: Left Back Leg (Hip, Knee, Ankle, Paw)
    "l_hip", "l_knee", "l_ankle", "l_back_paw", "l_back_toe_1", "l_back_toe_2",
    # 27-32: Right Back Leg
    "r_hip", "r_knee", "r_ankle", "r_back_paw", "r_back_toe_1", "r_back_toe_2",
    # 33-34: Tail
    "tail_1", "tail_2"
]

# Directed Tree: parent index for each joint (-1 for root joint 0)
PARENT_INDICES = [
    -1,            # 0: pelvis
    0, 1, 2, 3,    # 1-4: spine_1 -> spine_2 -> spine_3 -> neck_base
    4, 5,          # 5-6: head -> jaw
    4, 7, 8, 9, 10, 11, 11,   # 7-13: left front leg
    4, 14, 15, 16, 17, 18, 18, # 14-20: right front leg
    0, 21, 22, 23, 24, 24,    # 21-26: left back leg
    0, 27, 28, 29, 30, 30,    # 27-32: right back leg
    0, 33          # 33-34: tail
]

FAMILIES = ["felidae", "canidae", "equidae", "bovidae", "hippopotamidae"]

class SMALQuadrupedGenerator:
    """Procedural generator for quadruped skeletons and skinned surface point clouds."""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        
        # Family base proportions [spine_len, leg_len, neck_len, body_thickness, leg_thickness]
        self.family_profiles = {
            "felidae": {"spine": 0.55, "leg": 0.45, "neck": 0.20, "head": 0.18, "r_body": 0.16, "r_leg": 0.05, "tail": 0.35},
            "canidae": {"spine": 0.50, "leg": 0.48, "neck": 0.22, "head": 0.22, "r_body": 0.15, "r_leg": 0.045, "tail": 0.25},
            "equidae": {"spine": 0.60, "leg": 0.65, "neck": 0.38, "head": 0.26, "r_body": 0.20, "r_leg": 0.055, "tail": 0.20},
            "bovidae": {"spine": 0.58, "leg": 0.52, "neck": 0.25, "head": 0.24, "r_body": 0.24, "r_leg": 0.065, "tail": 0.20},
            "hippopotamidae": {"spine": 0.62, "leg": 0.35, "neck": 0.15, "head": 0.28, "r_body": 0.32, "r_leg": 0.09, "tail": 0.15},
        }

    def _build_skeleton(self, family: str, variance: float = 0.03) -> np.ndarray:
        prof = self.family_profiles[family]
        jitter = lambda scale: (self.rng.rand() * 2 - 1) * variance * scale
        
        spine_len = prof["spine"] * (1.0 + jitter(1.0))
        leg_len = prof["leg"] * (1.0 + jitter(1.0))
        neck_len = prof["neck"] * (1.0 + jitter(1.0))
        head_len = prof["head"] * (1.0 + jitter(1.0))
        tail_len = prof["tail"] * (1.0 + jitter(1.0))
        
        joints = np.zeros((35, 3), dtype=np.float32)
        
        # 0: Pelvis
        joints[0] = [-spine_len * 0.4, 0.0, 0.0]
        # 1-4: Spine to neck base
        joints[1] = [-spine_len * 0.15, 0.02, 0.0]
        joints[2] = [spine_len * 0.10, 0.03, 0.0]
        joints[3] = [spine_len * 0.35, 0.04, 0.0]
        joints[4] = [spine_len * 0.45, 0.08, 0.0] # neck_base
        
        # 5-6: Head & Jaw
        joints[5] = [joints[4, 0] + neck_len * 0.6, joints[4, 1] + neck_len * 0.7, 0.0]
        joints[6] = [joints[5, 0] + head_len * 0.7, joints[5, 1] - 0.05, 0.0]
        
        # Front legs
        for side, sign in [("l", 1.0), ("r", -1.0)]:
            idx = 7 if side == "l" else 14
            w = 0.12 * sign
            joints[idx]     = [joints[4, 0] * 0.9, joints[4, 1] - 0.05, w * 0.6]  # collar
            joints[idx + 1] = [joints[idx, 0], joints[idx, 1] - 0.1, w]           # shoulder
            joints[idx + 2] = [joints[idx + 1, 0], joints[idx + 1, 1] - leg_len * 0.4, w * 0.9] # elbow
            joints[idx + 3] = [joints[idx + 2, 0], joints[idx + 2, 1] - leg_len * 0.4, w * 0.85] # wrist
            joints[idx + 4] = [joints[idx + 3, 0] + 0.03, -0.45, w * 0.85]        # paw
            joints[idx + 5] = [joints[idx + 4, 0] + 0.04, -0.47, w * 0.75]        # toe 1
            joints[idx + 6] = [joints[idx + 4, 0] + 0.04, -0.47, w * 0.95]        # toe 2
            
        # Back legs
        for side, sign in [("l", 1.0), ("r", -1.0)]:
            idx = 21 if side == "l" else 27
            w = 0.14 * sign
            joints[idx]     = [joints[0, 0] + 0.02, joints[0, 1] - 0.05, w]      # hip
            joints[idx + 1] = [joints[idx, 0] - 0.05, joints[idx, 1] - leg_len * 0.35, w * 0.95] # knee
            joints[idx + 2] = [joints[idx + 1, 0] + 0.03, joints[idx + 1, 1] - leg_len * 0.4, w * 0.9] # ankle
            joints[idx + 3] = [joints[idx + 2, 0] + 0.02, -0.45, w * 0.85]       # paw
            joints[idx + 4] = [joints[idx + 3, 0] + 0.04, -0.47, w * 0.75]       # toe 1
            joints[idx + 5] = [joints[idx + 3, 0] + 0.04, -0.47, w * 0.95]       # toe 2
            
        # Tail
        joints[33] = [joints[0, 0] - tail_len * 0.5, joints[0, 1] + 0.05, 0.0]
        joints[34] = [joints[33, 0] - tail_len * 0.5, joints[33, 1] - tail_len * 0.4, 0.0]
        
        # Center and normalize skeleton
        center = np.mean(joints, axis=0)
        joints -= center
        max_dist = np.max(np.linalg.norm(joints, axis=1))
        joints /= (max_dist * 1.35)
        
        return joints.astype(np.float32)

    def _sample_points_around_bones(self, joints: np.ndarray, family: str, num_points: int) -> np.ndarray:
        prof = self.family_profiles[family]
        points = []
        
        # Bone segments from parent indices
        segments = []
        for child_idx in range(1, 35):
            parent_idx = PARENT_INDICES[child_idx]
            segments.append((parent_idx, child_idx))
            
        points_per_seg = max(1, num_points // len(segments))
        
        for parent_idx, child_idx in segments:
            p_a = joints[parent_idx]
            p_b = joints[child_idx]
            
            # Radii based on anatomical region
            is_spine = child_idx in [1, 2, 3, 4]
            is_head = child_idx in [5, 6]
            is_tail = child_idx in [33, 34]
            
            if is_spine:
                radius = prof["r_body"]
            elif is_head:
                radius = prof["r_body"] * 0.75
            elif is_tail:
                radius = prof["r_body"] * 0.25
            else:
                radius = prof["r_leg"]
                
            # Random position along bone [0, 1]
            u = self.rng.rand(points_per_seg, 1).astype(np.float32)
            bone_centers = (1 - u) * p_a + u * p_b
            
            # Surface cylinder/ellipsoid coordinates
            theta = self.rng.rand(points_per_seg) * 2 * np.pi
            z_offset = self.rng.randn(points_per_seg) * 0.08 * radius
            r = radius * (1.0 + z_offset)
            
            # Tangent & normal vectors
            bone_vec = p_b - p_a
            bone_norm = np.linalg.norm(bone_vec) + 1e-6
            bone_dir = bone_vec / bone_norm
            
            up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            if np.abs(np.dot(bone_dir, up)) > 0.9:
                up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            tangent = np.cross(bone_dir, up)
            tangent /= np.linalg.norm(tangent) + 1e-6
            binormal = np.cross(bone_dir, tangent)
            
            offsets = (
                r[:, None] * np.cos(theta)[:, None] * tangent[None, :] +
                r[:, None] * np.sin(theta)[:, None] * binormal[None, :]
            )
            
            seg_points = bone_centers + offsets
            points.append(seg_points)
            
        all_points = np.vstack(points)
        # Resample or trim exactly to num_points
        if len(all_points) > num_points:
            indices = self.rng.choice(len(all_points), num_points, replace=False)
            all_points = all_points[indices]
        elif len(all_points) < num_points:
            extra_idx = self.rng.choice(len(all_points), num_points - len(all_points), replace=True)
            all_points = np.vstack([all_points, all_points[extra_idx] + self.rng.randn(len(extra_idx), 3) * 0.005])
            
        return all_points.astype(np.float32)

    def generate_instance(self, family: str = "felidae", num_points: int = 2048) -> Dict[str, np.ndarray]:
        assert family in FAMILIES, f"Unknown family {family}. Supported: {FAMILIES}"
        joints = self._build_skeleton(family)
        points = self._sample_points_around_bones(joints, family, num_points)
        
        return {
            "points": points,
            "joints": joints,
            "joint_parents": np.array(PARENT_INDICES, dtype=np.int64),
            "family_id": FAMILIES.index(family),
            "shape_betas": np.zeros(10, dtype=np.float32)
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_smal_quadrupeds.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add data/smal_quadrupeds.py tests/test_smal_quadrupeds.py
git commit -m "feat: implement SMAL kinematic quadruped generator and tests"
```

---

### Task 3: PyTorch Dataset, Augmentations & Dataset Pre-caching

**Files:**
- Create: `data/dataset.py`
- Create: `scripts/prepare_data.py`
- Test: `tests/test_dataset.py`

- [ ] **Step 1: Write the failing test for PyTorch Dataset**

```python
# tests/test_dataset.py
import torch
import numpy as np
from data.dataset import CreatureDataset

def test_creature_dataset_synthesis_and_augmentation():
    dataset = CreatureDataset(num_samples_per_family=5, num_points=1024, augment=True, cache_dir=None)
    assert len(dataset) == 25  # 5 families * 5 samples
    
    sample = dataset[0]
    assert "points" in sample
    assert "joints" in sample
    assert "family_id" in sample
    assert isinstance(sample["points"], torch.Tensor)
    assert isinstance(sample["joints"], torch.Tensor)
    
    assert sample["points"].shape == (1024, 3)
    assert sample["joints"].shape == (35, 3)
    assert sample["points"].dtype == torch.float32
    assert sample["joints"].dtype == torch.float32
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dataset.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'data.dataset')

- [ ] **Step 3: Implement `data/dataset.py` and `scripts/prepare_data.py`**

```python
# data/dataset.py
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional
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
            cached = torch.load(os.path.join(cache_dir, "dataset.pt"))
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
```

```python
# scripts/prepare_data.py
import argparse
import os
import yaml
from data.dataset import CreatureDataset

def main():
    parser = argparse.ArgumentParser(description="Pre-generate synthetic SMAL quadruped dataset.")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    args = parser.parse_args()
    
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    cache_dir = cfg["data"]["cache_dir"]
    samples_per_family = cfg["data"]["num_samples_per_family"]
    num_points = cfg["data"]["num_points"]
    
    print(f"Generating dataset: {samples_per_family} samples per family, {num_points} points...")
    dataset = CreatureDataset(
        num_samples_per_family=samples_per_family,
        num_points=num_points,
        augment=False,
        cache_dir=cache_dir
    )
    print(f"Dataset successfully created and cached at {cache_dir}/dataset.pt (Total: {len(dataset)} samples).")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_dataset.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add data/dataset.py scripts/prepare_data.py tests/test_dataset.py
git commit -m "feat: implement PyTorch CreatureDataset with 3D augmentations and pre-caching"
```

---

### Task 4: Skeleton Kinematic Transformer Encoder

**Files:**
- Create: `models/skeleton_encoder.py`
- Test: `tests/test_skeleton_encoder.py`

- [ ] **Step 1: Write the failing test for `SkeletonEncoder`**

```python
# tests/test_skeleton_encoder.py
import torch
from models/skeleton_encoder import SkeletonEncoder

def test_skeleton_encoder_forward():
    batch_size = 4
    num_joints = 35
    hidden_dim = 256
    
    encoder = SkeletonEncoder(num_joints=num_joints, hidden_dim=hidden_dim, num_layers=2)
    joints_3d = torch.randn(batch_size, num_joints, 3)
    
    encoded_joints = encoder(joints_3d)
    
    assert encoded_joints.shape == (batch_size, num_joints, hidden_dim)
    assert not torch.isnan(encoded_joints).any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_skeleton_encoder.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'models.skeleton_encoder')

- [ ] **Step 3: Implement `models/skeleton_encoder.py`**

```python
# models/skeleton_encoder.py
import torch
import torch.nn as nn

class SkeletonEncoder(nn.Module):
    """
    Kinematic Tree Encoder for 3D Skeletons.
    Processes K joint coordinates and anatomical learned embeddings via Multi-Head Attention.
    """
    def __init__(self, num_joints: int = 35, hidden_dim: int = 256, num_layers: int = 2, num_heads: int = 4):
        super().__init__()
        self.num_joints = num_joints
        self.hidden_dim = hidden_dim
        
        # 3D coordinate projection + learned anatomical ID embedding
        self.joint_proj = nn.Linear(3, hidden_dim)
        self.anatomical_embed = nn.Embedding(num_joints, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, joints: torch.Tensor) -> torch.Tensor:
        """
        Args:
            joints: (B, K, 3) 3D coordinates
        Returns:
            (B, K, hidden_dim) enriched joint memory tokens
        """
        B, K, _ = joints.shape
        joint_indices = torch.arange(K, device=joints.device).unsqueeze(0).expand(B, -1)
        
        # Combine geometric 3D coordinate projection with anatomical identity
        tokens = self.joint_proj(joints) + self.anatomical_embed(joint_indices)
        tokens = self.transformer(tokens)
        return self.norm(tokens)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_skeleton_encoder.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add models/skeleton_encoder.py tests/test_skeleton_encoder.py
git commit -m "feat: implement kinematic SkeletonEncoder with anatomical embeddings"
```

---

### Task 5: Point-DiT with AdaLN & Geometric Cross-Attention

**Files:**
- Create: `models/point_dit.py`
- Test: `tests/test_point_dit.py`

- [ ] **Step 1: Write the failing test for Point-DiT**

```python
# tests/test_point_dit.py
import torch
from models/point_dit import PointDiT

def test_point_dit_forward():
    batch_size = 2
    num_points = 512
    num_joints = 35
    hidden_dim = 128
    
    model = PointDiT(
        point_dim=3,
        hidden_dim=hidden_dim,
        num_layers=2,
        num_heads=4,
        num_joints=num_joints
    )
    
    x_t = torch.randn(batch_size, num_points, 3)
    t = torch.rand(batch_size)
    joints = torch.randn(batch_size, num_joints, 3)
    
    v_pred = model(x_t, t, joints)
    
    assert v_pred.shape == (batch_size, num_points, 3)
    assert not torch.isnan(v_pred).any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_point_dit.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'models.point_dit')

- [ ] **Step 3: Implement `models/point_dit.py`**

```python
# models/point_dit.py
import math
import torch
import torch.nn as nn
from models.skeleton_encoder import SkeletonEncoder

class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedder followed by 2-layer MLP."""
    def __init__(self, hidden_dim: int, frequency_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.frequency_dim = frequency_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.frequency_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=t.device) / half_dim
        )
        args = t.view(-1, 1) * freqs.view(1, -1)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(embedding)


class AdaLNModulation(nn.Module):
    """Adaptive Layer Norm generating scale, shift, and gate for AdaLN-Zero."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim)
        )
        # Initialize final projection with zeros for stable start
        nn.init.zeros_(self.mlp[1].weight)
        nn.init.zeros_(self.mlp[1].bias)

    def forward(self, t_emb: torch.Tensor):
        return self.mlp(t_emb).chunk(6, dim=-1)


class SpatialCrossAttention(nn.Module):
    """Cross-Attention from 3D points to skeleton joints with spatial distance bias."""
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, pts_feat: torch.Tensor, joint_mem: torch.Tensor, pts_pos: torch.Tensor, joint_pos: torch.Tensor) -> torch.Tensor:
        B, N, D = pts_feat.shape
        _, K, _ = joint_mem.shape
        
        q = self.q_proj(pts_feat).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(joint_mem).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(joint_mem).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Standard attention scores: (B, heads, N, K)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Spatial distance bias: penalize distant joints
        dist_sq = torch.cdist(pts_pos, joint_pos, p=2.0) ** 2 # (B, N, K)
        spatial_bias = -1.5 * dist_sq.unsqueeze(1) # (B, 1, N, K)
        
        attn = attn + spatial_bias
        attn = torch.softmax(attn, dim=-1)
        
        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)


class PointDiTBlock(nn.Module):
    """Point-DiT block with Self-Attention, Spatial Cross-Attention, and AdaLN modulation."""
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = SpatialCrossAttention(hidden_dim, num_heads)
        
        self.norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.adaln = AdaLNModulation(hidden_dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, joint_mem: torch.Tensor, pts_pos: torch.Tensor, joint_pos: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln(t_emb)
        
        # Self-Attention on points
        normed = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        sa_out, _ = self.self_attn(normed, normed, normed)
        x = x + gate_msa.unsqueeze(1) * sa_out
        
        # Spatial Cross-Attention to skeleton joints
        ca_out = self.cross_attn(self.norm2(x), joint_mem, pts_pos, joint_pos)
        x = x + ca_out
        
        # Feed-Forward Network
        normed_ffn = self.norm3(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        ffn_out = self.ffn(normed_ffn)
        x = x + gate_mlp.unsqueeze(1) * ffn_out
        
        return x


class PointDiT(nn.Module):
    """Complete Point Diffusion/Flow Transformer conditioned on continuous time and 3D skeleton."""
    def __init__(
        self,
        point_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        num_joints: int = 35
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.point_proj = nn.Linear(point_dim, hidden_dim)
        self.time_embedder = TimestepEmbedder(hidden_dim)
        self.skeleton_encoder = SkeletonEncoder(num_joints=num_joints, hidden_dim=hidden_dim)
        
        self.blocks = nn.ModuleList([
            PointDiTBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, point_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, joints: torch.Tensor) -> torch.Tensor:
        """
        Predicts velocity field v_t at point cloud x_t.
        Args:
            x_t: (B, N, 3) 3D points
            t: (B,) continuous timestep in [0, 1]
            joints: (B, K, 3) 3D joint positions
        Returns:
            (B, N, 3) predicted velocities
        """
        t_emb = self.time_embedder(t)
        joint_mem = self.skeleton_encoder(joints)
        
        h = self.point_proj(x_t)
        for block in self.blocks:
            h = block(h, t_emb, joint_mem, pts_pos=x_t, joint_pos=joints)
            
        return self.out_proj(self.final_norm(h))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_point_dit.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add models/point_dit.py tests/test_point_dit.py
git commit -m "feat: implement PointDiT with AdaLN and spatial cross-attention"
```

---

### Task 6: Optimal Transport Flow Matching Engine & ODE Solvers

**Files:**
- Create: `models/flow_matching.py`
- Test: `tests/test_flow_matching.py`

- [ ] **Step 1: Write the failing test for Flow Matching loss and sampling**

```python
# tests/test_flow_matching.py
import torch
from models.point_dit import PointDiT
from models.flow_matching import OptimalTransportFlowMatching

def test_flow_matching_loss_and_sample():
    model = PointDiT(point_dim=3, hidden_dim=64, num_layers=2, num_heads=2, num_joints=35)
    fm = OptimalTransportFlowMatching(model)
    
    batch_size = 2
    num_points = 256
    x_1 = torch.randn(batch_size, num_points, 3)
    joints = torch.randn(batch_size, 35, 3)
    
    loss = fm.compute_loss(x_1, joints)
    assert loss.dim() == 0
    assert loss.item() > 0.0
    
    # Test Euler solver sampling
    traj, x_final = fm.sample(joints[:1], num_points=num_points, num_steps=5, solver="euler", record_trajectory=True)
    assert x_final.shape == (1, num_points, 3)
    assert len(traj) == 6 # t=0 plus 5 steps
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_flow_matching.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'models.flow_matching')

- [ ] **Step 3: Implement `models/flow_matching.py`**

```python
# models/flow_matching.py
import torch
import torch.nn as nn
from typing import Tuple, List, Optional

class OptimalTransportFlowMatching:
    """Optimal Transport Flow Matching (OT-FM) framework with Euler and Heun ODE solvers."""
    
    def __init__(self, model: nn.Module):
        self.model = model

    def compute_loss(self, x_1: torch.Tensor, joints: torch.Tensor) -> torch.Tensor:
        """
        Computes MSE loss along linear OT interpolants:
        x_t = (1 - t) * x_0 + t * x_1
        Target velocity u_t = x_1 - x_0
        """
        B, N, _ = x_1.shape
        device = x_1.device
        
        # Sample x_0 ~ N(0, I) and t ~ U[0, 1]
        x_0 = torch.randn_like(x_1)
        t = torch.rand(B, device=device)
        
        # OT interpolation
        t_expand = t.view(B, 1, 1)
        x_t = (1.0 - t_expand) * x_0 + t_expand * x_1
        target_v = x_1 - x_0
        
        pred_v = self.model(x_t, t, joints)
        loss = torch.mean((pred_v - target_v) ** 2)
        return loss

    @torch.no_grad()
    def sample(
        self,
        joints: torch.Tensor,
        num_points: int = 2048,
        num_steps: int = 20,
        solver: str = "heun",
        record_trajectory: bool = False
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Integrates dx/dt = v_theta(x, t, joints) from t=0 to t=1.
        """
        self.model.eval()
        B = joints.shape[0]
        device = joints.device
        
        x = torch.randn(B, num_points, 3, device=device)
        dt = 1.0 / num_steps
        
        trajectory = [x.clone().cpu()] if record_trajectory else []
        
        for i in range(num_steps):
            t_val = i / num_steps
            t = torch.full((B,), t_val, device=device)
            
            if solver == "euler":
                v = self.model(x, t, joints)
                x = x + dt * v
            elif solver == "heun":
                v1 = self.model(x, t, joints)
                x_next_euler = x + dt * v1
                t_next = torch.full((B,), min(1.0, (i + 1) / num_steps), device=device)
                v2 = self.model(x_next_euler, t_next, joints)
                x = x + 0.5 * dt * (v1 + v2)
            else:
                raise ValueError(f"Unknown solver {solver}. Choose 'euler' or 'heun'.")
                
            if record_trajectory:
                trajectory.append(x.clone().cpu())
                
        return trajectory, x
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_flow_matching.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add models/flow_matching.py tests/test_flow_matching.py
git commit -m "feat: implement OT-Flow Matching engine with Euler and Heun ODE solvers"
```

---

### Task 7: Kinematic Hybrid Splicer

**Files:**
- Create: `pipeline/hybrid_splicer.py`
- Test: `tests/test_hybrid_splicer.py`

- [ ] **Step 1: Write the failing test for Kinematic Hybrid Splicing**

```python
# tests/test_hybrid_splicer.py
import numpy as np
from data.smal_quadrupeds import SMALQuadrupedGenerator
from pipeline.hybrid_splicer import KinematicHybridSplicer

def test_hybrid_splicer():
    generator = SMALQuadrupedGenerator(seed=42)
    skel_host = generator.generate_instance("felidae")["joints"]
    skel_donor = generator.generate_instance("equidae")["joints"]
    
    splicer = KinematicHybridSplicer()
    chimera_skel = splicer.splice(
        skel_host=skel_host,
        skel_donor=skel_donor,
        junction_joint="neck_base" # Graft horse neck & head onto lion body
    )
    
    assert chimera_skel.shape == (35, 3)
    # Check that junction point matches smoothly
    neck_idx = 4
    head_idx = 5
    assert not np.isnan(chimera_skel).any()
    # Torso joints (0 to 3) should match host
    np.testing.assert_allclose(chimera_skel[0:4], skel_host[0:4])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hybrid_splicer.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'pipeline.hybrid_splicer')

- [ ] **Step 3: Implement `pipeline/hybrid_splicer.py`**

```python
# pipeline/hybrid_splicer.py
import numpy as np
from typing import Set, List
from data.smal_quadrupeds import JOINT_NAMES, PARENT_INDICES

class KinematicHybridSplicer:
    """Combines subtrees from different animal skeletons to create chimeric skeletons."""
    
    def __init__(self):
        self.joint_names = JOINT_NAMES
        self.parents = PARENT_INDICES
        self.children = {i: [] for i in range(len(JOINT_NAMES))}
        for child, parent in enumerate(self.parents):
            if parent >= 0:
                self.children[parent].append(child)

    def _get_subtree_indices(self, root_joint_idx: int) -> Set[int]:
        """Collect all joint indices belonging to the subtree rooted at root_joint_idx."""
        subtree = set()
        queue = [root_joint_idx]
        while queue:
            curr = queue.pop(0)
            subtree.add(curr)
            queue.extend(self.children[curr])
        return subtree

    def splice(
        self,
        skel_host: np.ndarray,
        skel_donor: np.ndarray,
        junction_joint: str = "neck_base"
    ) -> np.ndarray:
        """
        Grafts the subtree of skel_donor rooted at junction_joint onto skel_host.
        """
        assert junction_joint in self.joint_names, f"Unknown junction joint: {junction_joint}"
        junction_idx = self.joint_names.index(junction_joint)
        
        subtree_indices = self._get_subtree_indices(junction_idx)
        # Exclude the junction root itself from translation offset base
        subtree_descendants = subtree_indices - {junction_idx}
        
        chimera = skel_host.copy()
        
        # Translation offset to align donor subtree with host junction position
        donor_junction_pos = skel_donor[junction_idx]
        host_junction_pos = skel_host[junction_idx]
        offset = host_junction_pos - donor_junction_pos
        
        for idx in subtree_descendants:
            chimera[idx] = skel_donor[idx] + offset
            
        return chimera.astype(np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_hybrid_splicer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/hybrid_splicer.py tests/test_hybrid_splicer.py
git commit -m "feat: implement KinematicHybridSplicer for chimeric skeleton creation"
```

---

### Task 8: Quantitative Metrics (Chamfer Distance, Skeleton Coverage, Novelty)

**Files:**
- Create: `pipeline/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test for metrics**

```python
# tests/test_metrics.py
import torch
from pipeline.metrics import compute_chamfer_distance, compute_skeleton_coverage, compute_novelty_score

def test_metrics():
    x = torch.randn(2, 500, 3)
    y = x.clone() # Identical
    
    # Chamfer distance between identical point clouds must be 0
    cd = compute_chamfer_distance(x, y)
    assert torch.isclose(cd, torch.tensor(0.0), atol=1e-5)
    
    # Skeleton coverage
    joints = torch.randn(2, 35, 3)
    cov = compute_skeleton_coverage(joints, x)
    assert cov.item() >= 0.0
    
    # Novelty score
    dataset = [torch.randn(500, 3) for _ in range(3)]
    novelty = compute_novelty_score(x[0], dataset)
    assert novelty > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'pipeline.metrics')

- [ ] **Step 3: Implement `pipeline/metrics.py`**

```python
# pipeline/metrics.py
import torch
from typing import List

def compute_chamfer_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes symmetric squared Chamfer Distance between two point clouds.
    x: (B, N, 3) or (N, 3)
    y: (B, M, 3) or (M, 3)
    """
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if y.dim() == 2:
        y = y.unsqueeze(0)
        
    # Pairwise squared distances: (B, N, M)
    dist_sq = torch.cdist(x, y, p=2.0) ** 2
    
    min_x_to_y = torch.min(dist_sq, dim=2)[0].mean(dim=1)
    min_y_to_x = torch.min(dist_sq, dim=1)[0].mean(dim=1)
    
    return torch.mean(min_x_to_y + min_y_to_x)

def compute_skeleton_coverage(joints: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """
    Average Euclidean distance from each skeleton joint to its closest point in the point cloud.
    joints: (B, K, 3)
    points: (B, N, 3)
    """
    if joints.dim() == 2:
        joints = joints.unsqueeze(0)
    if points.dim() == 2:
        points = points.unsqueeze(0)
        
    dist = torch.cdist(joints, points, p=2.0)
    min_dist = torch.min(dist, dim=2)[0]
    return torch.mean(min_dist)

def compute_novelty_score(gen_points: torch.Tensor, dataset_points: List[torch.Tensor]) -> float:
    """
    Finds the minimum Chamfer Distance to any sample in the training set.
    """
    min_cd = float("inf")
    for sample in dataset_points:
        cd = compute_chamfer_distance(gen_points, sample).item()
        if cd < min_cd:
            min_cd = cd
    return min_cd
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/metrics.py tests/test_metrics.py
git commit -m "feat: implement vectorized Chamfer Distance, Skeleton Coverage, and Novelty metrics"
```

---

### Task 9: Training Pipeline & CLI Script

**Files:**
- Create: `pipeline/trainer.py`
- Create: `scripts/train.py`
- Test: `tests/test_trainer.py`

- [ ] **Step 1: Write the failing test for the trainer**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_trainer.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'pipeline.trainer')

- [ ] **Step 3: Implement `pipeline/trainer.py` and `scripts/train.py`**

```python
# pipeline/trainer.py
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional
from models.flow_matching import OptimalTransportFlowMatching

class Trainer:
    """Training manager for Flow Matching Point-DiT with automatic device selection."""
    
    def __init__(
        self,
        flow_matching: OptimalTransportFlowMatching,
        dataset,
        batch_size: int = 16,
        lr: float = 2e-4,
        weight_decay: float = 1e-5,
        checkpoint_dir: str = "checkpoints",
        device: str = "cuda"
    ):
        self.fm = flow_matching
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.fm.model.to(self.device)
        
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.optimizer = torch.optim.AdamW(self.fm.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def train_epoch(self, epoch: int) -> float:
        self.fm.model.train()
        total_loss = 0.0
        
        for batch in self.dataloader:
            pts = batch["points"].to(self.device)
            jts = batch["joints"].to(self.device)
            
            self.optimizer.zero_grad()
            loss = self.fm.compute_loss(pts, jts)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.fm.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(self.dataloader)
        return avg_loss

    def save_checkpoint(self, epoch: int) -> str:
        path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch:03d}.pt")
        torch.save({
            "epoch": epoch,
            "model_state": self.fm.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict()
        }, path)
        return path
```

```python
# scripts/train.py
import argparse
import yaml
import torch
from data.dataset import CreatureDataset
from models.point_dit import PointDiT
from models.flow_matching import OptimalTransportFlowMatching
from pipeline.trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description="Train 3D Creature Flow Matching model.")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    args = parser.parse_args()
    
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    dataset = CreatureDataset(
        num_samples_per_family=cfg["data"]["num_samples_per_family"],
        num_points=cfg["data"]["num_points"],
        augment=True,
        cache_dir=cfg["data"]["cache_dir"]
    )
    
    model = PointDiT(
        point_dim=cfg["model"]["point_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        num_heads=cfg["model"]["num_heads"],
        num_joints=cfg["model"]["num_joints"]
    )
    fm = OptimalTransportFlowMatching(model)
    
    trainer = Trainer(
        flow_matching=fm,
        dataset=dataset,
        batch_size=cfg["training"]["batch_size"],
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        checkpoint_dir=cfg["training"]["checkpoint_dir"],
        device=cfg["training"]["device"]
    )
    
    epochs = cfg["training"]["num_epochs"]
    print(f"Starting training on device: {trainer.device} for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        loss = trainer.train_epoch(epoch)
        print(f"Epoch {epoch:03d}/{epochs:03d} | Loss: {loss:.5f}")
        if epoch % cfg["training"]["save_every"] == 0 or epoch == epochs:
            path = trainer.save_checkpoint(epoch)
            print(f"--> Saved checkpoint to {path}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_trainer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/trainer.py scripts/train.py tests/test_trainer.py
git commit -m "feat: implement training pipeline and CLI script"
```

---

### Task 10: Sampling, Trajectory Export & Evaluation CLI

**Files:**
- Create: `scripts/sample_and_export.py`
- Create: `scripts/evaluate.py`
- Test: `tests/test_cli_pipeline.py`

- [ ] **Step 1: Write integration test for sample export and evaluation**

```python
# tests/test_cli_pipeline.py
import json
import os
import torch
import numpy as np
from data.smal_quadrupeds import SMALQuadrupedGenerator
from models.point_dit import PointDiT
from models.flow_matching import OptimalTransportFlowMatching
from pipeline.hybrid_splicer import KinematicHybridSplicer
from pipeline.metrics import compute_chamfer_distance

def test_sampling_and_export_json(tmp_path):
    model = PointDiT(hidden_dim=64, num_layers=2, num_heads=2)
    fm = OptimalTransportFlowMatching(model)
    generator = SMALQuadrupedGenerator(seed=42)
    
    skel = generator.generate_instance("felidae")["joints"]
    skel_t = torch.from_numpy(skel).unsqueeze(0)
    
    traj, x_final = fm.sample(skel_t, num_points=128, num_steps=5, record_trajectory=True)
    
    export_data = {
        "species": "lion",
        "joints": skel.tolist(),
        "trajectory": [step.squeeze(0).numpy().tolist() for step in traj]
    }
    
    out_file = os.path.join(tmp_path, "creatures.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"creatures": [export_data]}, f)
        
    assert os.path.exists(out_file)
    with open(out_file, "r") as f:
        loaded = json.load(f)
    assert len(loaded["creatures"]) == 1
    assert len(loaded["creatures"][0]["trajectory"]) == 6
```

- [ ] **Step 2: Run test to verify it fails/passes**

Run: `python -m pytest tests/test_cli_pipeline.py -v`
Expected: PASS

- [ ] **Step 3: Implement `scripts/sample_and_export.py` and `scripts/evaluate.py`**

```python
# scripts/sample_and_export.py
import os
import json
import yaml
import torch
import numpy as np
from data.smal_quadrupeds import SMALQuadrupedGenerator, PARENT_INDICES
from pipeline.hybrid_splicer import KinematicHybridSplicer
from models.point_dit import PointDiT
from models.flow_matching import OptimalTransportFlowMatching

def main():
    cfg_path = "configs/default_config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointDiT(
        point_dim=cfg["model"]["point_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        num_heads=cfg["model"]["num_heads"],
        num_joints=cfg["model"]["num_joints"]
    )
    
    ckpt_dir = cfg["training"]["checkpoint_dir"]
    ckpt_path = os.path.join(ckpt_dir, f"model_epoch_{cfg['training']['num_epochs']:03d}.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        print("Warning: No trained checkpoint found. Using freshly initialized model for demonstration.")
        
    model.to(device)
    fm = OptimalTransportFlowMatching(model)
    generator = SMALQuadrupedGenerator(seed=42)
    splicer = KinematicHybridSplicer()
    
    specimens = [
        {"name": "Lion (Felidae)", "type": "pure", "family": "felidae"},
        {"name": "Horse (Equidae)", "type": "pure", "family": "equidae"},
        {"name": "Wolf (Canidae)", "type": "pure", "family": "canidae"},
        {"name": "Deer (Bovidae)", "type": "pure", "family": "bovidae"},
        {"name": "Chimera (Lion Body + Horse Neck/Head)", "type": "hybrid", "host": "felidae", "donor": "equidae", "junction": "neck_base"}
    ]
    
    exported = []
    num_points = cfg["data"]["num_points"]
    steps = cfg["sampling"]["num_steps"]
    solver = cfg["sampling"]["solver"]
    
    for spec in specimens:
        print(f"Generating {spec['name']}...")
        if spec["type"] == "pure":
            joints = generator.generate_instance(spec["family"])["joints"]
        else:
            skel_h = generator.generate_instance(spec["host"])["joints"]
            skel_d = generator.generate_instance(spec["donor"])["joints"]
            joints = splicer.splice(skel_h, skel_d, junction_joint=spec["junction"])
            
        j_tensor = torch.from_numpy(joints).unsqueeze(0).to(device)
        traj, _ = fm.sample(j_tensor, num_points=num_points, num_steps=steps, solver=solver, record_trajectory=True)
        
        exported.append({
            "name": spec["name"],
            "type": spec["type"],
            "joints": joints.tolist(),
            "joint_parents": PARENT_INDICES,
            "trajectory": [step.squeeze(0).numpy().tolist() for step in traj]
        })
        
    out_dir = cfg["sampling"]["export_dir"]
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "creatures.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"creatures": exported}, f)
    print(f"Exported {len(exported)} creature trajectories to {out_file}")

if __name__ == "__main__":
    main()
```

```python
# scripts/evaluate.py
import os
import yaml
import torch
from data.dataset import CreatureDataset
from models.point_dit import PointDiT
from models.flow_matching import OptimalTransportFlowMatching
from pipeline.metrics import compute_chamfer_distance, compute_skeleton_coverage

def main():
    with open("configs/default_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    dataset = CreatureDataset(num_samples_per_family=10, num_points=1024, augment=False, cache_dir=None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = PointDiT(hidden_dim=cfg["model"]["hidden_dim"], num_layers=cfg["model"]["num_layers"]).to(device)
    fm = OptimalTransportFlowMatching(model)
    
    print("Evaluating test metrics on generated creatures...")
    all_cd, all_cov = [], []
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        joints = sample["joints"].unsqueeze(0).to(device)
        target_pts = sample["points"].unsqueeze(0).to(device)
        
        _, gen_pts = fm.sample(joints, num_points=1024, num_steps=15)
        cd = compute_chamfer_distance(gen_pts, target_pts).item()
        cov = compute_skeleton_coverage(joints, gen_pts).item()
        all_cd.append(cd)
        all_cov.append(cov)
        
    print(f"Mean Chamfer Distance: {sum(all_cd)/len(all_cd):.5f}")
    print(f"Mean Skeleton Coverage: {sum(all_cov)/len(all_cov):.5f}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cli_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/sample_and_export.py scripts/evaluate.py tests/test_cli_pipeline.py
git commit -m "feat: add CLI scripts for sampling, trajectory export, and evaluation"
```

---

### Task 11: Interactive 3D Three.js Web Viewer

**Files:**
- Create: `viewer/index.html`
- Create: `viewer/style.css`
- Create: `viewer/app.js`

- [ ] **Step 1: Create `viewer/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>3D Creature Flow Matching Viewer</title>
  <link rel="stylesheet" href="style.css">
  <!-- Three.js and OrbitControls via CDN -->
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
  <div id="canvas-container"></div>
  
  <div id="ui-overlay">
    <div class="glass-panel">
      <h1>3D Creature Flow Matching</h1>
      <p class="subtitle">Skeleton-Conditioned Optimal Transport Flow</p>
      
      <div class="control-group">
        <label for="creature-select">Specimen:</label>
        <select id="creature-select"></select>
      </div>

      <div class="control-group">
        <div class="slider-header">
          <span>Flow Time (t)</span>
          <span id="time-val">0.00</span>
        </div>
        <input type="range" id="time-slider" min="0" max="1" step="0.01" value="0">
      </div>

      <div class="btn-group">
        <button id="play-btn">▶ Play Flow</button>
        <button id="reset-cam-btn">Reset View</button>
        <button id="toggle-skeleton-btn">Toggle Skeleton</button>
      </div>

      <div class="stats-panel">
        <div><strong>Points:</strong> <span id="point-count">2048</span></div>
        <div><strong>Joints:</strong> 35 (SMAL Tree)</div>
        <div><strong>Status:</strong> <span id="status-label">Ready</span></div>
      </div>
    </div>
  </div>

  <script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Create `viewer/style.css`**

```css
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body, html {
  width: 100%;
  height: 100%;
  overflow: hidden;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  background-color: #0b0f19;
  color: #e2e8f0;
}

#canvas-container {
  width: 100%;
  height: 100%;
  position: absolute;
  top: 0;
  left: 0;
}

#ui-overlay {
  position: absolute;
  top: 24px;
  left: 24px;
  z-index: 10;
  pointer-events: none;
}

.glass-panel {
  pointer-events: auto;
  background: rgba(15, 23, 42, 0.75);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 16px;
  padding: 24px;
  width: 340px;
  box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
}

h1 {
  font-size: 1.25rem;
  font-weight: 700;
  color: #f8fafc;
  letter-spacing: -0.02em;
}

.subtitle {
  font-size: 0.8rem;
  color: #94a3b8;
  margin-bottom: 20px;
}

.control-group {
  margin-bottom: 16px;
}

label {
  display: block;
  font-size: 0.85rem;
  margin-bottom: 6px;
  color: #cbd5e1;
}

select {
  width: 100%;
  padding: 8px 12px;
  border-radius: 8px;
  background: rgba(30, 41, 59, 0.9);
  border: 1px solid rgba(255, 255, 255, 0.15);
  color: #f8fafc;
  outline: none;
  font-size: 0.85rem;
  cursor: pointer;
}

.slider-header {
  display: flex;
  justify-content: space-between;
  font-size: 0.85rem;
  margin-bottom: 6px;
  color: #cbd5e1;
}

input[type="range"] {
  width: 100%;
  accent-color: #38bdf8;
  cursor: pointer;
}

.btn-group {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-bottom: 16px;
}

button {
  padding: 9px 14px;
  border-radius: 8px;
  border: 1px solid rgba(255, 255, 255, 0.12);
  background: #1e293b;
  color: #f8fafc;
  font-size: 0.85rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
}

button:hover {
  background: #334155;
  border-color: #38bdf8;
}

#play-btn {
  background: #0284c7;
  border: none;
}

#play-btn:hover {
  background: #0369a1;
}

.stats-panel {
  border-top: 1px solid rgba(255, 255, 255, 0.1);
  padding-top: 12px;
  font-size: 0.75rem;
  color: #94a3b8;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
```

- [ ] **Step 3: Create `viewer/app.js`**

```javascript
// viewer/app.js
let scene, camera, renderer, controls;
let pointCloud, skeletonGroup;
let creaturesData = [];
let currentCreature = null;
let isPlaying = false;
let playProgress = 0.0;

function init() {
  const container = document.getElementById("canvas-container");
  
  // Scene
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0f1d);
  scene.fog = new THREE.FogExp2(0x0a0f1d, 0.25);

  // Camera
  camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
  camera.position.set(2.0, 1.5, 2.5);

  // Renderer
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  container.appendChild(renderer.domElement);

  // Controls
  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.target.set(0, 0, 0);

  // Lights & Grid
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
  scene.add(ambientLight);
  const dirLight = new THREE.DirectionalLight(0x38bdf8, 0.8);
  dirLight.position.set(5, 10, 5);
  scene.add(dirLight);

  const grid = new THREE.GridHelper(4, 20, 0x1e293b, 0x0f172a);
  grid.position.y = -0.6;
  scene.add(grid);

  skeletonGroup = new THREE.Group();
  scene.add(skeletonGroup);

  setupUI();
  loadData();
  animate();

  window.addEventListener("resize", onWindowResize);
}

function setupUI() {
  const slider = document.getElementById("time-slider");
  const timeVal = document.getElementById("time-val");
  const playBtn = document.getElementById("play-btn");
  const select = document.getElementById("creature-select");
  const skelBtn = document.getElementById("toggle-skeleton-btn");
  const resetBtn = document.getElementById("reset-cam-btn");

  slider.addEventListener("input", (e) => {
    const val = parseFloat(e.target.value);
    timeVal.textContent = val.toFixed(2);
    updatePointCloud(val);
  });

  playBtn.addEventListener("click", () => {
    isPlaying = !isPlaying;
    playBtn.textContent = isPlaying ? "⏸ Pause Flow" : "▶ Play Flow";
  });

  select.addEventListener("change", (e) => {
    const idx = parseInt(e.target.value);
    loadCreature(idx);
  });

  skelBtn.addEventListener("click", () => {
    skeletonGroup.visible = !skeletonGroup.visible;
  });

  resetBtn.addEventListener("click", () => {
    camera.position.set(2.0, 1.5, 2.5);
    controls.target.set(0, 0, 0);
  });
}

function loadData() {
  const select = document.getElementById("creature-select");
  const statusLabel = document.getElementById("status-label");

  fetch("data/creatures.json")
    .then((res) => {
      if (!res.ok) throw new Error("creatures.json not found");
      return res.json();
    })
    .then((data) => {
      creaturesData = data.creatures;
      select.innerHTML = "";
      creaturesData.forEach((c, i) => {
        const opt = document.createElement("option");
        opt.value = i;
        opt.textContent = c.name;
        select.appendChild(opt);
      });
      loadCreature(0);
      statusLabel.textContent = "Data Loaded";
    })
    .catch((err) => {
      console.warn("Using procedural fallback:", err);
      statusLabel.textContent = "Offline Mode (Run sample_and_export.py)";
    });
}

function loadCreature(idx) {
  if (!creaturesData[idx]) return;
  currentCreature = creaturesData[idx];
  
  // Rebuild Skeleton
  while (skeletonGroup.children.length > 0) {
    skeletonGroup.remove(skeletonGroup.children[0]);
  }

  const joints = currentCreature.joints;
  const parents = currentCreature.joint_parents;
  const jointGeo = new THREE.SphereGeometry(0.015, 12, 12);
  const jointMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8 });
  const boneMat = new THREE.LineBasicMaterial({ color: 0x0284c7, transparent: true, opacity: 0.6 });

  joints.forEach((j) => {
    const m = new THREE.Mesh(jointGeo, jointMat);
    m.position.set(j[0], j[1], j[2]);
    skeletonGroup.add(m);
  });

  for (let c = 1; c < joints.length; c++) {
    const p = parents[c];
    if (p >= 0) {
      const geo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(joints[p][0], joints[p][1], joints[p][2]),
        new THREE.Vector3(joints[c][0], joints[c][1], joints[c][2])
      ]);
      const line = new THREE.Line(geo, boneMat);
      skeletonGroup.add(line);
    }
  }

  updatePointCloud(parseFloat(document.getElementById("time-slider").value));
}

function updatePointCloud(tVal) {
  if (!currentCreature || !currentCreature.trajectory) return;
  const traj = currentCreature.trajectory;
  const numSteps = traj.length - 1;
  
  const stepFloat = tVal * numSteps;
  const stepIdx = Math.min(numSteps - 1, Math.floor(stepFloat));
  const alpha = stepFloat - stepIdx;
  
  const ptsA = traj[stepIdx];
  const ptsB = traj[Math.min(numSteps, stepIdx + 1)];
  const numPoints = ptsA.length;
  
  document.getElementById("point-count").textContent = numPoints;

  const positions = new Float32Array(numPoints * 3);
  const colors = new Float32Array(numPoints * 3);

  for (let i = 0; i < numPoints; i++) {
    const x = (1 - alpha) * ptsA[i][0] + alpha * ptsB[i][0];
    const y = (1 - alpha) * ptsA[i][1] + alpha * ptsB[i][1];
    const z = (1 - alpha) * ptsA[i][2] + alpha * ptsB[i][2];

    positions[i * 3] = x;
    positions[i * 3 + 1] = y;
    positions[i * 3 + 2] = z;

    // Gradient based on elevation and flow progression
    colors[i * 3] = 0.2 + 0.6 * (y + 0.5);
    colors[i * 3 + 1] = 0.5 + 0.4 * tVal;
    colors[i * 3 + 2] = 0.9 - 0.3 * (y + 0.5);
  }

  if (pointCloud) scene.remove(pointCloud);

  const pGeo = new THREE.BufferGeometry();
  pGeo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  pGeo.setAttribute("color", new THREE.BufferAttribute(colors, 3));

  const pMat = new THREE.PointsMaterial({
    size: 0.016,
    vertexColors: true,
    transparent: true,
    opacity: 0.88
  });

  pointCloud = new THREE.Points(pGeo, pMat);
  scene.add(pointCloud);
}

function onWindowResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}

function animate() {
  requestAnimationFrame(animate);

  if (isPlaying) {
    playProgress += 0.005;
    if (playProgress > 1.0) playProgress = 0.0;
    document.getElementById("time-slider").value = playProgress;
    document.getElementById("time-val").textContent = playProgress.toFixed(2);
    updatePointCloud(playProgress);
  }

  controls.update();
  renderer.render(scene, camera);
}

window.onload = init;
```

- [ ] **Step 4: Verify files exist and commit**

```bash
git add viewer/index.html viewer/style.css viewer/app.js
git commit -m "feat: implement interactive Three.js 3D web viewer with timeline flow controls"
```
