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
