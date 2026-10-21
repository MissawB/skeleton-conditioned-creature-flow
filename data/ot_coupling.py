"""
Per-shape optimal transport coupling between Gaussian noise and surface points.
Pairing each noise point with a surface point (instead of index i with index i) removes the
permutation ambiguity of independent coupling and yields straighter flow trajectories.
"""
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def ot_match_noise(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draws N(0, I) noise and reorders it so noise[i] is the exact OT partner of points[i]."""
    noise = rng.standard_normal(points.shape).astype(np.float32)
    noise_idx, point_idx = linear_sum_assignment(cdist(noise, points, "sqeuclidean"))
    matched = np.empty_like(noise)
    matched[point_idx] = noise[noise_idx]
    return matched

def _match_many(args) -> np.ndarray:
    points, num_noise, seed = args
    rng = np.random.default_rng(seed)
    return np.stack([ot_match_noise(points, rng) for _ in range(num_noise)])

def precompute_ot_noise(
    all_points: List[np.ndarray],
    num_noise: int,
    seed: int = 0,
    num_workers: Optional[int] = None
) -> List[np.ndarray]:
    """Returns, for each shape, a (num_noise, N, 3) stack of OT-matched noise. Exact matching costs ~2 s per 2048-point shape."""
    jobs = [(points, num_noise, seed + i) for i, points in enumerate(all_points)]
    if num_workers == 1 or len(jobs) <= 1:
        return [_match_many(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        return list(pool.map(_match_many, jobs))
