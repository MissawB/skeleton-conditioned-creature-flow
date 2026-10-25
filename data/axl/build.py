"""
Turns Articulation-XL row groups into compact training shards: one .npz per row group of the mirror, so an
interrupted build resumes where it stopped.
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, Optional
import numpy as np
from data.axl.normalize import canonical_rotation, normalize
from data.axl.remote_parquet import open_parquet
from data.axl.sampling import point_skin, sample_surface, vertex_skin
from data.axl.selection import group_of, load_tags
from data.axl.skeleton import parents_from_bones

MAX_JOINTS = 255   # skin indices are stored as uint8
SKIN_K = 4
TABLE_COLUMNS = {
    "models": ["uuid", "root_index"],
    "skeletons": ["uuid", "joints", "bones", "joint_names"],
    "meshes": ["uuid", "vertices", "faces"],
    "skins": ["uuid", "row", "col", "value", "n_rows", "n_cols"],
}

def process_model(row: dict, num_points: int, rng: np.random.Generator) -> dict:
    """One joined mirror row -> normalised, quantised creature. Raises ValueError(reason) to reject it."""
    joints = np.asarray(row["joints"], dtype=np.float64).reshape(-1, 3)
    num_joints = len(joints)
    if num_joints > MAX_JOINTS:
        raise ValueError("too many joints")
    if num_joints < 2:
        raise ValueError("single joint")
    if not np.isfinite(joints).all():
        raise ValueError("non-finite joints")
    parents = parents_from_bones(np.asarray(row["bones"]), int(row["root_index"]), num_joints)

    vertices = np.asarray(row["vertices"], dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(row["faces"], dtype=np.int64).reshape(-1, 3)
    points, face_idx, bary = sample_surface(vertices, faces, num_points, rng)
    skin = vertex_skin(row["row"], row["col"], row["value"], row["n_rows"], row["n_cols"], vertices, joints, parents)
    skin_idx, skin_w = point_skin(skin, faces, face_idx, bary, top_k=SKIN_K)

    names = list(row["joint_names"])
    R, orient = canonical_rotation(vertices, joints, parents, names)
    points, joints, scale = normalize(points, joints, R)

    quantised = np.round(skin_w * 255).astype(np.uint8)
    return {
        "uuid": row["uuid"],
        "points": points.astype(np.float16),
        "skin_idx": skin_idx,
        "skin_w": quantised,
        "joints": joints.astype(np.float32),
        "parents": parents.astype(np.int16),
        "names": names,
        "orient": orient,
        "scale": scale,
    }

def _read_row_group(split: str, index: int, open_fn: Callable) -> list:
    tables = {name: open_fn(f"{split}/{name}.parquet").read_row_group(index, columns=cols).to_pylist()
              for name, cols in TABLE_COLUMNS.items()}
    uuids = [r["uuid"] for r in tables["models"]]
    for name, rows in tables.items():
        if [r["uuid"] for r in rows] != uuids:
            raise RuntimeError(f"row group {index}: {name} is not aligned with models")
    joined = []
    for parts in zip(*tables.values()):
        row = {}
        for part in parts:
            row.update(part)
        joined.append(row)
    return joined

def _write_shard(path: Path, models: list, groups: list) -> None:
    tmp = path.with_name(path.stem + ".tmp.npz")
    point_counts = [len(m["points"]) for m in models]
    joint_counts = [len(m["joints"]) for m in models]
    np.savez(
        tmp,
        points=np.concatenate([m["points"] for m in models]) if models else np.zeros((0, 3), np.float16),
        skin_idx=np.concatenate([m["skin_idx"] for m in models]) if models else np.zeros((0, SKIN_K), np.uint8),
        skin_w=np.concatenate([m["skin_w"] for m in models]) if models else np.zeros((0, SKIN_K), np.uint8),
        point_offsets=np.concatenate([[0], np.cumsum(point_counts)]).astype(np.int64),
        joints=np.concatenate([m["joints"] for m in models]) if models else np.zeros((0, 3), np.float32),
        parents=np.concatenate([m["parents"] for m in models]) if models else np.zeros(0, np.int16),
        joint_offsets=np.concatenate([[0], np.cumsum(joint_counts)]).astype(np.int64),
        uuids=np.array([m["uuid"] for m in models], dtype=str),
        groups=np.array(groups, dtype=str),
        orient=np.array([m["orient"] for m in models], dtype=str),
        scale=np.array([m["scale"] for m in models], dtype=np.float32),
        names=np.array(json.dumps([m["names"] for m in models])),
    )
    os.replace(tmp, path)

def build_row_group(split: str, index: int, out_dir: Path, num_points: int, tags: dict, open_fn: Callable = open_parquet, seed: int = 0) -> tuple:
    """Builds one shard; returns (kept, rejected) counts."""
    shard = Path(out_dir) / split / f"rg{index:04d}.npz"
    rng = np.random.default_rng([seed, index])
    kept, groups, rejects = [], [], {}
    for row in _read_row_group(split, index, open_fn):
        group = group_of(tags.get(row["uuid"], set()))
        if group is None:
            rejects[row["uuid"]] = "excluded category"
            continue
        try:
            kept.append(process_model(row, num_points, rng))
            groups.append(group)
        except ValueError as error:
            rejects[row["uuid"]] = str(error)
    shard.with_name(shard.stem + ".rejects.json").write_text(json.dumps(rejects, indent=0))
    _write_shard(shard, kept, groups)
    return len(kept), len(rejects)

def build_split(split: str, out_dir: Path, num_points: int = 8192, row_groups: Optional[Iterable[int]] = None,
                open_fn: Callable = open_parquet, workers: int = 1, log: Callable = lambda message: None) -> None:
    """One shard per row group under out_dir/split; existing shards are skipped, leftovers of a crash are replaced."""
    out_dir = Path(out_dir)
    (out_dir / split).mkdir(parents=True, exist_ok=True)
    tags = load_tags(open_fn("meta_categories.parquet").read(columns=["uuid", "category"]))
    total = open_fn(f"{split}/models.parquet").metadata.num_row_groups
    todo = [i for i in (row_groups if row_groups is not None else range(total))
            if not (out_dir / split / f"rg{i:04d}.npz").exists()]
    if workers <= 1:
        for i in todo:
            log(f"rg{i:04d}: kept {{}} rejected {{}}".format(*build_row_group(split, i, out_dir, num_points, tags, open_fn)))
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {i: pool.submit(build_row_group, split, i, out_dir, num_points, tags, open_fn) for i in todo}
        for i, future in futures.items():
            log(f"rg{i:04d}: kept {{}} rejected {{}}".format(*future.result()))
