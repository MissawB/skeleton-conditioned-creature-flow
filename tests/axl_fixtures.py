"""Synthetic Articulation-XL-like tables: a skinned prism mesh around each bone of a skeleton."""
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def prism_model(joints, parents, uuid, radius=0.03):
    vertices, faces, rows, cols = [], [], [], []
    for c, p in enumerate(parents):
        if p < 0:
            continue
        a, b = joints[p], joints[c]
        axis = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        u = np.cross(axis, [0, 0, 1] if abs(axis[2]) < 0.9 else [1, 0, 0])
        u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        ring = [radius * (np.cos(t) * u + np.sin(t) * v) for t in (0, 2.1, 4.2)]
        base = len(vertices)
        vertices += [a + r for r in ring] + [b + r for r in ring]
        faces += [[base, base + 1, base + 2], [base + 3, base + 5, base + 4]]
        for k in range(3):
            n = (k + 1) % 3
            faces += [[base + k, base + 3 + k, base + 3 + n], [base + k, base + 3 + n, base + n]]
        rows += list(range(base, base + 6))
        cols += [p] * 6  # a joint owns the segment towards its children
    bones = [[p, c] for c, p in enumerate(parents) if p >= 0]
    return {
        "uuid": uuid, "root_index": int(np.where(np.asarray(parents) == -1)[0][0]),
        "joints": np.asarray(joints, float).tolist(), "bones": bones,
        "joint_names": [f"j{i}" for i in range(len(joints))],
        "vertices": np.asarray(vertices).tolist(), "faces": faces,
        "row": rows, "col": cols, "value": [1.0] * len(rows),
        "n_rows": len(vertices), "n_cols": len(joints),
    }


def write_mirror(root, split, models, tags, rows_per_group=2):
    """Writes models/skeletons/meshes/skins tables plus meta_categories.parquet under root."""
    (root / split).mkdir(parents=True, exist_ok=True)
    tables = {
        "models": ["uuid", "root_index"],
        "skeletons": ["uuid", "joints", "bones", "joint_names"],
        "meshes": ["uuid", "vertices", "faces"],
        "skins": ["uuid", "row", "col", "value", "n_rows", "n_cols"],
    }
    for name, columns in tables.items():
        table = pa.table({c: [m[c] for m in models] for c in columns})
        pq.write_table(table, root / split / f"{name}.parquet", row_group_size=rows_per_group)
    uuids = [u for u, ts in tags.items() for _ in ts]
    cats = [t for ts in tags.values() for t in ts]
    pq.write_table(pa.table({"uuid": uuids, "ordinal": list(range(len(cats))), "category": cats}), root / "meta_categories.parquet")
