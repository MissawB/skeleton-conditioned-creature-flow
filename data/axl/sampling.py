"""Area-weighted surface sampling of a skinned mesh, with skin weights carried to each sampled point."""
import numpy as np
import scipy.sparse as sp

def sample_surface(vertices: np.ndarray, faces: np.ndarray, num_points: int, rng: np.random.Generator):
    """Returns (points (N,3) float32, face_idx (N,) int64, bary (N,3) float32), faces drawn in proportion to their area."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if not np.isfinite(vertices).all():
        raise ValueError("non-finite vertices")
    if faces.size == 0 or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("face index out of range")
    tri = vertices[faces]
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    total = areas.sum()
    if not total > 0:
        raise ValueError("zero surface area")

    face_idx = rng.choice(len(faces), size=num_points, p=areas / total)
    r1, r2 = rng.random(num_points), rng.random(num_points)
    sqrt_r1 = np.sqrt(r1)
    bary = np.stack([1 - sqrt_r1, sqrt_r1 * (1 - r2), sqrt_r1 * r2], axis=1)
    points = np.einsum("nk,nkd->nd", bary, tri[face_idx])
    return points.astype(np.float32), face_idx, bary.astype(np.float32)

def _segment_distances(points: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """(P, B) distance from each point to each segment."""
    ab = ends - starts
    t = np.einsum("pbd,bd->pb", points[:, None] - starts[None], ab) / np.maximum((ab * ab).sum(1), 1e-12)
    closest = starts[None] + np.clip(t, 0, 1)[..., None] * ab[None]
    return np.linalg.norm(points[:, None] - closest, axis=2)

def vertex_skin(row, col, value, n_rows: int, n_cols: int, vertices: np.ndarray, joints: np.ndarray, parents: np.ndarray) -> sp.csr_matrix:
    """
    Sparse (V, J) skin matrix from the mirror's COO triplets (row = vertex, col = joint), renormalised to sum to 1.
    Sparse because a dense matrix of a multi-million-vertex mesh would take hundreds of MB.
    A joint's weights cover the segment from that joint to its children, so a vertex with no weight goes to the
    head joint of the nearest bone.
    """
    skin = sp.csr_matrix((np.asarray(value, dtype=np.float32), (np.asarray(row), np.asarray(col))), shape=(n_rows, n_cols))
    sums = np.asarray(skin.sum(1)).ravel()
    weighted = sums > 0
    skin = sp.diags(np.where(weighted, 1.0 / np.where(weighted, sums, 1.0), 0.0).astype(np.float32)) @ skin

    orphans = np.where(~weighted)[0]
    if len(orphans):
        children = np.where(parents >= 0)[0]
        if len(children) == 0:
            owners = np.zeros(len(orphans), dtype=np.int64)
        else:
            heads = parents[children]
            orphan_vertices = np.asarray(vertices, dtype=np.float64)[orphans]
            owners = np.empty(len(orphans), dtype=np.int64)
            # Chunked so a mesh with many unweighted vertices stays within memory
            for start in range(0, len(orphans), 4096):
                dist = _segment_distances(orphan_vertices[start:start + 4096], joints[heads], joints[children])
                owners[start:start + 4096] = heads[dist.argmin(1)]
        skin = skin + sp.csr_matrix((np.ones(len(orphans), np.float32), (orphans, owners)), shape=(n_rows, n_cols))
    return skin.tocsr().astype(np.float32)

def point_skin(skin, faces: np.ndarray, face_idx: np.ndarray, bary: np.ndarray, top_k: int = 4):
    """Barycentric skin weights of sampled points, cut to the top_k joints and renormalised: (idx uint8 (N,k), w float32 (N,k)).
    skin may be dense or sparse; only the rows of the sampled faces' corners are densified."""
    corners = faces[face_idx]
    weights = np.zeros((len(face_idx), skin.shape[1]), dtype=np.float32)
    for c in range(3):
        rows = skin[corners[:, c]]
        weights += bary[:, c, None] * (rows.toarray() if sp.issparse(rows) else rows)
    k = min(top_k, weights.shape[1])
    idx = np.argsort(-weights, axis=1)[:, :k]
    w = np.take_along_axis(weights, idx, axis=1)
    w /= np.maximum(w.sum(1, keepdims=True), 1e-12)
    if k < top_k:
        idx = np.pad(idx, ((0, 0), (0, top_k - k)))
        w = np.pad(w, ((0, 0), (0, top_k - k)))
    return idx.astype(np.uint8), w.astype(np.float32)
