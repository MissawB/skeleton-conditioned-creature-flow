import numpy as np
import pytest
from data.axl.sampling import sample_surface, vertex_skin, point_skin


def test_sample_surface_is_area_weighted():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 0, 0], [8, 0, 0], [5, 2, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2], [3, 4, 5]])  # areas 0.5 and 3.0
    points, face_idx, bary = sample_surface(vertices, faces, 20000, np.random.default_rng(0))
    assert abs((face_idx == 1).mean() - 3.0 / 3.5) < 0.02
    assert np.abs(points[:, 2]).max() < 1e-6
    assert np.allclose(bary.sum(1), 1, atol=1e-5) and (bary >= -1e-6).all()
    assert np.allclose(points, np.einsum("nk,nkd->nd", bary, vertices[faces[face_idx]]), atol=1e-5)


@pytest.mark.parametrize("vertices,faces", [
    (np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], float), np.array([[0, 1, 2]])),
    (np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float), np.array([[0, 1, 99]])),
    (np.array([[0, 0, 0], [1, 0, 0], [0, np.nan, 0]], float), np.array([[0, 1, 2]])),
])
def test_sample_surface_rejects_degenerate_meshes(vertices, faces):
    with pytest.raises(ValueError):
        sample_surface(vertices, faces, 10, np.random.default_rng(0))


def test_skin_weights_are_renormalised_and_unweighted_vertices_get_nearest_bone():
    joints = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], float)
    parents = np.array([-1, 0, 1])
    vertices = np.array([[0.2, 0.1, 0], [1.8, 0.1, 0], [1.5, 0, 0]], float)
    row, col, value = np.array([0, 0, 2]), np.array([0, 1, 2]), np.array([0.3, 0.3, 0.5])
    skin = vertex_skin(row, col, value, 3, 3, vertices, joints, parents)
    assert skin.dtype == np.float32 and skin.shape == (3, 3)
    assert np.allclose(skin[0], [0.5, 0.5, 0])
    assert np.allclose(skin[2], [0, 0, 1])
    assert np.allclose(skin[1], [0, 1, 0])  # nearest bone 1->2 belongs to its head joint 1


def test_point_skin_interpolates_and_keeps_top_k():
    skin = np.eye(3, dtype=np.float32)
    faces = np.array([[0, 1, 2]])
    idx, w = point_skin(skin, faces, np.array([0]), np.array([[0.5, 0.3, 0.2]], np.float32), top_k=2)
    assert idx.dtype == np.uint8 and idx[0].tolist() == [0, 1]
    assert np.allclose(w[0], [0.625, 0.375])
