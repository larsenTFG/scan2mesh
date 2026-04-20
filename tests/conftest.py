"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest


@pytest.fixture
def sample_pcd() -> o3d.geometry.PointCloud:
    """A small synthetic point cloud (sphere) with colors."""
    # Generate points on a sphere
    n = 5000
    phi = np.random.uniform(0, 2 * np.pi, n)
    theta = np.random.uniform(0, np.pi, n)
    r = 1.0
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.column_stack((x, y, z)))

    # Color based on position (normalized to [0, 1])
    colors = np.column_stack(((x + 1) / 2, (y + 1) / 2, (z + 1) / 2))
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


@pytest.fixture
def sample_pcd_with_normals(sample_pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Sample point cloud with normals estimated."""
    sample_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    sample_pcd.orient_normals_consistent_tangent_plane(k=15)
    return sample_pcd


@pytest.fixture
def sample_mesh(sample_pcd_with_normals: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
    """A small synthetic mesh from Poisson reconstruction."""
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        sample_pcd_with_normals, depth=6
    )
    # Transfer colors
    pcd_tree = o3d.geometry.KDTreeFlann(sample_pcd_with_normals)
    source_colors = np.asarray(sample_pcd_with_normals.colors)
    mesh_vertices = np.asarray(mesh.vertices)
    vertex_colors = np.zeros_like(mesh_vertices)
    for i, v in enumerate(mesh_vertices):
        _, idx, _ = pcd_tree.search_knn_vector_3d(v, 1)
        vertex_colors[i] = source_colors[idx[0]]
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    return mesh


@pytest.fixture
def tmp_ply(tmp_path, sample_pcd: o3d.geometry.PointCloud) -> str:
    """Write sample point cloud to a temporary PLY file."""
    path = str(tmp_path / "test.ply")
    o3d.io.write_point_cloud(path, sample_pcd)
    return path


@pytest.fixture
def tmp_pts(tmp_path, sample_pcd: o3d.geometry.PointCloud) -> str:
    """Write sample point cloud to a temporary PTS file."""
    path = tmp_path / "test.pts"
    points = np.asarray(sample_pcd.points)
    colors = (np.asarray(sample_pcd.colors) * 255).astype(int)
    intensity = np.zeros(len(points), dtype=int)

    with open(path, "w") as f:
        f.write(f"{len(points)}\n")
        for i in range(len(points)):
            f.write(
                f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} "
                f"{intensity[i]} {colors[i, 0]} {colors[i, 1]} {colors[i, 2]}\n"
            )
    return str(path)
