"""Post-processing: decimation and vertex color transfer."""

from __future__ import annotations

import logging

import numpy as np
import open3d as o3d

from scan2mesh.config import PipelineConfig

logger = logging.getLogger(__name__)


def decimate(mesh: o3d.geometry.TriangleMesh, target_faces: int) -> o3d.geometry.TriangleMesh:
    """Decimate mesh to a target triangle count using quadric decimation."""
    original = len(mesh.triangles)
    if original <= target_faces:
        logger.info(f"Mesh has {original} triangles, already below target {target_faces}")
        return mesh

    logger.info(f"Decimating: {original} → {target_faces} triangles")
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
    logger.info(f"Decimation result: {len(mesh.triangles)} triangles")
    return mesh


def decimate_by_deviation(
    mesh: o3d.geometry.TriangleMesh,
    reference_pcd: o3d.geometry.PointCloud,
    max_deviation_mm: float,
) -> o3d.geometry.TriangleMesh:
    """Decimate mesh while keeping geometric deviation within a threshold.

    Uses binary search on face count to find the lowest count that stays
    within the max deviation tolerance. Deviation is measured as the max
    distance from original point cloud points to the decimated mesh surface.
    """
    original_faces = len(mesh.triangles)
    if original_faces == 0:
        return mesh

    ref_points = np.asarray(reference_pcd.points)

    def check_deviation(candidate_mesh: o3d.geometry.TriangleMesh) -> float:
        """Compute max point-to-mesh distance in mm (assuming m units)."""
        scene = o3d.t.geometry.RaycastingScene()
        candidate_t = o3d.t.geometry.TriangleMesh.from_legacy(candidate_mesh)
        scene.add_triangles(candidate_t)
        distances = scene.compute_distance(
            o3d.core.Tensor(ref_points.astype(np.float32))
        )
        return float(distances.numpy().max()) * 1000.0  # m → mm

    # Binary search on face count
    low = max(original_faces // 100, 100)
    high = original_faces
    best_mesh = mesh

    logger.info(
        f"Deviation-aware decimation: max {max_deviation_mm}mm, "
        f"searching in [{low}, {high}] faces"
    )

    for _ in range(20):  # max iterations
        if high - low < max(original_faces // 200, 50):
            break
        mid = (low + high) // 2
        candidate = mesh.simplify_quadric_decimation(target_number_of_triangles=mid)
        dev = check_deviation(candidate)
        if dev <= max_deviation_mm:
            best_mesh = candidate
            high = mid
        else:
            low = mid

    logger.info(
        f"Deviation decimation: {original_faces} → {len(best_mesh.triangles)} triangles"
    )
    return best_mesh


def transfer_vertex_colors(
    mesh: o3d.geometry.TriangleMesh,
    source_pcd: o3d.geometry.PointCloud,
) -> o3d.geometry.TriangleMesh:
    """Transfer colors from source point cloud to mesh vertices via nearest-neighbor.

    Poisson reconstruction doesn't preserve point colors — this step
    copies the color of the nearest source point to each mesh vertex.
    """
    if not source_pcd.has_colors():
        logger.info("Source point cloud has no colors, skipping color transfer")
        return mesh

    logger.info(f"Transferring colors to {len(mesh.vertices)} mesh vertices")

    pcd_tree = o3d.geometry.KDTreeFlann(source_pcd)
    source_colors = np.asarray(source_pcd.colors)
    mesh_vertices = np.asarray(mesh.vertices)

    vertex_colors = np.zeros_like(mesh_vertices)
    for i, vertex in enumerate(mesh_vertices):
        _, idx, _ = pcd_tree.search_knn_vector_3d(vertex, 1)
        vertex_colors[i] = source_colors[idx[0]]

    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    logger.info("Color transfer complete")
    return mesh
