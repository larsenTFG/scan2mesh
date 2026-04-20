"""Surface reconstruction: Poisson and Ball Pivoting."""

from __future__ import annotations

import logging

import numpy as np
import open3d as o3d

from scan2mesh.config import PipelineConfig, ReconstructionMethod

logger = logging.getLogger(__name__)


def reconstruct(pcd: o3d.geometry.PointCloud, config: PipelineConfig) -> o3d.geometry.TriangleMesh:
    """Run surface reconstruction using the configured method."""
    if config.method == ReconstructionMethod.POISSON:
        return _poisson(pcd, config)
    elif config.method == ReconstructionMethod.BALL_PIVOTING:
        return _ball_pivoting(pcd, config)
    else:
        raise ValueError(f"Unknown reconstruction method: {config.method}")


def _poisson(pcd: o3d.geometry.PointCloud, config: PipelineConfig) -> o3d.geometry.TriangleMesh:
    """Poisson surface reconstruction.

    Produces smooth, watertight meshes. Good for visualization.
    Density-based trimming removes low-support boundary artifacts.
    """
    logger.info(f"Running Poisson reconstruction (depth={config.poisson_depth})")

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=config.poisson_depth
    )

    # Trim low-density faces (Poisson boundary artifacts)
    densities = np.asarray(densities)
    threshold = np.quantile(densities, config.poisson_density_quantile)
    vertices_to_remove = densities < threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)

    logger.info(
        f"Poisson complete: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles "
        f"(trimmed {np.sum(vertices_to_remove)} low-density vertices)"
    )

    return mesh


def _ball_pivoting(pcd: o3d.geometry.PointCloud, config: PipelineConfig) -> o3d.geometry.TriangleMesh:
    """Ball Pivoting Algorithm.

    Only creates triangles where points exist — no hallucinated geometry.
    Good for precision mode. May leave holes in sparse regions.
    """
    if config.ball_pivot_radii is not None:
        radii = config.ball_pivot_radii
    else:
        # Auto-compute radii from point spacing: 1x, 2x, 4x average NN distance
        nn_dist = np.mean(pcd.compute_nearest_neighbor_distance())
        radii = [nn_dist * m for m in (1.0, 2.0, 4.0)]

    logger.info(f"Running Ball Pivoting (radii={[f'{r:.4f}' for r in radii]})")

    radii_vec = o3d.utility.DoubleVector(radii)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii_vec)

    logger.info(
        f"Ball Pivoting complete: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles"
    )

    return mesh
