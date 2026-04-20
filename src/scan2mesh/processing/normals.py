"""Normal estimation and orientation."""

from __future__ import annotations

import logging

import numpy as np
import open3d as o3d

from scan2mesh.config import PipelineConfig

logger = logging.getLogger(__name__)


def estimate_normals(pcd: o3d.geometry.PointCloud, config: PipelineConfig) -> o3d.geometry.PointCloud:
    """Estimate normals if not present, then orient them consistently.

    Uses a hybrid search (radius + max_nn) scaled to the point cloud's density.
    Precision mode uses more neighbors for smoother, more accurate normals.
    """
    if pcd.has_normals():
        logger.info("Normals already present, skipping estimation")
        return pcd

    # Estimate search radius from average nearest-neighbor distance
    nn_dist = np.mean(pcd.compute_nearest_neighbor_distance())
    search_radius = nn_dist * 3.0

    logger.info(
        f"Estimating normals (neighbors={config.normal_neighbors}, "
        f"radius={search_radius:.4f})"
    )

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=search_radius,
            max_nn=config.normal_neighbors,
        )
    )

    # Orient normals consistently — critical for Poisson reconstruction
    logger.info("Orienting normals consistently")
    pcd.orient_normals_consistent_tangent_plane(k=min(config.normal_neighbors, 15))

    return pcd
