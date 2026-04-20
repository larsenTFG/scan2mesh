"""Normal estimation and orientation."""

from __future__ import annotations

import logging

import numpy as np
import open3d as o3d

from scan2mesh.config import PipelineConfig, ReconstructionMethod

logger = logging.getLogger(__name__)


def estimate_normals(pcd: o3d.geometry.PointCloud, config: PipelineConfig) -> o3d.geometry.PointCloud:
    """Estimate normals if not present, then orient them.

    Uses hybrid search (radius + max_nn) scaled to point spacing.
    Orientation strategy depends on reconstruction method:
      - Poisson needs globally consistent orientation (expensive MST over the cloud).
      - Ball Pivoting uses normals locally, so a cheap viewpoint-based flip is enough.
    """
    if pcd.has_normals():
        logger.info("Normals already present, skipping estimation")
        return pcd

    # Reuse precomputed spacing instead of running NN over the whole cloud again.
    spacing = config.point_spacing
    if spacing is None or spacing <= 0:
        spacing = float(np.mean(pcd.compute_nearest_neighbor_distance()))
    search_radius = spacing * 3.0

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

    if config.method == ReconstructionMethod.BALL_PIVOTING:
        # BPA uses normals locally — a cheap viewpoint flip is sufficient and
        # avoids the O(N log N) MST in orient_normals_consistent_tangent_plane,
        # which is the dominant cost on large clouds.
        bbox = pcd.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        # Flip outward from the scene center — good heuristic for scanner data
        # where the surface generally faces away from the interior.
        pcd.orient_normals_towards_camera_location(camera_location=center)
        # Flip again so normals point outward (the call above orients *toward* center)
        normals_arr = np.asarray(pcd.normals)
        normals_arr *= -1.0
        pcd.normals = o3d.utility.Vector3dVector(normals_arr)
        logger.info("Oriented normals via viewpoint flip (fast path for Ball Pivoting)")
    else:
        # Poisson requires globally consistent orientation to produce a clean surface.
        logger.info("Orienting normals consistently (Poisson requires global consistency)")
        pcd.orient_normals_consistent_tangent_plane(k=min(config.normal_neighbors, 15))

    return pcd
