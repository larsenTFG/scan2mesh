"""Preprocessing: outlier removal and voxel downsampling."""

from __future__ import annotations

import logging

import numpy as np
import open3d as o3d

from scan2mesh.config import PipelineConfig

logger = logging.getLogger(__name__)


def estimate_point_spacing(
    pcd: o3d.geometry.PointCloud,
    sample_size: int = 10000,
    k: int = 6,
) -> float:
    """Estimate local point spacing as the median distance to the k-th neighbor.

    Using k>1 is important for **anisotropic scan data** (scanner points are
    ~1mm apart along a scanline but 3-5mm apart between scanlines). The 1-NN
    distance only measures along-scanline spacing, which underestimates the
    gap BPA must bridge. k≈6 covers a full local patch including perpendicular
    neighbors, giving a spacing estimate that better reflects the true surface
    sampling.

    For large clouds, queries from a random subset but against a KDTree on
    the *full* cloud so distances reflect true neighborhoods, not the
    artificially sparser sampled set.
    """
    n = len(pcd.points)
    if n == 0:
        return 0.0

    points = np.asarray(pcd.points)
    tree = o3d.geometry.KDTreeFlann(pcd)

    if n > sample_size:
        rng = np.random.default_rng(42)
        indices = rng.choice(n, sample_size, replace=False)
    else:
        indices = np.arange(n)

    # Average distance to nearest k neighbors (excluding the point itself).
    # Using mean over k values smooths over anisotropic neighborhoods.
    distances = np.empty(len(indices), dtype=np.float64)
    for i, idx in enumerate(indices):
        _, _, sq_dists = tree.search_knn_vector_3d(points[idx], k + 1)
        if len(sq_dists) >= 2:
            nn_dists = np.sqrt(np.asarray(sq_dists[1:]))
            distances[i] = float(np.mean(nn_dists))
        else:
            distances[i] = 0.0

    distances = distances[distances > 0]
    if len(distances) == 0:
        return 0.0
    return float(np.median(distances))


def remove_outliers(pcd: o3d.geometry.PointCloud, config: PipelineConfig) -> o3d.geometry.PointCloud:
    """Remove outliers from the point cloud.

    Uses radius-based removal when point spacing is known (much faster than
    statistical on large clouds — no per-point std computation). Falls back to
    statistical removal otherwise.
    """
    if config.skip_outlier_removal:
        logger.info("Skipping outlier removal (disabled)")
        return pcd

    original_count = len(pcd.points)

    if config.point_spacing and config.point_spacing > 0:
        # Radius outlier: a point needs >=N neighbors within `radius` to survive.
        # Scale radius off the *post-downsample* spacing — voxel_size when we
        # downsampled, otherwise native point_spacing. Using native spacing
        # after downsampling would kill everything.
        effective_spacing = config.voxel_size if config.voxel_size else config.point_spacing
        radius = effective_spacing * 3.0
        min_neighbors = max(4, config.outlier_nb_neighbors // 4)
        logger.info(
            f"Removing outliers from {original_count:,} points "
            f"(radius={radius:.4f}, min_neighbors={min_neighbors})..."
        )
        pcd, _ = pcd.remove_radius_outlier(nb_points=min_neighbors, radius=radius)
    else:
        logger.info(f"Removing outliers from {original_count:,} points (statistical)...")
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=config.outlier_nb_neighbors,
            std_ratio=config.outlier_std_ratio,
        )

    removed = original_count - len(pcd.points)
    logger.info(f"Outlier removal: {original_count:,} → {len(pcd.points):,} ({removed:,} removed)")
    return pcd


def voxel_downsample(pcd: o3d.geometry.PointCloud, config: PipelineConfig) -> o3d.geometry.PointCloud:
    """Downsample using a voxel grid. Skips if voxel_size is None (precision/high)."""
    if config.voxel_size is None:
        logger.info("Skipping voxel downsampling (no voxel size set)")
        return pcd

    original_count = len(pcd.points)
    pcd = pcd.voxel_down_sample(voxel_size=config.voxel_size)
    logger.info(f"Voxel downsample (size={config.voxel_size:.4f}): {original_count} → {len(pcd.points)}")
    return pcd
