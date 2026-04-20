"""Preprocessing: outlier removal and voxel downsampling."""

from __future__ import annotations

import logging

import open3d as o3d

from scan2mesh.config import PipelineConfig

logger = logging.getLogger(__name__)


def remove_outliers(pcd: o3d.geometry.PointCloud, config: PipelineConfig) -> o3d.geometry.PointCloud:
    """Remove statistical outliers from the point cloud."""
    if config.skip_outlier_removal:
        logger.info("Skipping outlier removal (disabled)")
        return pcd

    original_count = len(pcd.points)
    logger.info(f"Removing outliers from {original_count:,} points (this may take a moment)...")
    pcd, mask = pcd.remove_statistical_outlier(
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
