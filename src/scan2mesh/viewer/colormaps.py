"""Color generation for point cloud visualization."""

from __future__ import annotations

import numpy as np
import open3d as o3d

# Distinct colors for per-scan coloring (up to 12 scans before cycling)
SCAN_PALETTE = np.array([
    [0.90, 0.30, 0.25],  # red
    [0.30, 0.70, 0.35],  # green
    [0.25, 0.50, 0.90],  # blue
    [0.95, 0.70, 0.20],  # amber
    [0.70, 0.30, 0.85],  # purple
    [0.20, 0.80, 0.80],  # cyan
    [0.95, 0.50, 0.30],  # orange
    [0.60, 0.80, 0.30],  # lime
    [0.85, 0.35, 0.70],  # pink
    [0.40, 0.65, 0.55],  # teal
    [0.75, 0.60, 0.40],  # tan
    [0.55, 0.55, 0.80],  # lavender
], dtype=np.float64)


def height_colormap(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    """Generate a blue-to-red height (Z) colormap. Returns Nx3 float64."""
    points = np.asarray(pcd.points)
    z = points[:, 2]
    z_min, z_max = z.min(), z.max()

    if z_max - z_min < 1e-10:
        return np.full((len(points), 3), 0.5)

    t = (z - z_min) / (z_max - z_min)

    colors = np.zeros((len(points), 3))
    colors[:, 0] = t          # red increases with height
    colors[:, 1] = 1.0 - np.abs(t - 0.5) * 2  # green peaks at mid-height
    colors[:, 2] = 1.0 - t    # blue decreases with height
    return colors


def uniform_color(pcd: o3d.geometry.PointCloud, rgb: tuple[float, float, float] = (0.7, 0.7, 0.7)) -> np.ndarray:
    """Generate a uniform color array. Returns Nx3 float64."""
    n = len(pcd.points)
    return np.tile(np.array(rgb, dtype=np.float64), (n, 1))


def scan_colors(point_counts: list[int]) -> np.ndarray:
    """Generate per-point colors where each scan gets a distinct color.

    Args:
        point_counts: Number of points in each scan.

    Returns:
        Nx3 color array for all points concatenated.
    """
    all_colors = []
    for i, count in enumerate(point_counts):
        color = SCAN_PALETTE[i % len(SCAN_PALETTE)]
        all_colors.append(np.tile(color, (count, 1)))
    return np.concatenate(all_colors)
