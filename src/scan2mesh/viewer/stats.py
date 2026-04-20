"""Point cloud statistics computation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d


@dataclass
class PointCloudStats:
    point_count: int
    has_colors: bool
    has_normals: bool
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    extent: np.ndarray
    centroid: np.ndarray
    density_estimate: float | None

    def format(self) -> str:
        lines = [
            f"Points:   {self.point_count:,}",
            f"Colors:   {'yes' if self.has_colors else 'no'}",
            f"Normals:  {'yes' if self.has_normals else 'no'}",
            f"Bounds X: [{self.bounds_min[0]:.2f}, {self.bounds_max[0]:.2f}]",
            f"Bounds Y: [{self.bounds_min[1]:.2f}, {self.bounds_max[1]:.2f}]",
            f"Bounds Z: [{self.bounds_min[2]:.2f}, {self.bounds_max[2]:.2f}]",
            f"Extent:   {self.extent[0]:.2f} x {self.extent[1]:.2f} x {self.extent[2]:.2f}",
        ]
        if self.density_estimate is not None:
            lines.append(f"Density:  ~{self.density_estimate:.0f} pts/m³")
        return "\n".join(lines)


def compute_stats(pcd: o3d.geometry.PointCloud) -> PointCloudStats:
    points = np.asarray(pcd.points)
    n = len(points)

    if n == 0:
        zero = np.zeros(3)
        return PointCloudStats(0, False, False, zero, zero, zero, zero, None)

    bounds_min = points.min(axis=0)
    bounds_max = points.max(axis=0)
    extent = bounds_max - bounds_min

    volume = float(np.prod(extent[extent > 0])) if np.all(extent >= 0) else None
    density = n / volume if volume and volume > 0 else None

    return PointCloudStats(
        point_count=n,
        has_colors=pcd.has_colors(),
        has_normals=pcd.has_normals(),
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        extent=extent,
        centroid=points.mean(axis=0),
        density_estimate=density,
    )
