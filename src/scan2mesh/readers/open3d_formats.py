"""Reader for formats natively supported by Open3D: PLY, PCD, XYZ, XYZN, XYZRGB."""

from __future__ import annotations

from typing import Callable, Optional

import open3d as o3d

from scan2mesh.readers import register


@register([".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb"])
def read_open3d(
    path: str,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> o3d.geometry.PointCloud:
    """Read point cloud using Open3D's native I/O."""
    if progress_callback:
        progress_callback("Reading point cloud", 0.0)
    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) == 0:
        raise ValueError(f"No points loaded from {path}")
    if progress_callback:
        progress_callback("Read complete", 1.0)
    return pcd
