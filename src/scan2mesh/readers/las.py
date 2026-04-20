"""Reader for LAS/LAZ lidar point cloud files via laspy."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import open3d as o3d

from scan2mesh.readers import register


@register([".las", ".laz"])
def read_las(
    path: str,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> o3d.geometry.PointCloud:
    """Read a LAS/LAZ file and return an Open3D PointCloud.

    Colors are normalized from uint16 (0-65535) to float64 (0-1).
    Normals are not typically present in LAS files.
    """
    import laspy

    if progress_callback:
        progress_callback("Reading LAS/LAZ file", 0.0)

    las = laspy.read(path)
    points = np.vstack((las.x, las.y, las.z)).T

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Extract colors if available
    try:
        red = np.asarray(las.red, dtype=np.float64)
        green = np.asarray(las.green, dtype=np.float64)
        blue = np.asarray(las.blue, dtype=np.float64)
        # LAS stores colors as uint16 (0-65535)
        max_val = 65535.0
        colors = np.vstack((red / max_val, green / max_val, blue / max_val)).T
        pcd.colors = o3d.utility.Vector3dVector(colors)
    except (AttributeError, LookupError):
        pass  # No color fields in this file

    if len(pcd.points) == 0:
        raise ValueError(f"No points loaded from {path}")

    if progress_callback:
        progress_callback("Read complete", 1.0)

    return pcd
