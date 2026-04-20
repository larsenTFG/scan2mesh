"""Reader for PTS point cloud files.

PTS format:
  - First line: point count (integer)
  - Subsequent lines: x y z intensity [r g b]
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import open3d as o3d

from scan2mesh.readers import register


@register([".pts"])
def read_pts(
    path: str,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> o3d.geometry.PointCloud:
    """Read a PTS file and return an Open3D PointCloud."""
    if progress_callback:
        progress_callback("Reading PTS file", 0.0)

    with open(path, "r") as f:
        # First line is point count
        num_points = int(f.readline().strip())

        # Read all data lines
        data = np.loadtxt(f, dtype=np.float64)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data[:, :3])

    # Colors are in columns 4-6 (0-indexed) if present (x y z intensity r g b)
    if data.shape[1] >= 7:
        colors = data[:, 4:7]
        # Normalize to [0, 1] if values are > 1 (uint8 range)
        if colors.max() > 1.0:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    if len(pcd.points) == 0:
        raise ValueError(f"No points loaded from {path}")

    if progress_callback:
        progress_callback("Read complete", 1.0)

    return pcd
