"""Reader for Leica PTX scan files.

PTX format: multiple scan blocks, each with:
  - columns (int)
  - rows (int)
  - scanner position (3 floats on one line)
  - scanner axes (3 lines of 3 floats)
  - 4x4 transform matrix (4 lines of 4 floats)
  - rows*cols lines of: x y z intensity [r g b]

Points with x==y==z==0 are invalid scan returns and are filtered out.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import open3d as o3d

from scan2mesh.readers import register


def _parse_ptx_blocks(path: str) -> list[tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """Parse all scan blocks from a PTX file.

    Returns list of (points_Nx3, transform_4x4, colors_Nx3_or_None) tuples.
    """
    blocks = []

    with open(path, "r") as f:
        while True:
            # Read columns and rows
            line = f.readline()
            if not line:
                break
            try:
                cols = int(line.strip())
            except ValueError:
                break
            rows = int(f.readline().strip())

            # Scanner position (1 line)
            f.readline()
            # Scanner axes (3 lines)
            for _ in range(3):
                f.readline()
            # Transform matrix (4 lines)
            transform = np.zeros((4, 4), dtype=np.float64)
            for r in range(4):
                transform[r] = [float(v) for v in f.readline().split()]

            # Read point data
            num_points = rows * cols
            raw_data = []
            for _ in range(num_points):
                raw_data.append(f.readline().split())

            if not raw_data:
                continue

            # Determine if colors are present (7 columns = x y z intensity r g b)
            num_cols = len(raw_data[0])
            data = np.array(raw_data, dtype=np.float64)

            points = data[:, :3]
            colors = None
            if num_cols >= 7:
                colors = data[:, 4:7] / 255.0  # r g b as uint8 → float

            blocks.append((points, transform, colors))

    return blocks


@register([".ptx"])
def read_ptx(
    path: str,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> o3d.geometry.PointCloud:
    """Read a PTX file, applying transforms and filtering invalid points."""
    if progress_callback:
        progress_callback("Parsing PTX file", 0.0)

    blocks = _parse_ptx_blocks(path)

    all_points = []
    all_colors = []

    for i, (points, transform, colors) in enumerate(blocks):
        if progress_callback:
            progress_callback(f"Processing scan block {i + 1}/{len(blocks)}", i / len(blocks))

        # Filter invalid points (0, 0, 0)
        valid = ~np.all(points == 0, axis=1)
        points = points[valid]

        # Apply 4x4 transform (skip if identity)
        if not np.allclose(transform, np.eye(4)):
            ones = np.ones((len(points), 1), dtype=np.float64)
            pts_h = np.hstack((points, ones))
            points = (transform @ pts_h.T).T[:, :3]

        all_points.append(points)

        if colors is not None:
            all_colors.append(colors[valid])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.concatenate(all_points))

    if all_colors and len(all_colors) == len(blocks):
        pcd.colors = o3d.utility.Vector3dVector(np.concatenate(all_colors))

    if len(pcd.points) == 0:
        raise ValueError(f"No valid points loaded from {path}")

    if progress_callback:
        progress_callback("Read complete", 1.0)

    return pcd
