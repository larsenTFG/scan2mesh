"""Reader for E57 point cloud files via pye57.

This module is imported conditionally — if pye57 is not installed,
the reader is simply not registered and E57 files will produce
a clear error message.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import open3d as o3d
import pye57

from scan2mesh.readers import register


@dataclass
class ScanInfo:
    """Metadata for a single scan within an E57 file."""

    index: int
    name: str
    point_count: int
    has_color: bool
    has_normals: bool
    has_intensity: bool
    bounds_min: tuple[float, float, float] | None
    bounds_max: tuple[float, float, float] | None

    @property
    def bounds_str(self) -> str:
        if self.bounds_min is None or self.bounds_max is None:
            return "unknown"
        mn, mx = self.bounds_min, self.bounds_max
        return (
            f"X[{mn[0]:.2f}, {mx[0]:.2f}] "
            f"Y[{mn[1]:.2f}, {mx[1]:.2f}] "
            f"Z[{mn[2]:.2f}, {mx[2]:.2f}]"
        )


def get_scan_info(path: str) -> list[ScanInfo]:
    """Read scan headers from an E57 file without loading point data."""
    e57_file = pye57.E57(path)
    scans: list[ScanInfo] = []

    for i in range(e57_file.scan_count):
        header = e57_file.get_header(i)

        point_count = header.point_count
        name = getattr(header, "name", None) or getattr(header, "description", None) or f"scan_{i}"

        fields = set(header.point_fields)
        has_color = "colorRed" in fields
        has_normals = "nor:normalX" in fields
        has_intensity = "intensity" in fields

        # Bounds from header if available, otherwise None (would need full read)
        bounds_min = None
        bounds_max = None
        x_min = getattr(header, "x_minimum", None)
        x_max = getattr(header, "x_maximum", None)
        y_min = getattr(header, "y_minimum", None)
        y_max = getattr(header, "y_maximum", None)
        z_min = getattr(header, "z_minimum", None)
        z_max = getattr(header, "z_maximum", None)
        if all(v is not None for v in (x_min, x_max, y_min, y_max, z_min, z_max)):
            bounds_min = (x_min, y_min, z_min)
            bounds_max = (x_max, y_max, z_max)

        scans.append(ScanInfo(
            index=i,
            name=str(name),
            point_count=point_count,
            has_color=has_color,
            has_normals=has_normals,
            has_intensity=has_intensity,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
        ))

    return scans


def _read_single_scan(e57_file: pye57.E57, scan_index: int) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Read one scan and return (points, colors_or_none, normals_or_none)."""
    data = e57_file.read_scan_raw(scan_index)

    x = np.asarray(data["cartesianX"], dtype=np.float64)
    y = np.asarray(data["cartesianY"], dtype=np.float64)
    z = np.asarray(data["cartesianZ"], dtype=np.float64)
    points = np.column_stack((x, y, z))

    colors = None
    if "colorRed" in data:
        r = np.asarray(data["colorRed"], dtype=np.float64) / 255.0
        g = np.asarray(data["colorGreen"], dtype=np.float64) / 255.0
        b = np.asarray(data["colorBlue"], dtype=np.float64) / 255.0
        colors = np.column_stack((r, g, b))

    normals = None
    if "nor:normalX" in data:
        nx = np.asarray(data["nor:normalX"], dtype=np.float64)
        ny = np.asarray(data["nor:normalY"], dtype=np.float64)
        nz = np.asarray(data["nor:normalZ"], dtype=np.float64)
        normals = np.column_stack((nx, ny, nz))

    return points, colors, normals


def read_e57_scan(
    path: str,
    scan_index: int,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> o3d.geometry.PointCloud:
    """Read a single scan from an E57 file as a PointCloud."""
    e57_file = pye57.E57(path)
    if scan_index < 0 or scan_index >= e57_file.scan_count:
        raise ValueError(
            f"Scan index {scan_index} out of range (file has {e57_file.scan_count} scans)"
        )

    if progress_callback:
        progress_callback(f"Reading scan {scan_index}", 0.0)

    points, colors, normals = _read_single_scan(e57_file, scan_index)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)

    if progress_callback:
        progress_callback("Read complete", 1.0)

    return pcd


def read_e57_scans(
    path: str,
    scan_indices: list[int],
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> list[o3d.geometry.PointCloud]:
    """Read multiple scans from an E57 file, returning each as a separate PointCloud."""
    e57_file = pye57.E57(path)
    num_scans = e57_file.scan_count

    for idx in scan_indices:
        if idx < 0 or idx >= num_scans:
            raise ValueError(f"Scan index {idx} out of range (file has {num_scans} scans)")

    results = []
    for step, i in enumerate(scan_indices):
        if progress_callback:
            progress_callback(f"Reading scan {i} ({step + 1}/{len(scan_indices)})", step / len(scan_indices))

        points, colors, normals = _read_single_scan(e57_file, i)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals)
        results.append(pcd)

    if progress_callback:
        progress_callback("Read complete", 1.0)

    return results


@register([".e57"])
def read_e57(
    path: str,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    scan_indices: list[int] | None = None,
) -> o3d.geometry.PointCloud:
    """Read an E57 file, concatenating selected scans into a single PointCloud.

    If scan_indices is None, all scans are loaded.
    """
    if progress_callback:
        progress_callback("Reading E57 file", 0.0)

    e57_file = pye57.E57(path)
    num_scans = e57_file.scan_count

    if scan_indices is None:
        indices = list(range(num_scans))
    else:
        for idx in scan_indices:
            if idx < 0 or idx >= num_scans:
                raise ValueError(f"Scan index {idx} out of range (file has {num_scans} scans)")
        indices = scan_indices

    all_points = []
    all_colors = []
    all_normals = []

    for step, i in enumerate(indices):
        if progress_callback:
            progress_callback(f"Reading scan {i + 1}/{num_scans}", step / len(indices))

        points, colors, normals = _read_single_scan(e57_file, i)
        all_points.append(points)
        if colors is not None:
            all_colors.append(colors)
        if normals is not None:
            all_normals.append(normals)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.concatenate(all_points))

    if all_colors and len(all_colors) == len(indices):
        pcd.colors = o3d.utility.Vector3dVector(np.concatenate(all_colors))
    if all_normals and len(all_normals) == len(indices):
        pcd.normals = o3d.utility.Vector3dVector(np.concatenate(all_normals))

    if len(pcd.points) == 0:
        raise ValueError(f"No points loaded from {path}")

    if progress_callback:
        progress_callback("Read complete", 1.0)

    return pcd
