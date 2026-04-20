"""Point cloud reader registry. All readers produce open3d.geometry.PointCloud."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import open3d as o3d

ReaderFn = Callable[[str, Optional[Callable[[str, float], None]]], o3d.geometry.PointCloud]

READER_REGISTRY: dict[str, ReaderFn] = {}


def register(extensions: list[str]):
    """Decorator to register a reader function for one or more file extensions."""
    def decorator(fn: ReaderFn) -> ReaderFn:
        for ext in extensions:
            READER_REGISTRY[ext.lower()] = fn
        return fn
    return decorator


def read_point_cloud(
    path: str | Path,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> o3d.geometry.PointCloud:
    """Read a point cloud file, dispatching to the appropriate format reader."""
    path = Path(path)
    ext = path.suffix.lower()
    reader = READER_REGISTRY.get(ext)
    if reader is None:
        supported = ", ".join(sorted(READER_REGISTRY.keys()))
        raise ValueError(f"Unsupported format: '{ext}'. Supported: {supported}")
    return reader(str(path), progress_callback=progress_callback)


# Import readers to trigger registration
from scan2mesh.readers import las, open3d_formats, ptx, pts  # noqa: E402, F401

# E57 is optional — only register if pye57 is available
try:
    from scan2mesh.readers import e57  # noqa: E402, F401
except ImportError:
    pass
