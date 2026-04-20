"""Mesh writer registry. All writers accept open3d.geometry.TriangleMesh."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import open3d as o3d

WriterFn = Callable[[o3d.geometry.TriangleMesh, str], None]

WRITER_REGISTRY: dict[str, WriterFn] = {}


def register(extensions: list[str]):
    """Decorator to register a writer function for one or more file extensions."""
    def decorator(fn: WriterFn) -> WriterFn:
        for ext in extensions:
            WRITER_REGISTRY[ext.lower()] = fn
        return fn
    return decorator


def write_mesh(mesh: o3d.geometry.TriangleMesh, path: str | Path) -> None:
    """Write a mesh to file, dispatching to the appropriate format writer."""
    path = Path(path)
    ext = path.suffix.lower()
    writer = WRITER_REGISTRY.get(ext)
    if writer is None:
        supported = ", ".join(sorted(WRITER_REGISTRY.keys()))
        raise ValueError(f"Unsupported output format: '{ext}'. Supported: {supported}")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer(mesh, str(path))


# Import writers to trigger registration
from scan2mesh.writers import obj, gltf, stl  # noqa: E402, F401
