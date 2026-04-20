"""STL mesh writer via trimesh."""

from __future__ import annotations

import logging

import numpy as np
import open3d as o3d
import trimesh

from scan2mesh.writers import register

logger = logging.getLogger(__name__)


@register([".stl"])
def write_stl(mesh: o3d.geometry.TriangleMesh, path: str) -> None:
    """Export mesh as binary STL. Note: STL does not support vertex colors."""
    t_mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles),
    )
    t_mesh.export(path, file_type="stl")
    logger.info(f"Exported STL: {path}")
