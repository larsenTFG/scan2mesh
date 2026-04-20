"""glTF/GLB mesh writer via trimesh."""

from __future__ import annotations

import logging

import numpy as np
import open3d as o3d
import trimesh

from scan2mesh.writers import register

logger = logging.getLogger(__name__)


@register([".gltf", ".glb"])
def write_gltf(mesh: o3d.geometry.TriangleMesh, path: str) -> None:
    """Export mesh as glTF or GLB with vertex colors."""
    vertex_colors = None
    if mesh.has_vertex_colors():
        vertex_colors = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)
        alpha = np.full((len(vertex_colors), 1), 255, dtype=np.uint8)
        vertex_colors = np.hstack((vertex_colors, alpha))

    t_mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles),
        vertex_colors=vertex_colors,
    )

    file_type = "glb" if path.lower().endswith(".glb") else "gltf"
    t_mesh.export(path, file_type=file_type)
    logger.info(f"Exported {file_type.upper()}: {path}")
