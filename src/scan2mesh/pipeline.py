"""Main conversion pipeline: point cloud → mesh."""

from __future__ import annotations

import concurrent.futures
import logging
import warnings
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import open3d as o3d

from scan2mesh.config import Mode, PipelineConfig
from scan2mesh.processing import preprocess, normals, reconstruct, postprocess
from scan2mesh.progress import ProgressReporter
from scan2mesh.readers import read_point_cloud
from scan2mesh.writers import write_mesh

logger = logging.getLogger(__name__)

LARGE_POINT_CLOUD_THRESHOLD = 50_000_000


def _run_interruptible(fn, *args, **kwargs):
    """Run a blocking function in a thread so Ctrl+C is not swallowed.

    Open3D's C++ calls hold the GIL and block signal delivery.
    Running them in a thread lets the main thread handle KeyboardInterrupt.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn, *args, **kwargs)
        # future.result() with timeout allows the main thread to wake
        # periodically and check for signals (KeyboardInterrupt).
        while True:
            try:
                return future.result(timeout=0.5)
            except concurrent.futures.TimeoutError:
                continue


def convert(
    input_path: str,
    output_path: str,
    config: PipelineConfig | None = None,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    scan_indices: list[int] | None = None,
) -> None:
    """Full point-cloud-to-mesh conversion pipeline.

    Args:
        input_path: Path to input point cloud file.
        output_path: Path for output mesh file.
        config: Pipeline configuration. Uses defaults if None.
        progress_callback: Optional callback(message, progress_0_to_1).
        scan_indices: For E57 files, which scans to include. None means all.
    """
    if config is None:
        config = PipelineConfig()

    progress = ProgressReporter(progress_callback)

    # Stage 1: Read
    progress.report("read", "Reading point cloud", 0.0)
    if scan_indices is not None and Path(input_path).suffix.lower() == ".e57":
        try:
            from scan2mesh.readers.e57 import read_e57
            pcd = read_e57(input_path, scan_indices=scan_indices)
        except ImportError:
            raise ValueError("pye57 is required for E57 support. Install with: pip install scan2mesh[e57]")
    else:
        pcd = read_point_cloud(input_path)
    point_count = len(pcd.points)
    logger.info(f"Loaded {point_count} points from {input_path}")

    if point_count > LARGE_POINT_CLOUD_THRESHOLD:
        warnings.warn(
            f"Large point cloud ({point_count:,} points). "
            f"Consider using --quality low for faster processing.",
            stacklevel=2,
        )

    progress.report("read", "Read complete", 1.0)

    # Resolve config with actual point cloud geometry
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox_diagonal = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
    config = config.resolve(point_count, bbox_diagonal)

    # Stage 2: Preprocess (downsample first — outlier removal is much faster on fewer points)
    progress.report("preprocess", "Downsampling", 0.0)
    pcd = _run_interruptible(preprocess.voxel_downsample, pcd, config)
    progress.report("preprocess", f"Removing outliers ({len(pcd.points):,} points)", 0.3)
    pcd = _run_interruptible(preprocess.remove_outliers, pcd, config)
    progress.report("preprocess", "Preprocessing complete", 1.0)

    # Stage 3: Normal estimation
    progress.report("normals", "Estimating normals", 0.0)
    pcd = _run_interruptible(normals.estimate_normals, pcd, config)
    progress.report("normals", "Normals ready", 1.0)

    # Stage 4: Surface reconstruction
    progress.report("reconstruct", "Reconstructing surface", 0.0)
    mesh = _run_interruptible(reconstruct.reconstruct, pcd, config)
    progress.report("reconstruct", "Reconstruction complete", 1.0)

    # Stage 5: Post-process
    progress.report("postprocess", "Post-processing", 0.0)

    # Decimation
    if config.target_faces and len(mesh.triangles) > config.target_faces:
        mesh = _run_interruptible(postprocess.decimate, mesh, config.target_faces)
    elif config.max_deviation_mm is not None:
        mesh = _run_interruptible(postprocess.decimate_by_deviation, mesh, pcd, config.max_deviation_mm)

    # Color transfer
    if config.transfer_colors and pcd.has_colors():
        mesh = _run_interruptible(postprocess.transfer_vertex_colors, mesh, pcd)

    progress.report("postprocess", "Post-processing complete", 1.0)

    # Stage 6: Export
    progress.report("export", "Exporting mesh", 0.0)
    write_mesh(mesh, output_path)
    progress.report("export", "Export complete", 1.0)

    logger.info(
        f"Conversion complete: {len(mesh.vertices)} vertices, "
        f"{len(mesh.triangles)} triangles → {output_path}"
    )
