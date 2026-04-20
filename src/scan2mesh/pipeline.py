"""Main conversion pipeline: point cloud → mesh."""

from __future__ import annotations

import logging
import math
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import open3d as o3d

from scan2mesh.config import Mode, PipelineConfig, ReconstructionMethod
from scan2mesh.processing import preprocess, normals, reconstruct, postprocess
from scan2mesh.progress import ProgressReporter
from scan2mesh.readers import read_point_cloud
from scan2mesh.writers import write_mesh

logger = logging.getLogger(__name__)

LARGE_POINT_CLOUD_THRESHOLD = 50_000_000
# Ball Pivoting becomes effectively hung above ~1M points on a single machine.
# When the resolved config would exceed this, we auto-bump voxel_spacing_multiple.
BPA_MAX_POINTS = 1_500_000


def _run_interruptible(fn, *args, heartbeat=None, **kwargs):
    """Run a blocking function in a daemon thread so Ctrl+C actually exits.

    Open3D's C++ calls hold the GIL and block signal delivery. Running them
    in a daemon thread lets the main thread handle KeyboardInterrupt, and
    because the thread is a daemon, the Python interpreter does not wait
    for it at shutdown — sys.exit() terminates the process immediately.

    (ThreadPoolExecutor cannot be used here: its context manager blocks on
    pool shutdown until running tasks complete, which defeats the point.)

    Args:
        heartbeat: Optional callable(elapsed_seconds) invoked ~5x/second while
            the worker is running. Useful for animating a progress bar during
            long C++ calls that emit no native progress signal.
    """
    result = {}

    def target():
        try:
            result["value"] = fn(*args, **kwargs)
        except BaseException as e:  # noqa: BLE001 — propagate everything
            result["error"] = e

    thread = threading.Thread(target=target, daemon=True)
    start = time.monotonic()
    thread.start()
    try:
        while thread.is_alive():
            thread.join(timeout=0.2)
            if heartbeat is not None and thread.is_alive():
                try:
                    heartbeat(time.monotonic() - start)
                except Exception:  # noqa: BLE001 — never let UI kill the pipeline
                    pass
    except KeyboardInterrupt:
        logger.warning("Interrupted by user — exiting.")
        # Daemon thread will be killed when the interpreter exits.
        sys.exit(130)

    if "error" in result:
        raise result["error"]
    return result["value"]


def _stage_heartbeat(progress: "ProgressReporter", stage: str, message: str, tau: float = 30.0):
    """Build a heartbeat callback that animates a stage's progress bar.

    Advances asymptotically: at t=tau reaches ~63%, at 3*tau ~95%. Never hits
    100% until the stage actually completes — that's reported explicitly by
    the caller. This keeps the bar visibly moving during long operations
    without pretending to know how long they'll take.
    """
    def beat(elapsed: float) -> None:
        frac = 1.0 - math.exp(-elapsed / tau)
        # Cap at 0.95 so the bar only reaches 100% when the stage really ends.
        progress.report(stage, f"{message} ({int(elapsed)}s)", min(frac, 0.95))
    return beat


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
    read_beat = _stage_heartbeat(progress, "read", "Reading point cloud", tau=15.0)
    if scan_indices is not None and Path(input_path).suffix.lower() == ".e57":
        try:
            from scan2mesh.readers.e57 import read_e57
            pcd = _run_interruptible(
                read_e57, input_path, scan_indices=scan_indices, heartbeat=read_beat
            )
        except ImportError:
            raise ValueError("pye57 is required for E57 support. Install with: pip install scan2mesh[e57]")
    else:
        pcd = _run_interruptible(read_point_cloud, input_path, heartbeat=read_beat)
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
    bbox_diagonal = float(np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound()))
    progress.report("read", "Estimating point spacing", 1.0)
    point_spacing = _run_interruptible(
        preprocess.estimate_point_spacing, pcd,
        heartbeat=_stage_heartbeat(progress, "read", "Estimating point spacing", tau=10.0),
    )
    logger.info(
        f"Scene geometry: bbox diagonal={bbox_diagonal:.3f}, "
        f"median point spacing={point_spacing:.4f}"
    )
    config = config.resolve(point_count, bbox_diagonal, point_spacing)
    logger.info(
        f"Resolved config: voxel_size={config.voxel_size}, "
        f"poisson_depth={config.poisson_depth}, method={config.method.value}"
    )

    # Stage 2: Preprocess (downsample first — outlier removal is much faster on fewer points)
    # Auto-cap BPA input size: even with a voxel downsample, a huge cloud can still
    # exceed BPA's practical limit. Estimate post-downsample count and bump the
    # voxel size if needed.
    if config.method == ReconstructionMethod.BALL_PIVOTING and config.voxel_size:
        # Points roughly scale inversely with voxel_size^2 for surface-like clouds.
        est_post = point_count * (point_spacing / config.voxel_size) ** 2 if config.voxel_size > point_spacing else point_count
        if est_post > BPA_MAX_POINTS:
            scale = (est_post / BPA_MAX_POINTS) ** 0.5
            new_voxel = config.voxel_size * scale
            logger.warning(
                f"Ball Pivoting input (~{int(est_post):,} pts) exceeds practical limit "
                f"({BPA_MAX_POINTS:,}); bumping voxel_size {config.voxel_size:.4f} → "
                f"{new_voxel:.4f} to keep runtime bounded."
            )
            config.voxel_size = new_voxel

    progress.report("preprocess", "Downsampling", 0.0)
    pcd = _run_interruptible(
        preprocess.voxel_downsample, pcd, config,
        heartbeat=_stage_heartbeat(progress, "preprocess", "Downsampling", tau=15.0),
    )
    progress.report("preprocess", f"Removing outliers ({len(pcd.points):,} points)", 0.3)
    pcd = _run_interruptible(
        preprocess.remove_outliers, pcd, config,
        heartbeat=_stage_heartbeat(progress, "preprocess", "Removing outliers", tau=30.0),
    )
    progress.report("preprocess", "Preprocessing complete", 1.0)

    # Stage 3: Normal estimation
    progress.report("normals", "Estimating normals", 0.0)
    pcd = _run_interruptible(
        normals.estimate_normals, pcd, config,
        heartbeat=_stage_heartbeat(progress, "normals", "Estimating normals", tau=30.0),
    )
    progress.report("normals", "Normals ready", 1.0)

    # Stage 4: Surface reconstruction — typically the longest stage.
    progress.report("reconstruct", "Reconstructing surface", 0.0)
    mesh = _run_interruptible(
        reconstruct.reconstruct, pcd, config,
        heartbeat=_stage_heartbeat(progress, "reconstruct", "Reconstructing surface", tau=60.0),
    )
    progress.report("reconstruct", "Reconstruction complete", 1.0)

    # Stage 5: Post-process
    progress.report("postprocess", "Post-processing", 0.0)

    # Decimation
    if config.target_faces and len(mesh.triangles) > config.target_faces:
        mesh = _run_interruptible(
            postprocess.decimate, mesh, config.target_faces,
            heartbeat=_stage_heartbeat(progress, "postprocess", "Decimating", tau=20.0),
        )
    elif config.max_deviation_mm is not None:
        mesh = _run_interruptible(
            postprocess.decimate_by_deviation, mesh, pcd, config.max_deviation_mm,
            heartbeat=_stage_heartbeat(progress, "postprocess", "Decimating (deviation)", tau=30.0),
        )

    # Color transfer
    if config.transfer_colors and pcd.has_colors():
        mesh = _run_interruptible(
            postprocess.transfer_vertex_colors, mesh, pcd,
            heartbeat=_stage_heartbeat(progress, "postprocess", "Transferring colors", tau=15.0),
        )

    progress.report("postprocess", "Post-processing complete", 1.0)

    # Stage 6: Export
    progress.report("export", "Exporting mesh", 0.0)
    _run_interruptible(
        write_mesh, mesh, output_path,
        heartbeat=_stage_heartbeat(progress, "export", "Exporting mesh", tau=10.0),
    )
    progress.report("export", "Export complete", 1.0)

    logger.info(
        f"Conversion complete: {len(mesh.vertices)} vertices, "
        f"{len(mesh.triangles)} triangles → {output_path}"
    )
