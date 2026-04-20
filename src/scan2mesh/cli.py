"""Command-line interface for scan2mesh."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from scan2mesh.config import Mode, PipelineConfig, QualityPreset, ReconstructionMethod
from scan2mesh.pipeline import convert


class DefaultGroup(click.Group):
    """Click group that falls back to 'convert' when no subcommand matches.

    Preserves backward compatibility: `scan2mesh input.las output.glb` still works.
    """

    def parse_args(self, ctx, args):
        # If the first arg isn't a known subcommand, inject 'convert'
        if args and args[0] not in self.commands and not args[0].startswith("-"):
            args = ["convert"] + args
        return super().parse_args(ctx, args)


@click.group(cls=DefaultGroup)
@click.version_option(package_name="scan2mesh")
def main():
    """scan2mesh: Convert lidar point clouds to meshes.

    Run 'scan2mesh convert' for mesh conversion, or use the subcommands
    below to inspect and manipulate point cloud files.
    """


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.option(
    "--mode", "-m",
    type=click.Choice(["visualization", "precision"]),
    default="visualization",
    help="Conversion mode: visualization (smooth, colorful) or precision (geometrically faithful).",
)
@click.option(
    "--quality", "-q",
    type=click.Choice(["low", "medium", "high"]),
    default="medium",
    help="Quality preset within the selected mode.",
)
@click.option(
    "--method",
    type=click.Choice(["poisson", "ball_pivoting"]),
    default=None,
    help="Override reconstruction method.",
)
@click.option(
    "--target-faces", "-f",
    type=int,
    default=None,
    help="Target triangle count (overrides preset, visualization mode).",
)
@click.option(
    "--max-deviation",
    type=float,
    default=None,
    help="Max geometric deviation in mm (overrides preset, precision mode).",
)
@click.option(
    "--voxel-size",
    type=float,
    default=None,
    help="Voxel size for downsampling (overrides preset).",
)
@click.option(
    "--poisson-depth",
    type=int,
    default=None,
    help="Poisson octree depth (overrides preset).",
)
@click.option(
    "--color/--no-color",
    default=None,
    help="Enable or disable vertex color transfer (default: on for visualization, off for precision).",
)
@click.option(
    "--bpa-radii",
    type=str,
    default=None,
    help=(
        "Comma-separated Ball Pivoting radii as multiples of point spacing "
        "(e.g. '1,2,4,8' for more coverage across scan gaps). Default: '1,2,4'. "
        "Each extra radius is another BPA pass — more coverage, more time."
    ),
)
@click.option(
    "--no-outlier-removal",
    is_flag=True,
    default=False,
    help="Skip outlier removal (faster, but may produce noisy meshes).",
)
@click.option(
    "--scans",
    type=str,
    default=None,
    help="For E57 files: comma-separated scan indices to include (e.g. '0,2,3'). Default: all.",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Enable verbose logging.",
)
def convert_cmd(
    input_file: str,
    output_file: str,
    mode: str,
    quality: str,
    method: str | None,
    target_faces: int | None,
    max_deviation: float | None,
    voxel_size: float | None,
    poisson_depth: int | None,
    color: bool | None,
    bpa_radii: str | None,
    no_outlier_removal: bool,
    scans: str | None,
    verbose: bool,
) -> None:
    """Convert a point cloud file to a mesh.

    Supports input formats: LAS, LAZ, E57, PLY, PCD, XYZ, PTX, PTS.
    Supports output formats: OBJ, glTF, GLB, STL.

    Examples:

      scan2mesh convert input.las output.glb

      scan2mesh input.e57 output.obj --mode precision --quality high

      scan2mesh input.e57 output.glb --scans 0,1,2

      scan2mesh input.ptx output.glb -q low -f 100000
    """
    _setup_logging(verbose)

    config = PipelineConfig(
        mode=Mode(mode),
        quality=QualityPreset(quality),
    )

    if method is not None:
        config.method = ReconstructionMethod(method)
    if target_faces is not None:
        config.target_faces = target_faces
    if max_deviation is not None:
        config.max_deviation_mm = max_deviation
    if voxel_size is not None:
        config.voxel_size = voxel_size
    if poisson_depth is not None:
        config.poisson_depth = poisson_depth
    if color is not None:
        config.transfer_colors = color
    if bpa_radii is not None:
        try:
            config.bpa_radius_multiples = [float(x) for x in bpa_radii.split(",") if x.strip()]
        except ValueError:
            raise click.BadParameter("--bpa-radii must be comma-separated numbers, e.g. '1,2,4,8'")
        if not config.bpa_radius_multiples:
            raise click.BadParameter("--bpa-radii must contain at least one value")
    if no_outlier_removal:
        config.skip_outlier_removal = True

    scan_indices = _parse_scan_indices(scans)

    bar = None

    def progress_callback(message: str, pct: float) -> None:
        nonlocal bar
        if bar is None:
            bar = click.progressbar(length=100, label="Converting", file=sys.stderr)
            bar.__enter__()
        new_pos = int(pct * 100)
        delta = new_pos - bar.pos
        if delta > 0:
            bar.update(delta)
        if verbose:
            click.echo(f"  {message}", err=True)
        if pct >= 1.0 and bar is not None:
            bar.__exit__(None, None, None)

    try:
        convert(
            input_file,
            output_file,
            config,
            progress_callback=progress_callback,
            scan_indices=scan_indices,
        )
        click.echo(f"Done: {output_file}", err=True)
    except Exception as e:
        if bar is not None:
            bar.__exit__(None, None, None)
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except (KeyboardInterrupt, SystemExit):
        if bar is not None:
            bar.__exit__(None, None, None)
        click.echo("\nAborted.", err=True)
        raise


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
def inspect(input_file: str) -> None:
    """Inspect a point cloud file and show scan/layer metadata.

    For E57 files with multiple scans, displays a table showing each scan's
    point count, available fields, and bounding box.

    For other formats, shows basic point cloud statistics.

    Examples:

      scan2mesh inspect building.e57

      scan2mesh inspect cloud.ply
    """
    path = Path(input_file)

    if path.suffix.lower() == ".e57":
        _inspect_e57(input_file)
    else:
        _inspect_generic(input_file)


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "--format", "output_format",
    type=click.Choice(["ply", "pcd", "las"]),
    default="ply",
    help="Output format for split scan files.",
)
@click.option(
    "--scans",
    type=str,
    default=None,
    help="Comma-separated scan indices to export (e.g. '0,2,3'). Default: all.",
)
def split(input_file: str, output_dir: str, output_format: str, scans: str | None) -> None:
    """Split an E57 file into individual scan files.

    Exports each scan (or selected scans) as a separate point cloud file.

    Examples:

      scan2mesh split building.e57 ./scans/

      scan2mesh split building.e57 ./scans/ --format pcd --scans 0,2,3
    """
    path = Path(input_file)
    if path.suffix.lower() != ".e57":
        click.echo("Error: split currently only supports E57 files.", err=True)
        sys.exit(1)

    try:
        from scan2mesh.readers.e57 import get_scan_info, read_e57_scan
    except ImportError:
        click.echo("Error: pye57 is required for E57 support. Install with: pip install scan2mesh[e57]", err=True)
        sys.exit(1)

    scan_infos = get_scan_info(input_file)
    scan_indices = _parse_scan_indices(scans)

    if scan_indices is None:
        indices = [s.index for s in scan_infos]
    else:
        max_idx = len(scan_infos) - 1
        bad = [i for i in scan_indices if i > max_idx]
        if bad:
            click.echo(f"Error: scan indices {bad} out of range (file has {len(scan_infos)} scans).", err=True)
            sys.exit(1)
        indices = scan_indices

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = path.stem

    import open3d as o3d

    for i in indices:
        info = scan_infos[i]
        out_name = f"{stem}_scan_{i:03d}.{output_format}"
        out_path = out_dir / out_name

        click.echo(f"  Writing scan {i} → {out_path} ({info.point_count:,} points)")
        pcd = read_e57_scan(input_file, i)

        if output_format == "las":
            _write_pcd_as_las(pcd, str(out_path))
        else:
            o3d.io.write_point_cloud(str(out_path), pcd)

    click.echo(f"Done: {len(indices)} scan(s) written to {out_dir}", err=True)


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--voxel-size",
    type=float,
    default=None,
    help="Downsample for faster preview (voxel size in scene units).",
)
@click.option(
    "--background",
    type=click.Choice(["dark", "light"]),
    default="dark",
    help="Viewer background color.",
)
def view(input_file: str, voxel_size: float | None, background: str) -> None:
    """Open an interactive 3D viewer for a point cloud file.

    Supports all input formats: LAS, LAZ, E57, PLY, PCD, XYZ, PTX, PTS.
    For E57 files with multiple scans, use S to color by scan and 1-9 to
    isolate individual scans.

    Examples:

      scan2mesh view building.e57

      scan2mesh view cloud.ply --voxel-size 0.05

      scan2mesh view large_scan.las --background light
    """
    from scan2mesh.viewer.app import PointCloudViewer

    viewer = PointCloudViewer()

    try:
        viewer.load_file(
            input_file,
            voxel_size=voxel_size,
            progress_callback=lambda msg: click.echo(f"  {msg}", err=True),
        )
    except Exception as e:
        click.echo(f"Error loading file: {e}", err=True)
        sys.exit(1)

    title = f"scan2mesh — {Path(input_file).name}"
    viewer.run(title=title, background=background)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> None:
    log_level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )


def _parse_scan_indices(scans: str | None) -> list[int] | None:
    """Parse a comma-separated string of scan indices."""
    if scans is None:
        return None
    try:
        return [int(s.strip()) for s in scans.split(",") if s.strip()]
    except ValueError:
        click.echo(f"Error: invalid --scans value '{scans}'. Expected comma-separated integers.", err=True)
        sys.exit(1)


def _inspect_e57(input_file: str) -> None:
    try:
        from scan2mesh.readers.e57 import get_scan_info
    except ImportError:
        click.echo("Error: pye57 is required for E57 support. Install with: pip install scan2mesh[e57]", err=True)
        sys.exit(1)

    scan_infos = get_scan_info(input_file)
    total_points = sum(s.point_count for s in scan_infos)

    click.echo()
    click.echo(f"  {Path(input_file).name} — {len(scan_infos)} scan(s), {total_points:,} points total")
    click.echo()

    # Table header
    click.echo(f"  {'#':<4} {'Name':<24} {'Points':>14}   {'Color':<6} {'Normals':<8} {'Intensity':<10} {'Bounds'}")
    click.echo(f"  {'—'*4} {'—'*24} {'—'*14}   {'—'*6} {'—'*8} {'—'*10} {'—'*40}")

    for s in scan_infos:
        click.echo(
            f"  {s.index:<4} {s.name:<24} {s.point_count:>14,}   "
            f"{'yes' if s.has_color else 'no':<6} "
            f"{'yes' if s.has_normals else 'no':<8} "
            f"{'yes' if s.has_intensity else 'no':<10} "
            f"{s.bounds_str}"
        )

    click.echo()


def _inspect_generic(input_file: str) -> None:
    import numpy as np
    from scan2mesh.readers import read_point_cloud

    click.echo(f"  Loading {Path(input_file).name}...")
    pcd = read_point_cloud(input_file)
    points = np.asarray(pcd.points)
    n = len(points)

    click.echo()
    click.echo(f"  {Path(input_file).name} — 1 layer, {n:,} points")
    click.echo()
    click.echo(f"  Has colors:  {'yes' if pcd.has_colors() else 'no'}")
    click.echo(f"  Has normals: {'yes' if pcd.has_normals() else 'no'}")

    if n > 0:
        mn = points.min(axis=0)
        mx = points.max(axis=0)
        extent = mx - mn
        click.echo(f"  Bounds:      X[{mn[0]:.2f}, {mx[0]:.2f}] Y[{mn[1]:.2f}, {mx[1]:.2f}] Z[{mn[2]:.2f}, {mx[2]:.2f}]")
        click.echo(f"  Extent:      {extent[0]:.2f} × {extent[1]:.2f} × {extent[2]:.2f}")

    click.echo()


def _write_pcd_as_las(pcd, output_path: str) -> None:
    """Write an Open3D PointCloud to LAS format."""
    import numpy as np
    import laspy

    points = np.asarray(pcd.points)
    header = laspy.LasHeader(point_format=2, version="1.2")
    header.offsets = points.min(axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])

    las = laspy.LasData(header)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]

    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        las.red = (colors[:, 0] * 65535).astype(np.uint16)
        las.green = (colors[:, 1] * 65535).astype(np.uint16)
        las.blue = (colors[:, 2] * 65535).astype(np.uint16)

    las.write(output_path)


if __name__ == "__main__":
    main()
