"""Interactive Open3D point cloud viewer."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import open3d as o3d

from scan2mesh.viewer.colormaps import height_colormap, uniform_color, scan_colors
from scan2mesh.viewer.stats import PointCloudStats, compute_stats


class PointCloudViewer:
    """Interactive 3D viewer for point cloud inspection.

    Keyboard controls:
        H - Height (Z) colormap
        C - Original RGB colors (if available)
        U - Uniform gray
        B - Toggle bounding box
        N - Toggle normal vectors
        S - Toggle per-scan coloring (E57 multi-scan)
        1-9 - Show only scan N (E57 multi-scan), 0 to show all
        I - Print stats to console
        R - Reset view
    """

    MAX_DISPLAY_POINTS = 10_000_000

    def __init__(self) -> None:
        self._pcd: o3d.geometry.PointCloud | None = None
        self._original_colors: np.ndarray | None = None
        self._stats: PointCloudStats | None = None
        self._bbox_visible = False
        self._normals_visible = False
        self._scan_point_counts: list[int] | None = None
        self._scan_clouds: list[o3d.geometry.PointCloud] | None = None
        self._scan_visibility: list[bool] | None = None
        self._vis: o3d.visualization.VisualizerWithKeyCallback | None = None
        self._bbox_lineset: o3d.geometry.LineSet | None = None
        self._downsampled = False

    def load_file(
        self,
        path: str,
        voxel_size: float | None = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Load a point cloud from any supported format."""
        ext = Path(path).suffix.lower()

        if ext == ".e57":
            self._load_e57(path, progress_callback)
        else:
            self._load_generic(path, progress_callback)

        if self._pcd is None or len(self._pcd.points) == 0:
            raise ValueError(f"No points loaded from {path}")

        # Auto-downsample for display if needed
        n = len(self._pcd.points)
        if voxel_size is not None:
            self._pcd = self._pcd.voxel_down_sample(voxel_size)
            self._downsampled = True
        elif n > self.MAX_DISPLAY_POINTS:
            bbox = self._pcd.get_axis_aligned_bounding_box()
            diag = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
            auto_voxel = diag * 0.001
            while True:
                test = self._pcd.voxel_down_sample(auto_voxel)
                if len(test.points) <= self.MAX_DISPLAY_POINTS:
                    self._pcd = test
                    self._downsampled = True
                    break
                auto_voxel *= 1.5

        self._original_colors = (
            np.asarray(self._pcd.colors).copy() if self._pcd.has_colors() else None
        )
        self._stats = compute_stats(self._pcd)

        if progress_callback:
            label = f"Loaded {self._stats.point_count:,} points"
            if self._downsampled:
                label += " (downsampled for display)"
            progress_callback(label)

    def _load_e57(self, path: str, progress_callback: Optional[Callable[[str], None]] = None) -> None:
        try:
            from scan2mesh.readers.e57 import get_scan_info, read_e57_scans
        except ImportError:
            raise ImportError("pye57 is required for E57 support. Install with: pip install scan2mesh[e57]")

        if progress_callback:
            progress_callback("Reading E57 scan headers...")

        scan_infos = get_scan_info(path)
        indices = list(range(len(scan_infos)))

        if progress_callback:
            progress_callback(f"Loading {len(indices)} scan(s)...")

        clouds = read_e57_scans(path, indices)
        self._scan_clouds = clouds
        self._scan_point_counts = [len(c.points) for c in clouds]
        self._scan_visibility = [True] * len(clouds)

        self._pcd = _merge_clouds(clouds)

    def _load_generic(self, path: str, progress_callback: Optional[Callable[[str], None]] = None) -> None:
        from scan2mesh.readers import read_point_cloud

        if progress_callback:
            progress_callback(f"Loading {Path(path).name}...")

        self._pcd = read_point_cloud(path)

    def run(self, title: str = "scan2mesh viewer", background: str = "dark") -> None:
        """Open the interactive viewer window."""
        if self._pcd is None:
            raise RuntimeError("No point cloud loaded. Call load_file() first.")

        bg = [0.1, 0.1, 0.1] if background == "dark" else [1.0, 1.0, 1.0]

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name=title, width=1280, height=800)
        self._vis = vis

        opt = vis.get_render_option()
        opt.background_color = np.array(bg)
        opt.point_size = 2.0

        if not self._pcd.has_colors():
            self._pcd.colors = o3d.utility.Vector3dVector(
                uniform_color(self._pcd)
            )

        vis.add_geometry(self._pcd)

        self._register_keys(vis)

        self._print_help()
        if self._stats:
            print()
            print(self._stats.format())
            if self._scan_point_counts:
                print(f"Scans:    {len(self._scan_point_counts)}")
                for i, count in enumerate(self._scan_point_counts):
                    print(f"  [{i}] {count:,} points")
            if self._downsampled:
                print(f"(Downsampled to {len(self._pcd.points):,} points for display)")
            print()

        vis.run()
        vis.destroy_window()

    def _register_keys(self, vis: o3d.visualization.VisualizerWithKeyCallback) -> None:
        vis.register_key_callback(ord("H"), self._on_height_colormap)
        vis.register_key_callback(ord("C"), self._on_original_colors)
        vis.register_key_callback(ord("U"), self._on_uniform_color)
        vis.register_key_callback(ord("B"), self._on_toggle_bbox)
        vis.register_key_callback(ord("N"), self._on_toggle_normals)
        vis.register_key_callback(ord("I"), self._on_print_stats)
        vis.register_key_callback(ord("S"), self._on_scan_colors)
        vis.register_key_callback(ord("R"), self._on_reset_view)

        for digit in range(10):
            vis.register_key_callback(
                ord(str(digit)),
                lambda vis_ref, d=digit: self._on_select_scan(vis_ref, d),
            )

    # -- Key callbacks --

    def _on_height_colormap(self, vis) -> bool:
        colors = height_colormap(self._pcd)
        self._pcd.colors = o3d.utility.Vector3dVector(colors)
        vis.update_geometry(self._pcd)
        print("[viewer] Height colormap")
        return False

    def _on_original_colors(self, vis) -> bool:
        if self._original_colors is not None:
            self._pcd.colors = o3d.utility.Vector3dVector(self._original_colors.copy())
            vis.update_geometry(self._pcd)
            print("[viewer] Original colors")
        else:
            print("[viewer] No original colors available")
        return False

    def _on_uniform_color(self, vis) -> bool:
        colors = uniform_color(self._pcd)
        self._pcd.colors = o3d.utility.Vector3dVector(colors)
        vis.update_geometry(self._pcd)
        print("[viewer] Uniform color")
        return False

    def _on_toggle_bbox(self, vis) -> bool:
        if not self._bbox_visible:
            bbox = self._pcd.get_axis_aligned_bounding_box()
            bbox.color = (0.0, 1.0, 0.0)
            self._bbox_lineset = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bbox)
            self._bbox_lineset.paint_uniform_color([0.0, 1.0, 0.0])
            vis.add_geometry(self._bbox_lineset, reset_bounding_box=False)
            self._bbox_visible = True
            print("[viewer] Bounding box ON")
        else:
            if self._bbox_lineset is not None:
                vis.remove_geometry(self._bbox_lineset, reset_bounding_box=False)
            self._bbox_visible = False
            print("[viewer] Bounding box OFF")
        return False

    def _on_toggle_normals(self, vis) -> bool:
        if not self._pcd.has_normals():
            print("[viewer] No normals available")
            return False

        if not self._normals_visible:
            opt = vis.get_render_option()
            opt.point_show_normal = True
            self._normals_visible = True
            print("[viewer] Normals ON")
        else:
            opt = vis.get_render_option()
            opt.point_show_normal = False
            self._normals_visible = False
            print("[viewer] Normals OFF")
        vis.update_renderer()
        return False

    def _on_print_stats(self, vis) -> bool:
        if self._stats:
            print()
            print(self._stats.format())
            if self._scan_point_counts:
                print(f"Scans:    {len(self._scan_point_counts)}")
            print()
        return False

    def _on_scan_colors(self, vis) -> bool:
        if not self._scan_point_counts:
            print("[viewer] Not a multi-scan file")
            return False

        colors = scan_colors(self._scan_point_counts)
        # If we downsampled the merged cloud, the counts won't match.
        # Fall back to even distribution.
        if len(colors) != len(self._pcd.points):
            n = len(self._pcd.points)
            n_scans = len(self._scan_point_counts)
            from scan2mesh.viewer.colormaps import SCAN_PALETTE
            chunk_size = n // n_scans
            parts = []
            for i in range(n_scans):
                start = i * chunk_size
                end = n if i == n_scans - 1 else (i + 1) * chunk_size
                c = SCAN_PALETTE[i % len(SCAN_PALETTE)]
                parts.append(np.tile(c, (end - start, 1)))
            colors = np.concatenate(parts)

        self._pcd.colors = o3d.utility.Vector3dVector(colors)
        vis.update_geometry(self._pcd)
        print("[viewer] Per-scan coloring")
        return False

    def _on_select_scan(self, vis, scan_digit: int) -> bool:
        if not self._scan_clouds:
            return False

        if scan_digit == 0:
            # Show all scans
            merged = _merge_clouds(self._scan_clouds)
            self._pcd.points = merged.points
            if merged.has_colors():
                self._pcd.colors = merged.colors
            if merged.has_normals():
                self._pcd.normals = merged.normals
            self._original_colors = (
                np.asarray(self._pcd.colors).copy() if self._pcd.has_colors() else None
            )
            vis.update_geometry(self._pcd)
            vis.reset_view_point(True)
            print(f"[viewer] Showing all {len(self._scan_clouds)} scans")
        else:
            idx = scan_digit - 1
            if idx >= len(self._scan_clouds):
                print(f"[viewer] No scan {idx} (file has {len(self._scan_clouds)} scans)")
                return False

            cloud = self._scan_clouds[idx]
            self._pcd.points = cloud.points
            if cloud.has_colors():
                self._pcd.colors = cloud.colors
            else:
                self._pcd.colors = o3d.utility.Vector3dVector(
                    uniform_color(cloud)
                )
            if cloud.has_normals():
                self._pcd.normals = cloud.normals
            vis.update_geometry(self._pcd)
            vis.reset_view_point(True)
            print(f"[viewer] Showing scan {idx} only ({len(cloud.points):,} points)")

        return False

    def _on_reset_view(self, vis) -> bool:
        vis.reset_view_point(True)
        print("[viewer] View reset")
        return False

    def _print_help(self) -> None:
        print("╔══════════════════════════════════════╗")
        print("║       scan2mesh viewer controls      ║")
        print("╠══════════════════════════════════════╣")
        print("║  H - Height (Z) colormap             ║")
        print("║  C - Original RGB colors             ║")
        print("║  U - Uniform gray                    ║")
        print("║  B - Toggle bounding box             ║")
        print("║  N - Toggle normals                  ║")
        print("║  S - Per-scan coloring (E57)         ║")
        print("║  1-9 - Show only scan N (E57)        ║")
        print("║  0 - Show all scans                  ║")
        print("║  I - Print stats                     ║")
        print("║  R - Reset view                      ║")
        print("╚══════════════════════════════════════╝")


def _merge_clouds(clouds: list[o3d.geometry.PointCloud]) -> o3d.geometry.PointCloud:
    """Merge multiple point clouds into one."""
    merged = o3d.geometry.PointCloud()
    all_points = [np.asarray(c.points) for c in clouds]
    merged.points = o3d.utility.Vector3dVector(np.concatenate(all_points))

    all_colors = [np.asarray(c.colors) for c in clouds if c.has_colors()]
    if len(all_colors) == len(clouds):
        merged.colors = o3d.utility.Vector3dVector(np.concatenate(all_colors))

    all_normals = [np.asarray(c.normals) for c in clouds if c.has_normals()]
    if len(all_normals) == len(clouds):
        merged.normals = o3d.utility.Vector3dVector(np.concatenate(all_normals))

    return merged
