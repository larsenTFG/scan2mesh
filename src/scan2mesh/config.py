"""Pipeline configuration, modes, and quality presets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Mode(Enum):
    VISUALIZATION = "visualization"
    PRECISION = "precision"


class ReconstructionMethod(Enum):
    POISSON = "poisson"
    BALL_PIVOTING = "ball_pivoting"


class QualityPreset(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Presets use voxel_spacing_multiple (multiples of median point-to-point distance)
# instead of an absolute or bbox-relative size, so detail is preserved regardless
# of scene scale.
# Poisson depth is resolved automatically from scene bbox / point spacing unless
# explicitly set; the preset value is a minimum floor.
_PRESETS: dict[tuple[Mode, QualityPreset], dict] = {
    # Visualization presets
    (Mode.VISUALIZATION, QualityPreset.LOW): {
        "voxel_spacing_multiple": 8.0,
        "method": ReconstructionMethod.POISSON,
        "poisson_depth_min": 7,
        "poisson_density_quantile": 0.01,
        "normal_neighbors": 20,
        "target_faces": 500_000,
        "max_deviation_mm": None,
        "transfer_colors": True,
    },
    (Mode.VISUALIZATION, QualityPreset.MEDIUM): {
        "voxel_spacing_multiple": 4.0,
        "method": ReconstructionMethod.POISSON,
        "poisson_depth_min": 9,
        "poisson_density_quantile": 0.01,
        "normal_neighbors": 20,
        "target_faces": 2_000_000,
        "max_deviation_mm": None,
        "transfer_colors": True,
    },
    (Mode.VISUALIZATION, QualityPreset.HIGH): {
        "voxel_spacing_multiple": 2.0,
        "method": ReconstructionMethod.POISSON,
        "poisson_depth_min": 10,
        "poisson_density_quantile": 0.01,
        "normal_neighbors": 20,
        "target_faces": None,
        "max_deviation_mm": None,
        "transfer_colors": True,
    },
    # Precision presets
    (Mode.PRECISION, QualityPreset.LOW): {
        "voxel_spacing_multiple": 3.0,
        "method": ReconstructionMethod.BALL_PIVOTING,
        "poisson_depth_min": 9,
        "poisson_density_quantile": 0.05,
        "normal_neighbors": 30,
        "target_faces": None,
        "max_deviation_mm": 5.0,
        "transfer_colors": False,
    },
    (Mode.PRECISION, QualityPreset.MEDIUM): {
        "voxel_spacing_multiple": 1.5,
        "method": ReconstructionMethod.BALL_PIVOTING,
        "poisson_depth_min": 10,
        "poisson_density_quantile": 0.05,
        "normal_neighbors": 50,
        "target_faces": None,
        "max_deviation_mm": 2.0,
        "transfer_colors": False,
    },
    (Mode.PRECISION, QualityPreset.HIGH): {
        # voxel_spacing_multiple=1.0 removes duplicate coverage from overlapping
        # scans without sacrificing resolution (all unique surface points survive).
        # Ball Pivoting is O(N * neighbors) per radius pass — a raw multi-scan
        # cloud with 5x overlap runs 5x slower AND produces worse triangles than
        # the deduplicated equivalent.
        "voxel_spacing_multiple": 1.0,
        "method": ReconstructionMethod.BALL_PIVOTING,
        "poisson_depth_min": 11,
        "poisson_density_quantile": 0.10,
        "normal_neighbors": 50,
        "target_faces": None,
        "max_deviation_mm": None,
        "transfer_colors": False,
    },
}

# Auto-computed Poisson depth is clamped to this range.
POISSON_DEPTH_MIN = 7
POISSON_DEPTH_MAX = 13


@dataclass
class PipelineConfig:
    """Full pipeline configuration. Created from mode/quality presets, then overridden by CLI flags."""

    mode: Mode = Mode.VISUALIZATION
    quality: QualityPreset = QualityPreset.MEDIUM

    # Preprocessing
    voxel_size: Optional[float] = None  # absolute size; computed from spacing multiple if None
    voxel_spacing_multiple: Optional[float] = None  # multiples of median point spacing
    skip_outlier_removal: bool = False
    outlier_nb_neighbors: int = 20
    outlier_std_ratio: float = 2.0

    # Normals
    normal_neighbors: int = 20

    # Reconstruction
    method: Optional[ReconstructionMethod] = None  # None = use preset default
    poisson_depth: Optional[int] = None  # None = auto-computed from bbox / spacing
    poisson_density_quantile: float = 0.01
    ball_pivot_radii: Optional[list[float]] = None  # absolute radii; auto-computed if None
    bpa_radius_multiples: Optional[list[float]] = None  # multiples of point spacing; overrides default [1,2,4]

    # Post-processing
    target_faces: Optional[int] = None
    max_deviation_mm: Optional[float] = None
    transfer_colors: Optional[bool] = None

    # Populated by resolve() — reused by later stages to avoid recomputing
    point_spacing: Optional[float] = None
    bbox_diagonal: Optional[float] = None

    def resolve(
        self,
        point_count: int,
        bbox_diagonal: float,
        point_spacing: float,
    ) -> PipelineConfig:
        """Fill in unset values from the mode/quality preset and point cloud geometry.

        Args:
            point_count: Number of points in the cloud.
            bbox_diagonal: Diagonal length of axis-aligned bounding box.
            point_spacing: Median nearest-neighbor distance (scene's native resolution).
        """
        preset = _PRESETS[(self.mode, self.quality)]
        self.point_spacing = point_spacing
        self.bbox_diagonal = bbox_diagonal

        if self.method is None:
            self.method = preset["method"]
        if self.transfer_colors is None:
            self.transfer_colors = preset["transfer_colors"]
        if self.target_faces is None and preset["target_faces"] is not None:
            self.target_faces = preset["target_faces"]
        if self.max_deviation_mm is None and preset["max_deviation_mm"] is not None:
            self.max_deviation_mm = preset["max_deviation_mm"]

        self.normal_neighbors = max(self.normal_neighbors, preset["normal_neighbors"])
        self.poisson_density_quantile = preset["poisson_density_quantile"]

        # Voxel size: scale with point spacing so large scenes don't over-decimate.
        if self.voxel_size is None:
            multiple = self.voxel_spacing_multiple
            if multiple is None:
                multiple = preset["voxel_spacing_multiple"]
            if multiple is not None and point_spacing > 0:
                self.voxel_size = multiple * point_spacing

        # Poisson depth: auto-compute so octree resolution matches the post-downsample
        # point spacing. Depth d gives ~bbox/2^d cells; we want cell size ~= effective
        # spacing (voxel_size if downsampled, else native spacing).
        if self.poisson_depth is None:
            effective_spacing = self.voxel_size if self.voxel_size else point_spacing
            if effective_spacing > 0 and bbox_diagonal > 0:
                auto_depth = int(math.ceil(math.log2(bbox_diagonal / effective_spacing)))
            else:
                auto_depth = preset["poisson_depth_min"]
            self.poisson_depth = max(
                preset["poisson_depth_min"],
                min(POISSON_DEPTH_MAX, max(POISSON_DEPTH_MIN, auto_depth)),
            )

        return self
