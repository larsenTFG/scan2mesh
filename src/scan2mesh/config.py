"""Pipeline configuration, modes, and quality presets."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


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


# Presets keyed by (mode, quality)
_PRESETS: dict[tuple[Mode, QualityPreset], dict] = {
    # Visualization presets
    (Mode.VISUALIZATION, QualityPreset.LOW): {
        "voxel_size_factor": 0.005,
        "method": ReconstructionMethod.POISSON,
        "poisson_depth": 7,
        "poisson_density_quantile": 0.01,
        "normal_neighbors": 20,
        "target_faces": 500_000,
        "max_deviation_mm": None,
        "transfer_colors": True,
    },
    (Mode.VISUALIZATION, QualityPreset.MEDIUM): {
        "voxel_size_factor": 0.002,
        "method": ReconstructionMethod.POISSON,
        "poisson_depth": 9,
        "poisson_density_quantile": 0.01,
        "normal_neighbors": 20,
        "target_faces": 2_000_000,
        "max_deviation_mm": None,
        "transfer_colors": True,
    },
    (Mode.VISUALIZATION, QualityPreset.HIGH): {
        "voxel_size_factor": 0.001,
        "method": ReconstructionMethod.POISSON,
        "poisson_depth": 11,
        "poisson_density_quantile": 0.01,
        "normal_neighbors": 20,
        "target_faces": None,
        "max_deviation_mm": None,
        "transfer_colors": True,
    },
    # Precision presets
    (Mode.PRECISION, QualityPreset.LOW): {
        "voxel_size_factor": 0.002,
        "method": ReconstructionMethod.BALL_PIVOTING,
        "poisson_depth": 9,
        "poisson_density_quantile": 0.05,
        "normal_neighbors": 30,
        "target_faces": None,
        "max_deviation_mm": 5.0,
        "transfer_colors": False,
    },
    (Mode.PRECISION, QualityPreset.MEDIUM): {
        "voxel_size_factor": 0.0005,
        "method": ReconstructionMethod.BALL_PIVOTING,
        "poisson_depth": 11,
        "poisson_density_quantile": 0.05,
        "normal_neighbors": 50,
        "target_faces": None,
        "max_deviation_mm": 2.0,
        "transfer_colors": False,
    },
    (Mode.PRECISION, QualityPreset.HIGH): {
        "voxel_size_factor": None,  # no downsampling
        "method": ReconstructionMethod.BALL_PIVOTING,
        "poisson_depth": 12,
        "poisson_density_quantile": 0.10,
        "normal_neighbors": 50,
        "target_faces": None,
        "max_deviation_mm": None,
        "transfer_colors": False,
    },
}


@dataclass
class PipelineConfig:
    """Full pipeline configuration. Created from mode/quality presets, then overridden by CLI flags."""

    mode: Mode = Mode.VISUALIZATION
    quality: QualityPreset = QualityPreset.MEDIUM

    # Preprocessing
    voxel_size: Optional[float] = None  # absolute size; computed from factor if None
    voxel_size_factor: Optional[float] = None  # relative to bounding box diagonal
    skip_outlier_removal: bool = False
    outlier_nb_neighbors: int = 20
    outlier_std_ratio: float = 2.0

    # Normals
    normal_neighbors: int = 20

    # Reconstruction
    method: Optional[ReconstructionMethod] = None  # None = use preset default
    poisson_depth: Optional[int] = None
    poisson_density_quantile: float = 0.01
    ball_pivot_radii: Optional[list[float]] = None  # auto-computed if None

    # Post-processing
    target_faces: Optional[int] = None
    max_deviation_mm: Optional[float] = None
    transfer_colors: Optional[bool] = None

    def resolve(self, point_count: int, bbox_diagonal: float) -> PipelineConfig:
        """Fill in unset values from the mode/quality preset and point cloud geometry."""
        preset = _PRESETS[(self.mode, self.quality)]

        if self.method is None:
            self.method = preset["method"]
        if self.poisson_depth is None:
            self.poisson_depth = preset["poisson_depth"]
        if self.transfer_colors is None:
            self.transfer_colors = preset["transfer_colors"]
        if self.target_faces is None and preset["target_faces"] is not None:
            self.target_faces = preset["target_faces"]
        if self.max_deviation_mm is None and preset["max_deviation_mm"] is not None:
            self.max_deviation_mm = preset["max_deviation_mm"]

        self.normal_neighbors = max(self.normal_neighbors, preset["normal_neighbors"])
        self.poisson_density_quantile = preset["poisson_density_quantile"]

        # Compute voxel size from factor if not explicitly set
        if self.voxel_size is None:
            factor = self.voxel_size_factor or preset["voxel_size_factor"]
            if factor is not None:
                self.voxel_size = bbox_diagonal * factor

        return self
