"""Tests for processing modules."""

from __future__ import annotations

import numpy as np
import open3d as o3d

from scan2mesh.config import PipelineConfig, Mode, QualityPreset, ReconstructionMethod
from scan2mesh.processing import preprocess, normals, reconstruct, postprocess


def test_remove_outliers(sample_pcd):
    config = PipelineConfig()
    result = preprocess.remove_outliers(sample_pcd, config)
    # Should remove some points but not all
    assert 0 < len(result.points) <= len(sample_pcd.points)


def test_voxel_downsample(sample_pcd):
    config = PipelineConfig(voxel_size=0.1)
    result = preprocess.voxel_downsample(sample_pcd, config)
    assert len(result.points) < len(sample_pcd.points)


def test_voxel_downsample_skips_when_none(sample_pcd):
    config = PipelineConfig(voxel_size=None)
    result = preprocess.voxel_downsample(sample_pcd, config)
    assert len(result.points) == len(sample_pcd.points)


def test_estimate_normals(sample_pcd):
    assert not sample_pcd.has_normals()
    config = PipelineConfig(normal_neighbors=20)
    result = normals.estimate_normals(sample_pcd, config)
    assert result.has_normals()
    assert len(result.normals) == len(result.points)


def test_estimate_normals_skips_if_present(sample_pcd_with_normals):
    config = PipelineConfig()
    original_normals = np.asarray(sample_pcd_with_normals.normals).copy()
    result = normals.estimate_normals(sample_pcd_with_normals, config)
    np.testing.assert_array_equal(np.asarray(result.normals), original_normals)


def test_poisson_reconstruction(sample_pcd_with_normals):
    config = PipelineConfig(
        method=ReconstructionMethod.POISSON,
        poisson_depth=6,
        poisson_density_quantile=0.01,
    )
    mesh = reconstruct.reconstruct(sample_pcd_with_normals, config)
    assert len(mesh.vertices) > 0
    assert len(mesh.triangles) > 0


def test_ball_pivoting_reconstruction(sample_pcd_with_normals):
    config = PipelineConfig(method=ReconstructionMethod.BALL_PIVOTING)
    mesh = reconstruct.reconstruct(sample_pcd_with_normals, config)
    assert len(mesh.vertices) > 0
    assert len(mesh.triangles) > 0


def test_decimate(sample_mesh):
    original_faces = len(sample_mesh.triangles)
    target = original_faces // 2
    result = postprocess.decimate(sample_mesh, target)
    assert len(result.triangles) <= target


def test_transfer_vertex_colors(sample_pcd_with_normals):
    config = PipelineConfig(
        method=ReconstructionMethod.POISSON,
        poisson_depth=6,
        poisson_density_quantile=0.01,
    )
    mesh = reconstruct.reconstruct(sample_pcd_with_normals, config)

    # Clear any existing colors to test transfer
    mesh.vertex_colors = o3d.utility.Vector3dVector()
    assert not mesh.has_vertex_colors()

    mesh = postprocess.transfer_vertex_colors(mesh, sample_pcd_with_normals)
    assert mesh.has_vertex_colors()
    colors = np.asarray(mesh.vertex_colors)
    assert colors.shape[1] == 3
    assert colors.min() >= 0.0
    assert colors.max() <= 1.0
