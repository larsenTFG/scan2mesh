"""Tests for the full conversion pipeline."""

from __future__ import annotations

from pathlib import Path

from scan2mesh.config import Mode, PipelineConfig, QualityPreset
from scan2mesh.pipeline import convert


def test_pipeline_visualization_ply_to_obj(tmp_ply, tmp_path):
    output = str(tmp_path / "output.obj")
    config = PipelineConfig(
        mode=Mode.VISUALIZATION,
        quality=QualityPreset.LOW,
    )
    convert(tmp_ply, output, config)
    assert Path(output).exists()
    assert Path(output).stat().st_size > 0


def test_pipeline_visualization_ply_to_glb(tmp_ply, tmp_path):
    output = str(tmp_path / "output.glb")
    config = PipelineConfig(
        mode=Mode.VISUALIZATION,
        quality=QualityPreset.LOW,
    )
    convert(tmp_ply, output, config)
    assert Path(output).exists()
    assert Path(output).stat().st_size > 0


def test_pipeline_precision_ply_to_obj(tmp_ply, tmp_path):
    output = str(tmp_path / "output.obj")
    config = PipelineConfig(
        mode=Mode.PRECISION,
        quality=QualityPreset.LOW,
    )
    convert(tmp_ply, output, config)
    assert Path(output).exists()
    assert Path(output).stat().st_size > 0


def test_pipeline_pts_to_stl(tmp_pts, tmp_path):
    output = str(tmp_path / "output.stl")
    config = PipelineConfig(
        mode=Mode.VISUALIZATION,
        quality=QualityPreset.LOW,
    )
    convert(tmp_pts, output, config)
    assert Path(output).exists()


def test_pipeline_progress_callback(tmp_ply, tmp_path):
    output = str(tmp_path / "output.obj")
    messages = []

    def callback(msg, pct):
        messages.append((msg, pct))

    config = PipelineConfig(
        mode=Mode.VISUALIZATION,
        quality=QualityPreset.LOW,
    )
    convert(tmp_ply, output, config, progress_callback=callback)

    assert len(messages) > 0
    # Progress should be monotonically non-decreasing
    pcts = [m[1] for m in messages]
    assert all(a <= b for a, b in zip(pcts, pcts[1:]))
    assert pcts[-1] == 1.0
