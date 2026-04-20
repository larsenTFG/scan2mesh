"""Tests for point cloud readers."""

from __future__ import annotations

import numpy as np

from scan2mesh.readers import read_point_cloud, READER_REGISTRY


def test_reader_registry_has_expected_formats():
    expected = {".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb", ".las", ".laz", ".ptx", ".pts"}
    assert expected.issubset(set(READER_REGISTRY.keys()))


def test_read_ply(tmp_ply):
    pcd = read_point_cloud(tmp_ply)
    assert len(pcd.points) > 0
    assert pcd.has_colors()


def test_read_pts(tmp_pts):
    pcd = read_point_cloud(tmp_pts)
    assert len(pcd.points) > 0
    assert pcd.has_colors()
    # Check colors are normalized to [0, 1]
    colors = np.asarray(pcd.colors)
    assert colors.max() <= 1.0
    assert colors.min() >= 0.0


def test_unsupported_format_raises():
    import pytest
    with pytest.raises(ValueError, match="Unsupported format"):
        read_point_cloud("file.unsupported")


def test_progress_callback(tmp_ply):
    messages = []
    def callback(msg, pct):
        messages.append((msg, pct))

    pcd = read_point_cloud(tmp_ply, progress_callback=callback)
    assert len(messages) > 0
    assert messages[-1][1] == 1.0
