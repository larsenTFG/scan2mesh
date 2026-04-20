"""Tests for mesh writers."""

from __future__ import annotations

from pathlib import Path

from scan2mesh.writers import write_mesh


def test_write_obj(tmp_path, sample_mesh):
    path = str(tmp_path / "output.obj")
    write_mesh(sample_mesh, path)
    assert Path(path).exists()
    assert Path(path).stat().st_size > 0


def test_write_glb(tmp_path, sample_mesh):
    path = str(tmp_path / "output.glb")
    write_mesh(sample_mesh, path)
    assert Path(path).exists()
    assert Path(path).stat().st_size > 0


def test_write_gltf(tmp_path, sample_mesh):
    path = str(tmp_path / "output.gltf")
    write_mesh(sample_mesh, path)
    assert Path(path).exists()


def test_write_stl(tmp_path, sample_mesh):
    path = str(tmp_path / "output.stl")
    write_mesh(sample_mesh, path)
    assert Path(path).exists()
    assert Path(path).stat().st_size > 0


def test_unsupported_output_raises(tmp_path, sample_mesh):
    import pytest
    with pytest.raises(ValueError, match="Unsupported output format"):
        write_mesh(sample_mesh, str(tmp_path / "output.abc"))


def test_creates_parent_dirs(tmp_path, sample_mesh):
    path = str(tmp_path / "sub" / "dir" / "output.obj")
    write_mesh(sample_mesh, path)
    assert Path(path).exists()
