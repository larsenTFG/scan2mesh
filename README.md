# scan2mesh

Convert lidar point cloud files to meshes, with support for both visualization
and precision-measurement workflows.

## Features

- **Input formats:** LAS, LAZ, E57, PLY, PCD, XYZ, PTX, PTS
- **Output formats:** OBJ, glTF, GLB, STL
- **Two modes:**
  - **Visualization** — smooth, watertight meshes with vertex colors (Poisson reconstruction)
  - **Precision** — geometrically faithful meshes with deviation-controlled decimation (Ball Pivoting)
- Quality presets (`low` / `medium` / `high`) per mode
- Vertex color transfer via KDTree nearest-neighbor
- Interruptible pipeline (Ctrl+C works even during Open3D C++ calls)

## Installation

```bash
git clone https://github.com/<your-username>/scan2mesh.git
cd scan2mesh
pip install -e .

# Optional: E57 support (requires C++ build tools on first install)
pip install -e ".[e57]"
```

## Usage

Sample E57 files are available in [`examples/`](examples/) to try out the tool.

```bash
# Visualization (default)
scan2mesh input.las output.glb

# Precision mode
scan2mesh input.e57 output.obj --mode precision --quality high

# Tune quality
scan2mesh input.ply output.glb -q low -f 100000

# Precision with colors enabled
scan2mesh input.e57 output.glb --mode precision --color

# Skip outlier removal for speed
scan2mesh large.las output.glb --no-outlier-removal
```

See `scan2mesh --help` for the full option list.

## Pipeline

1. **Read** — format-specific reader → unified point cloud
2. **Downsample** — voxel-grid downsampling (skippable in precision/high)
3. **Outlier removal** — statistical outlier filter (skippable via `--no-outlier-removal`)
4. **Normal estimation** — with consistent orientation
5. **Surface reconstruction** — Poisson or Ball Pivoting
6. **Post-process** — decimation (face-count or max-deviation) + vertex color transfer
7. **Export** — OBJ / GLB / glTF / STL

## Development

```bash
pip install -e ".[dev]"
pytest tests/
```

## License

MIT — see [LICENSE](LICENSE).
