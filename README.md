# scan2mesh

Convert, view, inspect, and split lidar point cloud files. Mesh output
supports both visualization (smooth, watertight) and precision (geometrically
faithful) workflows.

## Features

- **Input formats:** LAS, LAZ, E57, PLY, PCD, XYZ, PTX, PTS
- **Output formats:** OBJ, glTF, GLB, STL
- Interactive 3D viewer with per-scan coloring (E57)
- Metadata inspection, per-scan splitting
- Interruptible pipeline — Ctrl+C exits immediately even during Open3D C++ calls
- Animated progress bar during long stages

## Installation

```bash
git clone https://github.com/larsenTFG/scan2mesh.git
cd scan2mesh
pip install -e .

# Optional: E57 support (requires C++ build tools on first install)
pip install -e ".[e57]"
```

Sample E57 files are available in [`examples/`](examples/) to try out the tool.

---

## Commands

scan2mesh has four subcommands. The `convert` command is the default — it can
be omitted from the command line.

| Command | Purpose |
|---------|---------|
| [`convert`](#convert) | Point cloud → mesh (default) |
| [`view`](#view) | Open an interactive 3D viewer |
| [`inspect`](#inspect) | Show file / per-scan metadata |
| [`split`](#split) | Split an E57 file into per-scan files |

Run any command with `--help` for the full option list.

---

## `convert`

Convert a point cloud file to a mesh.

```bash
scan2mesh convert <input> <output>
scan2mesh <input> <output>              # 'convert' is implicit
```

Convert runs in one of two **reconstruction modes**, each with three quality
presets:

### Visualization mode — `--mode visualization` *(default)*

Smooth, watertight meshes for viewers, web, and game engines. Uses **Poisson
reconstruction**, which fills gaps and produces render-ready surfaces.

**Options**

| Flag | Description | Default |
|------|-------------|---------|
| `-q`, `--quality {low,medium,high}` | Preset within the mode. | `medium` |
| `-f`, `--target-faces INT` | Cap the output triangle count. | Preset |
| `--poisson-depth INT` | Poisson octree depth. | Auto (scene-scaled) |
| `--voxel-size FLOAT` | Absolute voxel size. | Auto |
| `--color / --no-color` | Vertex color transfer. | `--color` |

**Presets**

| Preset | Voxel (× spacing) | Poisson depth floor | Target faces |
|--------|-------------------|---------------------|--------------|
| `low`    | 8× | 7  | 500K      |
| `medium` | 4× | 9  | 2M        |
| `high`   | 2× | 10 | Unlimited |

**Examples**

```bash
# Quick preview
scan2mesh examples/trimble.e57 preview.glb -q low

# Web-ready mesh with a fixed polygon budget
scan2mesh examples/pump.e57 pump.glb -q medium -f 250000
```

### Precision mode — `--mode precision`

Geometrically faithful meshes — only triangulates where points were actually
measured. Uses **Ball Pivoting**, which leaves real gaps as holes rather than
hallucinating surface.

**Options**

| Flag | Description | Default |
|------|-------------|---------|
| `-q`, `--quality {low,medium,high}` | Preset within the mode. | `medium` |
| `--max-deviation FLOAT` | Max mesh deviation from source (mm). | Preset |
| `--bpa-radii "a,b,c"` | Ball Pivoting radii as multiples of point spacing. More/larger values bridge more gaps at the cost of runtime. | `1,2,4` |
| `--voxel-size FLOAT` | Absolute voxel size. | Auto |
| `--color / --no-color` | Vertex color transfer. | `--no-color` |

**Presets**

| Preset | Voxel (× spacing) | Normal neighbors | Decimation |
|--------|-------------------|------------------|------------|
| `low`    | 3×   | 30 | ≤ 5 mm deviation |
| `medium` | 1.5× | 50 | ≤ 2 mm deviation |
| `high`   | 1× (dedupe only) | 50 | None |

**Examples**

```bash
# Keep every measured point, transfer scanner colors
scan2mesh examples/pump.e57 pump.glb --mode precision -q high --color -v

# Tight deviation budget for QA / metrology
scan2mesh input.e57 out.obj --mode precision --max-deviation 1.0

# Extend BPA radii to bridge sparse scan regions
scan2mesh input.e57 out.glb --mode precision -q high --bpa-radii 1,2,4,8 --color
```

### Options available in both modes

| Flag | Description |
|------|-------------|
| `--method {poisson,ball_pivoting}` | Override reconstruction algorithm (decouples from mode default). |
| `--scans "0,2,3"` | For E57: include only these scan indices. |
| `--no-outlier-removal` | Skip the outlier filter (faster; safe for clean scans). |
| `-v`, `--verbose` | Per-stage logs including live elapsed time. |

---

## `view`

Open an interactive 3D viewer for a point cloud file.

```bash
scan2mesh view <input>
```

Uses Open3D's visualizer. For multi-scan E57 files, scans can be coloured and
isolated at runtime.

**Options**

| Flag | Description | Default |
|------|-------------|---------|
| `--voxel-size FLOAT` | Downsample before display (handy for huge clouds). | No downsampling |
| `--background {dark,light}` | Viewer background color. | `dark` |

**Keybindings (E57 with multiple scans)**

| Key | Action |
|-----|--------|
| `S` | Color points by scan index |
| `1`–`9` | Isolate an individual scan |

**Examples**

```bash
# View the whole file
scan2mesh view examples/trimble.e57

# Downsample a huge cloud before displaying
scan2mesh view cloud.ply --voxel-size 0.05

# Light-background view
scan2mesh view large_scan.las --background light
```

---

## `inspect`

Show metadata about a point cloud file without loading it fully into a mesh
pipeline.

```bash
scan2mesh inspect <input>
```

For E57 files, prints a table with per-scan point count, available fields
(x/y/z, intensity, color), and bounding box. For other formats, shows basic
point cloud statistics (count, bounds, has-colors, has-normals).

**Examples**

```bash
scan2mesh inspect examples/pump.e57
scan2mesh inspect cloud.ply
```

Use this first on unfamiliar E57 files so you know what `--scans` selectors
to pass to `convert` or `split`.

---

## `split`

Split an E57 file into individual per-scan files.

```bash
scan2mesh split <input.e57> <output_dir>
```

Each scan is written as a separate point cloud. Useful when a multi-scan E57
needs to be processed scan-by-scan or fed to a tool that can't read E57.

**Options**

| Flag | Description | Default |
|------|-------------|---------|
| `--format {ply,pcd,las}` | Output format for each scan file. | `ply` |
| `--scans "0,2,3"` | Only export these scan indices. | All |

**Examples**

```bash
# Split every scan into PLY files
scan2mesh split examples/pump.e57 ./pump_scans/

# Only scans 0, 2, 3 — as PCD
scan2mesh split building.e57 ./scans/ --format pcd --scans 0,2,3
```

Currently only E57 input is supported (no other format stores independent
scans).

---

## Pipeline (convert)

1. **Read** — format-specific reader → unified point cloud
2. **Estimate spacing** — k-NN sample to size voxels and reconstruction radii
   from the scene's native resolution
3. **Downsample** — voxel-grid (skippable in some presets)
4. **Outlier removal** — radius-based when spacing is known, statistical
   otherwise (skippable via `--no-outlier-removal`)
5. **Normal estimation** — fast viewpoint flip (BPA) or consistent MST
   orientation (Poisson)
6. **Surface reconstruction** — Poisson or Ball Pivoting
7. **Post-process** — decimation (face-count or deviation) + vertex color
   transfer
8. **Export** — OBJ / GLB / glTF / STL

## Tips

- **Huge clouds + precision/high:** the pipeline auto-bumps `voxel_size` if
  the post-downsample count would exceed the Ball Pivoting practical limit
  (~1.5M points). Watch for the `WARNING: Ball Pivoting input ... bumping
  voxel_size` line in `-v` mode.
- **Holes with BPA?** Extend radii: `--bpa-radii 1,2,4,8`, or switch to
  `--mode visualization` to fill gaps with Poisson.
- **Ctrl+C** works at any stage — blocking calls run in daemon threads.

## Development

```bash
pip install -e ".[dev]"
pytest tests/
```

## License

MIT — see [LICENSE](LICENSE).
