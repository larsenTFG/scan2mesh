# Example E57 Point Cloud Files

These sample point clouds are provided to try out scan2mesh.

| File | Size | Points | Scans | Color | Description |
|------|------|--------|-------|-------|-------------|
| `bunnyDouble.e57` | 743 KB | 30,571 | 1 | No | Stanford bunny — quick sanity test |
| `trimble.e57` | 14 MB | 899,896 | 13 | Yes | Multi-scan Trimble scanner capture |
| `pump.e57` | 51 MB | 2,878,964 | 5 | Yes | Industrial pump scan with structured grid |

## Try it

```bash
# Quick visualization test
scan2mesh examples/trimble.e57 trimble.glb -v

# Precision mesh from the pump scan
scan2mesh examples/pump.e57 pump.glb --mode precision --quality medium --color -v
```

## Attribution

These files are from the [E57 3D Imaging Format project](https://sourceforge.net/projects/e57-3d-imgfmt/)
on SourceForge (libe57.org), distributed under the
[Boost Software License 1.0](https://www.boost.org/LICENSE_1_0.txt).
