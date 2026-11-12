# NURBS Proto: Development

Building the kernel, running the tests and packaging a release. For users, see the [README](../README.md).

## Install from source

Symlink or copy `blender_nurbs/` into
`~/Library/Application Support/Blender/5.2/extensions/user_default/`, then enable
**NURBS Proto** in Preferences → Add-ons. It needs the kernel module built for
your system in `blender_nurbs/kernel/` (see [Build the kernel](#build-the-kernel-macos-windows-linux-blender-52--python-313)).

## Build the kernel (macOS, Windows, Linux; Blender 5.2 / Python 3.13)

The kernel is NVIDIA's USD BRep (SMLib), written in C++. `kernel_build/` compiles
it, its upstream pybind11 bindings and our extension (`kernel_build/ext/`: CV
access, surfaces from CVs, surface → Brep, the tighter trim) into ONE Python
module, without OpenUSD. The module is per platform; all can sit side by side in
`blender_nurbs/kernel/`, and Python loads the one for its platform:

| Platform | Module |
|---|---|
| macOS arm64 (Apple Silicon) | `_nproto_kernel.cpython-313-darwin.so` |
| Windows x64 | `_nproto_kernel.cp313-win_amd64.pyd` |
| Linux x64 | `_nproto_kernel.cpython-313-x86_64-linux-gnu.so` |

Upstream's PyPI wheels can't be used: they are for Python 3.12 (Blender 5.2 runs 3.13) and have none of our additions.

Sources, the same on every platform:

```bash
git clone https://github.com/NVIDIA-Omniverse/usd-brep.git third_party/usd-brep
```
```bash
git -C third_party/usd-brep checkout 3fd5b9548d19aa8c12c29bc9637a96d42a06f8a5
```
```bash
python3 -m pip install --target third_party/pybind11_pkg pybind11==2.13.6
```

The build needs a CPython **3.13 with its headers**, the version Blender 5.x runs. Blender's bundled Python has only `pyconfig.h`, so any normal 3.13 install is used for the headers; the module then loads into Blender's own 3.13. `-DPython3_EXECUTABLE` names it, else a `python3.13` on the PATH is used.

**macOS** (`brew install python@3.13`):

```bash
cmake -S kernel_build -B kernel_build/build -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=/opt/homebrew/bin/python3.13
```
```bash
cmake --build kernel_build/build -j10
```

The module is built for Apple Silicon only; Intel Macs are not supported. A build tree configured before this change keeps Homebrew's include path in its cache and still works.

**Linux** (GCC 9+ and `python3.13-dev`, or the manylinux_2_28 container the workflow uses, so the module also loads on older distributions):

```bash
cmake -S kernel_build -B kernel_build/build -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=/usr/bin/python3.13
```
```bash
cmake --build kernel_build/build -j
```

**Windows** (Visual Studio 2022 with C++, and Python 3.13 from python.org, whose `libs\python313.lib` is linked):

```bash
cmake -S kernel_build -B kernel_build/build -A x64 -DPython3_EXECUTABLE=C:/Python313/python.exe
```
```bash
cmake --build kernel_build/build --config Release
```

The Windows module uses the static C runtime, so it needs no Visual C++ redistributable: the worker jobs run Blender's `python.exe`, not `blender.exe`, and must find everything without it.

The build copies the module into `blender_nurbs/kernel/` under a temporary name and renames it over the old one, so a running Blender keeps the module it loaded (writing into a loaded library can crash it on macOS); restart Blender to use the new build. On Windows a loaded `.pyd` can't be replaced at all, so close Blender before building. Worker jobs start a fresh process and use the new module at once.

**Status:** the macOS arm64, Windows x64 and Linux x64 modules all build on GitHub Actions (below), and the headless test suite runs in Blender on each. Locally only the macOS build has been tried. Linux and Windows builds of the kernel refuse to evaluate a curve one rounding step past the end of its range, where the macOS build clamps; `kernel/harden.py` wraps those calls so every build behaves the same.

**All three at once (GitHub Actions):** `.github/workflows/kernel.yml` builds the module on macOS (arm64), Windows x64 and Linux x64 (manylinux_2_28). It then runs the headless test suite in Blender 5.2.1 on each and packages the extension: one zip per platform and one with all three, each checked with Blender's `extension validate`. It runs on a push that touches the add-on, the kernel build, the tests or the packaging, or by hand (workflow_dispatch). The project has to be a GitHub repository for this; `.gitignore` keeps `third_party/`, build trees and the built modules out of it.

**Release:** run the workflow by hand with **release** ticked (Actions > Kernel > Run workflow), or push a tag `v<version>`. Once the tests pass on all three systems, the zips are published as the GitHub Release `v<version>`, with the version from `blender_manifest.toml` (raise it first). It's a pre-release unless the commit is on `main`. A release that already exists gets its zips replaced.

```bash
gh workflow run kernel.yml --ref <branch> -f release=true
```

**Package:** `python3 tools/package_extension.py` zips `blender_nurbs/` into `dist/nurbs_modeling-<version>.zip`. `--split` also writes one zip per platform (`dist/nurbs_modeling-<version>-macos_arm64.zip`, `-windows_x64`, `-linux_x64`), each with only its own module. A zip's manifest lists exactly the platforms whose module it holds (a macOS module is read to tell arm64, Intel or both), so Blender won't install it where it has no kernel. `--check` only lists the modules found. Check a zip with:

```bash
/Applications/Blender.app/Contents/MacOS/Blender --factory-startup --command extension validate dist/nurbs_modeling-0.3.0.zip
```

Kernel files written by the add-on use `\n` line endings on every platform. Python on Windows would otherwise write `\r\n` into the B-rep text the C++ reader parses.

`count_missing_uv_trims(brep)` counts edge uses without a UV trim curve (diagnostics, tests). The extension also has `trim_surface_with_3d_curves_tol`: upstream's
`trim_surface_with_3d_curves` fits the UV trim curves within a fixed 0.01 model
units, which on a small part let a UV curve stray up to 0.5 mm from its 3D edge
(Edit CVs re-lifts edges from the UV curves, so a Patch edge pulled off its
neighbour). Patch now trims with a fit within 1e-5 of its size (edges re-lift
within 4e-6 of their length instead of 7e-4); the Brep's own tolerance stays
0.01, as before. Without the new module (Blender not restarted) Patch falls back
to the old trim.

## Benchmark

`tools/bench.py` times the add-on's main work on a real scene, object by object: a full rebuild from its history, the display tessellation, the continuity labels, the zebra's and the Curvature Map's own meshes, and one Edit CVs update. Add `-- --profile <object>` for a cProfile of that object's rebuild. It only reads the scene.

```bash
/Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup scene.blend --python tools/bench.py
```

## Test

```bash
/Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup --python tests/test_headless.py
```

## How G1/G2 lofts work

The kernel has no blend-loft call. So for each surface-edge input the add-on
samples the edge's position, its outward cross-boundary derivative and its
second derivative. From those it builds the CV rows of the loft:

- G1: 1 extra CV row
- G2: 2 extra CV rows

Across the loft the surface is a Bezier of degree `rows - 1`:

- G0–G0: degree 1
- G2–G2: degree 5

Along the loft, all rows are refit with one shared cubic knot vector. The
result is handed to the kernel as an exact B-spline surface.

## Known limits

* A G1/G2 loft fits its rows through samples (24 by default). Its boundary matches
  the source to about 1e-3 of the loft size, not exactly.
* The kernel's Python bindings never free objects (prototype memory leak).
* B-reps are stored as SMLib ASCII in object custom properties. There's no USD export yet.
* Fillet uses each surface's first face, untrimmed underneath.
* Tessellation repairs (all objects), continued: folds the edge flips can't undo (a sliver strip along a crack in the kernel's mesh, its two sides crossing over) are relaxed on single-face objects -- each inner vertex of a folded triangle moves to the surface point at the average (u, v) of its neighbours, exactly on the surface; folds are also repaired before the long-edge refinement, which would otherwise split one fold into several.
* Tessellation repairs (all objects): **cracks** -- the kernel's mesh of a trimmed face can leave thin gaps (two needle triangles fanning out of a corner without sharing an edge, or T-junctions); refinement would open them into visible lines, so they are zippered first (wherever an open edge arrives at a vertex and another leaves it in nearly the same direction, the thin triangle between them is added) and small leftover loops inside the face are filled (a real trimmed hole, whose rim lies on the B-rep's edges, stays open). The kernel's tessellation of a surface with a strongly uneven parameterization can also leave **folded triangles** (their plane stands across or against the surface normals); they overlap their neighbours and showed as spikes in shading and zebra. They are repaired by flipping the edge they share with a neighbour; vertices and normals stay. On single-face objects, edges whose end normals differ by more than the angle tolerance are also split (exact points and normals): the zebra interpolates normals linearly across a triangle, and long triangles over a bend drew straight-edged jags. **Folds inside one surface:** a surface with C0 knot lines where the normal jumps, like a loft through rectangles (straight-sided sections give sharp corners along the loft), is tessellated piece by piece for the display and the zebra. It is split exactly at those knot lines, so each fold is a sharp crease in the shading. The B-rep stays one face. Tessellated whole, the vertices on a fold carried one side's normal only, which smeared dark jagged streaks over the other side. The refinement also kept splitting the triangles across the fold, whose normals really do differ, into slivers: the user's 4-rectangle loft had 16,860 vertices and 5,086 badly shaded triangles; split, it has 2,055 and none. A surface that is only parametrically C0 but smooth (an exact circle's double knots) stays whole. The continuity check, following an input round a fold or past a closed surface's seam, could stop on the wrong side and report a false gap (a red G0 ring on a rectangle loft). A sample that seems off is now looked up again from scratch. On a result with several faces (a loft split at its folds) each sample is measured on the face it lies on. The labels belong to the feature, so after history steps (an Edge Fillet, a trim) they are checked on the feature's own result before those steps. A fillet rounding a loft's corners away from its sections is not a G0 failure of the loft: it used to turn both labels red, 0.003 off at the rounded corners. Writing the mesh, a face using one vertex twice (the kernel's mesh of a closed seam can have them) is dropped, and the per-corner normals are set only when they match the mesh's corners: a list of the wrong length crashed Blender inside normals_split_custom_set.
* The kernel's tessellator ignores Max Edge on trimmed faces and fills a trim crossing its cells with long fans, which shaded as facets and wedges on curved trimmed surfaces (e.g. a Patch). Single-face objects are now refined after tessellation: over-long edges are split at their midpoint in the surface's (u, v), so the new points and normals are exact. Joined objects (several faces) are not refined yet.
* **Display mesh defaults** (Display > Tessellation): Chord Tolerance **0.0005** and Angle Tolerance **5°** (were 0.002 and 15°). At 15° a triangle on a bend could span almost 15° of normal change, and the highlight on the Thermometer elbow showed it as steps; at 5° it spans at most 5°, and the mesh sits 2-4 times closer to the surface. **Flat faces** get no length limit any more (the kernel's Max Edge and the refinement both skip a planar single face: it shades the same however it is split), so a flat extrude is 2 triangles instead of 1,344. On the Thermometer's 12 objects: 35k triangles in 0.30 s instead of 25k in 0.23 s; the Patch re-meshes in 40-60 ms (was 16 ms) next to its 0.4 s solve. Scenes that never changed these two settings use the new defaults (re-mesh with Rebuild / Re-tessellate); a scene that set them keeps its values.
* The refinement and repairs work on whole arrays (one triangle array, vectorized split passes, plain-float Newton steps for the closest (u, v)): the same mesh as before, 2-3x faster (the elbow Patch's 1° zebra mesh, 125k triangles: 2.2 s -> 0.8 s; display meshes about twice as fast). On a **periodic** surface (tube, torus, cone, sphere) the closest (u, v) now wraps around the seam instead of stopping at the domain's end: points just past the seam landed on it, and the refinement folded the mesh there (a tube's display mesh had ~150-800 folded triangles along its seam; now none). Known limit: the kernel's own mesh of a rational sphere has slivers along its seam at fine zebra settings (~300 folded triangles at 1°), which the repairs don't remove.
* Kernel fillet surfaces along curved edges have dense CV nets, so **CVs / Hulls** display gets crowded on filleted objects (Edge Fillet > **Class-A Patches** rebuilds them light).
* The **Display** toggles (Edges, CVs / Hulls, Continuity) and the Analyze overlays (Zebra, Draft Angle, Curvature Map, Comb, Matcap) **and their settings** (stripes, opacity, quality, draft pull, curvature range, matcap and environment, comb) live on the window manager, not the scene. Changing them adds no undo step, so a tool's Adjust Last Operation panel stays open while you switch or tune an overlay, and they are not saved with the .blend. The same controls are in the viewport header's **Analyze** popover (wave icon, shows which overlay is on), reachable while a tool is picking inputs or its F9 panel is open. **Opacity** blends any overlay with the surface's own shading; **See-Through** there is the viewport's X-Ray. Clicks on the sidebar or header pass through while a tool is picking inputs.
* Objects touching an edited surface (a fillet, a joined neighbour) don't follow its CV edits unless they were built from it with construction history.
* Not built yet: toolbar (WorkSpaceTool) tools, and a live preview while picking.
