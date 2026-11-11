"""Package the add-on as a Blender extension zip with the kernel builds it has.

    python3 tools/package_extension.py            # -> dist/nurbs_modeling-<version>.zip
    python3 tools/package_extension.py --check    # only list the kernel builds found

The kernel is a compiled module, one file per platform, all side by side in
blender_nurbs/kernel/ (Python loads the one whose suffix matches it):

    _nproto_kernel.cpython-313-darwin.so            macOS (arm64, Intel, or both: read from the file)
    _nproto_kernel.cp313-win_amd64.pyd              Windows x64
    _nproto_kernel.cp313-win_arm64.pyd              Windows arm64
    _nproto_kernel.cpython-313-x86_64-linux-gnu.so  Linux x64

The zip's blender_manifest.toml lists exactly the platforms found, so Blender
refuses to install it where it has no kernel. The add-on folder itself is not
changed.
"""

import argparse
import fnmatch
import os
import re
import shutil
import struct
import sys
import tempfile
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADDON = os.path.join(ROOT, "blender_nurbs")
KERNEL = os.path.join(ADDON, "kernel")
SKIP = ["__pycache__", "*.pyc", ".DS_Store", "*.new"]

MACHO_CPU = {0x0100000C: "macos-arm64", 0x01000007: "macos-x64"}


def macho_platforms(path):
    """Platforms of a macOS binary from its Mach-O header (thin or universal)."""
    with open(path, "rb") as f:
        head = f.read(8)
        if head[:4] == b"\xca\xfe\xba\xbe":                       # universal (fat, big-endian)
            n = struct.unpack(">I", head[4:8])[0]
            cpus = [struct.unpack(">I", f.read(20)[:4])[0] for _ in range(n)]
        elif head[:4] == b"\xcf\xfa\xed\xfe":                     # 64-bit Mach-O, little-endian
            cpus = [struct.unpack("<I", head[4:8])[0]]
        else:
            return []
    return sorted({MACHO_CPU[c] for c in cpus if c in MACHO_CPU})


def kernel_builds():
    """{file name: [platforms]} of the kernel modules in blender_nurbs/kernel."""
    out = {}
    for name in sorted(os.listdir(KERNEL)):
        if not name.startswith("_nproto_kernel."):
            continue
        path = os.path.join(KERNEL, name)
        if name.endswith("-darwin.so"):
            out[name] = macho_platforms(path)
        elif name.endswith("-win_amd64.pyd"):
            out[name] = ["windows-x64"]
        elif name.endswith("-win_arm64.pyd"):
            out[name] = ["windows-arm64"]
        elif name.endswith("-x86_64-linux-gnu.so"):
            out[name] = ["linux-x64"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="only list the kernel builds found")
    ap.add_argument("--out", default=os.path.join(ROOT, "dist"), help="output folder")
    args = ap.parse_args()

    builds = kernel_builds()
    for name, plats in builds.items():
        print(f"  {name}: {', '.join(plats) or 'unknown'}")
    platforms = sorted({p for plats in builds.values() for p in plats})
    if not platforms:
        print("no kernel build found in blender_nurbs/kernel -- build it with kernel_build/ first")
        return 1
    print("platforms:", ", ".join(platforms))
    if args.check:
        return 0

    manifest = open(os.path.join(ADDON, "blender_manifest.toml"), encoding="utf-8").read()
    version = re.search(r'^version\s*=\s*"([^"]+)"', manifest, re.M).group(1)
    ext_id = re.search(r'^id\s*=\s*"([^"]+)"', manifest, re.M).group(1)
    manifest = re.sub(r'^platforms\s*=.*$', "platforms = [" + ", ".join(f'"{p}"' for p in platforms) + "]",
                      manifest, count=1, flags=re.M)

    os.makedirs(args.out, exist_ok=True)
    zpath = os.path.join(args.out, f"{ext_id}-{version}.zip")
    with tempfile.TemporaryDirectory() as tmp:
        stage = os.path.join(tmp, ext_id)
        shutil.copytree(ADDON, stage, ignore=shutil.ignore_patterns(*SKIP))
        with open(os.path.join(stage, "blender_manifest.toml"), "w", encoding="utf-8", newline="\n") as f:
            f.write(manifest)
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for dirpath, dirs, files in os.walk(stage):
                dirs[:] = sorted(d for d in dirs if not any(fnmatch.fnmatch(d, s) for s in SKIP))
                for fn in sorted(files):
                    if any(fnmatch.fnmatch(fn, s) for s in SKIP):
                        continue
                    full = os.path.join(dirpath, fn)
                    z.write(full, os.path.relpath(full, stage))
    print("wrote", zpath, f"({os.path.getsize(zpath) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
