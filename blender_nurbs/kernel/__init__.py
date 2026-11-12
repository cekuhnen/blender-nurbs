"""Loads the USD BRep (SMLib) kernel module built by kernel_build/.

Everything else in the add-on talks to the kernel through `sm` (the upstream
Python API, `usd_brep` without the USD submodule) and `ext` (our
`blender_ext` additions: CV access, surfaces from CVs, surface -> Brep).
"""

import os
import sys

_HERE = os.path.dirname(__file__)

sm = None
ext = None
error = None


def load():
    global sm, ext, error
    if sm is not None:
        return sm
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    try:
        import _nproto_kernel as _sm  # noqa: E402
    except ImportError as exc:  # wrong platform / Python version / not built
        error = (
            f"USD BRep kernel not available ({exc}). Build it with kernel_build/ "
            f"for Python {sys.version_info.major}.{sys.version_info.minor}."
        )
        return None
    try:      # end-of-range curve evaluation that every kernel build accepts
        from .harden import harden
        harden(_sm)
    except Exception as exc:
        print("NURBS kernel: evaluation guard not installed:", exc)
    sm = _sm
    ext = _sm.blender_ext
    error = None
    return sm


def available():
    return load() is not None
