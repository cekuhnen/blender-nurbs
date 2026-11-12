"""Loads the USD BRep (SMLib) kernel module built by kernel_build/.

Everything else in the add-on talks to the kernel through `sm` (the upstream
Python API, `usd_brep` without the USD submodule) and `ext` (our
`blender_ext` additions: CV access, surfaces from CVs, surface -> Brep).
"""

import sys

sm = None
ext = None
error = None


def load():
    global sm, ext, error
    if sm is not None:
        return sm
    try:
        # A submodule of this package (bl_ext.<repo>.nurbs_modeling.kernel._nproto_kernel):
        # Blender's extension policy allows no top-level modules and no sys.path changes.
        # (The trim / fillet worker runs in its own Python and imports it from its folder.)
        from . import _nproto_kernel as _sm
    except ImportError as exc:  # wrong platform / Python version / not built
        _sm = _loaded_by_older_version()
        if _sm is None:
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


def _loaded_by_older_version():
    """The add-on was updated in a running Blender whose older version (0.3.0
    and before) loaded the kernel as the top-level module _nproto_kernel. A
    compiled module cannot load twice (its types are registered already):
    use that one until Blender restarts."""
    old = sys.modules.get("_nproto_kernel")
    if old is not None and hasattr(old, "blender_ext"):
        print("NURBS kernel: using the kernel loaded before the add-on was updated; "
              "restart Blender to load the new one.")
        return old
    return None


def available():
    return load() is not None
