"""Extend / shrink a surface at one boundary, keeping its flow (Alias "Natural").

Exact: the surface's last span is continued as the SAME polynomial (rational:
the same conic / rational patch). With knots U, degree p and new end
parameter b', the new knot vector is U[0..k] + [b']*(p+1) (k = span holding b';
the last span when extending) and every CV whose knot window touches b' becomes
the blossom of span k's piece at that window. The old part of the surface stays
identical; extending past the end is the analytic continuation of the last
span, a negative distance cuts the surface back exactly.
"""

from . import store
from .kernel import load as _load
from .kernel.natext import curve_point, deboor as _deboor, end_curve, extend_grid, span as _span  # noqa: F401

EDGES = ("U0", "U1", "V0", "V1")


def extend_surface_text(text, edge, distance):
    """Brep text of the single untrimmed surface extended at `edge` (U0/U1/V0/V1)."""
    ext = _load().blender_ext
    brep = store.brep_from_text(text)
    if brep.face_count() != 1:
        raise RuntimeError("only single surfaces can be extended (Unjoin first)")
    d = ext.surface_cvs(brep.faces()[0].surface())
    s = ext.surface_from_cvs(*extend_grid(d, edge, distance))
    return store.brep_to_text(ext.brep_from_surface(s))


def apply_extend_surface(obj, text, inputs, p):
    return extend_surface_text(text, p["edge"], float(p["distance"]))


def _register_step():
    from . import steps
    steps.APPLY["extend_surface"] = apply_extend_surface
    steps.LABEL["extend_surface"] = lambda p: f"Extend · {p['edge']} {p['distance']:+.3g}"


_register_step()
