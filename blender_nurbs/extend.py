"""Extend / shrink a surface at one boundary, keeping its flow (Alias "Natural").

Exact: the surface's last span is continued as the SAME polynomial (rational:
the same conic / rational patch). With knots U, degree p and new end
parameter b', the new knot vector is U[0..k] + [b']*(p+1) (k = span holding b';
the last span when extending) and every CV whose knot window touches b' becomes
the blossom of span k's piece at that window. The old part of the surface stays
identical; extending past the end is the analytic continuation of the last
span, a negative distance cuts the surface back exactly.

A trimmed surface extends at the edges that still lie on its own boundary
(kernel/trimext.py): the carrier is extended the same way and the face is
trimmed again; its trim edges are refused (they are not a boundary the
surface can be continued from).
"""

from . import store
from .kernel import load as _load
from .kernel.natext import curve_point, deboor as _deboor, end_curve, extend_grid, span as _span  # noqa: F401

EDGES = ("U0", "U1", "V0", "V1")

TRIM_EDGE = "Trim edge: only edges on the surface's own boundary extend · Untrim or Refit to Trim first"
JOINED = "Joined surfaces: Unjoin first, then extend one surface"

_EDGE_CACHE = {}     # obj.session_uid -> (revision, {edge index: (iso, along start, along end)})


def boundary_edges(obj):
    """{E index: (iso, along start, along end)} of a one-face trimmed object:
    its edges on the carrier's own boundary (cached per geometry revision)."""
    from .kernel import trimext
    rev = obj.get(store.PROP_REV)
    hit = _EDGE_CACHE.get(obj.session_uid)
    if hit is not None and hit[0] == rev:
        return hit[1]
    try:
        out = trimext.boundary_edges(store.get_brep(obj))
    except Exception as exc:
        print("NURBS extend:", obj.name, exc)
        out = {}
    _EDGE_CACHE[obj.session_uid] = (rev, out)
    return out


def refusal(obj, ref):
    """Why Extend cannot take this edge ('' when it can)."""
    if store.is_natural(obj):
        return "" if ref in EDGES else TRIM_EDGE
    if store.get_brep(obj).face_count() != 1:
        return JOINED
    if ref.startswith("E") and ref[1:].isdigit() and int(ref[1:]) in boundary_edges(obj):
        return ""
    return TRIM_EDGE


def resolve(obj, ref):
    """(iso, at, (along lo, along hi), trimmed?) of a picked edge: the boundary
    it lies on, where it was picked along it (None: untrimmed, the whole edge)
    and the along range the extension spans. None when it cannot be extended."""
    if refusal(obj, ref):
        return None
    surf = store.get_surfaces(obj)[0]
    if store.is_natural(obj):
        (u0, u1), (v0, v1) = surf.u_domain(), surf.v_domain()
        rng = (v0, v1) if ref in ("U0", "U1") else (u0, u1)
        return ref, None, rng, False
    from .kernel import trimext
    iso, a0, a1 = boundary_edges(obj)[int(ref[1:])]
    at = 0.5 * (a0 + a1)
    run = trimext.run_range(store.get_brep(obj), iso, at)
    if run is None:
        return None
    return iso, at, (min(run), max(run)), True


def extend_brep(brep, edge, distance, at=None):
    """The one-face brep extended at `edge` (U0/U1/V0/V1). Untrimmed: the whole
    surface; trimmed: at the run of its edges on that boundary around along = at."""
    from .kernel import trimext
    ext = _load().blender_ext
    if brep.face_count() != 1:
        raise RuntimeError("only single surfaces can be extended (Unjoin first)")
    if at is None or trimext.is_natural(brep):        # no pick position: an untrimmed pick (as before)
        d = ext.surface_cvs(brep.faces()[0].surface())
        return ext.brep_from_surface(ext.surface_from_cvs(*extend_grid(d, edge, distance)))
    return trimext.extend_trimmed(brep, edge, at, distance)


def extend_surface_text(text, edge, distance, at=None):
    """Brep text of the single surface extended at `edge` (U0/U1/V0/V1)."""
    return store.brep_to_text(extend_brep(store.brep_from_text(text), edge, distance, at))


def result_natural(obj):
    """Is obj's geometry (after an extend) an untrimmed surface?"""
    from .kernel import trimext
    try:
        return trimext.is_natural(store.get_brep(obj))
    except Exception:
        return False


def apply_extend_surface(obj, text, inputs, p):
    return extend_surface_text(text, p["edge"], float(p["distance"]), p.get("at"))


def _register_step():
    from . import steps
    steps.APPLY["extend_surface"] = apply_extend_surface
    steps.LABEL["extend_surface"] = lambda p: f"Extend · {p['edge']} {p['distance']:+.3g}"


_register_step()
