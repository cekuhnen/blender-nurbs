"""Opening a file (or File > New) flushes everything the add-on kept about the
previous one.

Loading a file ends every running modal operator without calling its own
clean-up, so a tool's viewport overlay (Nudge CVs' CVs and G2 labels, a pick,
a sketch preview, the key hints) would keep drawing the old file's data over
the new one. Caches keyed by object name would also match a same-named object
in the new file. One load_post handler resets them all.

Modules with their own load handler (contcheck, subsel, snapping) keep it:
they do more than drop data (subsel restarts its picker)."""

import copy

import bpy
from bpy.app.handlers import persistent

from . import (anamesh, brep_mesh, contlabels, cvnudge, cvs, curvejoin, curveoffset, diagnostics, draw,
               drawtools, extend, feature_ui, flow, pattern, features, hints, history, isocombs, isodetach, loft_rails, merge, ops, patch, refit,
               scanfit, scanpatch, sections, snapping, square, steps, store, subd_brep, subsel, surfoffset, flange, network, railrevolve, measure, sweeps,
               unroll, workplanes)

# overlays and tool flags with a fixed shape: put back to how they start
_SHAPED = (cvnudge.overlay, draw.pick_overlay, draw.sketch_overlay, draw.ghost, contlabels.hover, ops.GIZMO,
           measure.OVERLAY)
_DEFAULTS = [copy.deepcopy(d) for d in _SHAPED]


def _emptied():
    """Per-file data: last-operation reports, pending rebuilds and caches of
    the old file's objects (emptied, not replaced: other modules hold them)."""
    return (
        # last-operation reports (Adjust Last Operation panels, notes)
        curvejoin.LAST_INFO, curveoffset.LAST_INFO, surfoffset.LAST_INFO, flange.LAST_INFO, network.LAST_INFO, railrevolve.LAST_INFO, loft_rails.LAST_INFO, merge.LAST_INFO, refit.LAST_INFO,
        square.LAST_INFO, square.LAST_NOTES, brep_mesh.LAST, features.LOFT_NOTES, features.LAST_EASES,
        sweeps.boundary_info, patch.info, diagnostics.LAST_RANGE, merge.DEVMAP, scanfit.LAST_INFO, scanpatch.LAST_INFO, unroll.LAST_INFO, flow.LAST_INFO,
        # pending work on object names of the old file
        history._dirty, history._just_built, history._held, cvs._dirty, cvs._modes, cvs._finish,
        feature_ui._resyncing, workplanes._anims, subd_brep._LAST,
        # caches of the old file's objects
        store._cache, store._disp_cache, draw._label_cache, draw._topo_cache, draw._span_cache,
        contlabels._cache, contlabels._marks, contlabels._base_cache, isocombs._cache,
        sections._cache, sections._breps, sections._kref_cache, anamesh._prep, patch._corner_cache,
        steps._split_cache, diagnostics._comb_cache, diagnostics._zebra_cache, diagnostics._curv_cache,
        diagnostics._sag_cache, diagnostics._curv_batches, diagnostics._value_batches,
        snapping._CENTERS, snapping._FACES, snapping._SURF_FLAT, snapping.MESH_CACHE,
        scanfit._SCANS, scanfit._SAMPLERS, scanfit.DEV_CACHE, scanfit.DEV_BATCHES, scanfit.STATS,
        extend._EDGE_CACHE, isodetach._trim_cache, pattern._BUILD_CACHE,
        subsel._valid, subsel._ftris, subsel._epts, subsel._stepf, subsel._mtris, subsel._cpoly,
    )


def flush():
    for d, d0 in zip(_SHAPED, _DEFAULTS):
        d.clear()
        d.update(copy.deepcopy(d0))
    for c in _emptied():
        c.clear()
    merge.WANT_DEVMAP[0] = False
    drawtools.LAST_INFO["text"] = ""
    HUD = hints.HUD                     # the tool's key hints (folded or not stays the user's choice)
    HUD["spec"], HUD["area"] = None, None


@persistent
def _on_load(*_):
    try:
        flush()
    except Exception as exc:
        print("NURBS file load flush:", exc)


def register():
    if _on_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_load)


def unregister():
    if _on_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load)
