"""NURBS Proto -- interactive NURBS surface modeling for Blender, built on
NVIDIA's USD BRep (SMLib) kernel."""

import bpy

from . import align, blendedit, brep_mesh, mesh_brep, subd_brep, dome, drawshapes, fillet, hints, snapping, subsel, contcheck, contlabels, isocombs, isodetach, loftslide, sections, patch, pivot, pivottool, movetool, aligntool, curvejoin, curveoffset, surfoffset, flange, network, railrevolve, measure, curvetools, curvetrim, drawtools, cvs, cvnudge, loft_rails, diagnostics, draw, extend, feature_ui, fileload, history, history_ui, ops, ops_sweep, ops_trim, pattern, pushpull, tween, unroll, flow, emboss, refit, merge, salign, scanfit, scanpatch, match, square, step_export, step_import, sweeps, symmetry, toolgroups, ui, workplanes  # noqa: F401 (sweeps registers builders)

_classes = hints.classes + feature_ui.classes + history_ui.classes + ui.classes + toolgroups.classes + ops.classes + ops_trim.classes + isodetach.classes + cvs.classes + cvnudge.classes + ops_sweep.classes + step_export.classes + step_import.classes + curvejoin.classes + curveoffset.classes + surfoffset.classes + flange.classes + network.classes + railrevolve.classes + measure.classes + curvetrim.classes + curvetools.classes + drawtools.classes + drawshapes.classes + fillet.classes + loft_rails.classes + contlabels.classes + loftslide.classes + contcheck.classes + sections.classes + salign.classes + match.classes + patch.classes + subsel.classes + dome.classes + brep_mesh.classes + mesh_brep.classes + subd_brep.classes + scanfit.classes + scanpatch.classes + workplanes.classes + pivottool.classes + movetool.classes + aligntool.classes + pattern.classes + symmetry.classes + pushpull.classes + tween.classes + unroll.classes + flow.classes + emboss.classes


def register():
    for c in _classes:
        bpy.utils.register_class(c)
    sections.register()
    workplanes.register()
    feature_ui.register()
    history_ui.register()
    ui.register()
    toolgroups.register()
    brep_mesh.register()
    mesh_brep.register()
    subd_brep.register()
    draw.register()
    history.register()
    pivot.register()
    pivottool.register()
    movetool.register()
    aligntool.register()
    tween.register()
    cvs.register()
    blendedit.register()
    symmetry.register()
    cvnudge.register()
    step_export.register()
    step_import.register()
    subsel.register()
    contcheck.register()
    snapping.register()
    fileload.register()


def unregister():
    fileload.unregister()
    snapping.unregister()
    contcheck.unregister()
    subsel.unregister()
    step_import.unregister()
    step_export.unregister()
    cvnudge.unregister()
    symmetry.unregister()
    blendedit.unregister()
    cvs.unregister()
    movetool.unregister()
    aligntool.unregister()
    tween.unregister()
    pivottool.unregister()
    pivot.unregister()
    history.unregister()
    draw.unregister()
    subd_brep.unregister()
    mesh_brep.unregister()
    brep_mesh.unregister()
    toolgroups.unregister()
    ui.unregister()
    history_ui.unregister()
    feature_ui.unregister()
    workplanes.unregister()
    sections.unregister()
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
