"""NURBS Proto -- interactive NURBS surface modeling for Blender, built on
NVIDIA's USD BRep (SMLib) kernel."""

import bpy

from . import align, brep_mesh, mesh_brep, subd_brep, dome, drawshapes, fillet, hints, snapping, subsel, contcheck, contlabels, isocombs, sections, patch, curvejoin, curveoffset, curvetools, curvetrim, drawtools, cvs, loft_rails, diagnostics, draw, extend, feature_ui, history, history_ui, ops, ops_sweep, ops_trim, refit, merge, salign, match, square, step_export, step_import, sweeps, toolgroups, ui, workplanes  # noqa: F401 (sweeps registers builders)

_classes = hints.classes + feature_ui.classes + history_ui.classes + ui.classes + toolgroups.classes + ops.classes + ops_trim.classes + cvs.classes + ops_sweep.classes + step_export.classes + step_import.classes + curvejoin.classes + curveoffset.classes + curvetrim.classes + curvetools.classes + drawtools.classes + drawshapes.classes + fillet.classes + loft_rails.classes + contlabels.classes + contcheck.classes + sections.classes + salign.classes + match.classes + patch.classes + subsel.classes + dome.classes + brep_mesh.classes + mesh_brep.classes + subd_brep.classes + workplanes.classes


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
    cvs.register()
    step_export.register()
    step_import.register()
    subsel.register()
    snapping.register()


def unregister():
    snapping.unregister()
    subsel.unregister()
    step_import.unregister()
    step_export.unregister()
    cvs.unregister()
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
