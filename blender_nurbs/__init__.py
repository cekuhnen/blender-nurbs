"""NURBS Modeling -- interactive NURBS surface tools for Blender, built on
NVIDIA's USD BRep (SMLib) kernel."""

import bpy

from . import align, dome, subsel, contcheck, contlabels, isocombs, sections, patch, curvejoin, curvetools, curvetrim, drawtools, cvs, loft_rails, diagnostics, draw, extend, feature_ui, history, history_ui, ops, ops_sweep, ops_trim, refit, salign, square, step_export, step_import, sweeps, toolgroups, ui  # noqa: F401 (sweeps registers builders)

_classes = feature_ui.classes + history_ui.classes + ui.classes + toolgroups.classes + ops.classes + ops_trim.classes + cvs.classes + ops_sweep.classes + step_export.classes + step_import.classes + curvejoin.classes + curvetrim.classes + curvetools.classes + drawtools.classes + loft_rails.classes + contlabels.classes + contcheck.classes + sections.classes + salign.classes + patch.classes + subsel.classes + dome.classes


def register():
    for c in _classes:
        bpy.utils.register_class(c)
    sections.register()
    feature_ui.register()
    history_ui.register()
    ui.register()
    toolgroups.register()
    draw.register()
    history.register()
    cvs.register()
    step_export.register()
    step_import.register()
    subsel.register()


def unregister():
    subsel.unregister()
    step_import.unregister()
    step_export.unregister()
    cvs.unregister()
    history.unregister()
    draw.unregister()
    toolgroups.unregister()
    ui.unregister()
    history_ui.unregister()
    feature_ui.unregister()
    sections.unregister()
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
