"""Curve selection in Edit Curve mode with simulated input in a real window:
the picker starts, clicks on curve lines / hull lines in CV, Hull and Curve
mode, Shift+click, double-click a CV, a box in Curve mode:
    blender -b --factory-startup --python tests/test_curve_select_gui.py
The background run writes an empty .blend, starts a GUI Blender with
--enable-event-simulate on it (a window opens for ~15 s), and reads back what
that run found. Screenshots go next to the results (the path is printed)."""
import json, os, subprocess, sys, tempfile

import bpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUI = "--curvesel-gui" in sys.argv


def background():
    tmp = tempfile.mkdtemp(prefix="nproto_curvesel_gui_")
    blend, out = os.path.join(tmp, "empty.blend"), os.path.join(tmp, "results.json")
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    bpy.ops.wm.save_as_mainfile(filepath=blend)
    cmd = [bpy.app.binary_path, "--factory-startup", "--enable-event-simulate", "--window-geometry", "0", "0",
           "1400", "900", blend, "--python", os.path.abspath(__file__), "--", "--curvesel-gui", out]
    try:
        r = subprocess.run(cmd, timeout=240, capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        print("FAIL the GUI run did not finish in 240 s")
        os._exit(1)
    if not os.path.isfile(out):
        print("FAIL the GUI run wrote no results\n" + (r.stdout or "")[-3000:] + (r.stderr or "")[-3000:])
        os._exit(1)
    res = json.load(open(out))
    ok = True
    for name, (good, extra) in res["checks"].items():
        print(("PASS " if good else "FAIL ") + name, extra if not good else "")
        ok &= bool(good)
    if res.get("error"):
        print("FAIL the GUI run raised\n" + res["error"])
        ok = False
    print("screenshots in", tmp)
    print("ALL OK" if ok else "SOME FAILED")
    sys.stdout.flush()
    os._exit(0 if ok else 1)


def gui():
    import traceback
    from mathutils import Vector
    from bpy_extras import view3d_utils
    sys.path.insert(0, ROOT)
    import blender_nurbs
    blender_nurbs.register()
    from blender_nurbs import curvesel

    out = sys.argv[sys.argv.index("--curvesel-gui") + 1]
    shots = os.path.dirname(out)
    checks = {}
    res = {"checks": checks}

    def check(name, cond, extra=""):
        checks[name] = (bool(cond), str(extra))

    ctx = bpy.context
    win = ctx.window_manager.windows[0]
    area = next(a for a in win.screen.areas if a.type == 'VIEW_3D')
    region = next(r for r in area.regions if r.type == 'WINDOW')
    over = lambda: ctx.temp_override(window=win, area=area, region=region)
    v = ctx.window_manager.nproto_view

    def wpx(P):
        q = view3d_utils.location_3d_to_region_2d(region, area.spaces.active.region_3d, Vector(P))
        return int(round(region.x + q[0])), int(round(region.y + q[1]))

    at = [0, 0]

    def move(p):
        for k in (3, 2, 1, 0):
            win.event_simulate('MOUSEMOVE', 'NOTHING', x=p[0] + k, y=p[1] + k)
        at[:] = p

    def click(shift=False):
        if shift:
            win.event_simulate('LEFT_SHIFT', 'PRESS', x=at[0], y=at[1])
        win.event_simulate('LEFTMOUSE', 'PRESS', x=at[0], y=at[1], shift=shift)
        win.event_simulate('LEFTMOUSE', 'RELEASE', x=at[0], y=at[1], shift=shift)
        if shift:
            win.event_simulate('LEFT_SHIFT', 'RELEASE', x=at[0], y=at[1])

    def shot(name):
        area.tag_redraw()
        with over():
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
        with ctx.temp_override(window=win, screen=win.screen):
            bpy.ops.screen.screenshot(filepath=os.path.join(shots, name + ".png"))

    def sel(ob):
        return [tuple(p.select for p in s.points) for s in ob.data.splines]

    def steps():
        cu = bpy.data.curves.new("Stadium", 'CURVE')
        cu.dimensions = '3D'

        def nurbs(pts, order):
            s = cu.splines.new('NURBS')
            s.points.add(len(pts) - 1)
            for p, c in zip(s.points, pts):
                p.co = (*c, 1.0)
            s.order_u = order
            s.use_endpoint_u = True

        nurbs([(-1, 3, 0), (-1, 4, 0), (1, 4, 0), (1, 3, 0)], 3)       # top arc
        nurbs([(1, 3, 0), (1, -3, 0)], 2)                              # right side
        nurbs([(1, -3, 0), (1, -4, 0), (-1, -4, 0), (-1, -3, 0)], 3)   # bottom arc
        nurbs([(-1, -3, 0), (-1, 3, 0)], 2)                            # left side
        ob = bpy.data.objects.new("Stadium", cu)
        ctx.scene.collection.objects.link(ob)
        ctx.view_layer.objects.active = ob
        ob.select_set(True)
        r3 = area.spaces.active.region_3d
        r3.view_perspective = 'ORTHO'
        r3.view_rotation = Vector((0, 0, 1)).to_track_quat('Z', 'Y')
        r3.view_location = Vector((0, 0, 0))
        r3.view_distance = 11.0
        with over():
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.curve.select_all(action='DESELECT')
        yield 0.6
        check("picker runs in Edit Curve mode", curvesel.RUN["on"],
              [op.bl_idname for op in win.modal_operators])
        tool = ctx.workspace.tools.from_space_view3d_mode('EDIT_CURVE', create=False)
        res["tool"] = tool and tool.idname

        # CV mode: hover and click the middle of the right side
        move(wpx((1.0, 0.0, 0.0)))
        yield 0.4
        check("hover on a side lights its curve", curvesel.HOVER["key"] is not None
              and curvesel.HOVER["key"][2] == "CURVE", curvesel.HOVER["key"])
        drawn = {"n": 0, "err": None}
        real = curvesel.draw_view

        def spy(c, dl):
            if curvesel.HOVER["key"] is not None:
                drawn["n"] += 1
            try:
                real(c, dl)
            except Exception as exc:
                drawn["err"] = repr(exc)
                raise
        curvesel.draw_view = spy
        area.tag_redraw()
        yield 0.4
        shot("1_hover_side")
        curvesel.draw_view = real
        check("the hover highlight is drawn", drawn["n"] > 0 and drawn["err"] is None, drawn)
        click()
        yield 0.3
        check("CV: click on a side selects the whole side",
              sel(ob) == [(False,) * 4, (True, True), (False,) * 4, (False, False)], sel(ob))
        # hull line of the top arc, away from the curve
        move(wpx((-0.7, 4.0, 0.0)))          # (at x = 0 the arc touches its hull: the curve's)
        yield 0.2
        click(shift=True)
        yield 0.3
        check("CV: Shift+click on a hull line adds that span",
              sel(ob)[0] == (False, True, True, False) and sel(ob)[1] == (True, True), sel(ob))
        shot("2_side_and_span")
        # a plain click near a CV is still Blender's
        move(wpx((-1.0, -4.0, 0.0)))
        yield 0.2
        click()
        yield 0.3
        check("CV: click on a CV picks just it (Blender)",
              sel(ob) == [(False,) * 4, (False, False), (False, False, True, False), (False, False)], sel(ob))
        # (double-click a CV is not covered: event_simulate sends no DOUBLE_CLICK)
        # click on nothing: Blender deselects
        move(wpx((0.0, 0.0, 0.0)))
        yield 0.2
        click()
        yield 0.3
        check("CV: click on empty space deselects (Blender)", not any(any(f) for f in sel(ob)), sel(ob))
        with over():
            bpy.ops.ed.undo_push(message="test")

        # Hull mode
        v.curve_sel_mode = "HULL"
        move(wpx((0.97, 3.0, 0.0)))
        yield 0.3
        click()
        yield 0.3
        n = sum(sum(f) for f in sel(ob))
        check("Hull: a click near a CV picks one span (two CVs)", n == 2, sel(ob))
        shot("3_hull_span")

        # Curve mode: a click on a CV takes its curve; a box takes every curve it touches
        v.curve_sel_mode = "CURVE"
        move(wpx((-1.0, 3.0, 0.0)))
        yield 0.3
        click()
        yield 0.3
        check("Curve: a click on a shared corner CV selects one whole curve",
              sel(ob)[0] == (True,) * 4 or sel(ob)[3] == (True, True), sel(ob))
        # a box across the left and right sides' middles (no CV inside)
        p0, p1 = wpx((-1.5, 0.5, 0.0)), wpx((1.5, -0.5, 0.0))
        move(p0)
        yield 0.2
        win.event_simulate('LEFTMOUSE', 'PRESS', x=p0[0], y=p0[1])
        for k in range(1, 11):
            q = (int(p0[0] + (p1[0] - p0[0]) * k / 10), int(p0[1] + (p1[1] - p0[1]) * k / 10))
            win.event_simulate('MOUSEMOVE', 'NOTHING', x=q[0], y=q[1])
            at[:] = q
        yield 0.1
        win.event_simulate('LEFTMOUSE', 'RELEASE', x=at[0], y=at[1])
        yield 0.5
        ops = [o.bl_idname for o in ctx.window_manager.operators][-3:]
        check("Curve: a box across both sides selects both whole sides (and only them)",
              sel(ob) == [(False,) * 4, (True, True), (False,) * 4, (True, True)], f"{sel(ob)} ops {ops}")
        shot("4_curve_box")
        UNDO = {"oskey": True} if sys.platform == "darwin" else {"ctrl": True}
        # Ctrl+Z goes back to the corner click's curve
        win.event_simulate('Z', 'PRESS', x=at[0], y=at[1], **UNDO)
        win.event_simulate('Z', 'RELEASE', x=at[0], y=at[1], **UNDO)
        yield 0.5
        check("Ctrl+Z undoes the box (the box does not come back)", sel(ob)[1] == (False, False), sel(ob))
        v.curve_sel_mode = "CV"
        with over():
            bpy.ops.object.mode_set(mode='OBJECT')
        move(wpx((2.0, 2.0, 0.0)))
        yield 0.3
        check("picker stops outside Edit Mode", not curvesel.RUN["on"])

    g = steps()

    def tick():
        try:
            return next(g)
        except StopIteration:
            pass
        except Exception:
            res["error"] = traceback.format_exc()
        json.dump(res, open(out, "w"), indent=1, default=str)
        bpy.ops.wm.quit_blender()
        return None

    bpy.app.timers.register(tick, first_interval=1.0)


if GUI:
    gui()
else:
    background()
