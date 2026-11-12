"""Key hints for the modal tools (sketch tools, picking, trims, checks).

Three places, each with one job:
  header      what is going on now: the tool, the step, live values and the
              modes that are on. One short line, so it fits a laptop screen.
  hint panel  every key, in groups, drawn in the 3D view at the bottom centre
              (of the part not covered by the toolbar and the sidebar). A key
              that cycles options shows all of them with the current one
              highlighted, a toggle shows on / off, an active lock lights up.
              Rows wrap to that free width; the panel steps aside for a shown
              Adjust Last Operation panel.
              ? or F1 collapses it to one line (kept in the add-on's
              preferences).
  status bar  the mouse basics, with Blender's own key icons.

A tool calls show() whenever its state changes and clear() when it ends:

    hints.show(context, "Line", ["place the end", "Length 2 ft"],
               [("Start", [hints.cycle("S", ["free", "tangent"], "free")]),
                ("Direction", [hints.key("H V", "horizontal / vertical", on=True)])],
               mouse=[("MOUSE_LMB", "Place"), ("EVENT_ESC", "Cancel")])
"""

import bpy
from bpy.props import BoolProperty

HUD = {"spec": None, "area": None, "collapsed": False, "top": 0.0}
FONT = 0
SIZE = 11                   # points, times the UI scale
COL_BG = (0.11, 0.11, 0.11, 0.84)
COL_LABEL = (0.56, 0.56, 0.56, 1.0)
COL_TEXT = (0.86, 0.86, 0.86, 1.0)
COL_ON = (1.0, 0.66, 0.22, 1.0)
COL_OFF = (0.5, 0.5, 0.5, 1.0)
COL_NOTE = (0.68, 0.68, 0.68, 1.0)
COL_CAP = (0.24, 0.24, 0.24, 0.95)
COL_CAP_EDGE = (0.46, 0.46, 0.46, 1.0)


# --------------------------------------------------------------------------
# Hint items
# --------------------------------------------------------------------------

def _keys(keys):
    return keys.split() if isinstance(keys, str) else list(keys)


def key(keys, text, on=None):
    """A key, or keys ("H V" = two caps), and what it does. on=True lights it up
    (that lock / mode is active), on=False dims it."""
    return {"keys": _keys(keys), "text": text, "on": on}


def cycle(keys, options, current, text=""):
    """A key that switches between options: all are shown, the current one lit."""
    return {"keys": _keys(keys), "text": text, "options": list(options), "current": current}


def toggle(keys, text, value):
    """An on / off key: its text, then on (lit) or off (dim)."""
    return {"keys": _keys(keys), "text": text, "options": ["on" if value else "off"],
            "current": "on" if value else None}


def note(text):
    """Plain advice, wrapped by words."""
    return {"note": " ".join(str(text).split())}


# --------------------------------------------------------------------------
# Layout (pure: measure(text) -> (width, height) at the panel's size)
# --------------------------------------------------------------------------

def _pieces(item, measure, s):
    """The item as (kind, text, style) pieces kept on one line, and its width."""
    if "note" in item:
        return None
    out = [("cap", k, "on" if item.get("on") else "cap") for k in item["keys"]]
    if item.get("text"):
        style = {True: "on", False: "off"}.get(item.get("on"), "text")
        out.append(("text", item["text"], style))
    if "options" in item:
        for i, o in enumerate(item["options"]):
            if i:
                out.append(("text", "·", "off"))
            out.append(("text", o, "on" if o == item.get("current") else "off"))
    return out


def layout(rows, measure, max_w, s=1.0):
    """Place the rows [(group label, [items])] in at most max_w px.
    Returns (width, height, ops) with ops ("text", x, y, text, style) and
    ("cap", x, y, w, h, text, style); x from the left, y the line's top,
    counted DOWN from the panel's top."""
    pad, gap, inner, capx = 8 * s, 12 * s, 4 * s, 4 * s
    th = measure("Ag")[1]
    line = th + 8 * s
    labels = [r[0] for r in rows if r[0]]
    lab_w = (max(measure(t)[0] for t in labels) + 10 * s) if labels else 0.0
    x0 = pad + lab_w
    right = max(max_w - pad, x0 + 40 * s)
    ops, y, widest = [], pad, 0.0

    def piece_w(kind, text):
        return measure(text)[0] + (2 * capx if kind == "cap" else 0.0)

    for label, items in rows:
        if label:
            ops.append(("text", pad, y, label, "label"))
        x = x0
        for item in items:
            if "note" in item:
                for word in item["note"].split():
                    w = measure(word)[0]
                    if x > x0 and x + w > right:
                        x, y = x0, y + line
                    ops.append(("text", x, y, word, "note"))
                    widest = max(widest, x + w)
                    x += w + measure(" ")[0]
                x += gap
                continue
            ps = _pieces(item, measure, s)
            w = sum(piece_w(k, t) for k, t, _ in ps) + inner * (len(ps) - 1)
            if x > x0 and x + w > right:
                x, y = x0, y + line
            for kind, text, style in ps:
                pw = piece_w(kind, text)
                if kind == "cap":
                    ops.append(("cap", x, y - 2 * s, pw, th + 4 * s, text, style))
                else:
                    ops.append(("text", x, y, text, style))
                x += pw + inner
            widest = max(widest, x - inner)
            x += gap - inner
        y += line
    return widest + pad, y - line + th + pad + 2 * s, ops


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------

def _area_key(area):
    try:
        return area.as_pointer()
    except Exception:
        return id(area)


def _prefs():
    try:
        a = bpy.context.preferences.addons.get(__package__)
        return a.preferences if a is not None else None
    except Exception:
        return None


def collapsed():
    p = _prefs()
    return bool(p.hints_collapsed) if p is not None else HUD["collapsed"]


def show(context, title, state=(), rows=(), mouse=()):
    """Header line, hint panel and status bar for the tool running in this area.
    state: short strings (the step, live values, what is on); rows: [(group,
    [items])]; mouse: [(icon, text)] for the status bar (MOUSE_LMB, EVENT_ESC...)."""
    head = "  ·  ".join([title] + [s for s in state if s])
    area = context.area
    if area is not None:
        try:
            area.header_text_set(head)
        except Exception:
            pass
    HUD["spec"] = {"title": title, "rows": list(rows), "mouse": list(mouse)}
    HUD["area"] = _area_key(area) if area is not None else None
    if mouse:
        try:
            context.workspace.status_text_set(_status_draw)
        except Exception:
            pass
    _redraw(context)
    return head


def clear(context):
    HUD["spec"], HUD["area"] = None, None
    if context.area is not None:
        try:
            context.area.header_text_set(None)
        except Exception:
            pass
    try:
        context.workspace.status_text_set(None)
    except Exception:
        pass
    _redraw(context)


def handle(event):
    """? or F1 pressed: collapse / expand the panel. True when handled."""
    if event.value != 'PRESS' or not (event.type == 'F1' or getattr(event, "unicode", "") == "?"):
        return False
    p = _prefs()
    if p is not None:
        p.hints_collapsed = not p.hints_collapsed
    else:
        HUD["collapsed"] = not HUD["collapsed"]
    return True


def _redraw(context):
    try:
        context.area.tag_redraw()
    except Exception:
        pass


def _status_draw(header, context):
    spec = HUD["spec"]
    if not spec:
        return
    row = header.layout.row(align=False)
    for icon, text in spec["mouse"]:
        try:
            row.label(text=text, icon=icon)
        except Exception:
            row.label(text=text)
        row.separator(factor=2.0)
    row.label(text="Show keys" if collapsed() else "Hide keys", icon='EVENT_F1')


# --------------------------------------------------------------------------
# Drawing (the add-on's POST_PIXEL handler calls draw())
# --------------------------------------------------------------------------

def _insets(area, region):
    """(left, right, redo) for the window region: px covered by the toolbar and
    the sidebar (region overlap), and the Adjust Last Operation panel as
    (x0, x1, top) in region px, or None. That panel keeps its size while it
    is hidden, parked below the view (y under the region's): only one inside
    the view is shown."""
    left = right = 0.0
    redo = None
    x0, x1, y0 = region.x, region.x + region.width, region.y
    for r in getattr(area, "regions", ()):
        if r.width <= 1 or r.height <= 1 or r == region:
            continue
        if r.type == 'TOOLS' and r.x <= x0 + 2:
            left = max(left, r.x + r.width - x0)
        elif r.type == 'UI' and r.x + r.width >= x1 - 2:
            right = max(right, x1 - r.x)
        elif r.type == 'HUD' and r.y >= y0 and r.x >= x0 - 1 and r.x < x1:
            redo = (r.x - x0, r.x + r.width - x0, r.y + r.height - y0)
    return max(left, 0.0), max(right, 0.0), redo


def place(w, h, region_w, insets, s=1.0):
    """Bottom-left corner (x, y) of a w x h panel: centred at the bottom of the
    free width; beside a shown redo panel when it would cover it and there is
    room, else above it."""
    left, right, redo = insets
    m = 10 * s
    L, R = left + m, region_w - right - m
    x, y = L + max(0.0, (R - L - w) * 0.5), m
    if redo is not None and x < redo[1] + m and x + w > redo[0] - m:
        beside = max(L, redo[1] + m)
        if R - beside >= w:
            x = beside + (R - beside - w) * 0.5
        else:
            y = redo[2] + m
    return x, y


def draw(context, region, box):
    """The hint panel, in the area the tool runs in. box(x, y, w, h, color)."""
    import blf
    spec = HUD["spec"]
    HUD["top"] = 0.0
    if not spec or context.area is None or _area_key(context.area) != HUD["area"]:
        return
    s = getattr(context.preferences.system, "ui_scale", 1.0) or 1.0
    blf.size(FONT, SIZE * s)
    measure = lambda t: blf.dimensions(FONT, t)
    ins = _insets(context.area, region)
    rows = spec["rows"]
    if collapsed():
        rows = [("", [key("?", f"{spec['title']} keys", on=None)])]
    elif rows:
        rows = rows[:-1] + [(rows[-1][0], list(rows[-1][1]) + [key("?", "hide", on=False)])]
    max_w = region.width - ins[0] - ins[1] - 20 * s
    w, h, ops = layout(rows, measure, max_w, s)
    X, Y = place(w, h, region.width, ins, s)        # bottom-left corner of the panel
    box(X, Y, w, h, COL_BG)
    th = measure("Ag")[1]
    asc = measure("A")[1]                           # text sits on its baseline: the descender hangs below
    top = Y + h
    col = {"label": COL_LABEL, "text": COL_TEXT, "on": COL_ON, "off": COL_OFF, "note": COL_NOTE, "cap": COL_TEXT}
    for op in ops:
        if op[0] == "cap":
            _, x, y, cw, ch, text, style = op
            yb = top - y - ch                       # flip: y counts down from the top
            edge = COL_ON if style == "on" else COL_CAP_EDGE
            box(X + x, yb, cw, ch, edge)
            box(X + x + s, yb + s, cw - 2 * s, ch - 2 * s, COL_CAP)
            blf.color(FONT, *(COL_ON if style == "on" else COL_TEXT))
            blf.position(FONT, X + x + 4 * s, yb + (ch - th) * 0.5 + (th - asc), 0)
            blf.draw(FONT, text)
        else:
            _, x, y, text, style = op
            blf.color(FONT, *col[style])
            blf.position(FONT, X + x, top - y - asc, 0)
            blf.draw(FONT, text)
    HUD["top"] = top


# --------------------------------------------------------------------------
# Preferences (where the collapsed panel is remembered)
# --------------------------------------------------------------------------

class NPROTO_AP_preferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    hints_collapsed: BoolProperty(name="Collapse Key Hints", default=False,
                                  description="Show the tools' key hints in the 3D view as one line "
                                              "(? or F1 while a tool runs switches it)")

    def draw(self, context):
        self.layout.prop(self, "hints_collapsed")


classes = (NPROTO_AP_preferences,)
