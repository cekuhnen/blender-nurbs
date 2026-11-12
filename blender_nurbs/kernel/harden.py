"""Make curve evaluation at the very end of a parameter range work on every
kernel build.

Sampling an edge as t0 + (t1 - t0) * k / n can land one rounding step past
the end of its range, and where that range ends differs by a rounding step
between the macOS, Linux and Windows builds. The Linux and Windows kernels
then refuse the end point (SmApiEvaluateCurve: SM_ERR_INVALID_INPUT) where
macOS evaluates it. Edge.point_at and Curve.evaluate are wrapped once, at
load: a call that fails at a parameter within a hair of the range is tried
again just inside it. A parameter really outside the range still raises.
"""

_HAIR = 1e-9          # how far past the range (x its length) still counts as rounding


def _wrap(orig):
    def safe(self, t, *args, **kwargs):
        try:
            return orig(self, t, *args, **kwargs)
        except Exception:
            try:
                t0, t1 = self.parameter_range()
            except Exception:
                raise
            h = (t1 - t0) or 1.0
            if not (t0 - _HAIR * h <= t <= t1 + _HAIR * h):
                raise
            last = None
            for eps in (1e-12, 1e-10, 1e-8):
                try:
                    return orig(self, min(max(t, t0 + eps * h), t1 - eps * h), *args, **kwargs)
                except Exception as exc:
                    last = exc
            raise last
    safe._nproto_safe = True
    safe.__doc__ = getattr(orig, "__doc__", None)
    return safe


def harden(sm):
    """Install the wrappers on the kernel module's classes (once)."""
    for cls_name, meth in (("Edge", "point_at"), ("Curve", "evaluate")):
        cls = getattr(sm, cls_name, None)
        orig = getattr(cls, meth, None) if cls is not None else None
        if orig is None or getattr(orig, "_nproto_safe", False):
            continue
        try:
            setattr(cls, meth, _wrap(orig))
        except (TypeError, AttributeError):
            pass
    return sm
