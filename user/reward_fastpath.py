# reward_fastpath.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional
import math

def _sign(x: float) -> float:
    return -1.0 if x < 0.0 else (1.0 if x > 0.0 else 0.0)

# anchor: rew_ctx_def
@dataclass(slots=True, frozen=True)
class RewCtx:
    dt: float
    # player
    px: float; py: float; pvx: float; pvy: float
    ppx: float; ppy: float
    p_state: Any; p_attacking: bool; p_grounded: bool
    # inferred facing: -1 left, +1 right, 0 unknown
    p_face: float
    # opponent
    ox: float; oy: float; ovx: float; ovy: float
    opx: float; opy: float
    o_state: Any
    # derived
    dx: float; dy: float; dist2: float
    half_w: float; half_h: float
    # tokens
    _tok_px: float; _tok_py: float; _tok_ox: float; _tok_oy: float

# anchor: unwrap_env
def _unwrap_env(env):
    for attr in ("raw_env", "_env", "env"):
        inner = getattr(env, attr, None)
        if inner is not None and hasattr(inner, "objects"):
            return inner
    return env


def clear_cached_ctx(env) -> None:
    # clears any cached RewCtx on the base sim object
    base = _unwrap_env(env)
    try:
        delattr(base, "_rew_ctx")
    except Exception:
        setattr(base, "_rew_ctx", None)

# anchor: compute_ctx
def _compute_ctx(base, dt: float) -> RewCtx:
    p = base.objects["player"]; o = base.objects["opponent"]

    px = float(p.body.position.x); py = float(p.body.position.y)
    pvx = float(p.body.velocity.x); pvy = float(p.body.velocity.y)

    ox = float(o.body.position.x); oy = float(o.body.position.y)
    ovx = float(o.body.velocity.x); ovy = float(o.body.velocity.y)

    ppx = float(getattr(p, "prev_x", px)); ppy = float(getattr(p, "prev_y", py))
    opx = float(getattr(o, "prev_x", ox)); opy = float(getattr(o, "prev_y", oy))

    dx = px - ox; dy = py - oy; dist2 = dx*dx + dy*dy

    half_w = float(math.floor(getattr(base, "stage_width_tiles", 0) / 2.0))
    half_h = float(math.floor(getattr(base, "stage_height_tiles", 0) / 2.0))

    p_state = getattr(p, "state", None)
    o_state = getattr(o, "state", None)

    # avoid imports to prevent cycles
    p_attacking = (getattr(p_state, "__class__", type).__name__ == "AttackState")
    p_grounded = bool(getattr(p, "is_on_floor", lambda: False)())

    # infer facing sign robustly
    pf_attr = getattr(p, "facing", None)
    p_face = 0.0
    if isinstance(pf_attr, (int, float)) and pf_attr != 0:
        p_face = -1.0 if pf_attr < 0 else 1.0
    else:
        s = str(pf_attr) if pf_attr is not None else ""
        if "LEFT" in s:  p_face = -1.0
        elif "RIGHT" in s: p_face =  1.0
        else:
            # fallback on recent motion
            dx_step = px - ppx
            if abs(pvx) > 0.1:
                p_face = _sign(pvx)
            elif abs(dx_step) > 1e-3:
                p_face = _sign(dx_step)
            else:
                p_face = 0.0

    return RewCtx(
        dt=float(dt),
        px=px, py=py, pvx=pvx, pvy=pvy, ppx=ppx, ppy=ppy, p_state=p_state,
        p_attacking=p_attacking, p_grounded=p_grounded, p_face=p_face,
        ox=ox, oy=oy, ovx=ovx, ovy=ovy, opx=opx, opy=opy, o_state=o_state,
        dx=dx, dy=dy, dist2=dist2, half_w=half_w, half_h=half_h,
        _tok_px=px, _tok_py=py, _tok_ox=ox, _tok_oy=oy,
    )

# anchor: get_ctx
def get_ctx(env, dt: Optional[float] = None) -> RewCtx:
    base = _unwrap_env(env)
    if dt is None:
        fps = float(getattr(base, "fps", 30.0))
        dt = 1.0 / fps if fps > 0 else 1.0 / 30.0
    ctx = _compute_ctx(base, float(dt))
    base._rew_ctx = ctx
    return ctx

# anchor: ctx_or_compute
def ctx_or_compute(env, dt: Optional[float] = None) -> RewCtx:
    base = _unwrap_env(env)
    ctx = getattr(base, "_rew_ctx", None)
    if isinstance(ctx, RewCtx):
        return ctx
    return get_ctx(base, dt)

__all__ = ["RewCtx", "get_ctx", "ctx_or_compute"]
