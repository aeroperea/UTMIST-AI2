
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Tuple

def _sign(x: float) -> float:
    return -1.0 if x < 0.0 else (1.0 if x > 0.0 else 0.0)

# robust mapper: enum/int/str -> {-1,0,+1}
def _facing_sign_from_attr(v: Any) -> float:
    # numeric value (preferred if present)
    val = getattr(v, "value", None)
    if isinstance(val, (int, float)) and val != 0:
        return _sign(float(val))
    # enum name or direct string
    name = getattr(v, "name", None)
    s = (name if isinstance(name, str) else (str(v) if v is not None else "")).upper()
    if "LEFT" in s:
        return -1.0
    if "RIGHT" in s:
        return 1.0
    # raw numeric attr
    if isinstance(v, (int, float)) and v != 0:
        return _sign(float(v))
    return 0.0

@dataclass(slots=True, frozen=True)
class RewCtx:
    dt: float
    # player
    px: float; py: float; pvx: float; pvy: float
    ppx: float; ppy: float
    p_state: Any; p_attacking: bool; p_grounded: bool
    p_face: float
    # opponent
    ox: float; oy: float; ovx: float; ovy: float
    opx: float; opy: float
    o_state: Any
    # actions
    p_act: Tuple[float, ...]
    o_act: Tuple[float, ...]
    # derived
    dx: float; dy: float; dist2: float
    half_w: float; half_h: float
    # platform
    pf_found: bool; pf_y: Optional[float]; pf_left: Optional[float]; pf_right: Optional[float]
    pf_vx: float; pf_vy: float
    # weapon
    w_obj: Any; wx: Optional[float]; wy: Optional[float]; w_dist2: float
    # anti-jitter (read-only; reward updates the score on the player object)
    p_attack_idx: int
    # tokens
    _tok_px: float; _tok_py: float; _tok_ox: float; _tok_oy: float


# anchor: unwrap_env
def _unwrap_env(e: Any) -> Any:
    # walk through common wrapper fields to reach the base env
    seen = set()
    cur = e
    for _ in range(8):  # shallow chain guard
        nxt = None
        if hasattr(cur, "raw_env"):
            nxt = getattr(cur, "raw_env")
        elif hasattr(cur, "env"):
            nxt = getattr(cur, "env")
        if not nxt or nxt is cur or nxt in seen:
            break
        seen.add(cur)
        cur = nxt
    return cur

def _resolve_agent_ids(env_like: Any) -> Tuple[int, int]:
    """
    returns (player_id, opponent_id) robustly.
    tries multiple common fields; falls back to (0,1).
    """
    b = _unwrap_env(env_like)

    # explicit id fields on base
    for a, bname in (("id_player", "id_opponent"),
                     ("player_id", "opponent_id"),
                     ("learner_id", "opponent_id"),
                     ("left_id", "right_id")):
        if hasattr(b, a) and hasattr(b, bname):
            try:
                return int(getattr(b, a)), int(getattr(b, bname))
            except Exception:
                pass

    # name->id maps
    for map_name in ("agent_name_to_id", "name_to_id", "agents_map"):
        m = getattr(b, map_name, None)
        if isinstance(m, dict):
            if "player" in m and "opponent" in m:
                try:
                    return int(m["player"]), int(m["opponent"])
                except Exception:
                    pass

    # vectorized self-play sometimes exports ordered ids
    for seq_name in ("agent_ids", "agents", "ids"):
        seq = getattr(b, seq_name, None)
        if isinstance(seq, (list, tuple)) and len(seq) >= 2:
            try:
                return int(seq[0]), int(seq[1])
            except Exception:
                pass

    # final fallback
    return 0, 1


def clear_cached_ctx(env) -> None:
    base = _unwrap_env(env)
    try:
        delattr(base, "_rew_ctx")
    except Exception:
        setattr(base, "_rew_ctx", None)

# anchor: compute_ctx

def _ctx_posvel(base) -> Tuple[Any, Any, float, float, float, float, float, float, float, float,
                                float, float, float, float, float, float, float]:
    p = base.objects["player"]; o = base.objects["opponent"]
    px = float(p.body.position.x); py = float(p.body.position.y)
    pvx = float(p.body.velocity.x); pvy = float(p.body.velocity.y)
    ox = float(o.body.position.x); oy = float(o.body.position.y)
    ovx = float(o.body.velocity.x); ovy = float(o.body.velocity.y)
    ppx = float(getattr(p, "prev_x", px)); ppy = float(getattr(p, "prev_y", py))
    opx = float(getattr(o, "prev_x", ox)); opy = float(getattr(o, "prev_y", oy))
    dx = px - ox; dy = py - oy; dist2 = dx*dx + dy*dy
    return p, o, px, py, pvx, pvy, ox, oy, ovx, ovy, ppx, ppy, opx, opy, dx, dy, dist2

def _ctx_stage_half_extents(base) -> Tuple[float, float]:
    sw = getattr(base, "stage_width_world", None)
    sh = getattr(base, "stage_height_world", None)
    if isinstance(sw, (int, float)) and isinstance(sh, (int, float)):
        return 0.5 * float(sw), 0.5 * float(sh)
    return 0.5 * float(getattr(base, "stage_width_tiles", 0)), \
           0.5 * float(getattr(base, "stage_height_tiles", 0))

def _ctx_infer_flags(base, p, p_state, px, ppx, pvx) -> Tuple[bool, bool, float]:
    # attacking
    p_attacking = False
    is_att_fn = getattr(p, "is_attacking", None)
    if callable(is_att_fn):
        try: p_attacking = bool(is_att_fn())
        except Exception: p_attacking = False
    if not p_attacking:
        p_attacking = (getattr(p_state, "__class__", type).__name__ == "AttackState")

    # grounded
    is_on_floor = getattr(p, "is_on_floor", None)
    if callable(is_on_floor):
        try: p_grounded = bool(is_on_floor())
        except Exception: p_grounded = False
    else:
        p_grounded = bool(getattr(p, "on_floor", False) or getattr(p, "grounded", False))

    # facing (sticky)
    last_face = float(getattr(base, "_rw_last_face", 0.0) or 0.0)
    fs = getattr(p, "facing_sign", None)
    p_face = _sign(float(fs)) if isinstance(fs, (int, float)) and float(fs) != 0.0 else 0.0
    if p_face == 0.0:
        p_face = _facing_sign_from_attr(getattr(p, "facing", None))
    if p_face == 0.0:
        dx_step = px - ppx
        if abs(pvx) > 0.1: p_face = _sign(pvx)
        elif abs(dx_step) > 1e-3: p_face = _sign(dx_step)
        else: p_face = last_face
    if p_face != 0.0:
        try: setattr(base, "_rw_last_face", p_face)
        except Exception: pass

    return p_attacking, p_grounded, p_face

def _ctx_action_len(base) -> int:
    try:
        if hasattr(base, "action_space") and getattr(base.action_space, "shape", None):
            return int(base.action_space.shape[0])
        if hasattr(base, "action_spaces") and isinstance(base.action_spaces, dict):
            sp = base.action_spaces.get(0) or next(iter(base.action_spaces.values()))
            return int(getattr(sp, "shape", (0,))[0])
        if hasattr(base, "act_helper") and hasattr(base.act_helper, "low"):
            return int(len(base.act_helper.low))
    except Exception:
        pass
    return 10

def _coerce_action(v, n: int) -> tuple[float, ...]:
    if v is None: return tuple(0.0 for _ in range(n))
    try: seq = [float(x) for x in list(v)]
    except Exception: return tuple(0.0 for _ in range(n))
    if len(seq) < n: seq += [0.0] * (n - len(seq))
    elif len(seq) > n: seq = seq[:n]
    return tuple(seq)

def _ctx_extract_actions(base, act_len: int) -> Tuple[tuple[float, ...], tuple[float, ...]]:
    pid, oid = _resolve_agent_ids(base)
    cur_action = (
        getattr(base, "cur_action", None)
        or getattr(_unwrap_env(base), "cur_action", None)
        or {}
    )
    p_src = None; o_src = None
    if isinstance(cur_action, dict):
        p_src = cur_action.get(pid); o_src = cur_action.get(oid)
        if (p_src is None or o_src is None) and ("player" in cur_action and "opponent" in cur_action):
            p_src = cur_action.get("player"); o_src = cur_action.get("opponent")
        if (p_src is None or o_src is None) and (0 in cur_action and 1 in cur_action):
            p_src = cur_action.get(0); o_src = cur_action.get(1)
    elif isinstance(cur_action, (list, tuple)) and len(cur_action) >= 2:
        ids = getattr(base, "agent_ids", None) or getattr(_unwrap_env(base), "agent_ids", None)
        if isinstance(ids, (list, tuple)) and len(ids) >= 2:
            try:
                idx_p = ids.index(pid); idx_o = ids.index(oid)
            except ValueError:
                idx_p, idx_o = 0, 1
        else:
            idx_p, idx_o = 0, 1
        p_src = cur_action[idx_p]; o_src = cur_action[idx_o]
    return _coerce_action(p_src, act_len), _coerce_action(o_src, act_len)

def _ctx_scan_env(base, px: float, py: float, x_pad: float = 0.4):
    try:
        from pymunk import ShapeFilter
    except Exception:
        ShapeFilter = None
    try:
        from environment.constants import WEAPON_CAT
    except Exception:
        WEAPON_CAT = 0

    pf_found = False
    pf_y = None; pf_left = None; pf_right = None; pf_vx = 0.0; pf_vy = 0.0
    w_obj = None; wx = None; wy = None; w_dist2 = float("inf")

    for obj in getattr(base, "objects", {}).values():
        sh = getattr(obj, "shape", None)
        if sh is None:
            continue
        clsname = obj.__class__.__name__

        # platform surfaces (assumes "Stage")
        if clsname == "Stage":
            try:
                bb = sh.cache_bb()
                if (px >= (bb.left - x_pad)) and (px <= (bb.right + x_pad)):
                    candidates = (bb.top, bb.bottom)
                    below = [y for y in candidates if y > py]
                    if below:
                        surface_y = min(below)
                        if (not pf_found) or (surface_y < float(pf_y)):
                            pf_found = True
                            pf_y = float(surface_y)
                            pf_left = float(bb.left)
                            pf_right = float(bb.right)
                            vel = getattr(obj, "body", None)
                            if vel is not None:
                                vxvy = getattr(obj.body, "velocity", (0.0, 0.0))
                                pf_vx = float(vxvy[0]) if len(vxvy) > 0 else 0.0
                                pf_vy = float(vxvy[1]) if len(vxvy) > 1 else 0.0
            except Exception:
                pass

        # nearest weapon by category
        if ShapeFilter is not None and WEAPON_CAT:
            try:
                filt = getattr(sh, "filter", None)
                if isinstance(filt, ShapeFilter) and (filt.categories & WEAPON_CAT):
                    oxw = float(obj.body.position.x); oyw = float(obj.body.position.y)
                    d2 = (oxw - px)*(oxw - px) + (oyw - py)*(oyw - py)
                    if d2 < w_dist2:
                        w_dist2 = d2; w_obj = obj; wx = oxw; wy = oyw
            except Exception:
                pass

    if w_obj is None:
        w_dist2 = float("inf")

    return pf_found, pf_y, pf_left, pf_right, pf_vx, pf_vy, w_obj, wx, wy, w_dist2

def _resolve_attack_idx(base, p_act: Tuple[float, ...],
                        attack_idxs: Tuple[int, ...],
                        thr: float = 0.5) -> int:
    # pick active attack idx among attack_idxs if above thr
    best_i = -1; best_v = thr
    for i in attack_idxs:
        if 0 <= i < len(p_act):
            v = float(p_act[i])
            if v > best_v:
                best_v = v; best_i = i
    return best_i


# replace your whole _compute_ctx with this version

def _compute_ctx(base, dt: float) -> RewCtx:
    p, o, px, py, pvx, pvy, ox, oy, ovx, ovy, ppx, ppy, opx, opy, dx, dy, dist2 = _ctx_posvel(base)
    half_w, half_h = _ctx_stage_half_extents(base)
    p_state = getattr(p, "state", None); o_state = getattr(o, "state", None)
    p_attacking, p_grounded, p_face = _ctx_infer_flags(base, p, p_state, px, ppx, pvx)
    act_len = _ctx_action_len(base)
    p_act, o_act = _ctx_extract_actions(base, act_len)
    pf_found, pf_y, pf_left, pf_right, pf_vx, pf_vy, w_obj, wx, wy, w_dist2 = _ctx_scan_env(base, px, py, x_pad=0.4)

    # anti-jitter: infer which attack index is active (ctx exposes it; reward will update score)
    attack_idxs = tuple(getattr(base, "attack_action_indices", (5, 6))) or (5, 6)
    p_attack_idx = _resolve_attack_idx(base, p_act, attack_idxs, thr=0.5)

    return RewCtx(
        dt=float(dt),
        px=px, py=py, pvx=pvx, pvy=pvy, ppx=ppx, ppy=ppy, p_state=p_state,
        p_attacking=p_attacking, p_grounded=p_grounded, p_face=p_face,
        ox=ox, oy=oy, ovx=ovx, ovy=ovy, opx=opx, opy=opy, o_state=o_state,
        p_act=p_act, o_act=o_act,
        dx=dx, dy=dy, dist2=dist2, half_w=half_w, half_h=half_h,
        pf_found=pf_found, pf_y=pf_y, pf_left=pf_left, pf_right=pf_right, pf_vx=pf_vx, pf_vy=pf_vy,
        w_obj=w_obj, wx=wx, wy=wy, w_dist2=w_dist2,
        p_attack_idx=p_attack_idx,
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

__all__ = ["RewCtx", "get_ctx", "ctx_or_compute", "clear_cached_ctx"]