from typing import Type, Tuple

#
from environment.agent import *
from user.reward_fastpath import ctx_or_compute

# --------------------------------------------------------------------------------
# ----------------------------- REWARD FUNCTIONS API -----------------------------
# --------------------------------------------------------------------------------

def _sign(x: float) -> float:
    return -1.0 if x < 0.0 else (1.0 if x > 0.0 else 0.0)

'''
Example Reward Functions:
- Find more [here](https://colab.research.google.com/drive/1qMs336DclBwdn6JBASa5ioDIfvenW8Ha?usp=sharing#scrollTo=-XAOXXMPTiHJ).
'''

def base_height_l2(
    env: WarehouseBrawl,
    target_height: float,
    obj_name: str = 'player'
) -> float:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # Extract the used quantities (to enable type-hinting)
    obj: GameObject = env.objects[obj_name]

    # Compute the L2 squared penalty
    return ((obj.body.position.y - target_height)**2) * env.dt

class RewardMode(Enum):
    ASYMMETRIC_OFFENSIVE = 0
    SYMMETRIC = 1
    ASYMMETRIC_DEFENSIVE = 2

def damage_interaction_reward(
    env: WarehouseBrawl,
    mode: RewardMode = RewardMode.SYMMETRIC,
) -> float:
    """
    Computes the reward based on damage interactions between players.

    Modes:
    - ASYMMETRIC_OFFENSIVE (0): Reward is based only on damage dealt to the opponent
    - SYMMETRIC (1): Reward is based on both dealing damage to the opponent and avoiding damage
    - ASYMMETRIC_DEFENSIVE (2): Reward is based only on avoiding damage

    Args:
        env (WarehouseBrawl): The game environment
        mode (DamageRewardMode): Reward mode, one of DamageRewardMode

    Returns:
        float: The computed reward.
    """
    # Getting player and opponent from the enviornment
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    # Reward dependent on the mode
    damage_taken = player.damage_taken_this_frame
    damage_dealt = opponent.damage_taken_this_frame

    if mode == RewardMode.ASYMMETRIC_OFFENSIVE:
        reward = damage_dealt
    elif mode == RewardMode.SYMMETRIC:
        reward = damage_dealt - damage_taken
    elif mode == RewardMode.ASYMMETRIC_DEFENSIVE:
        reward = -damage_taken
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return reward / 140



def _update_spam_tracker(p_obj: Any,
                         cur_attack_idx: int,
                         window_frames: int = 5,
                         inc: float = 1.0,
                         decay_per_step: float = 0.15) -> float:
    # persistent fields on the player object
    last_idx = int(getattr(p_obj, "_rw_last_attack_idx", -1))
    frames_since = int(getattr(p_obj, "_rw_frames_since_switch", 9999))
    score = float(getattr(p_obj, "_rw_spam_score", 0.0))

    switched = (cur_attack_idx >= 0 and last_idx >= 0 and cur_attack_idx != last_idx)
    if switched:
        if frames_since <= window_frames:
            score += inc
        frames_since = 0
    else:
        frames_since = min(frames_since + 1, 10_000)
        score = max(0.0, score - decay_per_step)

    setattr(p_obj, "_rw_last_attack_idx", cur_attack_idx)
    setattr(p_obj, "_rw_frames_since_switch", frames_since)
    setattr(p_obj, "_rw_spam_score", score)
    return score

def danger_zone_reward(
    env: WarehouseBrawl,
    zone_penalty: int = 1,
    zone_height: float = 4.2,
    symmetric: bool = False  # set True if you want "too high OR too low" to be penalized
) -> float:
    ctx = ctx_or_compute(env)
    if symmetric:
        overshoot = max(0.0, abs(ctx.py) - zone_height)
    else:
        # current behavior: only penalize when y exceeds +zone_height
        overshoot = max(0.0, ctx.py - zone_height)
    return -zone_penalty * overshoot * ctx.dt

def _nearest_platform_surface(env, px: float, py: float, x_pad: float):
    """find the closest platform surface below the player within horizontal gate"""
    best = None  # (surface_y, left, right, vx, vy)
    for obj in getattr(env, "objects", {}).values():
        if not hasattr(obj, "shape"): 
            continue
        # only consider moving/one-way platforms (Stage). ground is handled by collision anyway.
        if obj.__class__.__name__ != "Stage":
            continue
        bb = obj.shape.cache_bb()
        # horizontal gate with small padding so we encourage being roughly above center
        if px < (bb.left - x_pad) or px > (bb.right + x_pad):
            continue
        # choose the closest surface below the player in the engine's y+ = down convention
        candidates = [bb.top, bb.bottom]
        below = [y for y in candidates if y > py]
        if not below:
            continue
        surface_y = min(below)
        dist_y = surface_y - py  # >= 0 means below us
        # platform velocity (kinematic body)
        pvx = getattr(obj.body, "velocity", (0.0, 0.0))[0]
        pvy = getattr(obj.body, "velocity", (0.0, 0.0))[1]
        if (best is None) or (dist_y < best[0] - py):
            best = (surface_y, bb.left, bb.right, pvx, pvy)
    return best  # or None

# anchor: platform_soft_approach_ctx
def platform_soft_approach(env,
                           x_pad: float = 0.4,
                           y_window: float = 2.5,
                           vy_cap: float = 6.0,
                           reward_scale: float = 0.02) -> float:
    ctx = ctx_or_compute(env)
    p = env.objects["player"]

    is_on_floor = getattr(p, "is_on_floor", None)
    if bool(getattr(p, "on_platform", False)) or (callable(is_on_floor) and is_on_floor()):
        return 0.0
    if not getattr(ctx, "pf_found", False):
        return 0.0

    surface_y = float(ctx.pf_y)
    vy = float(ctx.pvy); pvy = float(ctx.pf_vy)
    vrel_y = max(0.0, vy - pvy)
    dy = surface_y - float(ctx.py)

    if dy <= 0.0 or vrel_y <= 1e-5:
        return 0.0

    w_dist = max(0.0, 1.0 - (dy / y_window))
    w_speed = min(1.0, vrel_y / max(1e-6, vy_cap))

    cx = 0.5 * (float(ctx.pf_left) + float(ctx.pf_right))
    halfw = max(1e-3, 0.5 * (float(ctx.pf_right) - float(ctx.pf_left)))
    w_center = max(0.0, 1.0 - abs(float(ctx.px) - cx) / halfw)

    r = reward_scale * w_dist * w_speed * (0.5 + 0.5 * w_center)
    return r * ctx.dt


def get_ko_bounds(env) -> Tuple[float, float]:
    """exactly match the KO check in PlayerObjectState.physics_process (uses // 2)"""
    half_w = math.floor(env.stage_width_tiles / 2.0)
    half_h = math.floor(env.stage_height_tiles / 2.0)
    return float(half_w), float(half_h)

def edge_safety(env,
                margin_x: float = 2.0,
                margin_y: float = 1.5,
                max_penalty: float = 0.02) -> float:
    ctx = ctx_or_compute(env)
    dx = ctx.half_w - abs(ctx.px)
    dy = ctx.half_h - abs(ctx.py)
    px = max(0.0, (margin_x - dx) / max(1e-6, margin_x))
    py = max(0.0, (margin_y - dy) / max(1e-6, margin_y))
    penalty = (px * px + py * py)
    return -max_penalty * penalty * ctx.dt

def in_state_reward(
    env: WarehouseBrawl,
    desired_state: Type[PlayerObjectState]=BackDashState,
) -> float:
   
    # Get player object from the environment
    player: Player = env.objects["player"]

    
    reward = 1 if isinstance(player.state, desired_state) else 0.0

    return reward * env.dt

def attack_quality_reward(
    env: WarehouseBrawl,
    distance_thresh: float = 2.5,
    near_bonus_scale: float = 0.6,
    far_penalty_scale: float = 1.2,
    misalign_scale: float = 1.0,
    away_scale: float = 0.35,
    v_cap: float = 6.0,
    edge_pad: float = 1.5
) -> float:
    ctx = ctx_or_compute(env)
    if not ctx.p_attacking:
        return 0.0

    r2 = distance_thresh * distance_thresh

    # facing alignment gate
    # tie-break: if overlapping on x now, use previous relative x
    dx_now = ctx.dx
    sdx = _sign(dx_now) if abs(dx_now) > 1e-5 else _sign(ctx.ppx - ctx.opx)
    desired = -sdx if sdx != 0.0 else ctx.p_face
    align = 0.5 * (1.0 + desired * ctx.p_face)  # 0..1

    # distance terms: gate only the near bonus by alignment (keep far penalty intact)
    if ctx.dist2 <= r2:
        gain = (r2 - ctx.dist2) * near_bonus_scale * (0.25 + 0.75 * align)
    else:
        gain = -(ctx.dist2 - r2) * far_penalty_scale

    # away-motion penalty (only when moving away)
    away = 1.0 if (sdx * ctx.pvx) > 0.0 else 0.0
    speed = min(1.0, abs(ctx.pvx) / max(1e-6, v_cap))
    away_term = away_scale * away * speed

    # near-edge penalty when attacking (assumes stage centered at x=0)
    edge = max(0.0, (abs(ctx.px) - (ctx.half_w - edge_pad)) / max(1e-6, edge_pad))
    edge_term = 0.5 * edge

    # misalignment penalty (explicit)
    mis_term = misalign_scale * (1.0 - align)

    return (gain - (mis_term + away_term + edge_term)) * ctx.dt


def attack_misalignment_penalty(
    env: WarehouseBrawl,
    v_cap: float = 6.0,
    edge_pad: float = 1.5
) -> float:
    # unified small penalty to discourage obvious bad swings
    ctx = ctx_or_compute(env)
    if not ctx.p_attacking:
        return 0.0
    sdx = _sign(ctx.dx)
    if sdx == 0.0:
        return 0.0

    desired = -sdx
    align = 0.5 * (1.0 + desired * ctx.p_face)  # 0..1
    mis = 1.0 - align

    away = 1.0 if (sdx * ctx.pvx) > 0.0 else 0.0
    speed = min(1.0, abs(ctx.pvx)/max(1e-6, v_cap))

    edge = max(0.0, (abs(ctx.px) - (ctx.half_w - edge_pad)) / max(1e-6, edge_pad))

    raw = 0.6*mis + 0.3*away*speed + 0.1*edge       # bounded <= 1.0
    return -min(1.0, raw) * ctx.dt

def penalize_useless_attacks_shaped(
    env: WarehouseBrawl,
    distance_thresh: float = 2.75,
    scale: float = 1.0,
) -> float:
    ctx = ctx_or_compute(env)
    if not ctx.p_attacking:
        return 0.0
    gap = ctx.dist2 - distance_thresh * distance_thresh
    penalty = max(0.0, gap) * (-scale)
    return penalty * ctx.dt


def head_to_middle_reward(
    env: WarehouseBrawl,
) -> float:
   
    # reward > 0 iff you moved closer to x=0
    p = env.objects["player"]
    x_prev = p.prev_x
    x_curr = p.body.position.x
    return abs(x_prev) - abs(x_curr)

# anchor: platform_aware_approach_ctx
def platform_aware_approach(env: WarehouseBrawl, y_thresh: float = 0.8, pos_only: bool = True) -> float:
    ctx = ctx_or_compute(env)

    dx0 = abs(ctx.ppx - ctx.opx); dy0 = abs(ctx.ppy - ctx.opy)
    dx1 = abs(ctx.px  - ctx.ox ); dy1 = abs(ctx.py  - ctx.oy )
    delta = (dy0 - dy1) if (dy0 > y_thresh or dy1 > y_thresh) else (dx0 - dx1)
    if pos_only and delta < 0.0:
        return 0.0

    jbo_rew = 0.0
    if getattr(ctx, "pf_found", False):
        pf_h = float(ctx.pf_y)
        # crossing the platform plane (top) this frame
        if ctx.ppy < pf_h <= ctx.py:
            jbo_rew = 1.0

    return (delta * ctx.dt) + jbo_rew


def head_to_opponent(env: WarehouseBrawl, threshold: float = 1.0, pos_only: bool = False) -> float:
    ctx = ctx_or_compute(env)
    r2 = threshold * threshold
    vp = max(0.0, (ctx.ppx-ctx.opx)**2 + (ctx.ppy-ctx.opy)**2 - r2)
    vc = max(0.0, (ctx.px -ctx.ox )**2 + (ctx.py -ctx.oy )**2 - r2)
    delta = vp - vc
    if pos_only and delta < 0.0:
        return 0.0
    return delta * ctx.dt

def holding_nokeys_or_more_than_3keys_penalty(env: WarehouseBrawl) -> float:
    ctx = ctx_or_compute(env)
    keys_held = sum(1 for v in ctx.p_act if float(v) > 0.5)
    return (-ctx.dt) if (keys_held > 3 or keys_held == 0) else (ctx.dt)

def on_win_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return 1.0
    else:
        return -1.0

def on_knockout_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return -1.0
    else:
        return 1.0
    
def on_equip_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == "player":
        if env.objects["player"].weapon == "Hammer":
            return 2.0
        elif env.objects["player"].weapon == "Spear":
            return 1.0
    return 0.0

def on_drop_penalty(env: WarehouseBrawl, agent: str) -> float:
    if agent == "player":
        if env.objects["player"].weapon == "Punch":
            return -2.0
    return 0.0

def on_combo_reward(env: WarehouseBrawl, agent: str) -> float:
    # reward combos for the player; penalize if opponent combos
    return 1.0 if agent == 'player' else -1.0

def spam_penalty(env: WarehouseBrawl,
                 window_frames: int = 5,
                 inc: float = 1.0,
                 decay_per_step: float = 0.15,
                 scale: float = 1.0) -> float:
    ctx = ctx_or_compute(env)
    p = env.objects["player"]
    cur_idx = int(getattr(ctx, "p_attack_idx", -1))
    score = _update_spam_tracker(p, cur_idx,
                                 window_frames=window_frames,
                                 inc=inc,
                                 decay_per_step=decay_per_step)
    return -scale * score * ctx.dt



def weapon_distance_reward(env: WarehouseBrawl) -> float:
    ctx = ctx_or_compute(env)
    d2 = float(getattr(ctx, "w_dist2", float("inf")))
    if math.isfinite(d2):
        return -d2 * ctx.dt
    return 0.0

def downslam_when_lower_than_platform_penalty(env: WarehouseBrawl, platform_height: float = 3.0, penalty: float = 1.0) -> float:
    ctx = ctx_or_compute(env)
    p = env.objects["player"]

    # Check if player is performing a downslam-like action.
    is_downslamming = False
    is_downslamming_attr = getattr(p, "is_downslamming", None)
    if callable(is_downslamming_attr):
        try:
            is_downslamming = bool(is_downslamming_attr())
        except Exception:
            is_downslamming = False

    if not is_downslamming:
        return 0.0

    # Penalize if player is below the platform height
    if ctx.py > platform_height:
        return -penalty * ctx.dt

    return 0.0

def jump_interval_reward(env: WarehouseBrawl, min_interval: float = 1.0, scale: float = 1.0) -> float:
    ctx = ctx_or_compute(env)
    p = env.objects["player"]

    # detect jump start: was on floor and now not, and upward velocity (y+ is down => vy < 0 is upward)
    on_floor = bool(p.is_on_floor()) if hasattr(p, "is_on_floor") else False
    was_on_floor = bool(getattr(p, "_rw_was_on_floor", on_floor))
    vy = float(getattr(p.body, "velocity", (0.0, 0.0))[1])
    jump_started = was_on_floor and (not on_floor) and (vy < -0.1)

    # maintain time-since-last-jump timer on the player
    prev_timer = float(getattr(p, "_rw_time_since_last_jump", 1e9))
    if jump_started:
        interval = prev_timer
        # reset timer
        p._rw_time_since_last_jump = 0.0
        # penalize if interval shorter than ideal `min_interval`
        if interval < min_interval:
            frac = max(0.0, 1.0 - (interval / max(1e-6, min_interval)))
            penalty = - (frac * frac) * scale
            out = penalty
        else:
            out = 0.0
    else:
        # accumulate time
        p._rw_time_since_last_jump = min(1e9, prev_timer + ctx.dt)
        out = 0.0

    # persist floor flag for next frame
    p._rw_was_on_floor = on_floor
    return out

def throw_quality_reward(env: WarehouseBrawl) -> float:
    ctx = ctx_or_compute(env)
    p = env.objects["player"]

    # Determine whether the player is performing a throw-like action.
    # The fast-path context does not expose `p_throwing`, so infer it:
    # 1) Prefer an explicit method/flag on the player if available (e.g., is_throwing())
    # 2) Otherwise, infer from input handler (historically key 'h' used for pickup/throw)
    # 3) Fallback: when no signal exists, gate by generic attacking to avoid constant firing
    p_throwing = False
    is_throwing_attr = getattr(p, "is_throwing", None)
    if callable(is_throwing_attr):
        try:
            p_throwing = bool(is_throwing_attr())
        except Exception:
            p_throwing = False
    if not p_throwing:
        inp = getattr(p, "input", None)
        try:
            if inp is not None and hasattr(inp, "key_status") and 'h' in inp.key_status:
                ks = inp.key_status['h']
                # treat either held or just_pressed as a throw-like event
                p_throwing = bool(getattr(ks, 'held', False) or getattr(ks, 'just_pressed', False))
        except Exception:
            p_throwing = False
    if not p_throwing:
        # graceful fallback: only evaluate when attacking at all
        p_throwing = bool(getattr(ctx, 'p_attacking', False))

    if not p_throwing:
        return 0.0

    # Check if player is facing opponent
    sdx = _sign(ctx.dx)
    if sdx == 0.0:
        # If overlapping, use previous positions
        sdx = _sign(ctx.ppx - ctx.opx)

    desired = -sdx if sdx != 0.0 else ctx.p_face
    align = 0.5 * (1.0 + desired * ctx.p_face)  # 0..1

    # Reward if facing opponent (align > 0.5), penalize if not
    reward = 1.0 if align > 0.5 else -0.5

    return reward * ctx.dt


def fell_off_map_event(env, pad: float = 0.0, only_bottom: bool = False, attack_pen: float = 0.5) -> float:
    """
    returns 1.0 exactly once when the player crosses the KO boundary.
    - pad > 0 shrinks the safe area slightly (fires earlier)
    - set only_bottom=True if you only want bottom falls
    """
    ctx = ctx_or_compute(env)
    p = env.objects["player"]
    added_pen = 0
    if ctx.p_attacking:
        added_pen += attack_pen

    if only_bottom:
        outside = (ctx.py > (ctx.half_h - pad))
    else:
        outside = (abs(ctx.px) > (ctx.half_w - pad)) or (abs(ctx.py) > (ctx.half_h - pad))

    was_outside = bool(getattr(p, "_rw_was_outside", False))
    fire = outside and not was_outside
    setattr(p, "_rw_was_outside", outside)

    return 1.0 + added_pen if fire else 0.0

def idle_penalty(
    env: WarehouseBrawl,
    speed_thresh: float = 0.6,   # ~units/sec that count as "moving"
    ema_tau: float = 0.35        # seconds; larger = slower to react
) -> float:
    """
    penalize standing still using a speed ema.
    skips during committed actions (attack/dash/dodge/backdash).
    """
    ctx = ctx_or_compute(env)
    p = env.objects["player"]

    vx = float(p.body.velocity.x)
    vy = float(p.body.velocity.y)
    speed = math.hypot(vx, vy)

    # ema of speed for stability
    ema_prev = float(getattr(p, "_rw_speed_ema", speed))
    alpha = 1.0 - math.exp(-ctx.dt / max(1e-6, ema_tau))
    ema = (1.0 - alpha) * ema_prev + alpha * speed
    setattr(p, "_rw_speed_ema", ema)

    # grace: don't punish when the agent is in committed movement/attack states
    s = getattr(p, "state", None)
    if isinstance(s, (AttackState, DashState, DodgeState, BackDashState)):
        return 0.0

    # smooth penalty as ema falls below threshold (bounded in [0,1])
    t = max(0.0, (speed_thresh - ema) / max(1e-6, speed_thresh))
    return -(t * t) * ctx.dt


def downslam_penalty(env: WarehouseBrawl, penalty_scale: float = 50.0) -> float:
    """
    Penalize downslam (DSig/DAir) attacks when player is below the nearest ground/platform.
    """
    ctx = ctx_or_compute(env)
    p = env.objects.get("player", None)
    if p is None:
        return 0.0

    # Check if player is in an attack state
    state = getattr(p, "state", None)
    if not isinstance(state, AttackState):
        return 0.0

    # Get the move type being executed
    move_type = getattr(state, "move_type", None)
    if move_type is None:
        return 0.0

    # Check if it's a downslam (DSig or DAir)
    is_downslam = False
    try:
        move_name = getattr(move_type, "name", str(move_type)).upper()
        if "DSIG" in move_name or "DAIR" in move_name or ("DOWN" in move_name and ("SIG" in move_name or "AIR" in move_name)):
            is_downslam = True
    except:
        pass

    if not is_downslam:
        return 0.0

    # Player is performing a downslam attack
    player_y = ctx.py

    # Find the highest platform/ground below the player
    highest_ground_below = -float('inf')

    for obj in getattr(env, "objects", {}).values():
        if obj is p:
            continue

        # --- SAFE: skip objects without a body or position ---
        body = getattr(obj, "body", None)
        if body is None or not hasattr(body, "position"):
            continue

        obj_pos = getattr(body, "position", None)
        if obj_pos is None:
            continue

        obj_y = float(obj_pos.y)

        # Check if it's below player
        if obj_y < player_y:
            highest_ground_below = max(highest_ground_below, obj_y)

    # Apply penalty
    if highest_ground_below > -float('inf'):
        distance_to_ground = player_y - highest_ground_below
        if distance_to_ground < 2.0:
            return -penalty_scale * ctx.dt
        else:
            proximity_factor = max(0.0, 1.0 - (distance_to_ground / 5.0))
            return -penalty_scale * 0.5 * proximity_factor * ctx.dt

    if player_y < ctx.half_h - 1.0:
        return -penalty_scale * 0.3 * ctx.dt

    return 0.0



'''
Add your dictionary of RewardFunctions here using RewTerms
'''
def gen_reward_manager(log_terms: bool=True):
    reward_functions = {
        #'target_height_reward': RewTerm(func=base_height_l2, weight=0.0, params={'target_height': -4, 'obj_name': 'player'}),
        'danger_zone_reward': RewTerm(func=danger_zone_reward, weight=1.0),
        'damage_reward':  RewTerm(func=damage_interaction_reward, weight=(140*67),
                                  params={"mode": RewardMode.ASYMMETRIC_OFFENSIVE}),
        'defence_reward': RewTerm(func=damage_interaction_reward, weight=10.0,
                                  params={"mode": RewardMode.ASYMMETRIC_DEFENSIVE}),
        #'head_to_middle_reward': RewTerm(func=head_to_middle_reward, weight=0.01),
        'platform_aware_approach': RewTerm(func=platform_aware_approach, weight=3.75,
                                           params={"y_thresh": 0.8, "pos_only": True}),
        'move_dir_reward': RewTerm(func=head_to_opponent, weight=10.0),
        'move_towards_reward': RewTerm(func=head_to_opponent, weight=75.0, params={"threshold" : 0.55, "pos_only": True}),
        # 'useless_attk_penalty': RewTerm(func=penalize_useless_attacks_shaped, weight=0.044, params={"distance_thresh" : 2.75, "scale" : 1.25}),
        'attack_quality': RewTerm(
            func=attack_quality_reward,
            weight=44.0,
            params=dict(distance_thresh=1.75, near_bonus_scale=1.0, far_penalty_scale=1.25),
        ),
        'idle_penalty': RewTerm(func=idle_penalty, weight=5.0, params={'speed_thresh': 0.7, 'ema_tau': 0.35}),
        # 'attack_misalign': RewTerm(func=attack_misalignment_penalty, weight=2.0),
        # gentle edge avoidance (dt inside: small)
        'edge_safety':             RewTerm(func=edge_safety, weight=0.77),
        'holding_more_than_3_keys': RewTerm(func=holding_nokeys_or_more_than_3keys_penalty, weight=7.0),
        'taunt_reward': RewTerm(func=in_state_reward, weight=-3.5, params={'desired_state': TauntState}),
        'spam_penalty': RewTerm(func=spam_penalty, weight=8.0, params={'attack_thresh': 3}),
        'jump_interval': RewTerm(func=jump_interval_reward, weight=4.0, params={'min_interval': 1.0, 'scale': 1.0}),
        'downslam_penalty': RewTerm(func=downslam_penalty, weight=1.0, params={'penalty_scale': 20.0}),
        'fell_off_map': RewTerm(func=fell_off_map_event, weight=-400.0, params={'pad': 1.0, 'only_bottom': False}),
        'throw_quality': RewTerm(func=throw_quality_reward, weight=11.0),
        'weapon_distance': RewTerm(func=weapon_distance_reward, weight=4.5),

        'spam_penalty': RewTerm(
            func=spam_penalty,
            weight=10.0,
            # you can tune these if needed:
            params={'window_frames': 5, 'inc': 1.0, 'decay_per_step': 0.15, 'scale': 1.0}
        ),

    }
    signal_subscriptions = {
        'on_win_reward': ('win_signal', RewTerm(func=on_win_reward, weight=50)),
        'on_knockout_reward': ('knockout_signal', RewTerm(func=on_knockout_reward, weight=150)),
        'on_combo_reward': ('hit_during_stun', RewTerm(func=on_combo_reward, weight=7)),
        'on_equip_reward': ('weapon_equip_signal', RewTerm(func=on_equip_reward, weight=45)),
        'on_drop_reward': ('weapon_drop_signal', RewTerm(func=on_drop_penalty, weight=10))
    }
    return RewardManager(reward_functions, signal_subscriptions, log_terms=log_terms)