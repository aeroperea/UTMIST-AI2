import numpy as np
from typing import Optional, Type, List, Tuple
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

def platform_soft_approach(env,
                           x_pad: float = 0.4,
                           y_window: float = 2.5,
                           vy_cap: float = 6.0,
                           reward_scale: float = 0.02) -> float:
    """
    reward clean approaches to a platform: horizontally aligned, descending toward the surface,
    stronger when time-to-contact is small but not zero. no penalties for air whiffs.
    """
    p = env.objects["player"]
    px, py = float(p.body.position.x), float(p.body.position.y)
    vx, vy = float(p.body.velocity.x), float(p.body.velocity.y)

    # already grounded or riding a platform? no shaping needed
    if getattr(p, "on_platform", None) is not None or p.is_on_floor():
        return 0.0

    surf = _nearest_platform_surface(env, px, py, x_pad)
    if surf is None:
        return 0.0

    surface_y, left, right, pvx, pvy = surf

    # relative vertical velocity (y+ is down in your codebase)
    vrel_y = max(0.0, vy - float(pvy))  # only reward if descending relative to platform
    dy = surface_y - py                  # > 0 only if surface is below the player

    if dy <= 0.0 or vrel_y <= 1e-5:
        return 0.0

    # distance and speed weights (smooth, bounded)
    w_dist = max(0.0, 1.0 - (dy / y_window))          # near surface -> stronger
    w_speed = min(1.0, vrel_y / vy_cap)               # reasonable fall speed cap
    # softly prefer being near platform center
    cx = 0.5 * (left + right)
    halfw = max(1e-3, 0.5 * (right - left))
    w_center = max(0.0, 1.0 - abs(px - cx) / halfw)

    r = reward_scale * w_dist * w_speed * (0.5 + 0.5 * w_center)
    return r * env.dt

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

def platform_aware_approach(env: WarehouseBrawl, y_thresh: float = 0.8, pos_only: bool = True) -> float:
    ctx = ctx_or_compute(env)
    dx0 = abs(ctx.ppx - ctx.opx); dy0 = abs(ctx.ppy - ctx.opy)
    dx1 = abs(ctx.px  - ctx.ox ); dy1 = abs(ctx.py  - ctx.oy )
    delta = (dy0 - dy1) if (dy0 > y_thresh or dy1 > y_thresh) else (dx0 - dx1)
    if pos_only and delta < 0.0:
        return 0.0
    return delta * ctx.dt

def head_to_opponent(env: WarehouseBrawl, threshold: float = 1.2, pos_only: bool = False) -> float:
    ctx = ctx_or_compute(env)
    r2 = threshold * threshold
    vp = max(0.0, (ctx.ppx-ctx.opx)**2 + (ctx.ppy-ctx.opy)**2 - r2)
    vc = max(0.0, (ctx.px -ctx.ox )**2 + (ctx.py -ctx.oy )**2 - r2)
    delta = vp - vc
    if pos_only and delta < 0.0:
        return 0.0
    return delta * ctx.dt

def holding_more_than_3_keys_penalty(env: WarehouseBrawl) -> float:
    a = env.objects["player"].cur_action
    return -env.dt if (a > 0.5).sum() > 3 else env.dt

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
            return -1.0
    return 0.0

def on_combo_reward(env: WarehouseBrawl, agent: str) -> float:
    # reward combos for the player; penalize if opponent combos
    return 1.0 if agent == 'player' else -1.0

def fell_off_map_event(env, pad: float = 0.0, only_bottom: bool = False) -> float:
    """
    returns 1.0 exactly once when the player crosses the KO boundary.
    - pad > 0 shrinks the safe area slightly (fires earlier)
    - set only_bottom=True if you only want bottom falls
    """
    ctx = ctx_or_compute(env)
    p = env.objects["player"]

    if only_bottom:
        outside = (ctx.py > (ctx.half_h - pad))
    else:
        outside = (abs(ctx.px) > (ctx.half_w - pad)) or (abs(ctx.py) > (ctx.half_h - pad))

    was_outside = bool(getattr(p, "_rw_was_outside", False))
    fire = outside and not was_outside
    setattr(p, "_rw_was_outside", outside)

    return 1.0 if fire else 0.0


'''
Add your dictionary of RewardFunctions here using RewTerms
'''
def gen_reward_manager(log_terms: bool=True):
    reward_functions = {
        #'target_height_reward': RewTerm(func=base_height_l2, weight=0.0, params={'target_height': -4, 'obj_name': 'player'}),
        'danger_zone_reward': RewTerm(func=danger_zone_reward, weight=0.7),
        'damage_reward':  RewTerm(func=damage_interaction_reward, weight=50,
                                  params={"mode": RewardMode.ASYMMETRIC_OFFENSIVE}),
        'defence_reward': RewTerm(func=damage_interaction_reward, weight=0.77,
                                  params={"mode": RewardMode.ASYMMETRIC_DEFENSIVE}),
        #'head_to_middle_reward': RewTerm(func=head_to_middle_reward, weight=0.01),
        'platform_aware_approach': RewTerm(func=platform_aware_approach, weight=0.88,
                                           params={"y_thresh": 0.8, "pos_only": True}),
        'move_dir_reward': RewTerm(func=head_to_opponent, weight=5.0),
        'move_towards_reward': RewTerm(func=head_to_opponent, weight=12.0, params={"threshold" : 0.75, "pos_only": True}),
        # 'useless_attk_penalty': RewTerm(func=penalize_useless_attacks_shaped, weight=0.044, params={"distance_thresh" : 2.75, "scale" : 1.25}),
        'attack_quality': RewTerm(
            func=attack_quality_reward,
            weight=4.0,
            params=dict(distance_thresh=1.75, near_bonus_scale=0.9, far_penalty_scale=1.25),
        ),
        # 'attack_misalign': RewTerm(func=attack_misalignment_penalty, weight=2.0),
        # gentle edge avoidance (dt inside: small)
        'edge_safety':             RewTerm(func=edge_safety, weight=0.044),
        'holding_more_than_3_keys': RewTerm(func=holding_more_than_3_keys_penalty, weight=7.0),
        'taunt_reward': RewTerm(func=in_state_reward, weight=-1.0, params={'desired_state': TauntState}),
        'fell_off_map': RewTerm(func=fell_off_map_event, weight=-40.0, params={'pad': 1.0, 'only_bottom': False}),
    }
    signal_subscriptions = {
        'on_win_reward': ('win_signal', RewTerm(func=on_win_reward, weight=20)),
        'on_knockout_reward': ('knockout_signal', RewTerm(func=on_knockout_reward, weight=20)),
        'on_combo_reward': ('hit_during_stun', RewTerm(func=on_combo_reward, weight=7)),
        'on_equip_reward': ('weapon_equip_signal', RewTerm(func=on_equip_reward, weight=11)),
        'on_drop_reward': ('weapon_drop_signal', RewTerm(func=on_drop_penalty, weight=14))
    }
    return RewardManager(reward_functions, signal_subscriptions, log_terms=log_terms)