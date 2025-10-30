'''
TRAINING: AGENT

This file contains all the types of Agent classes, the Reward Function API, and the built-in train function from our multi-agent RL API for self-play training.
- All of these Agent classes are each described below. 

Running this file will initiate the training function, and will:
a) Start training from scratch
b) Continue training from a specific timestep given an input `file_path`
'''

# -------------------------------------------------------------------
# ----------------------------- IMPORTS -----------------------------
# -------------------------------------------------------------------

import os
os.environ["TRAIN_MODE"] = "1"
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import glob, re
from functools import partial
import torch 
import gymnasium as gym
from torch.nn import functional as F
from torch import nn as nn
import numpy as np
import pygame
from stable_baselines3 import A2C, PPO, SAC, DQN, DDPG, TD3, HER 
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
#
from environment.agent import *
from user.reward_fastpath import ctx_or_compute
from typing import Optional, Type, List, Tuple

import time
import argparse

# -------------------------------------------------------------------------
# ----------------------------- AGENT CLASSES -----------------------------
# -------------------------------------------------------------------------

class SB3Agent(Agent):
    '''
    SB3Agent:
    - Defines an AI Agent that takes an SB3 class input for specific SB3 algorithm (e.g. PPO, SAC)
    Note:
    - For all SB3 classes, if you'd like to define your own neural network policy you can modify the `policy_kwargs` parameter in `self.sb3_class()` or make a custom SB3 `BaseFeaturesExtractor`
    You can refer to this for Custom Policy: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
    '''
    def __init__(
            self,
            sb3_class: Optional[Type[BaseAlgorithm]] = PPO,
            file_path: Optional[str] = None,
            extractor: BaseFeaturesExtractor = None,
            sb3_kwargs: Optional[dict] = None,       
            policy_kwargs: Optional[dict] = None
        ):
            self.sb3_class = sb3_class
            self.extractor = extractor
            self.sb3_kwargs = sb3_kwargs or {}
            self.policy_kwargs = policy_kwargs or {}
            super().__init__(file_path)

    def _initialize(self) -> None:
        if self.file_path is None:
            # merge extractor policy kwargs (if any) with user policy_kwargs
            ek = self.extractor.get_policy_kwargs() if self.extractor else {}
            pk = {**ek, **self.policy_kwargs}
            self.model = self.sb3_class(
                "MlpPolicy",
                self.env,
                policy_kwargs=pk,
                **self.sb3_kwargs
            )
            del self.env
        else:
            device = self.sb3_kwargs.get("device", "auto")
            self.model = self.sb3_class.load(self.file_path, device=device)

    def _gdown(self) -> str:
        # Call gdown to your link
        return

    #def set_ignore_grad(self) -> None:
        #self.model.set_ignore_act_grad(True)

    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path, include=['num_timesteps'])

    def learn(self, env, total_timesteps, log_interval: int = 1, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
        )

class RecurrentPPOAgent(Agent):
    '''
    RecurrentPPOAgent:
    - Defines an RL Agent that uses the Recurrent PPO (LSTM+PPO) algorithm
    '''
    def __init__(
            self,
            file_path: Optional[str] = None
    ):
        super().__init__(file_path)
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)

    def _initialize(self) -> None:
        if self.file_path is None:
            policy_kwargs = {
                'activation_fn': nn.ReLU,
                'lstm_hidden_size': 512,
                'net_arch': [dict(pi=[32, 32], vf=[32, 32])],
                'shared_lstm': True,
                'enable_critic_lstm': False,
                'share_features_extractor': True,

            }
            self.model = RecurrentPPO("MlpLstmPolicy",
                                      self.env,
                                      verbose=0,
                                      n_steps=30*90*20,
                                      batch_size=16,
                                      ent_coef=0.05,
                                      policy_kwargs=policy_kwargs)
            del self.env
        else:
            self.model = RecurrentPPO.load(self.file_path)

    def reset(self) -> None:
        self.episode_starts = np.ones((1,), dtype=bool)
        self.episode_starts = True

    def predict(self, obs):
        action, self.lstm_states = self.model.predict(obs, state=self.lstm_states, episode_start=self.episode_starts, deterministic=True)
        if self.episode_starts: self.episode_starts = False
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path)

    def learn(self, env, total_timesteps, log_interval: int = 2, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)

class BasedAgent(Agent):
    '''
    BasedAgent:
    - Defines a hard-coded Agent that predicts actions based on if-statements. Interesting behaviour can be achieved here.
    - The if-statement algorithm can be developed within the `predict` method below.
    '''
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.time = 0

    def predict(self, obs):
        self.time += 1
        pos = self.obs_helper.get_section(obs, 'player_pos')
        opp_pos = self.obs_helper.get_section(obs, 'opponent_pos')
        opp_KO = self.obs_helper.get_section(obs, 'opponent_state') in [5, 11]
        action = self.act_helper.zeros()

        # If off the edge, come back
        if pos[0] > 10.67/2:
            action = self.act_helper.press_keys(['a'])
        elif pos[0] < -10.67/2:
            action = self.act_helper.press_keys(['d'])
        elif not opp_KO:
            # Head toward opponent
            if (opp_pos[0] > pos[0]):
                action = self.act_helper.press_keys(['d'])
            else:
                action = self.act_helper.press_keys(['a'])

        # Note: Passing in partial action
        # Jump if below map or opponent is above you
        if (pos[1] > 1.6 or pos[1] > opp_pos[1]) and self.time % 2 == 0:
            action = self.act_helper.press_keys(['space'], action)

        # Attack if near
        if (pos[0] - opp_pos[0])**2 + (pos[1] - opp_pos[1])**2 < 4.0:
            action = self.act_helper.press_keys(['j'], action)
        return action

class UserInputAgent(Agent):
    '''
    UserInputAgent:
    - Defines an Agent that performs actions entirely via real-time player input
    '''
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def predict(self, obs):
        action = self.act_helper.zeros()
       
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            action = self.act_helper.press_keys(['w'], action)
        if keys[pygame.K_a]:
            action = self.act_helper.press_keys(['a'], action)
        if keys[pygame.K_s]:
            action = self.act_helper.press_keys(['s'], action)
        if keys[pygame.K_d]:
            action = self.act_helper.press_keys(['d'], action)
        if keys[pygame.K_SPACE]:
            action = self.act_helper.press_keys(['space'], action)
        # h j k l
        if keys[pygame.K_h]:
            action = self.act_helper.press_keys(['h'], action)
        if keys[pygame.K_j]:
            action = self.act_helper.press_keys(['j'], action)
        if keys[pygame.K_k]:
            action = self.act_helper.press_keys(['k'], action)
        if keys[pygame.K_l]:
            action = self.act_helper.press_keys(['l'], action)
        if keys[pygame.K_g]:
            action = self.act_helper.press_keys(['g'], action)

        return action

class ClockworkAgent(Agent):
    '''
    ClockworkAgent:
    - Defines an Agent that performs sequential steps of [duration, action]
    '''
    def __init__(
            self,
            action_sheet: Optional[List[Tuple[int, List[str]]]] = None,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.steps = 0
        self.current_action_end = 0  # Tracks when the current action should stop
        self.current_action_data = None  # Stores the active action
        self.action_index = 0  # Index in the action sheet

        if action_sheet is None:
            self.action_sheet = [
                (10, ['a']),
                (1, ['l']),
                (20, ['a']),
                (3, ['a', 'j']),
                (15, ['space']),
            ]
        else:
            self.action_sheet = action_sheet

    def predict(self, obs):
        """
        Returns an action vector based on the predefined action sheet.
        """
        # Check if the current action has expired
        if self.steps >= self.current_action_end and self.action_index < len(self.action_sheet):
            hold_time, action_data = self.action_sheet[self.action_index]
            self.current_action_data = action_data  # Store the action
            self.current_action_end = self.steps + hold_time  # Set duration
            self.action_index += 1  # Move to the next action

        # Apply the currently active action
        action = self.act_helper.press_keys(self.current_action_data)
        self.steps += 1  # Increment step counter
        return action
    
class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int = 64, features_dim: int = 64, hidden_dim: int = 64):
        """
        A 3-layer MLP policy:
        obs -> Linear(hidden_dim) -> ReLU -> Linear(hidden_dim) -> ReLU -> Linear(action_dim)
        """
        super(MLPPolicy, self).__init__()

        # Input layer
        self.fc1 = nn.Linear(obs_dim, hidden_dim, dtype=torch.float32)
        # Hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32)
        # Output layer
        self.fc3 = nn.Linear(hidden_dim, features_dim, dtype=torch.float32)

    def forward(self, obs):
        """
        obs: [batch_size, obs_dim]
        returns: [batch_size, action_dim]
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class MLPExtractor(BaseFeaturesExtractor):
    '''
    Class that defines an MLP Base Features Extractor
    '''
    def __init__(self, observation_space: gym.Space, features_dim: int = 64, hidden_dim: int = 64):
        super().__init__(observation_space, features_dim)
        self.model = MLPPolicy(
            obs_dim=observation_space.shape[0],
            features_dim=features_dim,
            hidden_dim=hidden_dim,
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs)
    
    @classmethod
    def get_policy_kwargs(cls, features_dim: int = 64, hidden_dim: int = 64) -> dict:
        return dict(
            features_extractor_class=cls,
            features_extractor_kwargs=dict(features_dim=features_dim, hidden_dim=hidden_dim) #NOTE: features_dim = 10 to match action space output
        )
    
class CustomAgent(Agent):
    def __init__(
        self,
        sb3_class: Optional[Type[BaseAlgorithm]] = PPO,
        file_path: Optional[str] = None,
        extractor: Optional[Type[BaseFeaturesExtractor]] = None,  # pass the class (e.g., MLPExtractor)
        sb3_kwargs: Optional[dict] = None,                        # new
        policy_kwargs: Optional[dict] = None                      # new
    ):
        self.sb3_class = sb3_class
        self.extractor = extractor
        self.sb3_kwargs = sb3_kwargs or {}
        self.policy_kwargs = policy_kwargs or {}
        super().__init__(file_path)

    def _initialize(self) -> None:
        if self.file_path is None:
            # merge extractor-provided policy kwargs with user overrides
            ek = self.extractor.get_policy_kwargs() if self.extractor else {}
            pk = {**ek, **self.policy_kwargs}
            self.model = self.sb3_class(
                "MlpPolicy",
                self.env,
                policy_kwargs=pk,
                **self.sb3_kwargs
            )
            del self.env
        else:
            # allow device override on load
            device = self.sb3_kwargs.get("device", "auto")
            self.model = self.sb3_class.load(self.file_path, device=device)

    def _gdown(self) -> Optional[str]:
        return

    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path, include=['num_timesteps'])

    def learn(self, env, total_timesteps, log_interval: int = 1, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)


# In[ ]:

# --------------------------------------------------------------------------------
# ----------------------------- REWARD FUNCTIONS API -----------------------------
# --------------------------------------------------------------------------------

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
    zone_height: float = 4.2
) -> float:
    """
    Applies a penalty for every time frame player surpases a certain height threshold in the environment.

    Args:
        env (WarehouseBrawl): The game environment.
        zone_penalty (int): The penalty applied when the player is in the danger zone.
        zone_height (float): The height threshold defining the danger zone.

    Returns:
        float: The computed penalty as a tensor.
    """

    ctx = ctx_or_compute(env)
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
) -> float:
    ctx = ctx_or_compute(env)
    # not attacking -> zero
    if not ctx.p_attacking:
        return 0.0
    r2 = distance_thresh * distance_thresh
    if ctx.dist2 <= r2:
        gain = (r2 - ctx.dist2) * near_bonus_scale
        return gain * ctx.dt
    penalty = (ctx.dist2 - r2) * (-far_penalty_scale)
    return penalty * ctx.dt

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

def platform_aware_approach(
    env: WarehouseBrawl,
    y_thresh: float = 0.8,
    pos_only: bool = True,
) -> float:
    ctx = ctx_or_compute(env)
    dx0 = abs(ctx.ppx - ctx.opx); dy0 = abs(ctx.ppy - ctx.opy)
    dx1 = abs(ctx.px  - ctx.ox ); dy1 = abs(ctx.py  - ctx.oy )
    if dy0 > y_thresh or dy1 > y_thresh:
        delta = (dy0 - dy1)
    else:
        delta = (dx0 - dx1)
    if pos_only and delta < 0.0:
        return 0.0
    return delta

def head_to_opponent(env: WarehouseBrawl, threshold: float = 1.0, pos_only: bool = True) -> float:
    ctx = ctx_or_compute(env)
    d_prev = (ctx.ppx - ctx.opx)
    d_curr = (ctx.px  - ctx.ox)
    r2 = threshold * threshold
    v_prev = max(0.0, d_prev * d_prev - r2)
    v_curr = max(0.0, d_curr * d_curr - r2)
    delta = v_prev - v_curr
    if pos_only and delta < 0.0:
        return 0.0
    return delta

def holding_more_than_3_keys(
    env: WarehouseBrawl,
) -> float:

    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is holding more than 3 keys
    a = player.cur_action
    if (a > 0.5).sum() > 3:
        return env.dt
    return 0

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

def on_drop_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == "player":
        if env.objects["player"].weapon == "Punch":
            return -1.0
    return 0.0

def on_combo_reward(env: WarehouseBrawl, agent: str) -> float:
    # reward combos for the player; penalize if opponent combos
    return 1.0 if agent == 'player' else -1.0

'''
Add your dictionary of RewardFunctions here using RewTerms
'''
def gen_reward_manager(log_terms: bool=True):
    reward_functions = {
        #'target_height_reward': RewTerm(func=base_height_l2, weight=0.0, params={'target_height': -4, 'obj_name': 'player'}),
        'danger_zone_reward': RewTerm(func=danger_zone_reward, weight=0.30),
        'damage_reward':  RewTerm(func=damage_interaction_reward, weight=3.0,
                                  params={"mode": RewardMode.ASYMMETRIC_OFFENSIVE}),
        'defence_reward': RewTerm(func=damage_interaction_reward, weight=0.30,
                                  params={"mode": RewardMode.ASYMMETRIC_DEFENSIVE}),
        #'head_to_middle_reward': RewTerm(func=head_to_middle_reward, weight=0.01),
        'platform_aware_approach': RewTerm(func=platform_aware_approach, weight=0.10,
                                           params={"y_thresh": 0.8, "pos_only": True}),
        'head_to_opponent': RewTerm(func=head_to_opponent, weight=0.33),
        # 'useless_attk_penalty': RewTerm(func=penalize_useless_attacks_shaped, weight=0.044, params={"distance_thresh" : 2.75, "scale" : 1.25}),
        'attack_quality': RewTerm(
            func=attack_quality_reward,
            weight=0.22,
            params=dict(distance_thresh=1, near_bonus_scale=0.9, far_penalty_scale=1.25),
        ),
        # gentle edge avoidance (dt inside: small)
        'edge_safety':             RewTerm(func=edge_safety, weight=0.02),
        'holding_more_than_3_keys': RewTerm(func=holding_more_than_3_keys, weight=-0.002),
        'taunt_reward': RewTerm(func=in_state_reward, weight=-0.33, params={'desired_state': TauntState}),
    }
    signal_subscriptions = {
        'on_win_reward': ('win_signal', RewTerm(func=on_win_reward, weight=50)),
        'on_knockout_reward': ('knockout_signal', RewTerm(func=on_knockout_reward, weight=8)),
        'on_combo_reward': ('hit_during_stun', RewTerm(func=on_combo_reward, weight=5)),
        'on_equip_reward': ('weapon_equip_signal', RewTerm(func=on_equip_reward, weight=15)),
        'on_drop_reward': ('weapon_drop_signal', RewTerm(func=on_drop_reward, weight=5))
    }
    return RewardManager(reward_functions, signal_subscriptions, log_terms=log_terms)

class RewardBreakdownCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._acc = {}
        self._n = 0

    def _on_step(self) -> bool:
        # on-policy: `infos` present during rollout collection
        infos = self.locals.get("infos", [])
        for inf in infos:
            if not inf:
                continue
            terms = inf.get("rew_terms")
            if terms:
                for k, v in terms.items():
                    self._acc[k] = self._acc.get(k, 0.0) + float(v)
                self._acc["_signals"] = self._acc.get("_signals", 0.0) + float(inf.get("rew_signals", 0.0))
                self._n += 1
        return True

    def _on_rollout_end(self) -> None:
        if self._n > 0:
            for k, s in self._acc.items():
                mean = s / self._n
                self.logger.record(f"reward_terms/{k}", mean)
                if self.verbose:
                    print(f"[rew] {k}: {mean:.5f}")
        self._acc.clear()
        self._n = 0

class PhaseTimerCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._t_rollout_start = None
        self._t_last_rollout_end = None

    # required by BaseCallback (abstract)
    def _on_step(self) -> bool:
        # do nothing per-step; keep training going
        return True

    def _on_rollout_start(self) -> None:
        now = time.time()
        # if we just finished a prior rollout, the time since then is training time
        if self._t_last_rollout_end is not None:
            train_sec = now - self._t_last_rollout_end
            self.logger.record("phases/train_sec", float(train_sec))
            if self.verbose:
                print(f"[phase] train {train_sec:.2f}s")
        self._t_rollout_start = now

    def _on_rollout_end(self) -> None:
        now = time.time()
        if self._t_rollout_start is not None:
            rollout_sec = now - self._t_rollout_start
            self.logger.record("phases/rollout_sec", float(rollout_sec))
            if self.verbose:
                print(f"[phase] rollout {rollout_sec:.2f}s")
        self._t_last_rollout_end = now

    def _on_training_end(self) -> None:
        # capture the last train segment (after the final rollout)
        if self._t_last_rollout_end is not None:
            train_sec = time.time() - self._t_last_rollout_end
            self.logger.record("phases/train_sec_final", float(train_sec))
            if self.verbose:
                print(f"[phase] train(final) {train_sec:.2f}s")

class RewardScheduleCallback(BaseCallback):
    def __init__(self, schedule: list[tuple[int, dict[str, float], bool]], verbose: int = 0):
        """
        schedule: list of (step_threshold, weight_updates, zero_missing)
        example: [(100_000, {"head_to_opponent": 0.1}, False),
                  (400_000, {"damage_interaction_reward": 3.0}, False),
                  (1_000_000, {"danger_zone_reward": 0.2, "head_to_opponent": 0.0}, True)]
        """
        super().__init__(verbose)
        self.schedule = sorted(schedule, key=lambda x: x[0])
        self._i = 0

    def _on_step(self) -> bool:
        t = self.num_timesteps
        env = self.model.get_env()
        # fire any stages we just crossed
        while self._i < len(self.schedule) and t >= self.schedule[self._i][0]:
            _, updates, zero_missing = self.schedule[self._i]
            env.env_method("set_reward_weights", updates, zero_missing=zero_missing)
            if self.verbose:
                print(f"[reward-schedule] applied stage {self._i} at {t} steps: {updates}, zero_missing={zero_missing}")
            self._i += 1
        return True
    
class EntropyScheduleCallback(BaseCallback):
    """
    decays PPO.ent_coef from `start` -> `end` with optional warmup.
    """
    def __init__(self, total_timesteps: int, start: float = 0.02, end: float = 0.005, warmup_frac: float = 0.02, verbose: int = 0):
        super().__init__(verbose)
        self.T = int(total_timesteps)
        self.s = float(start); self.e = float(end); self.w = float(warmup_frac)

    def _on_training_start(self) -> None:
        self.model.ent_coef = self.s

    def _on_step(self) -> bool:
        t = int(self.num_timesteps)
        p = min(1.0, max(0.0, t / self.T))        # 0->1 over training
        if p < self.w:                             # linear warmup to start
            val = self.s * (p / max(1e-9, self.w))
        else:                                      # cosine to end
            q = (p - self.w) / max(1e-9, (1.0 - self.w))
            val = self.e + 0.5 * (self.s - self.e) * (1.0 + float(np.cos(np.pi * q)))
        self.model.ent_coef = float(val)
        self.logger.record("train/ent_coef", float(val))
        return True

class LRScheduleCallback(BaseCallback):
    """
    cosine/linear/step lr scheduling for any sb3 algo.
    - updates model.policy.optimizer.param_groups[].lr
    - logs current lr to 'train/lr'
    """
    def __init__(
        self,
        total_timesteps: int,
        initial_lr: float,
        final_lr: float = 3e-5,
        warmup_frac: float = 0.02,
        schedule: str = "cosine",           # {"cosine","linear","step"}
        step_milestones: list[int] | None = None,  # used if schedule == "step" (absolute steps)
        step_factor: float = 0.5,           # multiply lr by this each milestone (min-capped by final_lr)
        verbose: int = 0,
    ):
        super().__init__(verbose)
        assert total_timesteps > 0, "total_timesteps must be > 0"
        self.T = int(total_timesteps)
        self.lr0 = float(initial_lr)
        self.lrf = float(final_lr)
        self.warm = float(max(0.0, min(0.95, warmup_frac)))
        self.schedule = schedule
        self.milestones = sorted(step_milestones or [])
        self.step_factor = float(step_factor)

    def _on_training_start(self) -> None:
        # ensure we start from the intended base lr
        self._set_lr(self.lr0)

    def _on_step(self) -> bool:
        t = int(self.num_timesteps)

        if self.schedule == "step":
            k = 0
            # count how many milestones we've passed
            while k < len(self.milestones) and t >= self.milestones[k]:
                k += 1
            lr = max(self.lrf, self.lr0 * (self.step_factor ** k))
        else:
            # normalized progress [0,1]
            p = min(1.0, max(0.0, t / self.T))
            if self.warm > 0.0 and p < self.warm:
                # linear warmup: 0 -> lr0
                lr = self.lr0 * (p / self.warm)
            else:
                q = 0.0 if p <= self.warm else (p - self.warm) / max(1e-9, (1.0 - self.warm))
                if self.schedule == "linear":
                    lr = self.lr0 + (self.lrf - self.lr0) * q
                else:
                    # cosine (default)
                    lr = self.lrf + 0.5 * (self.lr0 - self.lrf) * (1.0 + float(np.cos(np.pi * q)))

        self._set_lr(lr)
        self.logger.record("train/lr", float(lr))
        if self.verbose and (t % 10000 == 0):
            print(f"[lr] t={t} lr={lr:.6g}")
        return True

    def _set_lr(self, lr: float) -> None:
        opt = getattr(self.model.policy, "optimizer", None)
        if opt is None:
            return
        for g in opt.param_groups:
            g["lr"] = float(lr)


# --- env factory ---

def make_env(i: int,
             ckpt_dir: str,
             policy_partial: partial,
             opponent_mode: str = "random",
             resolution: CameraResolution = CameraResolution.LOW):
    """
    returns a thunk that builds ONE independent env (needed by VecEnv)
    """
    def _init():
        import os, torch
        # headless + single-thread hints
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        torch.set_num_threads(1)
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

        sp = DirSelfPlayRandom(policy_partial, ckpt_dir) if opponent_mode == "random" \
             else DirSelfPlayLatest(policy_partial, ckpt_dir)

        opponents = {
            'self_play': (1.0, sp),
            # you can mix in scripted opponents if you want, e.g.:
            # 'based_agent': (0.2, partial(BasedAgent)),
            # 'constant_agent': (0.1, partial(ConstantAgent)),
        }
        opp_cfg = OpponentsCfg(opponents=opponents)

	# do NOT pass a SaveHandler into workers
        rm = gen_reward_manager(log_terms=(i == 0))
        env = SelfPlayWarehouseBrawl(
            reward_manager=rm,
            opponent_cfg=opp_cfg,
            save_handler=None,
            resolution=resolution,
            train_mode=True,
            mode=RenderMode.NONE,
            debug_log_terms=(i == 0)
        )
        env.raw_env = PrevPosWrapper(env.raw_env)
        rm.subscribe_signals(env.raw_env)
        return env

    return _init

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--numworkers", "-nw", type=int, default=None,
                   help="how many env workers (omit to use the script default)")
    return p.parse_args()

if __name__ == "__main__":

    # ---- where checkpoints live (read by DirSelfPlay* and written by callback) ----
    EXP_ROOT = "checkpoints/experiment_nonrecurrent4"
    os.makedirs(EXP_ROOT, exist_ok=True)
    
    args = _parse_args()

    # single source of truth for the built-in default
    DEFAULT_NUM_WORKERS = 40

    # use cli override if provided; else keep the script's default
    n_envs = args.numworkers if args.numworkers is not None else DEFAULT_NUM_WORKERS

    def clip_sched(progress_remaining: float) -> float:
        # sb3 passes 1.0 -> 0.0 over training; start wide (0.3), end tighter (0.1)
        return 0.1 + 0.2 * progress_remaining

    # ---- sb3 hyperparams ----
    # note: with vectorized training, total rollout per update = n_steps * n_envs
    sb3_kwargs = dict(
        device="cuda",
        verbose=1,
        n_steps=2048,       # per-env rollout; 1024*8 = 8192 samples/update if n_envs=8
        batch_size=16384,    # must divide n_steps * n_envs
        n_epochs=12,
        learning_rate=3e-4,
        gamma=0.997,
        gae_lambda=0.96,
        ent_coef=0.02,
        clip_range=clip_sched,
        target_kl=0.09,
    )

    policy_kwargs = dict(
        activation_fn=nn.SiLU,
        net_arch=[dict(pi=[512, 256, 128], vf=[512, 256, 128])],
        features_extractor_class=MLPExtractor,
        features_extractor_kwargs=dict(features_dim=256, hidden_dim=512)
        )

    policy_partial_cpu = partial(
        CustomAgent,
        sb3_class=PPO,
        extractor=MLPExtractor,
        sb3_kwargs=dict(device="cpu"),
        policy_kwargs=policy_kwargs
    )

    # vec_env = SubprocVecEnv(
    #     [make_env(i, EXP_ROOT, policy_partial_cpu, opponent_mode="random", resolution=CameraResolution.LOW)
    #     for i in range(n_envs)],
    #     start_method="spawn"
    # )
    base_env = SubprocVecEnv(
        [make_env(i, EXP_ROOT, policy_partial_cpu, opponent_mode="random", resolution=CameraResolution.LOW)
        for i in range(n_envs)],
        # on linux, prefer default 'fork' unless you hit cuda/pytorch issues
        start_method="spawn"
    )

    mon_env = VecMonitor(base_env, filename=os.path.join(EXP_ROOT, "monitor"))

    # try to resume: load vecnormalize stats if present, otherwise create fresh
    vn_path = os.path.join(EXP_ROOT, "vecnormalize.pkl")  # we’ll save to this name below
    if os.path.exists(vn_path):
        vec_env = VecNormalize.load(vn_path, mon_env)
        vec_env.training = True   # important when resuming training
        vec_env.norm_reward = True
    else:
        vec_env = VecNormalize(mon_env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=sb3_kwargs["gamma"])


    def _latest_ckpt(ckpt_dir: str, prefix: str = "rl_model_") -> Optional[str]:
        zips = glob.glob(os.path.join(ckpt_dir, f"{prefix}*.zip"))
        if not zips:
            return None
        # sort by the trailing step number; fall back to mtime if needed
        def _key(p):
            m = re.search(rf"{re.escape(prefix)}(\d+)\.zip$", os.path.basename(p))
            return int(m.group(1)) if m else -1
        zips.sort(key=_key)
        return zips[-1]

    # resume model if there is a checkpoint, else start fresh
    load_checkpoint = True
    if load_checkpoint:
        latest = _latest_ckpt(EXP_ROOT, "rl_model_")
    if load_checkpoint and latest is not None:
         # ---- PPO model ----
        model = PPO.load(latest, env=vec_env, device=sb3_kwargs["device"])
        print(f"[resume] loaded {latest}")
    else:
         # ---- PPO model ----
        model = PPO(policy="MlpPolicy", env=vec_env, policy_kwargs=policy_kwargs, **sb3_kwargs)
        if load_checkpoint:
            print("[resume] no checkpoint found; starting fresh")
        else:
            print("load checkpoint was false starting fresh")

    # saving
    class SaveVecNormCallback(BaseCallback):
        def __init__(self, save_freq: int, path: str, verbose: int = 0):
            super().__init__(verbose)
            self.save_freq = max(1, int(save_freq))
            self.path = path
            self._vec = None
        def _on_step(self) -> bool:
            if self.n_calls % self.save_freq == 0:
                if self._vec is None:
                    self._vec = self.model.get_vec_normalize_env()
                if self._vec is not None:
                    self._vec.save(self.path)
                    if self.verbose >= 1:
                        print(f"[vecnorm] saved {self.path} at {self.num_timesteps} steps")
            return True
    target_save_every = 500_000
    ckpt_cb = CheckpointCallback(
        save_freq=max(1, target_save_every // n_envs),
        save_path=EXP_ROOT,
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    
    # TOTAL STEPS
    total_steps = 32_000_000

    # callbacks
    vec_cb = SaveVecNormCallback(save_freq=target_save_every, path=vn_path)
    rb_cb  = RewardBreakdownCallback(verbose=1)
    timing_cb = PhaseTimerCallback(verbose=1)
    entSched_cb = EntropyScheduleCallback(total_timesteps=total_steps, start=0.02, end=0.005, warmup_frac=0.02)

    # learning-rate scheduler (cosine decay with 2% warmup; 3e-4 -> 3e-5)
    lrSched_cb = LRScheduleCallback(
        total_timesteps=total_steps,
        initial_lr=sb3_kwargs["learning_rate"],
        final_lr=3e-5,
        warmup_frac=0.02,
        schedule="cosine",
        # if you prefer step decay instead:
        # schedule="step", step_milestones=[1_500_000, 3_000_000, 5_000_000], step_factor=0.5,
        verbose=0,
    )

    # rwSched_cb = RewardScheduleCallback(
    #             schedule=[
    #                 (100_000, {"head_to_opponent": 0.10}, False),
    #                 (400_000, {"damage_interaction_reward": 3.0}, False),
    #                 (1_000_000, {"danger_zone_reward": 0.2, "head_to_opponent": 0.0}, True),  # stage switch
    #             ],
    #             verbose=1,
    #         )

    # ---- train ----
    model.learn(total_timesteps=total_steps, callback=CallbackList(
                                                        [ckpt_cb, 
                                                         vec_cb, 
                                                         rb_cb, 
                                                         timing_cb,
                                                         lrSched_cb,
                                                        #  rwSched_cb
                                                         ]))

    # final save
    model.save(os.path.join(EXP_ROOT, "final_model"))
    model.get_vec_normalize_env().save(vn_path)
    vec_env.close()