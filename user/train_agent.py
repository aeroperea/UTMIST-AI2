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

import time
import argparse
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
from typing import Optional, Type, List, Tuple
#
from environment.agent import *
from user.custom_feature_extractor import *
from user.custom_callbacks import *
from user.reward_system import *


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
            features_extractor_kwargs=dict(features_dim=features_dim, hidden_dim=hidden_dim)
        )


def _has_get_policy_kwargs(extractor_cls) -> bool:
    return hasattr(extractor_cls, "get_policy_kwargs") and callable(getattr(extractor_cls, "get_policy_kwargs"))

from user.custom_feature_extractor import _mirror_action

class CustomAgent(Agent):
    def __init__(self, sb3_class: Optional[Type[BaseAlgorithm]] = PPO,
                 file_path: Optional[str] = None,
                 extractor: Optional[Type[BaseFeaturesExtractor]] = None,
                 sb3_kwargs: Optional[dict] = None,
                 policy_kwargs: Optional[dict] = None):
        self.sb3_class = sb3_class
        self.extractor = extractor
        self.sb3_kwargs = sb3_kwargs or {}
        self.policy_kwargs = policy_kwargs or {}
        super().__init__(file_path)

    def _initialize(self) -> None:
        if self.file_path is None:
            ek = self.extractor.get_policy_kwargs() if (self.extractor and _has_get_policy_kwargs(self.extractor)) else {}
            pk = {**ek, **self.policy_kwargs}
            self.model = self.sb3_class("MlpPolicy", self.env, policy_kwargs=pk, **self.sb3_kwargs)
            del self.env
        else:
            device = self.sb3_kwargs.get("device", "auto")
            self.model = self.sb3_class.load(self.file_path, device=device)


    def _gdown(self) -> Optional[str]:
        return

    def predict(self, obs):
        sign = 1.0 if (obs.shape[0] > 4 and float(obs[4]) > 0.5) else -1.0
        action, _ = self.model.predict(obs)
        return _mirror_action(action, sign)

    def save(self, file_path: str) -> None:
        self.model.save(file_path, include=['num_timesteps'])

    def learn(self, env, total_timesteps, log_interval: int = 1, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)


# In[ ]:

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

        # mirror spec (fill indices to match your action layout; defaults are no-op)
        swap_pairs   = []     # e.g. [(0,1)] if 0=left,1=right buttons
        mirror_axes  = []     # e.g. [2] if axis 2 is horiz in [0,1]
        mirror_angles= []     # e.g. [3] if 3 is aim angle in [0,1]

        # train-time mirroring for the learner
        env = ActionMirrorWrapper(
            env,
            facing_index=4,   # your facing bit (True=right, False=left)
            px_index=0, ox_index=32,
            swap_pairs=((1, 3),),  # A<->D
            mirror_axes=(), mirror_angles=(),
            invert_facing=False
        )
        return env
    return _init

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--numworkers", "-nw", type=int, default=None,
                   help="how many env workers (omit to use the script default)")
    return p.parse_args()

if __name__ == "__main__":

    # ---- where checkpoints live (read by DirSelfPlay* and written by callback) ----
    EXP_ROOT = "checkpoints/FusedFeatureExtractor1" # todo: make this name prfix too
    os.makedirs(EXP_ROOT, exist_ok=True)
    
    args = _parse_args()

    # single source of truth for the built-in default
    DEFAULT_NUM_WORKERS = 32

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
        n_epochs=4,
        learning_rate=2.5e-4,
        gamma=0.997,
        gae_lambda=0.96,
        ent_coef=0.02,
        clip_range=clip_sched,
        target_kl=0.08,
        clip_range_vf=0.2,
        normalize_advantage=True,
        max_grad_norm=0.5
    )

    FUSED_EXTRACTOR_KW = dict(
        features_dim=256,
        hidden_dim=512,
        enum_fields=observation_fields,
        xy_player=[0, 1],
        xy_opponent=[32, 33],
        use_pairwise=True,
        # new in fused extractor:
        facing_index=4,                  # uses your facing bit; set None to use dx fallback
        flip_x_indices=(0, 2, 32, 34),   # flip x and vx for both agents
        flip_pair_dx=True,               # also flip engineered pairwise dx
        invert_facing=False,             # set True only if your facing bit is inverted
        use_compile=False,               # leave off during early training
    )

    policy_kwargs = dict(
        activation_fn=nn.SiLU,
            net_arch=[dict(pi=[256, 256, 128], vf=[256, 256, 128])],
            features_extractor_class=FusedFeatureExtractor,
            features_extractor_kwargs=FUSED_EXTRACTOR_KW,
        )
    
    

    policy_partial_cpu = partial(
        CustomAgent,
        sb3_class=PPO,
        extractor=FusedFeatureExtractor,
        sb3_kwargs=dict(device="cpu"),
        policy_kwargs=policy_kwargs
    )

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
        vec_env.training = True
        vec_env.norm_reward = True
        vec_env.norm_obs = False   # fused extractor expects raw (unnormalized) obs
    else:
            vec_env = VecNormalize(
            mon_env,
            norm_obs=False,          # <-- important with fused extractor
            norm_reward=True,
            clip_obs=10.0,
            gamma=sb3_kwargs["gamma"],
        )

    name_prefix="FusedFeatureExtractor1"

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
        latest = _latest_ckpt(EXP_ROOT, name_prefix)
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
        name_prefix=name_prefix,
        save_replay_buffer=False,
        save_vecnormalize=True,
        verbose=1
    )
    
    # TOTAL STEPS
    total_steps = 64_000_000

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
        warmup_frac=0.01,
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
    vec_env.close()# train-time mirroring for the learner