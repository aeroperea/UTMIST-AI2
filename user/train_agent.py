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
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
#
from environment.agent import *
from typing import Optional, Type, List, Tuple

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
    def __init__(self, file_path: Optional[str] = None,
                 policy_kwargs: Optional[dict] = None,
                 sb3_kwargs: Optional[dict] = None):
        
        super().__init__(file_path)
        self._policy_kwargs = policy_kwargs or {}
        self._sb3_kwargs = sb3_kwargs or {}
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)

    def _initialize(self) -> None:
        dev = self._sb3_kwargs.get("device", "cpu")
        if self.file_path is None:
            self.model = RecurrentPPO("MlpLstmPolicy", self.env,
                                    policy_kwargs=self._policy_kwargs,
                                    device=dev,  # ensure cpu here for opponents
                                    **{k:v for k,v in self._sb3_kwargs.items() if k != "device"})
            del self.env
        else:
            self.model = RecurrentPPO.load(self.file_path, device=dev)
        self.model.set_training_mode(False)

    def reset(self) -> None:
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
    return (obj.body.position.y - target_height)**2

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


# In[ ]:


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
    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is in the danger zone
    reward = -zone_penalty if player.body.position.y >= zone_height else 0.0

    return reward * env.dt

def in_state_reward(
    env: WarehouseBrawl,
    desired_state: Type[PlayerObjectState]=BackDashState,
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
    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is in the danger zone
    reward = 1 if isinstance(player.state, desired_state) else 0.0

    return reward * env.dt

def head_to_middle_reward(
    env: WarehouseBrawl,
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
    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is in the danger zone
    multiplier = -1 if player.body.position.x > 0 else 1
    reward = multiplier * (player.body.position.x - player.prev_x)

    return reward

def head_to_opponent(
    env: WarehouseBrawl,
) -> float:

    # Get player object from the environment
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    # Apply penalty if the player is in the danger zone
    multiplier = -1 if player.body.position.x > opponent.body.position.x else 1
    reward = multiplier * (player.body.position.x - player.prev_x)

    return reward

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
    if agent == 'player':
        return -1.0
    else:
        return 1.0

'''
Add your dictionary of RewardFunctions here using RewTerms
'''
def gen_reward_manager():
    reward_functions = {
        #'target_height_reward': RewTerm(func=base_height_l2, weight=0.0, params={'target_height': -4, 'obj_name': 'player'}),
        'danger_zone_reward': RewTerm(func=danger_zone_reward, weight=0.5),
        'damage_interaction_reward': RewTerm(func=damage_interaction_reward, weight=1.0),
        #'head_to_middle_reward': RewTerm(func=head_to_middle_reward, weight=0.01),
        #'head_to_opponent': RewTerm(func=head_to_opponent, weight=0.05),
        'penalize_attack_reward': RewTerm(func=in_state_reward, weight=-0.04, params={'desired_state': AttackState}),
        'holding_more_than_3_keys': RewTerm(func=holding_more_than_3_keys, weight=-0.01),
        #'taunt_reward': RewTerm(func=in_state_reward, weight=0.2, params={'desired_state': TauntState}),
    }
    signal_subscriptions = {
        'on_win_reward': ('win_signal', RewTerm(func=on_win_reward, weight=50)),
        'on_knockout_reward': ('knockout_signal', RewTerm(func=on_knockout_reward, weight=8)),
        'on_combo_reward': ('hit_during_stun', RewTerm(func=on_combo_reward, weight=5)),
        'on_equip_reward': ('weapon_equip_signal', RewTerm(func=on_equip_reward, weight=10)),
        'on_drop_reward': ('weapon_drop_signal', RewTerm(func=on_drop_reward, weight=15))
    }
    return RewardManager(reward_functions, signal_subscriptions)

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
        # silence audio for headless workers
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        torch.set_num_threads(1)

        rm = gen_reward_manager()

        sp = DirSelfPlayRandom(policy_partial, ckpt_dir) if opponent_mode == "random" else DirSelfPlayLatest(policy_partial, ckpt_dir)

        opponents = {"self_play": (1.0, sp)}
        opp_cfg = OpponentsCfg(opponents=opponents)

        # do NOT pass a SaveHandler into workers
        rm = gen_reward_manager()
        env = SelfPlayWarehouseBrawl(
            reward_manager=rm, opponent_cfg=opp_cfg,
            save_handler=None, resolution=resolution,
            train_mode=True, mode=RenderMode.NONE
        )
        rm.subscribe_signals(env.raw_env)  # <-- take this from original
        return Monitor(env)

    return _init

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



if __name__ == "__main__":

    # ---- where checkpoints live (read by DirSelfPlay* and written by callback) ----
    EXP_ROOT = "checkpoints/experiment_11(Recurrent)"
    os.makedirs(EXP_ROOT, exist_ok=True)

    # ---- vectorized env build ----
    n_envs = 32

    # ---- sb3 hyperparams ----
    # note: with vectorized training, total rollout per update = n_steps * n_envs
    sb3_kwargs = dict(
        device="cuda",
        verbose=1,
        n_steps=1024,       # per-env rollout; 1024*8 = 8192 samples/update if n_envs=8
        batch_size=4096,    # must divide n_steps * n_envs
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.999,
        gae_lambda=0.95,
        ent_coef=0.0077,
        clip_range=0.2,
        target_kl=0.02,
        clip_range_vf = 0.2,
        vf_coef = 0.5,
        max_grad_norm = 0.5,     
    )

    policy_kwargs = dict(
        activation_fn=nn.SiLU,
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        lstm_hidden_size=256,                       # core recurrent capacity
        n_lstm_layers=1,
        shared_lstm=False,                           # shared torso for pi/vf, cheaper and stable
        enable_critic_lstm=True,                   
        ortho_init=True,
        features_extractor_class=MLPExtractor,
        features_extractor_kwargs=dict(features_dim=64, hidden_dim=128)
        )

    # what the opponent loads when env.reset() happens
    policy_partial = partial(RecurrentPPOAgent,
                         policy_kwargs=policy_kwargs,
                         sb3_kwargs={**sb3_kwargs, "device": "cpu"})  # <-- opponents on cpu

    vec_env = SubprocVecEnv(
        [make_env(i, EXP_ROOT, policy_partial, opponent_mode="random", resolution=CameraResolution.LOW)
        for i in range(n_envs)],
        start_method="spawn"  # safer with CUDA and SDL
    )

    # try to resume: load vecnormalize stats if present, otherwise create fresh
    vn_path = os.path.join(EXP_ROOT, "vecnormalize.pkl")  # we’ll save to this name below
    if os.path.exists(vn_path):
        vec_env = VecNormalize.load(vn_path, vec_env)
        vec_env.training = True   # important when resuming training
        vec_env.norm_reward = True
    else:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=sb3_kwargs["gamma"])


    # resume model if there is a checkpoint, else start fresh
    load_checkpoint = True
    if load_checkpoint:
        latest = _latest_ckpt(EXP_ROOT, "rl_model_")
    if load_checkpoint and latest is not None:
         # ---- PPO model ----
        model = RecurrentPPO.load(latest, env=vec_env, device=sb3_kwargs["device"])
        print(f"[resume] loaded {latest}")
    else:
         # ---- PPO model ----
        model = RecurrentPPO(
            policy="MlpLstmPolicy",
            env=vec_env,
            policy_kwargs=policy_kwargs,
            **sb3_kwargs
        )
        if load_checkpoint:
            print("[resume] no checkpoint found; starting fresh")
        else:
            print("load checkpoint was false starting fresh")

    # saving
    target_save_every = 100_000
    ckpt_cb = CheckpointCallback(
        save_freq=max(1, target_save_every // n_envs),
        save_path=EXP_ROOT,
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
        
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

    vec_cb = SaveVecNormCallback(save_freq=target_save_every, path=vn_path)
    rb_cb  = RewardBreakdownCallback(verbose=1)

    # ---- train ----
    total_steps = 5_000_000
    model.learn(total_timesteps=total_steps, callback=CallbackList([ckpt_cb, vec_cb, rb_cb]))

    # final save
    model.save(os.path.join(EXP_ROOT, "final_model"))
    model.get_vec_normalize_env().save(vn_path)
    vec_env.close()