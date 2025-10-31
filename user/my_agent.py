# submission: agent (box action space + policy-only load)
import os
import gdown
from typing import Optional
from environment.agent import Agent
from stable_baselines3 import PPO
from stable_baselines3.common.save_util import load_from_zip_file
from torch import nn
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from user.train_agent import MLPExtractor
from user.custom_feature_extractor import *

# fused extractor args (identical to trainer)
FUSED_EXTRACTOR_KW = dict(
    features_dim=256,
    hidden_dim=512,
    enum_fields=observation_fields,
    xy_player=[0, 1],
    xy_opponent=[32, 33],
    use_pairwise=True,
    facing_index=4,                  # uses facing bit; set None to use dx fallback
    flip_x_indices=(0, 2, 32, 34),   # flip x and vx for both agents
    flip_pair_dx=True,
    invert_facing=False,
    use_compile=False,
)

# --- tiny spec-only env (gymnasium) ---
class _SpecEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, obs_dim=64, act_dim=10):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        # gaussian policy -> continuous box with log_std in policy
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(act_dim,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, False, False, {}

class SubmittedAgent(Agent):
    def __init__(self, file_path: Optional[str] = None):
        super().__init__(file_path)

    def _initialize(self) -> None:
        # mirror training policy + fused features
        policy_kwargs = dict(
            activation_fn=nn.SiLU,
            net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
            features_extractor_class=FusedFeatureExtractor,
            features_extractor_kwargs=FUSED_EXTRACTOR_KW,
        )

        # env spec must match model io
        dummy = _SpecEnv(obs_dim=64, act_dim=10)

        # container for policy-only load
        self.model = PPO(
            "MlpPolicy",
            dummy,
            policy_kwargs=policy_kwargs,
            device="cuda",
            verbose=0,
            n_steps=32,
            batch_size=32,
            n_epochs=1,
        )

        if self.file_path is None:
            return

        # load policy params only; replace pickled objs
        _, params, _ = load_from_zip_file(
            self.file_path,
            device="cpu",
            custom_objects={
                "features_extractor_class": FusedFeatureExtractor,
                "activation_fn": nn.SiLU,
                "clip_range": 0.2,
                "lr_schedule": (lambda *_: 3e-4),
                "policy_kwargs": {},
            },
            print_system_info=False,
        )
        policy_state = params.get("policy")
        if policy_state is None:
            raise RuntimeError("checkpoint missing 'policy' state_dict")
        if "log_std" not in policy_state:
            raise RuntimeError("checkpoint lacks 'log_std' (was action space different at train time?)")

        self.model.policy.load_state_dict(policy_state, strict=True)

    def _gdown(self) -> str:
        data_path = "rl-model.zip"
        if not os.path.isfile(data_path):
            print(f"downloading {data_path}...")
            url = "https://drive.google.com/file/d/1JIokiBOrOClh8piclbMlpEEs6mj3H1HJ/view?usp=sharing"
            gdown.download(url, output=data_path, fuzzy=True)
        return data_path

    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path)

    def learn(self, env, total_timesteps, log_interval: int = 4):
        self.model.set_env(env)
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)