# submission: agent (box action space + policy-only load)
import os
import gdown
from typing import Optional
from environment.agent import Agent
from stable_baselines3 import PPO
from stable_baselines3.common.save_util import load_from_zip_file
from user.train_agent import MLPExtractor
from torch import nn

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# --- tiny spec-only env (gymnasium) ---
class _SpecEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, obs_dim=64, act_dim=10):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        # IMPORTANT: Box action space to match Gaussian policy (has log_std)
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
        # mirror training arch
        policy_kwargs = dict(
            activation_fn=nn.SiLU,
            net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
            features_extractor_class=MLPExtractor,
            features_extractor_kwargs=dict(features_dim=256, hidden_dim=512),
        )

        # env spec must match dims of the trained model
        dummy = _SpecEnv(obs_dim=64, act_dim=10)

        # minimal container; cpu is fine for MLP policies
        self.model = PPO(
            "MlpPolicy",
            dummy,
            policy_kwargs=policy_kwargs,
            device="cpu",
            verbose=0,
            n_steps=32,
            batch_size=32,
            n_epochs=1,
        )

        if self.file_path is None:
            return

        # load policy-only params; replace pickled objects to avoid deserialization errors
        _, params, _ = load_from_zip_file(
            self.file_path,
            device="cpu",
            custom_objects={
                "features_extractor_class": MLPExtractor,
                "activation_fn": nn.SiLU,
                "clip_range": 0.2,                    # replace schedules/callables
                "lr_schedule": (lambda *_: 3e-4),
                "policy_kwargs": {},                  # silence policy_kwargs pickle
            },
            print_system_info=False,
        )
        policy_state = params.get("policy")
        if policy_state is None:
            raise RuntimeError("checkpoint missing 'policy' state_dict")

        # sanity: checkpoint expects Gaussian policy (has log_std)
        if "log_std" not in policy_state:
            raise RuntimeError("checkpoint does not contain 'log_std'; did you train with a different action space?")

        self.model.policy.load_state_dict(policy_state, strict=True)

    def _gdown(self) -> str:
        data_path = "rl-model.zip"
        if not os.path.isfile(data_path):
            print(f"Downloading {data_path}...")
            url = "https://drive.google.com/file/d/1JIokiBOrOClh8piclbMlpEEs6mj3H1HJ/view?usp=sharing"
            gdown.download(url, output=data_path, fuzzy=True)
        return data_path

    def predict(self, obs):
        action, _ = self.model.predict(obs)
        # if your runtime env expects 0/1 keys, you can threshold:
        # action = (action > 0.5).astype(np.float32)
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path)

    def learn(self, env, total_timesteps, log_interval: int = 4):
        self.model.set_env(env)
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
