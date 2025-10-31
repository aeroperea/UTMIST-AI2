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
from user.custom_feature_extractor import _mirror_action

FUSED_EXTRACTOR_KW = dict(
        features_dim=256,
        hidden_dim=512,
        enum_fields=observation_fields,
        xy_player=[0, 1],
        xy_opponent=[32, 33],
        use_pairwise=True,
        facing_index=4,
        flip_x_indices=(0, 2, 32, 34),
        flip_pair_dx=True,
        invert_facing=False,
        use_compile=False,
        n_blocks=2  # <--- set 4 first; try 6 if value underfits
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

        dummy = _SpecEnv(obs_dim=64, act_dim=10)

        # use auto so cpu-only runtimes don't break
        self.model = PPO(
            "MlpPolicy",
            dummy,
            policy_kwargs=policy_kwargs,
            device="auto",
            verbose=0,
            n_steps=32,
            batch_size=32,
            n_epochs=1,
        )

        if self.file_path is None:
            return

        # policy-only load
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
            raise RuntimeError("checkpoint lacks 'log_std' (action space mismatch at train time?)")
        self.model.policy.load_state_dict(policy_state, strict=True)

    def _gdown(self) -> str:
        data_path = "rl-model.zip"
        if not os.path.isfile(data_path):
            print(f"downloading {data_path}...")
            url = "https://drive.google.com/file/d/1JIokiBOrOClh8piclbMlpEEs6mj3H1HJ/view?usp=sharing"
            gdown.download(url, output=data_path, fuzzy=True)
        return data_path

    # anchor: submitted_agent_predict
    def predict(self, obs):
        """
        mirror to match training wrapper: swap indices (1,3) when facing left.
        assumes Box[0,1] actions with button gates around 0.5.
        handles (64,) or (B,64).
        """
        import numpy as np

        def _swap_A_D(a: np.ndarray, facing_right_bit: float) -> np.ndarray:
            if facing_right_bit <= 0.5:
                a = a.copy()
                a[1], a[3] = a[3], a[1]   # A <-> D
            return a

        arr = np.asarray(obs, dtype=np.float32)

        if arr.ndim == 1:
            facing_right = float(arr[4]) if arr.shape[0] > 4 else 1.0
            # sample like training; switch to deterministic=True later if you prefer
            action, _ = self.model.predict(arr, deterministic=False)
            # safety clamp to [0,1] in case action_space low/high differs
            action = np.clip(action, 0.0, 1.0)
            action = _swap_A_D(action, facing_right)
            # small bias to cross 0.5 gates; remove if not needed
            action = np.clip(action + 0.02, 0.0, 1.0)
            return action

        # batched
        actions, _ = self.model.predict(arr, deterministic=False)
        actions = np.clip(actions, 0.0, 1.0)
        for i in range(arr.shape[0]):
            facing_right = float(arr[i, 4]) if arr.shape[1] > 4 else 1.0
            actions[i] = _swap_A_D(actions[i], facing_right)
            actions[i] = np.clip(actions[i] + 0.02, 0.0, 1.0)
        return actions


    def save(self, file_path: str) -> None:
        self.model.save(file_path)

    def learn(self, env, total_timesteps, log_interval: int = 4):
        self.model.set_env(env)
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)