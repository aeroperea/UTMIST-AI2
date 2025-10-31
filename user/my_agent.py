# submission: agent (box action space + policy-only load)
import os
import gdown
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from environment.agent import Agent
from stable_baselines3 import PPO
from stable_baselines3.common.save_util import load_from_zip_file
from torch import nn

# your fused extractor + field map
from user.custom_feature_extractor import (
    FusedFeatureExtractor,
    observation_fields,   # dict with enum field indices/sizes
)

# ------------------ obs-space with finite bounds (matches training expectations) ------------------

OBS_DIM = 64
def _make_submission_obs_space() -> spaces.Box:
    low  = np.full(OBS_DIM, -1.0, dtype=np.float32)
    high = np.full(OBS_DIM,  1.0, dtype=np.float32)

    # kinematics (conservative but finite)
    # self
    low[0],  high[0]  = -30.0, 30.0   # px
    low[1],  high[1]  = -10.0, 25.0   # py
    low[2],  high[2]  = -25.0, 25.0   # vx
    low[3],  high[3]  = -25.0, 25.0   # vy
    low[4],  high[4]  =   0.0,  1.0   # facing bit

    # opponent
    low[32], high[32] = -30.0, 30.0   # ox
    low[33], high[33] = -10.0, 25.0   # oy
    low[34], high[34] = -25.0, 25.0   # ovx
    low[35], high[35] = -25.0, 25.0   # ovy
    low[36], high[36] =   0.0,  1.0   # ofacing bit

    # enum-like columns: make them 0..(n-1) so embedding ids match training logic
    for name, cfg in observation_fields.items():
        idx, n = int(cfg["index"]), int(cfg["n"])
        low[idx], high[idx] = 0.0, float(n - 1)

    return spaces.Box(low=low, high=high, dtype=np.float32)

# ------------------ tiny spec env that carries real bounds ------------------

class _SpecEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, act_dim=10):
        super().__init__()
        self.observation_space = _make_submission_obs_space()
        # gaussian policy -> continuous box with log_std in policy
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(act_dim,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, False, False, {}

# ------------------ action mirroring (A<->D) + thresholding ------------------

def _ego_sign_from_obs(arr: np.ndarray, facing_index: int = 4, px_index: int = 0, ox_index: int = 32,
                       invert_facing: bool = False) -> float:
    if arr.shape[0] > facing_index:
        v = float(arr[facing_index])
        if invert_facing:
            v = -v
        return 1.0 if v > 0.5 else -1.0
    # fallback to dx
    return 1.0 if float(arr[px_index] - arr[ox_index]) >= 0.0 else -1.0

def _mirror_action(a: np.ndarray, sign: float) -> np.ndarray:
    # indices: 1 = A (left), 3 = D (right) per your mapping
    if sign >= 0.0:
        return a
    a = a.copy()
    a[1], a[3] = a[3], a[1]      # swap left/right
    return a

def _threshold01(a: np.ndarray, thr: float = 0.5) -> np.ndarray:
    # convert to 0/1 keys expected by the game input layer
    return (a >= thr).astype(np.float32)

# ------------------ submitted agent ------------------

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
)

class SubmittedAgent(Agent):
    def __init__(self, file_path: Optional[str] = None):
        super().__init__(file_path)

    def _initialize(self) -> None:
        policy_kwargs = dict(
            activation_fn=nn.SiLU,
            net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
            features_extractor_class=FusedFeatureExtractor,
            features_extractor_kwargs=FUSED_EXTRACTOR_KW,
        )

        dummy = _SpecEnv(act_dim=10)

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
        """mirror A<->D if facing left; then threshold to 0/1 keys."""
        arr = np.asarray(obs, dtype=np.float32)

        if arr.ndim == 1:
            sign = _ego_sign_from_obs(arr, facing_index=4, px_index=0, ox_index=32, invert_facing=False)
            a, _ = self.model.predict(arr, deterministic=True)
            a = _mirror_action(a, sign)
            a = _threshold01(a, 0.5)
            return a

        # batched
        actions, _ = self.model.predict(arr, deterministic=True)
        for i in range(arr.shape[0]):
            sign = _ego_sign_from_obs(arr[i], facing_index=4, px_index=0, ox_index=32, invert_facing=False)
            actions[i] = _threshold01(_mirror_action(actions[i], sign), 0.5)
        return actions

    def save(self, file_path: str) -> None:
        self.model.save(file_path)

    def learn(self, env, total_timesteps, log_interval: int = 4):
        self.model.set_env(env)
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
