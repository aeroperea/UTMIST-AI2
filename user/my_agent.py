# file: user/my_agent.py
# anchors: imports, defaults, spec_env, infer, class_submitted_agent

# anchor: imports
import os
import re
import gdown
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.save_util import load_from_zip_file

from environment.agent import Agent
from user.train_agent import MLPExtractor
from user.custom_feature_extractor import *
from user.custom_feature_extractor import _mirror_action

# anchor: defaults
FUSED_EXTRACTOR_KW = dict(
    features_dim=512,
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
    n_blocks=8
)

# anchor: spec_env
class _SpecEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, obs_dim=64, act_dim=10):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(act_dim,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, False, False, {}

# anchor: infer
def _infer_arch_from_state(policy_state: Dict[str, Any]) -> Tuple[List[int], List[int], Optional[int]]:
    def _layer_sizes(prefix: str) -> Tuple[List[int], Optional[int]]:
        wkeys = [k for k in policy_state.keys() if k.startswith(prefix) and k.endswith(".weight")]
        if not wkeys:
            return [], None
        def _idx(k: str) -> int:
            m = re.search(r"\.(\d+)\.weight$", k)
            return int(m.group(1)) if m else 0
        wkeys.sort(key=_idx)
        outs = []
        features_in = None
        for i, k in enumerate(wkeys):
            w = policy_state[k]
            out_dim, in_dim = int(w.shape[0]), int(w.shape[1])
            outs.append(out_dim)
            if i == 0:
                features_in = in_dim
        return outs, features_in

    pi_shape, feat_in_pi = _layer_sizes("mlp_extractor.policy_net.")
    vf_shape, feat_in_vf = _layer_sizes("mlp_extractor.value_net.")
    inferred_features_dim = feat_in_pi or feat_in_vf
    return pi_shape, vf_shape, inferred_features_dim

# anchor: class_submitted_agent
class SubmittedAgent(Agent):
    def __init__(
        self,
        file_path: Optional[str] = None,
        *,
        pi_shape: Optional[List[int]] = None,
        vf_shape: Optional[List[int]] = None,
        extractor_n_blocks: Optional[int] = None
    ):
        self._user_pi = list(pi_shape) if pi_shape is not None else None
        self._user_vf = list(vf_shape) if vf_shape is not None else None
        self._user_nblocks = int(extractor_n_blocks) if extractor_n_blocks is not None else None
        super().__init__(file_path)

    def _make_policy_kwargs(
        self,
        pi_shape: Optional[List[int]],
        vf_shape: Optional[List[int]],
        features_dim: Optional[int],
        n_blocks: Optional[int]
    ) -> Dict[str, Any]:
        net_arch = dict(pi=[512,256,256,128,128], vf=[512,256,256,128,128])
        if pi_shape is not None:
            net_arch["pi"] = list(pi_shape)
        if vf_shape is not None:
            net_arch["vf"] = list(vf_shape)

        fx_kw = dict(FUSED_EXTRACTOR_KW)
        if features_dim is not None:
            fx_kw["features_dim"] = int(features_dim)
        if n_blocks is not None:
            fx_kw["n_blocks"] = int(n_blocks)

        return dict(
            activation_fn=nn.SiLU,
            net_arch=net_arch,
            features_extractor_class=FusedFeatureExtractor,
            features_extractor_kwargs=fx_kw,
        )

    def _initialize(self) -> None:
        req_pi = self._user_pi
        req_vf = self._user_vf
        req_nblocks = self._user_nblocks

        policy_kwargs = self._make_policy_kwargs(
            pi_shape=req_pi,
            vf_shape=req_vf,
            features_dim=None,
            n_blocks=req_nblocks
        )

        dummy = _SpecEnv(obs_dim=64, act_dim=10)
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
            device="cuda",
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
            raise RuntimeError("checkpoint lacks 'log_std'")

        try:
            self.model.policy.load_state_dict(policy_state, strict=True)
        except Exception as e:
            print(f"[submitted_agent] shape mismatch; rebuilding to match checkpoint: {e}")
            pi_ckpt, vf_ckpt, feat_in = _infer_arch_from_state(policy_state)

            if self._user_pi and self._user_pi != pi_ckpt:
                print(f"[submitted_agent] ignoring requested pi_shape {self._user_pi} -> {pi_ckpt}")
            if self._user_vf and self._user_vf != vf_ckpt:
                print(f"[submitted_agent] ignoring requested vf_shape {self._user_vf} -> {vf_ckpt}")
            if self._user_nblocks is not None:
                print("[submitted_agent] extractor_n_blocks must match checkpoint; using checkpoint layout")

            rebuilt_kwargs = self._make_policy_kwargs(
                pi_shape=pi_ckpt or self._user_pi,
                vf_shape=vf_ckpt or self._user_vf,
                features_dim=feat_in,
                n_blocks=None
            )
            self.model = PPO(
                "MlpPolicy",
                _SpecEnv(obs_dim=64, act_dim=10),
                policy_kwargs=rebuilt_kwargs,
                device="auto",
                verbose=0,
                n_steps=32,
                batch_size=32,
                n_epochs=1,
            )
            self.model.policy.load_state_dict(policy_state, strict=True)

    # anchor: submitted_agent_predict
    def predict(self, obs):
        import numpy as np
        def _swap_A_D(a: np.ndarray, facing_right_bit: float) -> np.ndarray:
            if facing_right_bit <= 0.5:
                a = a.copy()
                a[1], a[3] = a[3], a[1]
            return a

        arr = np.asarray(obs, dtype=np.float32)

        if arr.ndim == 1:
            facing_right = float(arr[4]) if arr.shape[0] > 4 else 1.0
            action, _ = self.model.predict(arr, deterministic=False)
            action = np.clip(action, 0.0, 1.0)
            action = _swap_A_D(action, facing_right)
            action = np.clip(action + 0.02, 0.0, 1.0)
            return action

        actions, _ = self.model.predict(arr, deterministic=False)
        actions = np.clip(actions, 0.0, 1.0)
        for i in range(arr.shape[0]):
            facing_right = float(arr[i, 4]) if arr.shape[1] > 4 else 1.0
            actions[i] = _swap_A_D(actions[i], facing_right)
            actions[i] = np.clip(actions[i] + 0.02, 0.0, 1.0)
        return actions

    def _gdown(self) -> str:
        data_path = "rl-model.zip"
        if not os.path.isfile(data_path):
            print(f"downloading {data_path}...")
            url = "https://drive.google.com/file/d/1JIokiBOrOClh8piclbMlpEEs6mj3H1HJ/view?usp=sharing"
            gdown.download(url, output=data_path, fuzzy=True)
        return data_path

    def save(self, file_path: str) -> None:
        self.model.save(file_path)

    def learn(self, env, total_timesteps, log_interval: int = 4):
        self.model.set_env(env)
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
