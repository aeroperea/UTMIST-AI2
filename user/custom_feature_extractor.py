import torch
from torch import nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import math

observation_fields={
                # self
                "s_state":  {"index": 8,   "n": 13, "dim": 16},
                "s_move":   {"index": 14,  "n": 12, "dim": 8},
                "s_weapon": {"index": 15,  "n": 3,  "dim": 4},
                "s_sp0_t":  {"index": 18,  "n": 4,  "dim": 3},
                "s_sp1_t":  {"index": 21,  "n": 4,  "dim": 3},
                "s_sp2_t":  {"index": 24,  "n": 4,  "dim": 3},
                "s_sp3_t":  {"index": 27,  "n": 4,  "dim": 3},
                # opponent (shift +32)
                "o_state":  {"index": 8+32,   "n": 13, "dim": 16},
                "o_move":   {"index": 14+32,  "n": 12, "dim": 8},
                "o_weapon": {"index": 15+32,  "n": 3,  "dim": 4},
                "o_sp0_t":  {"index": 18+32,  "n": 4,  "dim": 3},
                "o_sp1_t":  {"index": 21+32,  "n": 4,  "dim": 3},
                "o_sp2_t":  {"index": 24+32,  "n": 4,  "dim": 3},
                "o_sp3_t":  {"index": 27+32,  "n": 4,  "dim": 3},
            }


class ResMLPExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256, hidden_dim=512):
        super().__init__(observation_space, features_dim)
        obs_dim = observation_space.shape[0]
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 256)    # bottleneck
        self.out = nn.Linear(256, features_dim)  # identity if features_dim==256
        self.act = nn.SiLU()
        self._init_weights()

    def _init_weights(self):
        for m in [self.fc1, self.fc2, self.fc3, self.out]:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.act(self.ln1(self.fc1(obs)))
        h = self.act(self.fc2(x))
        x = x + h
        x = self.act(self.fc3(x))
        return self.out(x)

    @classmethod
    def get_policy_kwargs(cls, features_dim: int = 256, hidden_dim: int = 512) -> dict:
        # convenience helper to match how you're calling extractor.get_policy_kwargs()
        return dict(
            features_extractor_class=cls,
            features_extractor_kwargs=dict(features_dim=features_dim, hidden_dim=hidden_dim)
        )
    
# best_fused_extractor_flatbox.py
# anchor: rmsnorm
# rmsnorm with minimal alloc
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # mean(x^2) along last dim, fused rsqrt, scale in-place where safe
        rms_inv = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True).add(self.eps))
        return self.weight * (x * rms_inv)

# rezero block; init chosen to keep behavior
class ReZeroBlock(nn.Module):
    def __init__(self, dim: int, act=nn.SiLU(inplace=True)):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.fc1  = nn.Linear(dim, dim)
        self.fc2  = nn.Linear(dim, dim)
        self.act  = act
        self.alpha = nn.Parameter(torch.zeros(1))
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.act(self.fc1(h))
        h = self.fc2(h)
        return x + self.alpha * h

class FusedFeatureExtractor(BaseFeaturesExtractor):
    """
    flat box obs with ego-flip, enum embeds, pairwise dx,dy,r2, rezero+rmsnorm trunk.
    fast path: precomputed enum ranges, in-place flips, single concat.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
        hidden_dim: int = 512,
        enum_fields: dict | None = None,
        xy_player: list[int] | tuple[int, int] | None = None,
        xy_opponent: list[int] | tuple[int, int] | None = None,
        use_pairwise: bool = True,
        facing_index: int | None = 4,
        flip_x_indices: list[int] | tuple[int, ...] | None = (0, 2, 32, 34),
        flip_pair_dx: bool = True,
        invert_facing: bool = False,
        use_compile: bool = False,
    ):
        assert isinstance(observation_space, spaces.Box) and len(observation_space.shape) == 1
        super().__init__(observation_space, features_dim)

        D = int(observation_space.shape[0])
        low  = torch.as_tensor(observation_space.low,  dtype=torch.float32)
        high = torch.as_tensor(observation_space.high, dtype=torch.float32)
        finite = torch.isfinite(low) & torch.isfinite(high)

        center = torch.where(finite, 0.5 * (high + low), torch.zeros_like(low))
        mul    = torch.where(finite, 2.0 / (high - low).clamp(min=1e-6), torch.ones_like(low))
        self.register_buffer("_center", center, persistent=False)
        self.register_buffer("_mul",    mul,    persistent=False)
        self.D = D

        # enum tables
        self._enum_items: list[tuple[int, int, int, str]] = []
        self.embs = nn.ModuleDict()
        if enum_fields:
            for name, cfg in enum_fields.items():
                idx = int(cfg["index"]); ncls = int(cfg["n"]); dim = int(cfg["dim"])
                assert 0 <= idx < D and ncls >= 2 and dim >= 1
                self._enum_items.append((idx, ncls, dim, name))
                self.embs[name] = nn.Embedding(ncls, dim)

        # anchor: enum_precompute_affine
        if self._enum_items:
            lows, highs, scales, biases = [], [], [], []
            for (idx, ncls, _dim, _name) in self._enum_items:
                inv_m = 1.0 / max(self._mul[idx].item(), 1e-6)
                lo = float(self._center[idx].item() - inv_m)
                hi = float(self._center[idx].item() + inv_m)
                lows.append(lo); highs.append(hi)
                rng = max(hi - lo, 1e-6)
                s = (ncls - 1) / rng
                b = -lo * s
                scales.append(s); biases.append(b)
            self.register_buffer("_enum_low",   torch.tensor(lows,   dtype=torch.float32), persistent=False)
            self.register_buffer("_enum_high",  torch.tensor(highs,  dtype=torch.float32), persistent=False)
            self.register_buffer("_enum_scale", torch.tensor(scales, dtype=torch.float32), persistent=False)
            self.register_buffer("_enum_bias",  torch.tensor(biases, dtype=torch.float32), persistent=False)
        else:
            self.register_buffer("_enum_low",   torch.empty(0), persistent=False)
            self.register_buffer("_enum_high",  torch.empty(0), persistent=False)
            self.register_buffer("_enum_scale", torch.empty(0), persistent=False)
            self.register_buffer("_enum_bias",  torch.empty(0), persistent=False)

        # xy and flip config as buffers
        self.register_buffer("xy_p",     torch.as_tensor(xy_player,   dtype=torch.long) if xy_player   is not None else torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("xy_o",     torch.as_tensor(xy_opponent, dtype=torch.long) if xy_opponent is not None else torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("flip_idx", torch.as_tensor(flip_x_indices, dtype=torch.long) if flip_x_indices else torch.empty(0, dtype=torch.long), persistent=False)

        self._use_pair     = bool(use_pairwise and self.xy_p.numel() == 2 and self.xy_o.numel() == 2)
        self.facing_index  = None if facing_index is None else int(facing_index)
        self.flip_pair_dx  = bool(flip_pair_dx)
        self.invert_facing = bool(invert_facing)

        # trunk
        pair_extra = 3 if self._use_pair else 0
        in_extra   = sum(dim for (_i, _n, dim, _name) in self._enum_items)
        in_dim     = D + in_extra + pair_extra

        self.in_norm = RMSNorm(in_dim)
        self.embed   = nn.Linear(in_dim, hidden_dim)
        self.block1  = ReZeroBlock(hidden_dim, act=nn.SiLU(inplace=True))
        self.block2  = ReZeroBlock(hidden_dim, act=nn.SiLU(inplace=True))
        self.head    = nn.Linear(hidden_dim, features_dim)
        nn.init.kaiming_uniform_(self.embed.weight, nonlinearity="relu")
        nn.init.zeros_(self.embed.bias)
        nn.init.orthogonal_(self.head.weight, gain=0.5)
        nn.init.zeros_(self.head.bias)
        self.act = nn.SiLU(inplace=True)

        if use_compile and hasattr(torch, "compile"):
            self.forward = torch.compile(self.forward, mode="reduce-overhead", fullgraph=False)  # type: ignore


    # map scalar -> class id using cached lows/highs; single pass loop   
    def _enum_embs(self, obs: torch.Tensor) -> torch.Tensor:
        if not self._enum_items:
            return obs.new_zeros((obs.shape[0], 0))
        with torch.no_grad():
            ids_list = [
                torch.round(obs[:, idx] * self._enum_scale[j] + self._enum_bias[j]) \
                    .to(torch.long).clamp_(0, ncls - 1)
                for j, (idx, ncls, _dim, _name) in enumerate(self._enum_items)
            ]
        outs = [self.embs[name](ids) for (_idx, _ncls, _dim, name), ids in zip(self._enum_items, ids_list)]
        return torch.cat(outs, dim=-1)

    def _ego_sign(self, obs: torch.Tensor) -> torch.Tensor | None:
        if self.facing_index is not None:
            col = obs[:, self.facing_index]
            if self.invert_facing:
                col = -col
            s = torch.where(col > 0.5, 1.0, -1.0)
            return s.unsqueeze(-1)
        if self.xy_p.numel() == 2 and self.xy_o.numel() == 2:
            dx = obs[:, self.xy_p[0]] - obs[:, self.xy_o[0]]
            s = torch.where(dx >= 0, 1.0, -1.0)
            return s.unsqueeze(-1)
        return None

    def _pair_features(self, obs: torch.Tensor, sign: torch.Tensor | None) -> torch.Tensor:
        if not self._use_pair:
            return obs.new_zeros((obs.shape[0], 0))
        p = obs[:, self.xy_p]  # [B,2]
        o = obs[:, self.xy_o]  # [B,2]
        d = p - o              # [B,2]
        if sign is not None and self.flip_pair_dx:
            d[:, 0].mul_(sign.squeeze(-1))  # in-place flip x component
        r2 = d.mul(d).sum(-1, keepdim=True)
        return torch.cat((d, r2), dim=-1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # scale to ~[-1,1]
        x = (obs - self._center) * self._mul

        # ego flip: in-place on selected columns
        sign = self._ego_sign(obs)
        if sign is not None and self.flip_idx.numel():
            cols = self.flip_idx
            x[:, cols] = x[:, cols] * sign

        # engineered extras (single concat)
        parts = [x]
        e = self._enum_embs(obs)
        if e.numel():
            parts.append(e)
        p = self._pair_features(obs, sign)
        if p.numel():
            parts.append(p)
        x = torch.cat(parts, dim=-1)

        x = self.in_norm(x)
        x = self.act(self.embed(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.act(x)
        return self.head(x)

import numpy as np
import gymnasium as gym

class ActionMirrorWrapper(gym.Wrapper):
    def __init__(self, env,
                 facing_index=4, px_index=0, ox_index=32,
                 swap_pairs=((1, 3),),   # <-- left/right
                 mirror_axes=(),         # none, actions are buttons
                 mirror_angles=(),       # none, no analog aim angle here
                 invert_facing=False):
        super().__init__(env)
        self.facing_index = facing_index
        self.px_index = px_index
        self.ox_index = ox_index
        self.swap_pairs = tuple(swap_pairs)
        self.mirror_axes = tuple(mirror_axes)
        self.mirror_angles = tuple(mirror_angles)
        self.invert_facing = bool(invert_facing)
        self._last_obs = None

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self._last_obs = obs
        return obs, info

    def _sign(self, obs: np.ndarray) -> float:
        if self.facing_index is not None:
            v = float(obs[self.facing_index])
            if self.invert_facing:
                v = -v
            return 1.0 if v > 0.5 else -1.0
        dx = float(obs[self.px_index] - obs[self.ox_index])
        return 1.0 if dx >= 0.0 else -1.0

    def _mirror(self, a: np.ndarray, sign: float) -> np.ndarray:
        if sign >= 0.0:
            return a
        a = a.copy()
        for i, j in self.swap_pairs:
            a[i], a[j] = a[j], a[i]
        for k in self.mirror_axes:
            a[k] = 1.0 - float(a[k])
        for k in self.mirror_angles:
            a[k] = 1.0 - float(a[k])
        return a

    def step(self, action):
        s = self._sign(self._last_obs) if self._last_obs is not None else 1.0
        action = self._mirror(np.asarray(action, dtype=np.float32), s)
        obs, r, t, tr, info = self.env.step(action)
        self._last_obs = obs
        return obs, r, t, tr, info

def _mirror_action(a: np.ndarray, sign: float) -> np.ndarray:
    if sign >= 0.0:
        return a
    a = a.copy()
    i, j = 1, 3  # A<->D
    a[i], a[j] = a[j], a[i]
    return a