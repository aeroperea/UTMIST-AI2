import torch
from torch import nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

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
# ------------------ core blocks ------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(torch.clamp(x.pow(2).mean(dim=-1, keepdim=True), min=self.eps))
        return self.weight * (x * rms)

class ReZeroBlock(nn.Module):
    def __init__(self, dim: int, act=nn.SiLU()):
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

# ------------------ extractor ------------------

class FusedFeatureExtractor(BaseFeaturesExtractor):
    """
    flat box obs. env stays unchanged.
    features:
      - bound-aware affine scaling for floats
      - small learned embeddings for discrete fields (by hard-coded indices)
      - optional engineered pair features (dx, dy, r2)
      - rezero + rmsnorm residual mlp
    note: assumes obs fed to policy are raw env values (not vecnormalized).
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
        hidden_dim: int = 512,
        # enum_fields: dict[name] = dict(index=int, n=int, dim=int)
        enum_fields: dict | None = None,
        # pairwise indices to add dx,dy,r2 if available
        xy_player: list[int] | tuple[int, int] | None = None,
        xy_opponent: list[int] | tuple[int, int] | None = None,
        use_pairwise: bool = True,
    ):
        assert isinstance(observation_space, spaces.Box) and len(observation_space.shape) == 1
        super().__init__(observation_space, features_dim)

        D = int(observation_space.shape[0])
        low  = torch.as_tensor(observation_space.low,  dtype=torch.float32)
        high = torch.as_tensor(observation_space.high, dtype=torch.float32)
        finite = torch.isfinite(low) & torch.isfinite(high)

        # bound-aware center/scale for floats; 0 where inf
        c = torch.where(finite, 0.5 * (high + low), torch.zeros_like(low))
        m = torch.where(finite, 2.0 / (high - low).clamp(min=1e-6), torch.ones_like(low))
        self.register_buffer("_center", c, persistent=False)
        self.register_buffer("_mul",    m, persistent=False)
        self.D = D

        # embeddings for integer-coded fields by index
        self.enum_cfg = {}
        self.embs = nn.ModuleDict()
        in_extra = 0
        if enum_fields:
            for name, cfg in enum_fields.items():
                idx = int(cfg["index"]); ncls = int(cfg["n"]); dim = int(cfg["dim"])
                assert 0 <= idx < D and ncls >= 2 and dim >= 1
                self.enum_cfg[name] = (idx, ncls, dim)
                self.embs[name] = nn.Embedding(ncls, dim)
                in_extra += dim

        # optional engineered pairwise features
        self.xy_p = torch.as_tensor(xy_player, dtype=torch.long) if xy_player is not None else None
        self.xy_o = torch.as_tensor(xy_opponent, dtype=torch.long) if xy_opponent is not None else None
        add_pair = int(use_pairwise and (self.xy_p is not None) and (self.xy_o is not None))
        pair_extra = 3 if add_pair else 0  # dx, dy, r2

        # trunk
        in_dim = D + in_extra + pair_extra
        self.in_norm = RMSNorm(in_dim)
        self.embed = nn.Linear(in_dim, hidden_dim)
        self.block1 = ReZeroBlock(hidden_dim, act=nn.SiLU())
        self.block2 = ReZeroBlock(hidden_dim, act=nn.SiLU())
        self.head = nn.Linear(hidden_dim, features_dim)
        nn.init.kaiming_uniform_(self.embed.weight, nonlinearity="relu")
        nn.init.zeros_(self.embed.bias)
        nn.init.orthogonal_(self.head.weight, gain=0.5)
        nn.init.zeros_(self.head.bias)
        self.act = nn.SiLU()

    def _enum_ids_from_scalar(self, col: torch.Tensor, low: float, high: float, ncls: int) -> torch.Tensor:
        # robust mapping: if finite bounds, assume linear map to [0..n-1]; else treat as raw ids
        if torch.isfinite(torch.tensor(low)) and torch.isfinite(torch.tensor(high)) and (high > low):
            t = (col - low) / max(high - low, 1e-6)
            ids = torch.round(t * (ncls - 1)).to(torch.long)
        else:
            ids = torch.round(col).to(torch.long)
        return ids.clamp_(0, ncls - 1)

    def _gather_enum_embs(self, obs: torch.Tensor) -> torch.Tensor:
        outs = []
        for name, (idx, ncls, _dim) in self.enum_cfg.items():
            col = obs[:, idx]
            ids = self._enum_ids_from_scalar(col, float(self._center[idx] - 1.0 / max(self._mul[idx].item(), 1e-6)), 
                                                  float(self._center[idx] + 1.0 / max(self._mul[idx].item(), 1e-6)), ncls)
            outs.append(self.embs[name](ids))
        return torch.cat(outs, dim=-1) if outs else obs.new_zeros((obs.shape[0], 0))

    def _pair_features(self, obs: torch.Tensor) -> torch.Tensor:
        if (self.xy_p is None) or (self.xy_o is None):
            return obs.new_zeros((obs.shape[0], 0))
        p = obs[:, self.xy_p]  # [B,2]
        o = obs[:, self.xy_o]  # [B,2]
        d = p - o              # dx,dy
        r2 = (d * d).sum(-1, keepdim=True)
        return torch.cat([d, r2], dim=-1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # scale floats to roughly [-1,1] where bounds are known
        x = obs
        if x.shape[-1] == self.D:
            x = (x - self._center) * self._mul

        e = self._gather_enum_embs(obs)   # use raw obs for ids
        p = self._pair_features(obs)      # use raw obs for geometry
        if e.numel() or p.numel():
            x = torch.cat([x, e, p], dim=-1)

        x = self.in_norm(x)
        x = self.act(self.embed(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.act(x)
        return self.head(x)