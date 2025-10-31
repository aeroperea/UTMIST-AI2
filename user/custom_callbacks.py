from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
import time
import numpy as np

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