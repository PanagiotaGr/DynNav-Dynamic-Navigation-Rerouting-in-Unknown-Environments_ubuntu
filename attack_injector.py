from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class AttackConfig:
    mode: str = "none"   # none | replay | step_bias | ramp_bias | burst_noise
    t_start: int = 100

    # replay
    replay_len: int = 30

    # bias attacks
    bias: np.ndarray | None = None   # (3,)
    rate: np.ndarray | None = None   # (3,) per step

    # burst noise
    sigma: float = 0.0
    duration: int = 20

class AttackInjector:
    def __init__(self, cfg: AttackConfig):
        self.cfg = cfg
        self.buf = []  # replay buffer of clean measurements

    def _on(self, t: int) -> bool:
        return self.cfg.mode != "none" and t >= self.cfg.t_start

    def observe_clean(self, z_clean: np.ndarray):
        """Store clean measurements for replay."""
        if self.cfg.mode != "replay":
            return
        self.buf.append(np.array(z_clean, dtype=float))
        if len(self.buf) > self.cfg.replay_len:
            self.buf = self.buf[-self.cfg.replay_len:]

    def apply(self, t: int, z_clean: np.ndarray) -> tuple[np.ndarray, bool]:
        """
        Returns (z_attacked, attack_on_now).
        """
        z = np.array(z_clean, dtype=float)
        if not self._on(t):
            return z, False

        mode = self.cfg.mode

        if mode == "replay":
            if len(self.buf) == 0:
                return z, True
            idx = (t - self.cfg.t_start) % len(self.buf)
            return np.array(self.buf[idx], dtype=float), True

        if mode == "step_bias":
            if self.cfg.bias is None:
                raise ValueError("step_bias requires --bias bx,by,byaw")
            return z + np.array(self.cfg.bias, dtype=float), True

        if mode == "ramp_bias":
            if self.cfg.rate is None:
                raise ValueError("ramp_bias requires --rate rx,ry,ryaw")
            dt = t - self.cfg.t_start
            return z + dt * np.array(self.cfg.rate, dtype=float), True

        if mode == "burst_noise":
            dt = t - self.cfg.t_start
            if 0 <= dt < int(self.cfg.duration) and self.cfg.sigma > 0:
                z = z + np.random.randn(*z.shape) * float(self.cfg.sigma)
                return z, True
            return z, False

        raise ValueError(f"Unknown mode: {mode}")
