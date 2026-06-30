"""Homeostasis / "boredom" (SPEC §4.4).

Each neuron tracks a low-pass estimate of its own firing rate. The further it is
*below* target_rate, the more random noise current it receives -- so a neuron
starved of real input drifts up until it fires spontaneously (seeding STDP and
new connections), and once real drive brings it to target_rate the boredom term
falls to ~0. This is the built-in exploration->exploitation transition.
"""
from __future__ import annotations

import numpy as np

from config import Config


class Homeostasis:
    def __init__(self, n: int, cfg: Config, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        d = cfg.derived()
        self.alpha_rate = d["alpha_rate"]
        # convert spikes/step into an estimate in Hz: 1 spike in dt ms -> 1000/dt Hz
        self.spike_to_hz = 1000.0 / cfg.dt
        self.rate_est = np.full(n, cfg.target_rate, dtype=np.float64)

    def reset(self) -> None:
        self.rate_est[:] = self.cfg.target_rate

    def noise_current(self) -> np.ndarray:
        """Boredom noise current to add to each neuron's input this step."""
        boredom = np.clip(self.cfg.target_rate - self.rate_est, 0.0, None)
        kick = self.rng.random(self.rate_est.shape)  # Poisson-ish positive kick
        return boredom * self.cfg.noise_gain * kick

    def observe(self, spikes: np.ndarray) -> None:
        """Update the low-pass rate estimate (Hz) from this step's spikes."""
        inst_hz = spikes.astype(np.float64) * self.spike_to_hz
        self.rate_est = self.alpha_rate * self.rate_est + (1 - self.alpha_rate) * inst_hz
