"""Sensory encoding and motor decoding (SPEC §12.3).

Vision -> input spikes:
    input layer is (n_rays x n_channels) neurons, one channel per meaningful hit
    type (reward / punishment / wall / agent). For each ray, the channel matching
    its hit fires as a Poisson process at a rate proportional to proximity; the
    other channels stay silent.

Output spikes -> x/y motion:
    four motor pools (+x, -x, +y, -y). A smooth EMA of each pool's firing rate is
    differenced per axis and scaled into a velocity vector.
"""
from __future__ import annotations

import numpy as np

from config import Config
from env.regions import TYPE_REWARD, TYPE_PUNISH, TYPE_WALL, TYPE_AGENT

# map hit-type code -> channel index within a ray's input block
_TYPE_TO_CHANNEL = {
    TYPE_REWARD: 0,
    TYPE_PUNISH: 1,
    TYPE_WALL: 2,
    TYPE_AGENT: 3,
}

# output pool indices (egocentric forward/turn motor, SPEC §12.3)
OUT_FORWARD, OUT_LEFT, OUT_RIGHT = 0, 1, 2


class SensoryEncoder:
    """Vision rays -> input spike vector (one per agent per step)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.n_channels = len(cfg.vision_channels)
        self.n_vision = cfg.n_rays * self.n_channels
        # interoception: 2 internal sensors [in_reward (satiation), in_punish
        # (nociception)] that fire when the agent is inside a region. Rays skip the
        # region you're in, so this is the only signal that you're in danger.
        self.n_intero = 2 if cfg.interoception else 0
        self.n_input = self.n_vision + self.n_intero
        self.p_scale = cfg.max_input_rate * cfg.dt / 1000.0  # spike prob at proximity=1
        self.last_rates = np.zeros(self.n_input)  # for the UI

    def encode(self, proximity: np.ndarray, hit_type: np.ndarray,
               in_reward: bool, in_punish: bool,
               rng: np.random.Generator) -> np.ndarray:
        """rays + interoception -> bool spikes (n_input,)."""
        rates = np.zeros(self.n_input)
        for ray in range(self.cfg.n_rays):
            ch = _TYPE_TO_CHANNEL.get(int(hit_type[ray]))
            if ch is None:
                continue
            rates[ray * self.n_channels + ch] = proximity[ray]
        if self.n_intero:
            rates[self.n_vision + 0] = 1.0 if in_reward else 0.0   # satiation
            rates[self.n_vision + 1] = 1.0 if in_punish else 0.0   # nociception
        self.last_rates = rates
        p = rates * self.p_scale
        return rng.random(self.n_input) < p


class MotorDecoder:
    """Output spikes -> (forward, turn) command, via EMA firing rate (SPEC §12.3).

    forward >= 0 (world units/step along heading); turn is signed (radians/step,
    + = left). The world (env/regions.py) integrates these into heading and pos.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        d = cfg.derived()
        self.alpha = d["alpha_motor"]
        self.spike_to_hz = 1000.0 / cfg.dt
        self.rate = np.zeros(cfg.n_output)  # Hz per output pool
        # rolling raster for the UI
        self.history = np.zeros((cfg.motor_window, cfg.n_output), dtype=bool)
        self._h = 0

    def reset(self) -> None:
        self.rate[:] = 0.0
        self.history[:] = False

    def decode(self, output_spikes: np.ndarray) -> np.ndarray:
        inst = output_spikes.astype(np.float64) * self.spike_to_hz
        self.rate = self.alpha * self.rate + (1 - self.alpha) * inst
        self.history[self._h] = output_spikes
        self._h = (self._h + 1) % self.cfg.motor_window

        forward = self.rate[OUT_FORWARD] * self.cfg.speed_scale
        turn = (self.rate[OUT_LEFT] - self.rate[OUT_RIGHT]) * self.cfg.turn_scale
        return np.array([forward, turn])
