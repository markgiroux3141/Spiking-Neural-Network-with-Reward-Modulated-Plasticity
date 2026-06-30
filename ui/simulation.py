"""Headless-capable simulation driver (SPEC §13.4).

Owns the world and every agent's network/encoder/decoder, and exposes
`advance(n_steps)` plus read-only `state()`. Both train.py (headless) and the
Pygame UI drive the *same* Simulation -- no simulation logic lives in the
renderer.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from config import Config
from env.regions import RegionWorld
from env import vision
from env.encoding import SensoryEncoder, MotorDecoder
from snn.network import FeedForwardSNN

TRAIL_LEN = 120
RASTER_LEN = 120


class Agent:
    def __init__(self, idx: int, cfg: Config):
        self.idx = idx
        self.net = FeedForwardSNN(cfg, cfg.rng(1000 + idx))
        self.encoder = SensoryEncoder(cfg)
        self.decoder = MotorDecoder(cfg)
        self.enc_rng = cfg.rng(5000 + idx)
        self.reward_ema = 0.0
        self.trail: deque = deque(maxlen=TRAIL_LEN)
        self.velocity = np.zeros(2)
        # rolling history for the inspector panel
        self.raster_in: deque = deque(maxlen=RASTER_LEN)
        self.raster_hidden: deque = deque(maxlen=RASTER_LEN)
        self.raster_out: deque = deque(maxlen=RASTER_LEN)
        self.reward_hist: deque = deque(maxlen=RASTER_LEN)
        self.delta_hist: deque = deque(maxlen=RASTER_LEN)

    def record(self) -> None:
        self.raster_in.append(self.net.in_spikes.copy())
        self.raster_hidden.append(self.net.hidden_spikes.copy())
        self.raster_out.append(self.net.output_spikes.copy())
        self.reward_hist.append(self.net.last_R)
        self.delta_hist.append(self.net.last_delta_R)


class Simulation:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.world = RegionWorld(self.cfg, self.cfg.rng(0))
        self.agents = [Agent(i, self.cfg) for i in range(self.cfg.n_agents)]
        self.step_count = 0
        self.learning_enabled = True
        # behavioral metric counters (SPEC §14 Phase 11)
        self.time_in_reward = np.zeros(self.cfg.n_agents)
        self.time_in_punish = np.zeros(self.cfg.n_agents)
        # latest vision, cached for the renderer
        self.proximity = None
        self.hit_type = None
        self.hit_point = None
        self._recast()

    def _recast(self) -> None:
        self.proximity, self.hit_type, self.hit_point = vision.cast(self.world, self.cfg)

    def set_learning(self, enabled: bool) -> None:
        self.learning_enabled = enabled
        for a in self.agents:
            a.net.learning_enabled = enabled

    def reset(self, reseed: int | None = None) -> None:
        if reseed is not None:
            self.cfg.seed = reseed
        self.world = RegionWorld(self.cfg, self.cfg.rng(0))
        for a in self.agents:
            a.net.reset_state()
            a.decoder.reset()
            a.reward_ema = 0.0
            a.trail.clear()
        self.step_count = 0
        self.time_in_reward[:] = 0.0
        self.time_in_punish[:] = 0.0
        self._recast()

    def _apply_motor(self, motor: np.ndarray) -> np.ndarray:
        """Step the world with a (M, 2) motor command; update trails/metrics."""
        R = self.world.step(motor)
        in_r, in_p = self.world.occupancy()
        self.time_in_reward += in_r
        self.time_in_punish += in_p
        for a in self.agents:
            a.reward_ema = 0.99 * a.reward_ema + 0.01 * float(R[a.idx])
            a.trail.append(self.world.pos[a.idx].copy())
            a.velocity = self.world.vel[a.idx].copy()
        self._recast()
        self.step_count += 1
        return R

    def step_once(self) -> None:
        """One SNN-driven step: sense -> spike -> motor -> reward update."""
        cfg = self.cfg
        motor = np.zeros((cfg.n_agents, 2))
        for a in self.agents:
            in_spikes = a.encoder.encode(self.proximity[a.idx], self.hit_type[a.idx],
                                         a.enc_rng)
            out_spikes = a.net.step(in_spikes)
            motor[a.idx] = a.decoder.decode(out_spikes)

        R = self._apply_motor(motor)
        for a in self.agents:
            a.net.apply_reward(float(R[a.idx]))
            a.record()

    def advance(self, n_steps: int) -> None:
        for _ in range(n_steps):
            self.step_once()

    # behavioral metric (fraction of agent-steps in each region type) ------
    def reset_metrics(self) -> None:
        """Zero the occupancy counters (e.g. to measure a trailing window only)."""
        self.time_in_reward[:] = 0.0
        self.time_in_punish[:] = 0.0
        self._metric_base_step = self.step_count

    def occupancy_fractions(self) -> tuple[float, float]:
        base = getattr(self, "_metric_base_step", 0)
        denom = max(1, self.step_count - base) * self.cfg.n_agents
        return (float(self.time_in_reward.sum() / denom),
                float(self.time_in_punish.sum() / denom))

    def best_agent(self) -> int:
        return int(np.argmax([a.reward_ema for a in self.agents]))

    def mean_reward(self) -> float:
        return float(np.mean([a.reward_ema for a in self.agents]))
