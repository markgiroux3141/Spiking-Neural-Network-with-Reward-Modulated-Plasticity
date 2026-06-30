"""Multi-agent reward/punishment region world (SPEC §12.1).

A continuous W x H box with walls, M agents stepped together, and a set of
circular regions: reward regions (sign +1, seek) and punishment regions
(sign -1, avoid). Each step takes a per-agent velocity command and returns each
agent's reward signal -- the third factor `R` for that agent's network.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import Config

# hit-type codes shared with vision.py / encoding.py
TYPE_NONE = 0
TYPE_REWARD = 1
TYPE_PUNISH = 2
TYPE_WALL = 3
TYPE_AGENT = 4


@dataclass
class Region:
    cx: float
    cy: float
    r: float
    sign: int          # +1 reward, -1 punishment
    magnitude: float

    @property
    def type_code(self) -> int:
        return TYPE_REWARD if self.sign > 0 else TYPE_PUNISH


class RegionWorld:
    def __init__(self, cfg: Config, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.W = cfg.world_w
        self.H = cfg.world_h
        self.M = cfg.n_agents

        self.regions: list[Region] = []
        self.pos = np.zeros((self.M, 2))
        self.heading = np.zeros(self.M)
        self.vel = np.zeros((self.M, 2))
        self.reset()

    # --- setup -----------------------------------------------------------
    def _spawn_regions(self) -> None:
        cfg = self.cfg
        self.regions = []
        for _ in range(cfg.n_reward_regions):
            self.regions.append(self._random_region(sign=+1, mag=cfg.reward_magnitude))
        for _ in range(cfg.n_punish_regions):
            self.regions.append(self._random_region(sign=-1, mag=cfg.punish_magnitude))

    def _random_region(self, sign: int, mag: float) -> Region:
        pad = self.cfg.region_radius
        cx = self.rng.uniform(pad, self.W - pad)
        cy = self.rng.uniform(pad, self.H - pad)
        return Region(cx, cy, self.cfg.region_radius, sign, mag)

    def reset(self) -> None:
        self._spawn_regions()
        # scatter agents, random headings
        self.pos = self.rng.uniform(
            [self.cfg.agent_radius, self.cfg.agent_radius],
            [self.W - self.cfg.agent_radius, self.H - self.cfg.agent_radius],
            size=(self.M, 2),
        )
        self.heading = self.rng.uniform(-np.pi, np.pi, size=self.M)
        self.vel = np.zeros((self.M, 2))
        self.prev_phi = self._potential(self.pos)

    def _potential(self, pos: np.ndarray) -> np.ndarray:
        """Smooth attraction/repulsion field: + near reward, - near punishment."""
        phi = np.zeros(pos.shape[0])
        if self.cfg.shaping_gain == 0:
            return phi
        for reg in self.regions:
            d = np.sqrt((pos[:, 0] - reg.cx) ** 2 + (pos[:, 1] - reg.cy) ** 2)
            phi += reg.sign * reg.magnitude * np.exp(-d / self.cfg.shaping_scale)
        return phi

    # --- dynamics --------------------------------------------------------
    def step(self, motor: np.ndarray) -> np.ndarray:
        """Advance all agents by motor command (M, 2) = (forward, turn).

        Heading is controlled directly (heading += turn); the agent then moves
        `forward` world units along its heading (SPEC §12.3). Returns reward (M,).
        """
        cfg = self.cfg
        motor = np.asarray(motor, dtype=np.float64).reshape(self.M, 2)
        forward = np.clip(motor[:, 0], 0.0, None)   # cannot move backward
        turn = motor[:, 1]

        # steer, then advance along the (new) heading
        self.heading = (self.heading + turn + np.pi) % (2 * np.pi) - np.pi
        self.vel = forward[:, None] * np.stack(
            [np.cos(self.heading), np.sin(self.heading)], axis=1)
        new_pos = self.pos + self.vel

        # walls: clamp into the box, flag agents that were clamped
        lo = cfg.agent_radius
        hi_x, hi_y = self.W - cfg.agent_radius, self.H - cfg.agent_radius
        clamped = (
            (new_pos[:, 0] < lo) | (new_pos[:, 0] > hi_x)
            | (new_pos[:, 1] < lo) | (new_pos[:, 1] > hi_y)
        )
        new_pos[:, 0] = np.clip(new_pos[:, 0], lo, hi_x)
        new_pos[:, 1] = np.clip(new_pos[:, 1], lo, hi_y)
        self.pos = new_pos

        # slow region drift so the task isn't a fixed lookup
        if cfg.region_drift > 0:
            for reg in self.regions:
                reg.cx = float(np.clip(reg.cx + self.rng.normal(0, cfg.region_drift),
                                       reg.r, self.W - reg.r))
                reg.cy = float(np.clip(reg.cy + self.rng.normal(0, cfg.region_drift),
                                       reg.r, self.H - reg.r))

        return self._reward(clamped)

    def _reward(self, clamped: np.ndarray) -> np.ndarray:
        R = np.full(self.M, self.cfg.R_step, dtype=np.float64)
        for reg in self.regions:
            d2 = (self.pos[:, 0] - reg.cx) ** 2 + (self.pos[:, 1] - reg.cy) ** 2
            inside = d2 <= reg.r ** 2
            R[inside] += reg.sign * reg.magnitude
        R[clamped] += self.cfg.R_wall
        # potential-based shaping: reward change in potential (approach reward,
        # retreat from punishment). Theory-preserving (Ng et al. 1999).
        if self.cfg.shaping_gain != 0:
            phi = self._potential(self.pos)
            R += self.cfg.shaping_gain * (phi - self.prev_phi)
            self.prev_phi = phi
        return R

    def occupancy(self):
        """Per-agent (in_reward, in_punish) booleans -- the behavioral metric base."""
        in_reward = np.zeros(self.M, dtype=bool)
        in_punish = np.zeros(self.M, dtype=bool)
        for reg in self.regions:
            d2 = (self.pos[:, 0] - reg.cx) ** 2 + (self.pos[:, 1] - reg.cy) ** 2
            inside = d2 <= reg.r ** 2
            if reg.sign > 0:
                in_reward |= inside
            else:
                in_punish |= inside
        return in_reward, in_punish

    # --- introspection ---------------------------------------------------
    def region_arrays(self):
        """(centers (K,2), radii (K,), signs (K,), mags (K,)) for vision/render."""
        if not self.regions:
            return (np.zeros((0, 2)), np.zeros(0), np.zeros(0, dtype=int), np.zeros(0))
        centers = np.array([[r.cx, r.cy] for r in self.regions])
        radii = np.array([r.r for r in self.regions])
        signs = np.array([r.sign for r in self.regions], dtype=int)
        mags = np.array([r.magnitude for r in self.regions])
        return centers, radii, signs, mags
