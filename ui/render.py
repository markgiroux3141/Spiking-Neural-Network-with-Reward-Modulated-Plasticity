"""World-view rendering: regions, agents, rays, trails (SPEC §13.1).

Pure drawing against a read-only Simulation.state -- no simulation logic here.
"""
from __future__ import annotations

import math

import numpy as np
import pygame

from env.regions import TYPE_REWARD, TYPE_PUNISH, TYPE_WALL, TYPE_AGENT, TYPE_NONE

BG = (18, 20, 28)
WALL = (70, 76, 90)
REWARD = (70, 200, 110)
PUNISH = (215, 70, 70)
AGENT = (120, 170, 240)
AGENT_SEL = (255, 220, 90)
TRAIL = (90, 100, 130)

RAY_COLOR = {
    TYPE_REWARD: (70, 200, 110),
    TYPE_PUNISH: (215, 70, 70),
    TYPE_WALL: (90, 96, 110),
    TYPE_AGENT: (120, 170, 240),
    TYPE_NONE: (45, 48, 60),
}


class WorldView:
    def __init__(self, rect: pygame.Rect, cfg):
        self.rect = rect
        self.cfg = cfg
        self.scale = min(rect.width, rect.height) / max(cfg.world_w, cfg.world_h)

    def w2s(self, p):
        """World coords -> screen pixel (y flipped so +y is up)."""
        x = self.rect.x + p[0] * self.scale
        y = self.rect.y + self.rect.height - p[1] * self.scale
        return int(x), int(y)

    def agent_at(self, sim, mouse):
        """Return the index of the agent under `mouse`, or None."""
        best, best_d = None, 18
        for a in sim.agents:
            sx, sy = self.w2s(sim.world.pos[a.idx])
            d = math.hypot(sx - mouse[0], sy - mouse[1])
            if d < best_d:
                best, best_d = a.idx, d
        return best

    def draw(self, surface, sim, selected, show_rays=True, show_trails=True):
        cfg = self.cfg
        pygame.draw.rect(surface, (10, 11, 16), self.rect)
        pygame.draw.rect(surface, WALL, self.rect, 2)

        # regions as translucent fills
        overlay = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        for reg in sim.world.regions:
            color = REWARD if reg.sign > 0 else PUNISH
            alpha = int(60 + 120 * min(1.0, reg.magnitude))
            cx = reg.cx * self.scale
            cy = self.rect.height - reg.cy * self.scale
            pygame.draw.circle(overlay, (*color, alpha), (int(cx), int(cy)),
                               int(reg.r * self.scale))
            pygame.draw.circle(overlay, (*color, 200), (int(cx), int(cy)),
                               int(reg.r * self.scale), 2)
        surface.blit(overlay, self.rect.topleft)

        # trails
        if show_trails:
            for a in sim.agents:
                if len(a.trail) > 1:
                    pts = [self.w2s(p) for p in a.trail]
                    pygame.draw.lines(surface, TRAIL, False, pts, 1)

        # rays from the selected agent
        if show_rays and selected is not None:
            origin = self.w2s(sim.world.pos[selected])
            for ray in range(cfg.n_rays):
                hp = sim.hit_point[selected, ray]
                t = int(sim.hit_type[selected, ray])
                pygame.draw.line(surface, RAY_COLOR[t], origin, self.w2s(hp), 1)

        # agents as oriented triangles, tinted by recent reward
        for a in sim.agents:
            self._draw_agent(surface, sim, a, selected)

    def _draw_agent(self, surface, sim, a, selected):
        pos = sim.world.pos[a.idx]
        heading = sim.world.heading[a.idx]
        size = max(6, self.cfg.agent_radius * self.scale * 1.6)
        cx, cy = self.w2s(pos)
        # triangle points: nose + two tails (screen y is flipped, so negate angle)
        ang = -heading
        nose = (cx + size * math.cos(ang), cy + size * math.sin(ang))
        left = (cx + size * math.cos(ang + 2.5), cy + size * math.sin(ang + 2.5))
        right = (cx + size * math.cos(ang - 2.5), cy + size * math.sin(ang - 2.5))

        if a.idx == selected:
            color = AGENT_SEL
        else:
            # green when doing well, red when doing badly
            r = np.clip(a.reward_ema * 6, -1, 1)
            color = (int(150 - 90 * r), int(150 + 90 * r), 150)
        pygame.draw.polygon(surface, color, [nose, left, right])
        if a.idx == selected:
            pygame.draw.polygon(surface, (255, 255, 255), [nose, left, right], 2)
