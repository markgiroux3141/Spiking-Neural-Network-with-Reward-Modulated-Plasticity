"""Pygame live viewer (SPEC §13).

Controls:
    Space  play / pause          S  single step          R  reset (re-seed)
    Up/Dn  sim steps per frame    L  freeze/unfreeze learning
    T      toggle trails          Y  toggle rays          Tab  cycle agent
    Click  select agent           Esc / window close  quit

Run:  python -m ui.app
"""
from __future__ import annotations

import sys

import numpy as np
import pygame

from config import Config
from ui.simulation import Simulation
from ui.render import WorldView
from ui import panels

W, H = 1280, 720
HUD = (210, 214, 224)


def main():
    cfg = Config()
    sim = Simulation(cfg)

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Reward-Modulated SNN — region world")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 13)
    big = pygame.font.SysFont("consolas", 18, bold=True)

    view = WorldView(pygame.Rect(10, 10, 700, 700), cfg)
    inspector = pygame.Rect(718, 10, W - 728, 700)

    selected = sim.best_agent()
    running = True
    playing = True
    steps_per_frame = 4
    show_trails = True
    show_rays = True

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE,):
                    running = False
                elif ev.key == pygame.K_SPACE:
                    playing = not playing
                elif ev.key == pygame.K_s:
                    sim.step_once()
                elif ev.key == pygame.K_r:
                    sim.reset(reseed=cfg.seed + 1)
                    selected = sim.best_agent()
                elif ev.key == pygame.K_l:
                    sim.set_learning(not sim.learning_enabled)
                elif ev.key == pygame.K_t:
                    show_trails = not show_trails
                elif ev.key == pygame.K_y:
                    show_rays = not show_rays
                elif ev.key == pygame.K_TAB:
                    selected = (selected + 1) % cfg.n_agents
                elif ev.key == pygame.K_UP:
                    steps_per_frame = min(64, steps_per_frame + 1)
                elif ev.key == pygame.K_DOWN:
                    steps_per_frame = max(1, steps_per_frame - 1)
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                hit = view.agent_at(sim, ev.pos)
                if hit is not None:
                    selected = hit

        if playing:
            sim.advance(steps_per_frame)

        screen.fill((18, 20, 28))
        view.draw(screen, sim, selected, show_rays=show_rays, show_trails=show_trails)
        panels.draw_inspector(screen, font, big, sim, selected, inspector)

        _hud(screen, font, sim, playing, steps_per_frame, selected, clock)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


def _hud(screen, font, sim, playing, spf, selected, clock):
    pop_h = np.mean([a.net.homeo_h.rate_est.mean() for a in sim.agents])
    pop_o = np.mean([a.net.homeo_o.rate_est.mean() for a in sim.agents])
    lines = [
        f"step {sim.step_count:>7d}   {'PLAY' if playing else 'PAUSE'}   "
        f"{spf} steps/frame   fps {clock.get_fps():.0f}",
        f"mean reward(ema) {sim.mean_reward():+.4f}   best {sim.best_agent()}   "
        f"sel {selected}   learning {'ON' if sim.learning_enabled else 'OFF'}",
        f"pop rate  hidden {pop_h:5.1f} Hz   output {pop_o:5.1f} Hz",
        "space:play  S:step  R:reset  L:learn  T:trails  Y:rays  Tab:cycle  click:select",
    ]
    backdrop = pygame.Surface((700, 8 + 16 * len(lines)), pygame.SRCALPHA)
    backdrop.fill((0, 0, 0, 150))
    screen.blit(backdrop, (10, 10))
    for i, ln in enumerate(lines):
        screen.blit(font.render(ln, True, HUD), (16, 14 + i * 16))


if __name__ == "__main__":
    main()
