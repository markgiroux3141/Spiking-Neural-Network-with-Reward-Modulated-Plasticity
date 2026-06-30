"""Selected-agent network inspector (SPEC §13.2).

Renders, for the selected agent: input sensor bars, a rolling spike raster
(input / hidden / output lanes), motor-pool activity + decoded velocity arrow,
the input->hidden weight heatmap, and the reward / (R - baseline) traces.
"""
from __future__ import annotations

import numpy as np
import pygame

PANEL_BG = (24, 26, 34)
TEXT = (210, 214, 224)
MUTED = (130, 136, 150)
SPIKE = (120, 200, 255)
HID = (180, 160, 255)
OUT = (255, 200, 110)
REWARD_C = (90, 220, 130)
DELTA_C = (255, 150, 90)


def _label(surface, font, text, x, y, color=TEXT):
    surface.blit(font.render(text, True, color), (x, y))


def _raster(surface, rect, history, color):
    """history: deque of bool vectors (newest last). Draws time->x, neuron->y."""
    pygame.draw.rect(surface, (14, 15, 20), rect)
    if not history:
        return
    n = len(history[0])
    cols = len(history)
    if n == 0 or cols == 0:
        return
    cw = rect.width / cols
    rh = rect.height / n
    for c, vec in enumerate(history):
        idx = np.nonzero(vec)[0]
        x = rect.x + int(c * cw)
        for i in idx:
            y = rect.y + int(i * rh)
            pygame.draw.rect(surface, color, (x, y, max(1, int(cw)), max(1, int(rh))))


def _heatmap(surface, rect, W):
    pygame.draw.rect(surface, (14, 15, 20), rect)
    n_post, n_pre = W.shape
    wmax = max(1e-6, W.max())
    cw = rect.width / n_pre
    rh = rect.height / n_post
    for i in range(n_post):
        for j in range(n_pre):
            v = W[i, j] / wmax
            if v <= 0.01:
                continue
            c = (int(40 + 120 * v), int(40 + 180 * v), int(60 + 80 * v))
            pygame.draw.rect(surface, c,
                             (rect.x + int(j * cw), rect.y + int(i * rh),
                              max(1, int(cw)), max(1, int(rh))))


def _trace(surface, rect, series, color, zero_center=True):
    pygame.draw.rect(surface, (14, 15, 20), rect)
    if len(series) < 2:
        return
    arr = np.array(series, dtype=float)
    lo, hi = arr.min(), arr.max()
    if zero_center:
        m = max(abs(lo), abs(hi), 1e-6)
        lo, hi = -m, m
        pygame.draw.line(surface, (60, 64, 76),
                         (rect.x, rect.centery), (rect.right, rect.centery), 1)
    span = max(hi - lo, 1e-6)
    pts = []
    for i, v in enumerate(arr):
        x = rect.x + int(i / (len(arr) - 1) * rect.width)
        y = rect.bottom - int((v - lo) / span * rect.height)
        pts.append((x, y))
    pygame.draw.lines(surface, color, False, pts, 1)


def _bars(surface, rect, values, color):
    pygame.draw.rect(surface, (14, 15, 20), rect)
    if len(values) == 0:
        return
    bw = rect.width / len(values)
    vmax = max(1e-6, float(np.max(values)))
    for i, v in enumerate(values):
        h = int(v / vmax * rect.height)
        pygame.draw.rect(surface, color,
                         (rect.x + int(i * bw), rect.bottom - h,
                          max(1, int(bw) - 1), h))


def draw_inspector(surface, font, big, sim, idx, rect):
    pygame.draw.rect(surface, PANEL_BG, rect)
    x = rect.x + 12
    y = rect.y + 10
    if idx is None:
        _label(surface, big, "Click an agent to inspect", x, y, MUTED)
        return
    a = sim.agents[idx]
    net = a.net

    _label(surface, big, f"Agent {idx}", x, y)
    _label(surface, font, f"reward(ema) {a.reward_ema:+.4f}    "
                          f"baseline {net.baseline:+.4f}", x + 110, y + 4)
    y += 30

    w = rect.width - 24

    _label(surface, font, "input sensor rates (n_rays x channels)", x, y, MUTED)
    y += 16
    _bars(surface, pygame.Rect(x, y, w, 34), a.encoder.last_rates, SPIKE)
    y += 44

    _label(surface, font, "spike raster — input", x, y, MUTED)
    y += 16
    _raster(surface, pygame.Rect(x, y, w, 40), list(a.raster_in), SPIKE)
    y += 46
    _label(surface, font, "hidden", x, y, MUTED)
    y += 16
    _raster(surface, pygame.Rect(x, y, w, 70), list(a.raster_hidden), HID)
    y += 76
    _label(surface, font, "output  (fwd  turn-L  turn-R)", x, y, MUTED)
    y += 16
    _raster(surface, pygame.Rect(x, y, w, 30), list(a.raster_out), OUT)
    y += 38

    # motor velocity arrow
    _label(surface, font, "decoded velocity", x, y, MUTED)
    cxv = x + w - 40
    cyv = y + 22
    pygame.draw.circle(surface, (60, 64, 76), (cxv, cyv), 22, 1)
    v = a.velocity
    vn = v / (np.linalg.norm(v) + 1e-9) * 20
    pygame.draw.line(surface, OUT, (cxv, cyv), (cxv + vn[0], cyv - vn[1]), 2)
    y += 48

    _label(surface, font, "input->hidden weights", x, y, MUTED)
    y += 16
    _heatmap(surface, pygame.Rect(x, y, w, 80), net.syn_ih.W)
    y += 88

    _label(surface, font, "reward (green) and R - baseline (orange)", x, y, MUTED)
    y += 16
    tr = pygame.Rect(x, y, w, 50)
    _trace(surface, tr, list(a.reward_hist), REWARD_C)
    _trace_overlay(surface, tr, list(a.delta_hist), DELTA_C)


def _trace_overlay(surface, rect, series, color):
    """Draw a second trace sharing the same box without clearing it."""
    if len(series) < 2:
        return
    arr = np.array(series, dtype=float)
    m = max(np.abs(arr).max(), 1e-6)
    lo, hi = -m, m
    span = hi - lo
    pts = []
    for i, v in enumerate(arr):
        px = rect.x + int(i / (len(arr) - 1) * rect.width)
        py = rect.bottom - int((v - lo) / span * rect.height)
        pts.append((px, py))
    pygame.draw.lines(surface, color, False, pts, 1)
