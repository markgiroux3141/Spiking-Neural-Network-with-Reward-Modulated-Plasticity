"""Egocentric raycasting vision (SPEC §12.2).

Each agent casts n_rays rays fanned across `fov`, centered on its heading. Each
ray reports the nearest hit within max_range as (proximity, hit_type), using
analytic ray-vs-circle (regions, other agents) and ray-vs-box (walls)
intersection -- no pixel marching.

Returns:
    proximity (M, n_rays) in [0, 1]   (1 - d/max_range; 0 if nothing in range)
    hit_type  (M, n_rays) int codes   (regions.TYPE_*)
    hit_point (M, n_rays, 2) world coords of the hit (for rendering)
"""
from __future__ import annotations

import numpy as np

from config import Config
from env.regions import (
    RegionWorld, TYPE_NONE, TYPE_REWARD, TYPE_PUNISH, TYPE_WALL, TYPE_AGENT,
)


def ray_angles(cfg: Config) -> np.ndarray:
    """Offsets of each ray relative to heading, spread across the FOV."""
    if cfg.n_rays == 1:
        return np.array([0.0])
    return np.linspace(-cfg.fov / 2.0, cfg.fov / 2.0, cfg.n_rays)


def _ray_circle_t(o, d, center, radius):
    """Nearest positive t where ray (o + t d) meets a circle, else +inf. Vectorized
    over rays (o, d shape (R, 2)); center (2,), radius scalar."""
    oc = o - center
    b = np.einsum("ij,ij->i", d, oc)        # d . oc  (d is unit)
    c = np.einsum("ij,ij->i", oc, oc) - radius ** 2
    disc = b * b - c
    t = np.full(o.shape[0], np.inf)
    # skip circles the ray origin is already inside (c < 0): an agent should not
    # "see" the region it is currently standing in on every ray.
    ok = (disc >= 0) & (c >= 0)
    if ok.any():
        sq = np.sqrt(disc[ok])
        t1 = -b[ok] - sq
        t2 = -b[ok] + sq
        # smallest strictly-positive root
        cand = np.where(t1 > 1e-9, t1, np.where(t2 > 1e-9, t2, np.inf))
        t[ok] = cand
    return t


def _ray_box_t(o, d, W, H):
    """Distance along each ray to the surrounding box wall (origin assumed inside)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = np.where(d[:, 0] > 0, (W - o[:, 0]) / d[:, 0],
                      np.where(d[:, 0] < 0, (0 - o[:, 0]) / d[:, 0], np.inf))
        ty = np.where(d[:, 1] > 0, (H - o[:, 1]) / d[:, 1],
                      np.where(d[:, 1] < 0, (0 - o[:, 1]) / d[:, 1], np.inf))
    tx = np.where(tx > 1e-9, tx, np.inf)
    ty = np.where(ty > 1e-9, ty, np.inf)
    return np.minimum(tx, ty)


def cast(world: RegionWorld, cfg: Config):
    M, n_rays = world.M, cfg.n_rays
    offsets = ray_angles(cfg)                                  # (n_rays,)
    abs_ang = world.heading[:, None] + offsets[None, :]        # (M, n_rays)
    dirs = np.stack([np.cos(abs_ang), np.sin(abs_ang)], axis=-1)  # (M, n_rays, 2)
    origins = np.broadcast_to(world.pos[:, None, :], (M, n_rays, 2))

    R = M * n_rays
    o = origins.reshape(R, 2)
    d = dirs.reshape(R, 2)

    best_t = np.full(R, np.inf)
    best_type = np.full(R, TYPE_NONE, dtype=int)

    # regions (circles)
    centers, radii, signs, _ = world.region_arrays()
    for k in range(len(radii)):
        t = _ray_circle_t(o, d, centers[k], radii[k])
        hit = t < best_t
        best_t[hit] = t[hit]
        best_type[hit] = TYPE_REWARD if signs[k] > 0 else TYPE_PUNISH

    # other agents (circles) -- exclude self
    ar = cfg.agent_radius
    ray_agent = np.repeat(np.arange(M), n_rays)  # which agent each ray belongs to
    for j in range(M):
        t = _ray_circle_t(o, d, world.pos[j], ar)
        t[ray_agent == j] = np.inf               # don't see yourself
        hit = t < best_t
        best_t[hit] = t[hit]
        best_type[hit] = TYPE_AGENT

    # walls (always present; only matters if nearest)
    t_wall = _ray_box_t(o, d, cfg.world_w, cfg.world_h)
    hit = t_wall < best_t
    best_t[hit] = t_wall[hit]
    best_type[hit] = TYPE_WALL

    # out of range -> nothing
    out = best_t > cfg.max_range
    best_t[out] = cfg.max_range
    best_type[out] = TYPE_NONE

    proximity = (1.0 - best_t / cfg.max_range).reshape(M, n_rays)
    hit_type = best_type.reshape(M, n_rays)
    hit_point = (o + best_t[:, None] * d).reshape(M, n_rays, 2)
    return proximity, hit_type, hit_point
