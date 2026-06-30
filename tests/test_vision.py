"""SPEC §12.2 acceptance: a ray aimed at a known region returns its type and a
distance matching geometry; empty space returns none; rotation changes hits."""
import numpy as np

from config import Config
from env.regions import RegionWorld, Region, TYPE_REWARD, TYPE_NONE
from env import vision


def _one_ray_world():
    cfg = Config()
    cfg.n_rays = 1            # single ray straight along heading
    cfg.fov = 0.0
    cfg.n_agents = 1
    cfg.max_range = 0.6
    world = RegionWorld(cfg, cfg.rng(0))
    world.regions = []        # clear random regions
    world.pos = np.array([[0.2, 0.5]])
    return cfg, world


def test_ray_hits_region_with_correct_type_and_distance():
    cfg, world = _one_ray_world()
    # reward region centered straight ahead (+x) at distance 0.3, radius 0.05
    world.regions = [Region(cx=0.5, cy=0.5, r=0.05, sign=+1, magnitude=1.0)]
    world.heading = np.array([0.0])   # facing +x
    prox, htype, _ = vision.cast(world, cfg)
    assert htype[0, 0] == TYPE_REWARD
    # near edge of region is at distance 0.3 - 0.05 = 0.25 -> proximity 1 - 0.25/0.6
    expected = 1.0 - 0.25 / cfg.max_range
    assert abs(prox[0, 0] - expected) < 0.02


def test_empty_direction_returns_none():
    cfg, world = _one_ray_world()
    cfg.max_range = 0.3
    world.regions = [Region(cx=0.5, cy=0.5, r=0.05, sign=+1, magnitude=1.0)]
    world.pos = np.array([[0.1, 0.5]])      # region is to the +x; wall +y is 0.5 away
    world.heading = np.array([np.pi / 2])   # face +y: no region, wall beyond max_range
    prox, htype, _ = vision.cast(world, cfg)
    assert htype[0, 0] == TYPE_NONE
    assert prox[0, 0] == 0.0


def test_rotation_changes_hit():
    cfg, world = _one_ray_world()
    world.regions = [Region(cx=0.5, cy=0.5, r=0.05, sign=+1, magnitude=1.0)]
    world.heading = np.array([0.0])           # toward region
    _, htype_toward, _ = vision.cast(world, cfg)
    world.heading = np.array([np.pi])         # away from region
    _, htype_away, _ = vision.cast(world, cfg)
    assert htype_toward[0, 0] == TYPE_REWARD
    assert htype_away[0, 0] != TYPE_REWARD
