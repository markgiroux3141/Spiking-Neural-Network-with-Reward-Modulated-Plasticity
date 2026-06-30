"""Behavioral evaluation: is the SNN actually doing the right things? (SPEC §14)

Beyond raw reward, this scores the *behaviors* we care about, at steady state,
across controllers (learn / frozen / random / reflex):

  go toward green   : alignment of motion with direction to a VISIBLE reward
                      region (cosine; +1 = straight toward, 0 = random, -1 = away)
  stay in green     : median dwell (consecutive steps) per reward-region visit
  avoid red         : alignment of motion AWAY from a visible punishment region
  exit red fast     : median dwell per punishment-region visit (small is good,
                      and should be << reward dwell)
  occupancy         : fraction of steady-state agent-steps in reward vs punishment

Metrics are recorded only after a settle period so they reflect the learned
policy, not the initial random phase.

    python evaluate.py                 # all controllers
    python evaluate.py --mode learn
    python evaluate.py --steps 40000
"""
from __future__ import annotations

import argparse
import numpy as np

from config import Config
from ui.simulation import Simulation
from env.reflex import reflex_motor, BASE_FORWARD
from env.regions import TYPE_REWARD, TYPE_PUNISH

MODES = ("reflex", "learn", "frozen", "random")


def _alignment(vel, pos, target):
    """cos angle between velocity and the vector pos->target (toward target)."""
    v = vel
    sp = np.hypot(*v)
    d = target - pos
    dn = np.hypot(*d)
    if sp < 1e-9 or dn < 1e-9:
        return None
    return float((v[0] * d[0] + v[1] * d[1]) / (sp * dn))


def _bearing(pos, heading, target):
    """Signed angle (rad) of `target` relative to `heading`, in [-pi, pi].
    |bearing| small = dead ahead; large = to the side / behind."""
    d = target - pos
    ang = np.arctan2(d[1], d[0]) - heading
    return (ang + np.pi) % (2 * np.pi) - np.pi


def evaluate(mode: str, steps: int, seed: int) -> dict:
    cfg = Config()
    cfg.seed = seed
    sim = Simulation(cfg)
    sim.set_learning(mode == "learn")
    rng = cfg.rng(999)
    settle = int(steps * 0.6)

    approach, avoid = [], []                     # alignment samples
    r_dwell, p_dwell = [], []                    # completed visit lengths
    r_run = np.zeros(cfg.n_agents, dtype=int)
    p_run = np.zeros(cfg.n_agents, dtype=int)

    for t in range(steps):
        # --- pre-step state (the policy acts on THIS vision) ---
        pos0 = sim.world.pos.copy()
        h0 = sim.world.heading.copy()
        htype0 = sim.hit_type.copy()
        hp0 = sim.hit_point.copy()          # where each ray actually hit
        centers, radii, signs, _ = sim.world.region_arrays()
        rr = radii[signs > 0] if len(signs) else np.zeros(0)
        pr = radii[signs < 0] if len(signs) else np.zeros(0)
        rc = centers[signs > 0] if len(signs) else np.zeros((0, 2))
        pc = centers[signs < 0] if len(signs) else np.zeros((0, 2))

        # --- choose + apply motor for this controller ---
        if mode in ("learn", "frozen"):
            sim.step_once()
        elif mode == "reflex":
            sim._apply_motor(reflex_motor(sim.proximity, sim.hit_type, cfg))
        else:  # random
            motor = np.stack([rng.uniform(0, 2 * BASE_FORWARD, cfg.n_agents),
                              rng.normal(0, 0.05, cfg.n_agents)], axis=1)
            sim._apply_motor(motor)

        if t == settle:
            sim.reset_metrics()
            approach, avoid, r_dwell, p_dwell = [], [], [], []

        # --- record behavioral metrics ---
        vel = sim.world.vel
        pos1 = sim.world.pos          # post-step pose
        h1 = sim.world.heading
        in_r, in_p = sim.world.occupancy()
        for a in range(cfg.n_agents):
            # inside-region status at the START of the step (from pre-move state).
            # Approach/avoid are only meaningful while TRAVELING (outside the
            # region) -- an agent parked inside green stops seeing it and would
            # otherwise look like it's fleeing distant greens.
            inside_r = bool(len(rc) and (((rc - pos0[a]) ** 2).sum(1) <= rr ** 2).any())
            inside_p = bool(len(pc) and (((pc - pos0[a]) ** 2).sum(1) <= pr ** 2).any())

            # target = where the agent's rays ACTUALLY saw green / red (its percept),
            # not the geometrically-nearest center (which may be behind it / out of FOV)
            green_rays = htype0[a] == TYPE_REWARD
            red_rays = htype0[a] == TYPE_PUNISH

            # approach: outside green, sees green -> velocity points toward it
            # (+1 = straight at the green it sees; works because moving forward
            #  toward a seen target is the natural approach)
            if not inside_r and green_rays.any():
                tgt = hp0[a, green_rays].mean(axis=0)
                c = _alignment(vel[a], pos0[a], tgt)
                if c is not None:
                    approach.append(c)
            # avoid: outside red, sees red -> TURN AWAY from the side it's on.
            # Velocity can't point away from something dead-ahead while moving
            # forward, so we score the signed turn relative to red's bearing:
            # red on the left (b>0) -> turning right (dpsi<0) is avoidance.
            # This gives random turns an expected score of 0 (unlike |bearing|
            # growth, which any motion inflates), so it isolates real avoidance.
            if not inside_p and red_rays.any():
                tgt = hp0[a, red_rays].mean(axis=0)
                b = _bearing(pos0[a], h0[a], tgt)
                dpsi = (h1[a] - h0[a] + np.pi) % (2 * np.pi) - np.pi
                avoid.append(np.degrees(-np.sign(b) * dpsi))
            # dwell runs (consecutive steps inside a region per visit)
            if in_r[a]:
                r_run[a] += 1
            elif r_run[a] > 0:
                r_dwell.append(r_run[a]); r_run[a] = 0
            if in_p[a]:
                p_run[a] += 1
            elif p_run[a] > 0:
                p_dwell.append(p_run[a]); p_run[a] = 0

    fr, fp = sim.occupancy_fractions()
    med = lambda x: float(np.median(x)) if len(x) else 0.0
    mean = lambda x: float(np.mean(x)) if len(x) else 0.0
    return {
        "mode": mode,
        "in_reward": fr, "in_punish": fp,
        "approach": mean(approach), "n_app": len(approach),
        "avoid": mean(avoid), "n_avo": len(avoid),
        "r_dwell": med(r_dwell), "p_dwell": med(p_dwell),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=MODES)
    args = ap.parse_args()

    modes = [args.mode] if args.mode else list(MODES)
    rows = [evaluate(m, args.steps, args.seed) for m in modes]

    print("\n" + "=" * 80)
    print(f"{'controller':<9}{'in_rew':>8}{'in_pun':>8}"
          f"{'->green':>9}{'turn_off_red':>14}{'green_dwell':>12}{'red_dwell':>11}")
    print("-" * 80)
    for r in rows:
        print(f"{r['mode']:<9}{r['in_reward']:>8.3f}{r['in_punish']:>8.3f}"
              f"{r['approach']:>9.2f}{r['avoid']:>14.2f}"
              f"{r['r_dwell']:>12.0f}{r['p_dwell']:>11.0f}")
    print("-" * 80)
    print("->green      : velocity alignment toward a VISIBLE reward "
          "(+1 = straight at it, 0 = random)")
    print("turn_off_red : deg/step the agent turns AWAY from the side a VISIBLE "
          "punishment is on (+ = avoiding, 0 = random)")
    print("dwell        : median consecutive steps per visit "
          "(green: high = stays; red: low = exits fast)")


if __name__ == "__main__":
    main()
