"""Headless training / evaluation (SPEC §6, §14 Phase 11).

Drives the same Simulation the UI uses, with no rendering, and scores controllers
on a frame-independent behavioral metric: the fraction of agent-steps spent inside
reward vs punishment regions. Compares four controllers:

    reflex   hand-wired Braitenberg reflex (upper-bound sanity, no SNN)
    learn    reward-modulated SNN with plasticity on
    frozen   SNN with plasticity off (lower bound for the network)
    random   random-walk baseline

    python train.py                      # run all four and print a table
    python train.py --mode learn         # a single controller
    python train.py --steps 40000
"""
from __future__ import annotations

import argparse

import numpy as np

from config import Config
from ui.simulation import Simulation
from env.reflex import reflex_motor, BASE_FORWARD

MODES = ("reflex", "learn", "frozen", "random")


def evaluate(mode: str, steps: int, seed: int, log: bool = True) -> dict:
    cfg = Config()
    cfg.seed = seed
    sim = Simulation(cfg)
    sim.set_learning(mode == "learn")
    rng = cfg.rng(999)

    log_every = max(1, steps // 10)
    settle = int(steps * 0.6)   # measure steady-state over the final 40%
    for t in range(steps):
        if mode in ("learn", "frozen"):
            sim.step_once()
        elif mode == "reflex":
            motor = reflex_motor(sim.proximity, sim.hit_type, cfg)
            sim._apply_motor(motor)
        else:  # random
            motor = np.stack([
                rng.uniform(0, 2 * BASE_FORWARD, cfg.n_agents),
                rng.normal(0, 0.05, cfg.n_agents),
            ], axis=1)
            sim._apply_motor(motor)

        if t == settle:
            sim.reset_metrics()   # discard the early (pre-convergence) phase

        if log and (t + 1) % log_every == 0:
            fr, fp = sim.occupancy_fractions()
            print(f"  [{mode:6s}] step {t+1:>7d}   in_reward {fr:6.3f}   "
                  f"in_punish {fp:6.3f}   mean_reward(ema) {sim.mean_reward():+.4f}")

    fr, fp = sim.occupancy_fractions()   # steady-state window
    return {"mode": mode, "in_reward": fr, "in_punish": fp,
            "score": fr - fp, "mean_reward": sim.mean_reward()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=MODES, help="run a single controller")
    args = ap.parse_args()

    modes = [args.mode] if args.mode else list(MODES)
    results = []
    for m in modes:
        print(f"\n=== {m} ===")
        results.append(evaluate(m, args.steps, args.seed))

    print("\n" + "=" * 64)
    print(f"{'controller':<10}{'in_reward':>12}{'in_punish':>12}"
          f"{'score':>10}{'mean_R':>12}")
    print("-" * 64)
    for r in results:
        print(f"{r['mode']:<10}{r['in_reward']:>12.3f}{r['in_punish']:>12.3f}"
              f"{r['score']:>10.3f}{r['mean_reward']:>12.4f}")
    print("score = in_reward - in_punish  (higher is better; reflex ~ upper bound)")


if __name__ == "__main__":
    main()
