# Spiking Neural Network with Reward-Modulated Plasticity — Build Spec

## 0. What we're building

A spiking neural network (SNN) that learns to drive an embodied agent toward
food in a 2D world, using **only local plasticity plus a global reward signal** —
no backpropagation. Neurons with little input fire spontaneously ("boredom"),
coincident firing forms Hebbian links, and a delayed reward/punishment signal
selectively strengthens or weakens whichever connections were recently active.
Over time, structured input→output pathways replace the random firing.

This is a known paradigm: **three-factor learning / reward-modulated STDP**, with
an **eligibility trace** as the credit-assignment mechanism. Reference papers
(read if stuck, do not need to reproduce): Izhikevich 2007 "Solving the distal
reward problem"; Frémaux & Gerstner 2016 "Neuromodulated STDP and theory of
three-factor learning rules".

**Success looks like:** an agent that, after training, reaches food faster than a
random walker, and whose synaptic weights show stable structure rather than
saturating to the bounds or collapsing to silence.

---

## 1. Design decisions (assumptions — change here if you disagree)

- **Language / core:** Python, pure **NumPy** for the network core. No SNN
  framework for the reference implementation — the eligibility trace and the
  three-factor update are the whole point, so they must be explicit and
  inspectable. (Brian2 / BindsNET / snnTorch are valid later ports; do not start
  there.)
- **Time:** discrete time steps of `dt = 1.0 ms`. The whole sim is integer-stepped.
- **Neuron model:** leaky integrate-and-fire (LIF).
- **Environment:** continuous 2D plane, single agent, food pellets that respawn.
  Reward on eating, small punishment on wall/timeout. Kept deliberately simple so
  learning, not the environment, is what's being tested.
- **Determinism:** every run takes a seed; the same seed reproduces the same run.

---

## 2. Glossary (intuition → mechanism → variable)

| Intuition | Mechanism | Code variable |
|---|---|---|
| neuron fires when poked enough | LIF threshold crossing | `V`, `V_thresh` |
| neuron "gets bored" with no input | homeostatic intrinsic noise/excitability | `boredom`, `noise_rate` |
| neurons that fire together wire together | STDP (timing-sensitive Hebb) | `dw_stdp` |
| a connection is "modifiable" then forgets | eligibility trace, decays exponentially | `e` (per-synapse) |
| reward/punishment | global third factor, prediction error | `R`, `baseline` |
| only recently-active synapses change | `dw = lr * (R - baseline) * e` | weight update |
| stop exploring once well-driven | homeostasis suppresses boredom | `target_rate` |

---

## 3. Repository layout

```
snn-agent/
  README.md
  requirements.txt          # numpy, matplotlib; (optional) gymnasium-free, self-contained
  config.py                 # all hyperparameters in one dataclass, seed-controlled
  snn/
    neuron.py               # LIFLayer
    synapses.py             # weight matrix, STDP, eligibility traces
    network.py              # wires layers, runs one timestep, applies reward
    homeostasis.py          # boredom / intrinsic plasticity / spontaneous firing
    stability.py            # weight bounds, normalization, competition
  env/
    forage.py               # 2D foraging world + reward signal (original single-agent)
    regions.py              # NEW: multi-agent reward/punishment region world (§12)
    vision.py               # NEW: egocentric raycasting sensors (§12.2)
    encoding.py             # sensors/vision -> input spike rates; output spikes -> x/y motor
  train.py                  # main loop: env <-> network, logging (headless)
  ui/
    app.py                  # NEW: Pygame live viewer + controls (§13)
    render.py               # NEW: world / agents / rays / trails rendering
    panels.py               # NEW: selected-agent network introspection panels
    simulation.py           # NEW: headless-capable sim driver shared by train.py and the UI
  viz/
    monitors.py             # spike raster, weight matrix, reward curve, trace decay
  tests/
    test_neuron.py
    test_eligibility.py
    test_three_factor.py
    test_stability.py
```

---

## 4. Component specs

### 4.1 LIF neuron (`snn/neuron.py`)
A vectorized layer of `N` neurons, all updated per timestep.

```
V[t+1] = V[t] + dt/tau_m * (-(V[t] - V_rest) + R_m * I[t])   # leak + input
spike  = V[t+1] >= V_thresh
V[spike] = V_reset                                            # reset after spike
# refractory: clamp V at V_reset for `refractory_ms` after a spike
```

- Inputs `I[t]` come from (a) synaptic current from presynaptic spikes, plus
  (b) boredom noise current (§4.4).
- Expose `spikes` (boolean vector) each step; that is the layer's output.

**Acceptance:** with constant suprathreshold `I`, the layer fires at a stable
rate that increases monotonically with `I`. With `I=0` and noise off, it stays
silent.

### 4.2 Synapses, STDP, eligibility trace (`snn/synapses.py`)
Dense weight matrix `W[post, pre]`, plus a same-shape eligibility matrix `e`.

Per timestep:
1. **Synaptic current:** `I_syn = W @ pre_spikes` (delivered to postsynaptic layer).
2. **Pre/post traces** for STDP: maintain low-pass traces `x_pre`, `x_post`
   that bump by 1 on a spike and decay with `tau_pre`, `tau_post`.
3. **STDP onto eligibility (not directly onto W):**
   - on a postsynaptic spike: `e += A_plus  * x_pre`   (pre-before-post → potentiate)
   - on a presynaptic spike:  `e -= A_minus * x_post`  (post-before-pre → depress)
4. **Eligibility decay:** `e *= exp(-dt / tau_e)` every step.

> Critical: STDP writes to `e`, NOT to `W`. `W` only changes when reward arrives
> (§4.3). This is what makes credit assignment work across the reward delay.

**Acceptance:** a single pre-then-post spike pair raises the corresponding `e`
entry; that entry then decays to ~37% after `tau_e` ms with no further activity.
A post-then-pre pair lowers it.

### 4.3 Three-factor weight update (`snn/network.py`)
When a reward signal `R` is delivered (can be every step; usually sparse):

```
dW = learning_rate * (R - baseline) * e
W  = clip(W + dW, w_min, w_max)
```

- **`baseline` is mandatory, not optional.** It is a running estimate of expected
  reward (e.g. exponential moving average of recent `R`). Without it the network
  strengthens *everything* recently active and learns "do more of all of it"
  instead of "do more of the good thing." Learning is driven by reward
  *prediction error* `(R - baseline)`, not raw reward.
- Punishment is just negative `R`; it weakens whatever was eligible.
- Apply the update to all synapses at once; `e` ensures only recently-active ones
  move meaningfully.

**Acceptance:** in an isolated 2-neuron test, drive pre→post into a positive `e`,
then deliver `R > baseline`: the weight grows. Deliver `R < baseline`: it shrinks.
With `e≈0` (stale), the same reward barely moves the weight.

### 4.4 Homeostasis / "boredom" (`snn/homeostasis.py`)
Each neuron tracks its own recent firing rate (low-pass of its spikes). The
further its rate is **below** `target_rate`, the more boredom noise current it
receives:

```
rate_est = lowpass(spikes, tau_rate)
boredom  = clip(target_rate - rate_est, 0, inf)
I_noise  = boredom * noise_gain * poisson_or_gaussian_kick()
```

- A neuron starved of real input drifts up in excitability until it fires
  spontaneously → seeds STDP → seeds new connections.
- Once real input drives it near `target_rate`, boredom → 0 and it stops firing
  randomly. **This is the built-in exploration→exploitation transition** you
  wanted: undriven regions keep exploring; wired-up regions go quiet and stable.

**Acceptance:** a disconnected neuron with no input eventually fires at roughly
`target_rate`. After it receives strong real drive, its boredom term falls to
near zero.

### 4.5 Stability (`snn/stability.py`)
Random firing + Hebbian growth will wire spurious connections to noise unless
contained. Implement and keep all of:

- **Hard bounds:** `w_min`, `w_max` clip (already in §4.3).
- **Synaptic normalization:** periodically rescale each postsynaptic neuron's
  incoming weights so their sum (or L2 norm) holds ~constant → creates competition
  between inputs.
- **Optional weight decay:** tiny constant pull of unused weights toward 0 to prune.

**Acceptance:** run 100k steps of pure noise (no meaningful reward). Mean `|W|`
must stay bounded and the network must not fall fully silent or fully saturated.

---

## 5. Environment & agent I/O

### 5.1 Foraging world (`env/forage.py`)
- Continuous 2D box (e.g. 1.0 × 1.0). Agent has position + heading.
- `K` food pellets at random positions; eating one (within `eat_radius`) gives
  `R_food = +1` and respawns the pellet elsewhere.
- Small per-step `R_step = -0.001` to encourage efficiency; `R_wall = -0.2` on
  hitting a wall. These feed §4.3 as the reward signal.
- `step(motor_command) -> (sensory_obs, reward, done)`.

### 5.2 Encoding (`env/encoding.py`)
- **Sensors → input spikes:** a handful of directional "food sensors" (e.g. 8
  rays or a simple egocentric bearing+distance to nearest pellet) converted to
  Poisson spike rates feeding input neurons. Closer/aligned food → higher rate.
- **Output spikes → motor:** two or more motor neuron pools (e.g. turn-left,
  turn-right, forward). Decode by spike count over a short window into a
  continuous motor command (population/rate coding).

**Acceptance:** with hand-set weights wiring "food-ahead sensor → forward motor,"
the agent moves toward a static pellet. (Sanity check that encoding works before
trusting learning.)

---

## 6. Main loop (`train.py`)

```
seed everything from config
build network (input layer, hidden layer(s), output layer; sparse random W)
for episode in episodes:
    reset env
    for t in steps:
        obs            = env.observe()
        in_spikes      = encode_sensors(obs)            # §5.2
        spikes         = network.step(in_spikes)        # §4.1, 4.2, 4.4
        motor          = decode_motor(spikes)           # §5.2
        obs, R, done   = env.step(motor)                # §5.1
        network.apply_reward(R)                         # §4.3  (baseline update inside)
        network.normalize_if_due()                      # §4.5
        log(...)
        if done: break
```

The network's internal order per `step`: deliver synaptic current → integrate LIF
→ detect spikes → update STDP traces → write to eligibility `e` → decay `e` →
update homeostasis. Reward is applied separately right after, using the current `e`.

---

## 7. Hyperparameters — starting values (`config.py`)

These are *plausible starting points*, expected to need tuning. Put every one in a
single seed-controlled dataclass.

| Param | Start | Note |
|---|---|---|
| `dt` | 1.0 ms | timestep |
| `tau_m` | 20 ms | membrane leak |
| `V_thresh / V_reset / V_rest` | -50 / -65 / -65 mV | standard LIF |
| `refractory_ms` | 5 ms | |
| `tau_pre / tau_post` | 20 / 20 ms | STDP traces |
| `A_plus / A_minus` | 0.01 / 0.012 | depression slightly > potentiation aids stability |
| `tau_e` | **~200–1000 ms** | **must ≈ typical action→reward delay; tune first** |
| `learning_rate` | 0.01 | three-factor `dW` scale |
| `baseline tau` | ~10 s | EMA window for expected reward |
| `target_rate` | 5 Hz | homeostatic setpoint |
| `noise_gain` | tune | enough to make a bored neuron reach target_rate |
| `w_min / w_max` | 0 / 1 | (use signed bounds if you allow inhibition) |
| `normalize_every` | 100 steps | competition cadence |
| network size | 8 in / 64 hidden / 3 out | scale up once it learns |

**Tuning order when it won't learn:** (1) confirm encoding works with hand-set
weights; (2) set `tau_e` to match reward delay; (3) confirm `baseline` is actually
subtracting (log `R - baseline`); (4) check weights aren't pinned at bounds
(stability); (5) only then touch STDP/learning_rate.

---

## 8. Build phases (do in order; each must pass its tests before the next)

**Phase 1 — Neuron.** Implement LIF layer + tests (§4.1). Visualize a spike raster
under constant and zero input.

**Phase 2 — Synapses + eligibility.** Weight matrix, STDP traces, eligibility
trace writing/decay + tests (§4.2). Visualize a single `e` entry rising on a
pre→post pair and decaying over `tau_e`.

**Phase 3 — Three-factor update.** Reward application with baseline + tests
(§4.3). Demonstrate a single synapse strengthening under `R>baseline` and
weakening under `R<baseline`, and *not* moving when `e` is stale.

**Phase 4 — Homeostasis.** Boredom firing + tests (§4.4). Show a disconnected
neuron self-firing to `target_rate`, then going quiet under real drive.

**Phase 5 — Stability.** Bounds + normalization + the 100k-step noise test (§4.5).

**Phase 6 — Environment + encoding.** Foraging world, sensors, motor decode, and
the hand-wired sanity agent (§5).

**Phase 7 — Integration.** Full loop (§6). Train. Compare learned agent vs random
walker on mean steps-to-food. Plot reward-over-episodes, weight-matrix evolution,
and spike rasters before/after learning.

---

## 9. Visualization / debugging (`viz/monitors.py`)
Build these early — this system is far easier to debug visually than numerically:
- spike raster (input / hidden / output lanes)
- weight-matrix heatmap over training (should develop structure, not saturate)
- eligibility-trace decay trace for a sample synapse
- reward and `(R - baseline)` over time
- agent trajectory overlaid on food positions, early vs late training

---

## 10. Known failure modes (expected — don't be alarmed)
- **Everything strengthens / weights saturate:** baseline missing or too slow, or
  no normalization. See §4.3 / §4.5.
- **Nothing learns:** `tau_e` too short for the reward delay, or noise too low so
  nothing explores, or encoding broken (test it in isolation).
- **Network goes silent or epileptic:** homeostasis/bounds mistuned; check Phase
  4 and 5 in isolation.
- **Fiddliness in general:** many interacting time constants is normal for this
  paradigm; tune one subsystem at a time using its own test, not end-to-end.

**Confirmed failure modes from the region-world build (and their fixes):**
- **Input has zero effect on output → network is a pure noise generator.** Weights
  are bounded [0,1] but the LIF needs ~15 mV to fire, so synaptic input is swamped
  by homeostatic noise. Fix: a synaptic current gain (`cfg.syn_gain`) decoupling
  weight bounds from the membrane-current scale. *Always verify vision actually
  drives the output before trusting any learning result.*
- **Learning erases structure (in_reward and in_punish converge to equal).** Dense
  high firing rates make STDP indiscriminate. Fix: a low firing setpoint
  (`target_rate ≈ 2 Hz`) so eligibility captures real pre→post causation.
- **A large wall penalty corrupts learning.** `R_wall` correlated with motor
  activity injects destructive negative reward. Keep it small.
- **Cumulative metrics hide learning.** Measure behavior over a steady-state
  trailing window, not the whole run (early exploration dilutes it).

## 11. Stretch goals (after Phase 7 works)
- Separate excitatory/inhibitory populations (Dale's law).
- Multiple hidden layers / recurrent hidden pool.
- Port core to Brian2 or BindsNET and compare.
- Swap foraging for a gym-style task once the learning core is trusted.

---

## 12. Extension — multi-agent, vision-driven region world

This generalizes §5. Instead of one agent chasing point pellets, we put **many
small agents** in a continuous 2D arena seeded with **reward regions** (places to
seek) and **punishment regions** (places to avoid). Agents *see* their surroundings
by **raycasting**, and move freely in the **x/y plane** via motor output neurons.
The SNN core (§4) is unchanged — this only swaps the environment, the sensory
encoding, and the motor decoding, and adds a live UI (§13).

The original §5 foraging world stays as the simplest learning sanity check; this
region world is the richer target task.

### 12.1 The world (`env/regions.py`)
- Continuous 2D box `W × H` (e.g. `1.0 × 1.0`), walls on all sides.
- **`M` agents** stepped simultaneously, each carrying its **own** SNN instance
  (same hyperparameters, independent weights, seed = `base_seed + agent_id`) so
  they learn independently and can be compared/ranked. State per agent:
  `pos (x, y)`, `heading θ`, `velocity (vx, vy)`, `alive`.
- **Regions:** a list of zones, each with:
  - a shape — circle `(cx, cy, r)` or axis-aligned rect,
  - a `sign` — reward (`+`) or punishment (`−`),
  - a `magnitude`,
  - optional slow drift / respawn so the task isn't memorizable as fixed coords.
- **Reward signal per agent per step** (this is the third factor `R` for that
  agent's network, §4.3):
  ```
  R_agent = Σ region.magnitude for regions containing the agent      # + for reward, − for punishment
          + R_step (small negative, efficiency pressure)
          + R_wall (negative, on hitting a wall this step)
  ```
- `step(motor_per_agent) -> per-agent (obs, R, done)`. `obs` is the raw ray hits
  (§12.2); encoding to spikes happens in `env/encoding.py`.

**Acceptance:** an agent placed inside a reward region receives positive `R`; the
same agent inside a punishment region receives negative `R`; regions, agents, and
walls are all detectable by rays before any learning is wired up.

### 12.2 Vision / raycasting (`env/vision.py`)
Each agent senses the world through `n_rays` rays fanned across a field of view
`fov`, centered on its **heading** (egocentric — rays rotate with the agent).

- For each ray, march/intersect outward to `max_range` and record the **nearest**
  hit:
  - `distance` to the hit (→ normalized proximity `1 − d/max_range`, 0 if nothing),
  - `hit_type` ∈ `{reward_region, punishment_region, wall, agent, none}`.
- Return an array shaped `(M, n_rays, ...)` of `(proximity, type)` per ray.
- Keep it cheap: analytic ray-vs-circle / ray-vs-rect intersection, not pixel
  marching. Vectorize across rays (and agents where practical).

**Acceptance:** a ray aimed straight at a known region returns that region's type
and a distance matching the geometry; a ray aimed at empty space returns `none`;
rotating the agent rotates which regions its rays strike.

### 12.3 Encoding & motor (`env/encoding.py`, extended)
- **Vision → input spikes:** input layer is `n_rays × n_channels` neurons, one
  channel per `hit_type` that matters (reward / punishment / wall / agent). For
  each ray, the channel matching its hit fires at a Poisson rate ∝ proximity; the
  others stay near zero. Closer + more salient → higher rate. This is "sight."
- **Output spikes → motion (egocentric forward/turn):** **three motor neuron
  pools** — `forward, turn-left, turn-right`. Heading is an independently
  controlled state variable; the agent moves forward along its heading and steers
  by turning. Decode each pool's smoothed firing rate (Hz) into a motor command:
  ```
  forward = rate(forward) * speed_scale                 # >= 0, world units/step
  turn    = (rate(turn-left) − rate(turn-right)) * turn_scale   # radians/step
  heading += turn
  pos     += forward * (cos(heading), sin(heading))
  ```
  **Why egocentric forward/turn rather than world-frame x/y:** vision is
  egocentric (rays fan around the heading), so a world-frame `+x/−x/+y/−y` motor
  would require the network to learn the heading-dependent ego→world rotation with
  no heading input — effectively unlearnable. With forward/turn, the map "reward
  ahead → forward, reward on the left → turn left" is fixed and heading-
  independent, so reward-modulated STDP can discover it. (See the resolved design
  fork in the build log.)

**Acceptance (hand-wired sanity, do before trusting learning):** a reflex
controller (`env/reflex.py`) reading the vision rays directly — "reward ahead →
forward, reward on the left → turn left, punishment/wall ahead → turn away" —
must drive agents into reward zones and away from punishment zones with no SNN at
all. This isolates the world/vision/motor geometry from the learning core, so a
later learning failure can be localized to the SNN. Measure it with the same
behavioral metric used for learning (fraction of time in reward vs punishment
regions, §14 Phase 11).

### 12.4 Multi-agent learning notes
- Each agent's `network.apply_reward(R_agent)` runs independently with its own
  `baseline`. No weight sharing by default.
- Stretch: every `N` episodes, clone the highest-mean-reward agent's weights into
  the worst performers (a crude evolutionary nudge on top of within-life learning).

---

## 13. UI — live visualization (`ui/`)

**Choice: Pygame** for the real-time interactive view; matplotlib (§9) remains for
offline/after-the-fact analysis plots.

*Why Pygame:* pure-Python and already in-stack, real-time per-frame drawing with
no web/GUI framework, runs natively on the Windows desktop, and immediate-mode
rendering makes drawing agents/rays/regions and overlaying network state simple.

### 13.1 World view
- Regions as translucent fills — **green** for reward, **red** for punishment,
  opacity ∝ magnitude.
- Agents as **oriented triangles**, tinted by recent reward (green = doing well,
  red = doing badly). Optional fading **motion trails**.
- **Rays** drawn from the selected agent (toggle: all agents), colored by hit type.

### 13.2 Inspector panel (selected agent)
Click an agent to select it; a side panel renders its network live (same data
`viz/monitors.py` computes, drawn in Pygame):
- input sensor activations as per-ray / per-channel bars,
- rolling **spike raster** (input / hidden / output lanes),
- output motor-pool activity + the decoded velocity vector (an arrow),
- **weight heatmap** (should develop structure, not saturate),
- `R` and `(R − baseline)` traces.

### 13.3 Controls / HUD
- Space = play/pause, `S` = single step, speed control = sim steps per rendered
  frame, `R` = reset (re-seed), `L` = freeze/unfreeze learning (weights), `T` =
  toggle trails, ray toggle, click = select agent, `Tab` = cycle agent.
- HUD: global step, FPS, mean reward, population firing rates, current best agent.

### 13.4 Architecture (decouple sim from render)
- `ui/simulation.py` holds a `Simulation` that owns the world + all agent networks
  and exposes `advance(n_steps)` and read-only `state()`. 
- `train.py` (headless) and `ui/app.py` (interactive) both drive the **same**
  `Simulation` — the UI just renders `state()` each frame and can advance many sim
  steps per frame for speed. No simulation logic lives in the renderer.

---

## 14. Build phases — extension (continue after Phase 7)

**Phase 8 — Region world + vision.** `regions.py`, `vision.py` + tests (§12.1,
§12.2): region containment → reward sign; analytic raycast returns correct
type/distance; rotation changes hits.

**Phase 9 — Vision encoding + x/y motor + hand-wired sanity.** Extend
`encoding.py` (§12.3). Hard-wire approach/avoid and confirm agents seek reward and
dodge punishment with learning off.

**Phase 10 — Multi-agent integration + UI.** `ui/simulation.py`, `ui/app.py`,
`render.py`, `panels.py` (§13). Many agents learning live; inspector on the
selected agent.

**Phase 11 — Behavioral evaluation.** A frame-independent **behavioral metric** —
fraction of agent-steps spent inside reward regions vs punishment regions (not
just `reward_ema`, which an agent can game by parking in one zone). Compare four
controllers on it: **reflex** (hand-wired, upper-bound sanity), **learning** SNN,
**frozen** SNN (lower bound), and **random** walk. Success (SPEC §0): the learning
agent beats random and frozen on time-in-reward / time-in-punishment, approaching
the reflex baseline, and its weights show structure rather than saturating.
