# Roadmap — toward a biologically-plausible, evolvable spiking brain

## Vision

Grow the project from "a reward-modulated SNN that learns to seek reward" into a
small but coherent computational-neuroscience program:

> A population of **genomes** that **develop** into spatially-embedded spiking
> networks (excitatory + inhibitory neurons, distance-dependent conduction
> delays); each network **learns within its lifetime** via reward-modulated
> plasticity; and the population **evolves** across generations under selection
> on behavior.

The unifying idea: **put neurons in space.** That one decision yields (a)
distance-based conduction delays, (b) a geometry a generative genome can grow
into, and (c) a directly visualizable brain. Spatial embedding is the spine that
connects all three of the directions below.

## Where we are now

- Core: LIF + STDP + eligibility three-factor plasticity, homeostasis, stability
  (all unit-tested). Egocentric raycasting vision, forward/turn motor.
- **Validated behavior** (`evaluate.py`): agents **learn to seek and stay in
  green** (steer toward visible reward 0.70 vs 0.40 random; dwell 168 vs 45
  steps) but **do not learn to avoid red** (`turn_off_red ≈ 0` vs reflex 0.55).
- Root cause of the gap: the network is **all-excitatory**, so it can only learn
  attraction. Avoidance wants inhibition. That sets the first phase.

---

## Phase A — Excitatory/inhibitory populations + inhibition (the avoidance fix)

**Goal:** give the network a withdrawal/veto pathway so it can learn to avoid red,
the biologically natural way.

**Biological grounding.** Dale's law (a neuron releases the same
neurotransmitter at all its synapses → it is either excitatory *or* inhibitory,
never both). Cortex is ~80% E / 20% I. Aversive behavior is largely inhibition-
mediated (withdrawal reflexes veto ongoing motor drive). Inhibitory synapses are
plastic too — Vogels, Sprekeler, Gerstner (2011) "inhibitory plasticity balances
excitation and inhibition."

**Design.**
- Tag every neuron E or I (genome-specifiable later). An I-neuron's outgoing
  current is negative; weight *magnitudes* stay ≥ 0 (sign comes from the
  presynaptic type), so STDP/eligibility math is unchanged — only the delivered
  current flips sign. (SPEC §4.5 already anticipates signed bounds.)
- Add an inhibitory interneuron pool to the hidden layer (~20%). Allow E→E, E→I,
  I→E, I→I connectivity.
- Plasticity: keep reward-modulated STDP on excitatory synapses; add inhibitory
  plasticity on I→E synapses (start with Vogels-style toward an E/I balance set-
  point, then make it reward-modulated so red→(suppress forward / gate turn) can
  be *learned*).
- Wiring intuition to reach for (let plasticity find, or seed structurally):
  "red-ahead → inhibit forward" (slow/stop before entering) and "red-left →
  inhibit turn-left" (net turn right = away) — the inhibitory Braitenberg coward.

**Touchpoints:** `config.py` (E/I ratio, signed weight bounds, iSTDP params),
`snn/neuron.py` (per-neuron E/I tag), `snn/synapses.py` (signed current, iSTDP
update), `snn/network.py` (interneuron pool, wiring).

**Success metric:** `evaluate.py` — `turn_off_red` rises clearly above 0, red
`in_pun` and `red_dwell` fall, while green-seeking is preserved. New unit test:
an inhibitory synapse delivers negative current and an I-driven post neuron's
rate drops.

---

## Phase B — Spatial embedding + conduction delays

**Goal:** place neurons in space; make each synapse's delay ∝ distance. Explore
the temporal computation this unlocks.

**Biological grounding.** Axonal conduction velocity is finite; delay ≈
distance / velocity (modulated by myelination/diameter). Delay lines do real
computation: sound localization (Jeffress model), and **polychronization**
(Izhikevich 2006) — heterogeneous delays make reproducible spatiotemporal spike
patterns ("polychronous groups"), a far larger representational store than rate.

**Design.**
- Give each neuron a position (2D or 3D). Per-synapse delay
  `d_ij = round(||x_i - x_j|| / v_cond / dt)` in timesteps.
- Implement delay lines: a presynaptic spike is deposited into a ring buffer and
  delivered `d_ij` steps later. STDP then sees the *delayed* arrival, so learning
  shapes which delayed paths cohere.
- Explore: looming/approach detection (red getting closer), motion-direction
  selectivity, sequence detection — things rate coding can't express.

**Touchpoints:** `snn/` (neuron positions, per-synapse delay buffers), a layout
module, `ui/` (render neurons in their space + signal propagation).

**Success metric:** a delay-line coincidence detector demonstrably fires for one
temporal order of inputs and not the reverse; polychronous groups are observable;
no regression on the foraging behavior.

---

## Phase C — Dual neuromodulation: appetitive vs aversive (optional, supports A)

**Goal:** stop collapsing reward and punishment into one signed scalar.

**Biological grounding.** The brain uses *distinct* systems for reward vs threat
(e.g. dopaminergic reward-prediction-error vs separate aversive circuitry), often
targeting different populations with different signs/time-constants.

**Design.** Two third-factors — an appetitive signal (gates potentiation of
approach pathways) and an aversive signal (gates potentiation of inhibitory/
withdrawal pathways). Each with its own baseline and eligibility coupling.
Composes naturally with the E/I split from Phase A.

---

## Phase D — Generative / developmental genome (genome → network by rules)

**Goal:** stop hand-specifying the architecture; let a compact genome *grow* it.

**Biological grounding.** Genomes don't encode every synapse — they encode a
*developmental program*. Two routes, increasing in biological fidelity:

- **D1 — CPPN / HyperNEAT (pragmatic first step).** A small "genome" network
  (compositional pattern-producing network) maps neuron *coordinates* →
  connection weight / sign / delay. Exploits the Phase-B geometry and symmetry;
  decouples genome size from network size. (Stanley et al., HyperNEAT.)
- **D2 — Developmental L-system / gene-regulatory growth (the biological dream).**
  Genome = rewrite/regulatory rules. Simulate development: cells divide,
  differentiate into E/I via "morphogen" gradients, grow axons along gradients
  and connect by proximity rules. Closest to "a gene sequence grows a network."
  (Gruau cellular encoding; gene-regulatory-network models of neural development.)

**Recommended order:** D1 first (fast payoff, reuses geometry), then D2 as the
ambitious morphogenetic version.

**Touchpoints:** new `genome/` package (encoding + a `develop(genome) -> Network`
builder); `snn/network.py` becomes a *product* of development rather than
hand-wired.

**Success metric:** a genome reliably develops into a working network;
small genome changes produce sensible structural changes; symmetry/repetition is
expressible compactly.

---

## Phase E — Evolutionary outer loop (evolving learners)

**Goal:** evolve genomes whose developed networks perform — *and learn* — well.

**Design.**
- Population of genomes → develop → each controls agent(s) in the region world →
  fitness from `evaluate.py` behavior (multi-objective: seek + avoid + efficiency,
  not just summed reward).
- Selection + mutation + crossover; consider speciation/novelty search to protect
  innovation (NEAT-style) and avoid premature convergence.
- **Keep within-life reward-modulated STDP on.** Then evolution discovers
  networks that are good *learners*, not just good fixed controllers — the
  Baldwin effect / learning-guided evolution. This is the payoff that ties the
  whole stack together.

**Touchpoints:** `evolution/` package (population, variation, selection);
reuse the headless `Simulation` for fitness rollouts; parallelize across genomes.

**Success metric:** mean population fitness climbs across generations; evolved-
then-learned agents beat hand-tuned ones on the `evaluate.py` behaviors —
crucially including the red-avoidance that pure within-life learning couldn't get.

---

## Cross-cutting

- **Visualization:** once neurons are spatial (Phase B), draw the brain itself —
  E/I colored, signals propagating along delayed axons — alongside the world.
- **Determinism & tests:** every phase keeps the seed→run reproducibility and
  adds its own acceptance test (per SPEC §8 discipline), tuned in isolation.
- **Measure, don't assume:** `evaluate.py` is the arbiter. Every phase is judged
  on behavior validated against the reflex ground truth, not on reward alone.

## Suggested sequence

A (avoidance via inhibition) → B (spatial + delays) → D1 (CPPN genome) →
E (evolution), with C and D2 folded in as depth allows. A is the immediate, high-
value, low-dependency step and directly closes the gap `evaluate.py` exposed.

## References (read-if-stuck, not to reproduce)

- Dale's principle; Vogels, Sprekeler & Gerstner 2011 (inhibitory plasticity).
- Izhikevich 2006, "Polychronization: computation with spikes"; Jeffress 1948
  (delay-line sound localization).
- Stanley & Miikkulainen 2002 (NEAT); Stanley et al. 2009 (HyperNEAT); Gruau 1994
  (cellular encoding); gene-regulatory-network developmental models.
- (Already in SPEC) Izhikevich 2007; Frémaux & Gerstner 2016 (three-factor rules).
