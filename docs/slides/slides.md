---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section { font-family: "Helvetica Neue", Arial, sans-serif; font-size: 26px; color: #0b0b0b; background: #fcfcfb; padding: 56px 64px; }
  h1 { font-size: 44px; color: #0b0b0b; margin-bottom: 0.3em; }
  h2 { font-size: 34px; color: #0b0b0b; margin: 0 0 0.4em; }
  strong { color: #2a78d6; }
  table { font-size: 21px; border-collapse: collapse; }
  th { background: #f1f0ec; }
  td, th { padding: 6px 12px; }
  .muted { color: #52514e; font-size: 20px; }
  .big { font-size: 64px; font-weight: 700; color: #2a78d6; line-height: 1; }
  .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 36px; align-items: start; }
  .wide-left { grid-template-columns: 1.35fr 1fr; gap: 28px; }
  .wide-left li { font-size: 23px; }
  img { display: block; margin: 0 auto; }
  footer { color: #8a8984; font-size: 16px; }
footer: CS4246 Group 23 · Carbon-Aware Smart Building Energy Controller
---

<!-- _paginate: false -->
<!-- _footer: "" -->

# Carbon-Aware Smart Building Energy Controller

Learning *when* to cool an office so the grid burns less carbon

<br>

**Group 23** · Ethan · Jaylon · Brandon · Sean · Patrick
<span class="muted">CS4246/5446 Reinforcement Learning and Sequential Decision Making · 12 Nov 2026</span>

---

## Why: when a building consumes matters as much as how much

- Buildings use **~30% of global final energy**; in Singapore, air-conditioning dominates commercial electricity
- Grid **carbon intensity changes hour to hour** with wind and solar output
- Buildings store "coolth" in their thermal mass → they *can* shift load in time
- But thermal inertia means an action now affects comfort **hours later** → a sequential decision problem

**Question:** can an RL controller cut HVAC carbon beyond a well-tuned carbon-aware rule, without hurting comfort?

<!-- Patrick -->

---

## What we built

1. **Simulator** of an office zone (1R1C thermal model) with a year of weather, occupancy, price and carbon data
2. **Formal MDP** with a multi-objective reward (carbon, cost, comfort)
3. **PPO and DQN** vs three rule-based controllers and an **exact DP upper bound**
4. **Analysis**: 5 seeds, held-out months, 95% CIs, Pareto sweep, state ablations, policy traces
5. **Responsible-AI** reading of what the agent actually learned

<!-- Patrick -->

---

## Simulator: a 1R1C office zone

<div class="cols">
<div>

C·dT/dt = (T_out − T_in)/R + Q_int + Q_sol − COP·P

- R = 0.4 K/kW, C = 15 kWh/K → **τ = 6 h**
- 14 kW HVAC, COP falls on hot afternoons
- Exact discretisation, 15-min steps
- **Validated:** step response hits 63% at τ; energy ≈ office ACMV range; DP oracle dominates every baseline (10 unit tests)

</div>
<div>

| Series | Source |
|---|---|
| Weather | Singapore-like, 25–33 °C, rainy days |
| Occupancy | Weekday 08–18 + jitter, holidays |
| Price | USEP-like, SGD/kWh |
| Carbon intensity | **UK-grid-like** (SG grid ~95% gas → nearly flat CI) |

Train Jan–Sep · **test Oct–Dec (92 unseen days)**

</div>
</div>

<!-- Ethan -->

---

## The MDP

- **State (20 features):** T_in, T_out, occupancy sensor, price, CI, time of day, weekday, forecasts of T_out / price / CI at +1, 2, 4, 8 h
- **Action:** HVAC power ∈ {0, 25, 50, 75, 100}% — the same space for PPO, DQN *and* exact DP
- **Reward:** r_t = −[ λ_C·CI_t·E_t + λ_$·p_t·E_t + λ_K·discomfort_t ], discomfort = °C·h outside **23–26 °C, only when occupied**
- **γ = 0.99** at 15 min ≈ 25 h effective horizon, long enough to credit 5 am pre-cooling for 2 pm comfort

**Key observation:** carbon and cost are both ∝ energy, so carbon-awareness = **shifting when you consume**. Price and CI are nearly uncorrelated here (r = −0.07): the cheapest hour is not the cleanest.

<!-- Brandon -->

---

## Controllers, and a fair fight

| Controller | What it does |
|---|---|
| Fixed setpoint | Holds 24 °C around the clock |
| Setback | Cools during working hours only; **setpoint and start time tuned** |
| Carbon heuristic | Cools deeper when CI is low, drifts warmer when high; **4 parameters tuned** |
| Carbon-proportional | Setpoint rises smoothly with CI; **gain and base tuned** |
| DQN / **PPO** | PPO: clipped surrogate, GAE, return normalisation; **12 settings × 2 seeds tuned** |
| **Oracle (DP)** | Backward induction with perfect foresight, a true upper bound |

**Same rules for everyone:** tuned on training days only (972 rule settings), scored with the same reward PPO maximises, then tested **once** on 92 unseen days. 5 seeds per learner, paired 95% bootstrap CIs.

<!-- Jaylon: baselines + tuning. Brandon: PPO/oracle rows -->

---

## Result: tuning matters most; PPO adds a little more

<div class="cols">
<div>

| Controller | kg CO₂/day | Viol.* |
|---|---|---|
| Fixed setpoint | 18.96 | 0% |
| Setback, 24 °C (untuned) | 11.10 | 0% |
| **Best tuned rule** (25.9 °C) | 8.52 | 0.7% |
| DQN | 8.52 | 7.6% |
| **PPO (tuned)** | **8.21** | 5.2% |
| Oracle (DP) | 7.83 | 3.7% |

<span class="muted">*occupied 15-min steps outside 23–26 °C</span>

</div>
<div>

Tuning one setpoint: **−23%** carbon

PPO vs the best tuned rule:

<div class="big">−3.7%</div>

carbon, 95% CI [−4.0, −3.2]%
**reward +1.1%** (p = 0.03), cost −4.1%

19% of the gap to the oracle closed

Price: more frequent but tiny violations (0.08 °C·h/day)

</div>
</div>

<!-- Sean -->

---

## Learning: PPO is stable, DQN is noisy

![h:440](../../results/figures/learning_curves.png)

PPO plateaus by ~700k steps; its 5 seeds land within 1.3% on test carbon. It ends **level with the tuned rule** on these 16 days and slightly ahead on 60 validation and 92 test days: a small edge.

<!-- Sean (Brandon for PPO stability questions) -->

---

## What did PPO learn? Ride the band edge, cool a little in carbon dips

![h:560](../../results/figures/day_trace.png)

<!-- Sean: Both hold ~25.7-25.9 °C. PPO cools slightly more in the 10:00 and 12:00-15:00 CI dips, then coasts to 26 °C as CI rises. Effective CI paid: PPO 0.115 vs rule 0.118 kg/kWh. -->

---

## Where is the remaining headroom? Not in the carbon signal

<div class="cols wide-left">
<div>

| Oracle variant (test) | Reward | kg CO₂ |
|---|---|---|
| Full foresight | −13.07 | 7.83 |
| No carbon foresight | −13.15 | 8.01 |
| **No occupancy foresight** | **−14.08** | 7.92 |
| *Best tuned rule* | *−13.88* | *8.52* |
| *PPO (tuned)* | *−13.72* | *8.21* |

</div>
<div>

- Knowing future CI is worth **0.08** reward/day
- Knowing when people arrive and leave is worth **~1.0**
- Tuning drove every carbon-reactive rule to **ignore CI**
- Removing CI from PPO's state costs only +1.3% carbon
- One room with τ = 6 h is a **leaky battery**: little to shift

</div>
</div>

<!-- Sean, with Brandon -->

---

## At matched comfort, it's close to a tie

![h:430](../../results/figures/pareto.png)

Inside the comfort band, the PPO front and the setpoint sweep overlap. PPO's extra savings come from **accepting more discomfort**.

<!-- Sean presents; Patrick links to RAI -->

---

## Responsible AI

- **Who sets λ?** The weights decide whose comfort is traded for carbon. The Pareto front makes that trade visible.
- **Specification gaming:** PPO sits at the edge of the band. Comfort is charged only when occupied, so a faulty sensor invites exploitation; we **score comfort on true occupancy**.
- **Comfort is a distribution:** **53%** of PPO's violations fall at 17:00–18:30. **Late leavers** bear them; the mean hides it.
- **Honest baselines:** an untuned baseline overstated PPO's gain from 3.7% to 11%. Comparing against weak rules is a quiet way to mislead.
- **Transparency:** a tuned thermostat gets most of the benefit and is fully explainable. That is a legitimate deployment choice.

<!-- Patrick -->

---

## Limitations and what is next

- Single zone, simplified thermal model, **synthetic profiles** (UK CI substituted for Singapore)
- Small RL margin; DQN, sweep and ablations used default PPO settings

**In progress for the report**
- **Occupancy prediction**: where the oracle's headroom actually is
- Robustness: shifted CI (reads signal vs memorises clock), noisy occupancy, tariff regimes
- **Constrained PPO** (Lagrangian): comfort as a constraint, not a weight
- **Battery storage**: a real lever for carbon shifting

<!-- Patrick -->

---

## Takeaways

1. **Tune the baseline first:** one setpoint gives −23% carbon; skipping this inflated our RL gain about 3×
2. Tuned PPO still wins, **−3.7% carbon and +1.1% reward**, with a little genuine load-shifting
3. The carbon signal is worth little in one room; **occupancy foresight** is the real headroom
4. An oracle plus fair tuning is what reveals all of this

<br>

**Thank you · Questions?**

<!-- Patrick -->
