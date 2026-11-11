---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  :root { --ink: #17202a; --muted: #5f6b7a; --rule: #d9dee5; --panel: #f3f6f9; --accent: #2a78d6; --accent-soft: #e6f0fb; }
  section { font-family: "Inter", "Segoe UI", Arial, sans-serif; font-size: 25px; line-height: 1.45; color: var(--ink); background: #fff; padding: 58px 72px 64px; justify-content: flex-start; }
  h1 { font-size: 50px; font-weight: 700; letter-spacing: -0.02em; line-height: 1.12; margin: 0 0 18px; color: inherit; }
  h2 { font-size: 34px; font-weight: 700; letter-spacing: -0.015em; line-height: 1.2; margin: 0 0 26px; color: var(--ink); }
  strong { font-weight: 600; color: inherit; }
  em { font-style: italic; }
  ul, ol { margin: 0; padding-left: 1.1em; }
  li { margin: 0 0 12px; }
  li::marker { color: var(--muted); }
  img { display: block; margin: 0 auto; background: transparent; }
  .kicker { font-size: 15px; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; color: var(--accent); margin-bottom: 10px; }
  .muted { color: var(--muted); font-size: 19px; }
  .caption { color: var(--muted); font-size: 20px; margin-top: 14px; }
  .hl { color: var(--accent); font-weight: 600; }
  /* tables: rules, no fills */
  section table { display: table; width: 100%; font-size: 21px; border-collapse: collapse; margin: 0; }
  section table tr, section table tr:nth-child(2n) { background: transparent !important; }
  section table th, section table td { border: none !important; padding: 8px 18px 8px 0; text-align: left; vertical-align: top; }
  section table th { font-size: 14px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); border-bottom: 2px solid var(--ink) !important; background: transparent !important; }
  section table td { border-bottom: 1px solid var(--rule) !important; }
  section table td:first-child { white-space: nowrap; }
  td strong { color: var(--accent); }
  /* layout */
  .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 48px; align-items: start; }
  .cols > div { min-width: 0; }
  .wide-left { grid-template-columns: 1.3fr 1fr; }
  .wide-right { grid-template-columns: 1fr 1.25fr; }
  .cards { display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px; }
  .card { background: var(--panel); border-radius: 10px; padding: 18px 20px; font-size: 20px; line-height: 1.4; }
  .card h3 { font-size: 20px; font-weight: 700; margin: 0 0 6px; color: var(--ink); }
  .card p { margin: 0; }
  .callout { border-left: 4px solid var(--accent); background: var(--accent-soft); padding: 14px 20px; border-radius: 0 8px 8px 0; margin-top: 26px; }
  .callout p { margin: 0; }
  .eq { font-family: "Cambria Math", "Times New Roman", serif; font-size: 26px; background: var(--panel); border-radius: 8px; padding: 12px 18px; margin-bottom: 20px; }
  .stat { margin-bottom: 26px; }
  .stat .n { font-size: 66px; font-weight: 700; letter-spacing: -0.03em; line-height: 1; color: var(--accent); }
  .stat .n.dark { color: var(--ink); }
  .stat .l { font-size: 19px; color: var(--muted); margin-top: 6px; line-height: 1.35; }
  /* title and closing slides */
  section.title { background: #10202e; color: #f4f6f8; justify-content: center; padding: 72px 88px; }
  section.title h1 { color: #fff; max-width: 900px; }
  section.title .kicker { color: #7fb3ec; }
  section.title .sub { font-size: 28px; color: #c9d3dd; margin-bottom: 56px; }
  section.title .people { font-size: 22px; color: #fff; }
  section.title .muted { color: #8fa1b3; }
  section.title::after { display: none; }
  footer { font-size: 13px; color: #9aa4b0; left: 72px; bottom: 26px; }
  section::after { font-size: 13px; color: #9aa4b0; right: 72px; bottom: 26px; }
footer: Group 23 · Carbon-aware HVAC control
---

<!-- _class: title -->
<!-- _paginate: false -->
<!-- _footer: "" -->

<div class="kicker">CS4246/5446 · Group 23</div>

# Carbon-Aware Smart Building Energy Controller

<div class="sub">Learning <em>when</em> to cool an office so the grid burns less carbon</div>

<div class="people">Ethan · Jaylon · Brandon · Sean · Patrick</div>
<div class="muted">Reinforcement Learning and Sequential Decision Making · 12 Nov 2026</div>

---

<div class="kicker">Motivation</div>

## When a building uses power matters as much as how much

- Buildings use about 30% of global final energy; in Singapore, air-conditioning dominates commercial electricity
- Grid carbon intensity swings hour to hour with wind and solar output
- A building's thermal mass stores cooling, so load can be moved in time
- But an action now affects comfort hours later, which makes this a sequential decision problem

<div class="callout">

Can an RL controller cut HVAC carbon beyond a well-tuned rule, without hurting comfort?

</div>

<!-- Patrick -->

---

<div class="kicker">Approach</div>

## What we built

<div class="cards">
<div class="card">

### Simulator
A 1R1C office zone driven by a year of weather, occupancy, price and carbon data

</div>
<div class="card">

### Formulation
An MDP whose reward combines carbon, cost and comfort

</div>
<div class="card">

### Controllers
PPO and DQN against tuned rule-based controllers and an exact DP upper bound

</div>
<div class="card">

### Evaluation
5 seeds, held-out months, 95% CIs, a Pareto sweep, state ablations and policy traces

</div>
<div class="card">

### Responsible AI
What the agent actually learned, and who pays for it

</div>
</div>

<!-- Patrick -->

---

<div class="kicker">Simulator</div>

## A single office zone with a 6-hour time constant

<div class="cols">
<div>

<div class="eq">C·dT/dt = (T<sub>out</sub> − T<sub>in</sub>)/R + Q<sub>int</sub> + Q<sub>sol</sub> − COP·P</div>

- R = 0.4 K/kW and C = 15 kWh/K, so τ = 6 h
- 14 kW HVAC whose COP falls on hot afternoons
- Exact discretisation at 15-minute steps
- Validated by 10 unit tests: step response, energy range, oracle dominance

</div>
<div>

| Series | Source |
|---|---|
| Weather | Singapore-like, 25–33 °C, rainy days |
| Occupancy | Weekdays 08–18, jitter, holidays |
| Price | USEP-like, SGD/kWh |
| Carbon intensity | UK-grid-like (Singapore's ~95% gas grid is nearly flat) |

<div class="caption">Train January–September · test October–December (92 unseen days)</div>

</div>
</div>

<!-- Ethan -->

---

<div class="kicker">Formulation</div>

## The decision problem

<div class="cols wide-left">
<div>

| Element | Definition |
|---|---|
| State | 20 features: T<sub>in</sub>, T<sub>out</sub>, occupancy sensor, price, CI, time, weekday, forecasts at +1, 2, 4, 8 h |
| Action | HVAC power ∈ {0, 25, 50, 75, 100}%, shared by PPO, DQN and the DP oracle |
| Reward | −[λ<sub>C</sub>·CI·E + λ<sub>$</sub>·price·E + λ<sub>K</sub>·discomfort], discomfort counted only when occupied, outside 23–26 °C |
| Discount | γ = 0.99, a ~25 h horizon: enough to credit 5 am pre-cooling for 2 pm comfort |

</div>
<div>

<div class="callout">

Carbon and cost both scale with energy, so being carbon-aware means **shifting when you consume**.

Price and CI are nearly uncorrelated here (r = −0.07): the cheapest hour is not the cleanest.

</div>
</div>
</div>

<!-- Brandon -->

---

<div class="kicker">Controllers</div>

## Rules and PPO are tuned under one protocol

| Controller | Behaviour | Tuned |
|---|---|---|
| Fixed setpoint | Holds 24 °C around the clock | — |
| Setback | Cools during working hours only | setpoint, start time |
| Carbon heuristic | Cools deeper when CI is low, drifts warmer when high | 4 parameters |
| Carbon-proportional | Setpoint rises with CI | gain, base |
| PPO · DQN | Clipped surrogate, GAE, return normalisation | PPO: 12 settings × 2 seeds |
| Oracle (DP) | Backward induction with perfect foresight | upper bound |

<div class="callout">

All tuning on training days only (972 rule settings), scored with the reward PPO maximises. Tested once on 92 unseen days; 5 seeds per learner, paired 95% bootstrap CIs.

</div>

<!-- Jaylon: baselines + tuning. Brandon: PPO/oracle rows -->

---

<div class="kicker">Main result</div>

## Tuning matters most; PPO's extra gain is small

<div class="cols wide-left">
<div>

| Controller | kg CO₂/day | Violations* |
|---|---|---|
| Fixed setpoint | 18.96 | 0% |
| Setback, 24 °C (untuned) | 11.10 | 0% |
| Best tuned rule (25.9 °C) | 8.52 | 0.7% |
| DQN | 8.52 | 7.6% |
| **PPO (tuned)** | **8.21** | **5.2%** |
| Oracle (DP) | 7.83 | 3.7% |

<div class="muted">*occupied 15-minute steps outside 23–26 °C</div>

</div>
<div>

<div class="stat"><div class="n dark">−23%</div><div class="l">carbon from tuning one setpoint</div></div>

<div class="stat"><div class="n">−3.7%</div><div class="l">PPO vs best tuned rule, 95% CI [−4.0, −3.2]<br>reward +1.1% (p = 0.03), cost −4.1%</div></div>

<div class="stat"><div class="n dark">72%</div><div class="l">of that saving is from 3 public holidays, when the rule cools an empty office. Other 89 days: −1.1% carbon, reward −1.6%</div></div>

</div>
</div>

<!-- Sean -->

---

<div class="kicker">Learning dynamics</div>

## PPO learns stably; DQN is noisy

![h:400](../../results/figures/learning_curves.png)

<div class="caption">PPO's 5 seeds settle by 0.75–1.3M steps and land within 1.3% on test carbon. On these 16 training days it ends level with the tuned rule. DQN drops out repeatedly.</div>

<!-- Sean (Brandon for PPO stability questions) -->

---

<div class="kicker">Policy behaviour</div>

## PPO rides the band edge and cools a little in carbon dips

![h:500](../../results/figures/day_trace.png)

<!-- Sean: Both hold ~25.7-25.9 °C. PPO cools slightly more in the 10:00 and 12:00-15:00 CI dips, then coasts to 26 °C as CI rises. Effective CI paid (energy-weighted): PPO 0.162 vs rule 0.164 vs oracle 0.159 kg/kWh; 63% vs 60% of energy in below-median-CI hours. -->

---

<div class="kicker">Where the headroom is</div>

## The bottleneck is occupancy, not the carbon signal

<div class="cols wide-left">
<div>

| Oracle variant (test) | Reward | kg CO₂ |
|---|---|---|
| Full foresight | −13.07 | 7.83 |
| No carbon foresight | −13.15 | 8.01 |
| **No occupancy foresight** | **−14.08** | 7.92 |
| Best tuned rule | −13.88 | 8.52 |
| PPO (tuned) | −13.72 | 8.21 |

</div>
<div>

<div class="stat"><div class="n dark">0.08</div><div class="l">reward/day from knowing future carbon intensity</div></div>

<div class="stat"><div class="n">~1.0</div><div class="l">reward/day from knowing when people arrive and leave (0.65 on ordinary days)</div></div>

<div class="muted">PPO's own win is occupancy too: it stays off on holidays. Removing CI from its state costs 1.3% carbon (not significant). One room with τ = 6 h is a leaky battery.</div>

</div>
</div>

<!-- Sean, with Brandon -->

---

<div class="kicker">Trade-off</div>

## At matched comfort, the tuned rule wins

![h:400](../../results/figures/pareto_normal_days.png)

<div class="caption">89 ordinary days. At strict comfort every PPO setting is beaten by the rule. PPO only wins at 0.08–0.1 °C·h/day, a band the thermostat skips (25.9 °C → 26.0 °C jumps from 0.7% to 35% violations).</div>

<!-- Sean presents; Patrick links to RAI -->

---

<div class="kicker">Responsible AI</div>

## What the numbers do not show on their own

<div class="cards">
<div class="card">

### Who sets λ?
The reward is a utility function: its weights decide whose comfort is traded for carbon. The Pareto front makes that trade visible.

</div>
<div class="card">

### Specification gaming
Comfort is charged only when occupied, so a faulty sensor invites reward over-optimisation. We score comfort on true occupancy.

</div>
<div class="card">

### Comfort is a distribution
More than half of PPO's violations fall between 17:00 and 18:30. Late leavers bear them; the mean hides it.

</div>
<div class="card">

### Honest baselines
A partly tuned rule inflated PPO's gain to 11%, an anchor that fair tuning removed; a holiday calendar shrinks it to 0.9%. Check where a gain comes from.

</div>
<div class="card">

### Transparency
A tuned thermostat gets most of the benefit and anyone can explain it. That is a legitimate choice. RL would need a human override.

</div>
</div>

<!-- Patrick -->

---

<div class="kicker">Limitations</div>

## What this study cannot claim yet

<div class="cols">
<div>

### Limits

- One zone and a simplified thermal model
- Synthetic profiles, with UK carbon intensity standing in for Singapore's
- A small RL margin; DQN, the λ sweep and the ablations used default PPO settings

</div>
<div>

### Next, for the report

- Occupancy prediction, where the oracle's headroom is
- Robustness: shifted CI, noisy occupancy, tariff regimes
- Constrained PPO: comfort as a constraint, not a weight
- Battery storage: a real lever for shifting carbon

</div>
</div>

<!-- Patrick -->

---

<div class="kicker">Summary</div>

## Takeaways

1. **Tune the baseline first.** One setpoint gives −23% carbon; skipping this inflated our RL gain about threefold.
2. **Tuned PPO wins by 3.7% carbon**, mostly by staying off on holidays; on ordinary days it is ~1%, paid for in comfort.
3. **In one room the carbon signal is worth little.** Occupancy foresight is the real headroom.
4. **An oracle plus fair tuning** is what reveals all of this.

<div class="callout">

Thank you. Questions?

</div>

<!-- Patrick -->
