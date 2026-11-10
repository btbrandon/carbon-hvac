# 4. Problem Formulation

*Draft — Brandon. Numbers marked ⟨⟩ are filled from `results/` after the final runs.*

We control the cooling of a single office zone over one day. Each episode starts at 00:00 on a calendar day and runs for H = 96 decision steps of Δt = 15 min. The controller sees the zone and the grid, picks an HVAC power level, and the zone's temperature evolves according to the thermal model in §5.

## 4.1 MDP

**State.** The observation at step t is

s_t = (T_in, T_out, ô_t, p_t, CI_t, sin(2πh/24), cos(2πh/24), w_t, {T̂_out, p̂, ĈI}_{t+τ} for τ ∈ {1, 2, 4, 8} h),

where T_in and T_out are the indoor and outdoor temperatures (°C), ô_t is the occupancy *sensor* reading, p_t is the wholesale electricity price (SGD/kWh), CI_t is the grid carbon intensity (kg CO₂/kWh), h is the hour of day and w_t flags a working day. The last block holds forecasts of outdoor temperature, price and CI at four lead times (12 forecast features; 20 in total). Each feature is affinely scaled to roughly [−1, 1] with fixed constants, so the observation is identical across runs and needs no running normaliser.

**Action.** a_t ∈ {0, 1, 2, 3, 4} selects HVAC electrical power P_t = (a_t/4)·P_max, with P_max = 14 kW. We chose five discrete levels for three reasons:

1. Staged compressors are common in commercial units.
2. The same action space admits PPO, DQN *and* an exact dynamic-programming solution (§6), so all controllers solve exactly the same problem.
3. Power resolution is coarse but temporal resolution is fine: averaged over an hour, four 15-min steps already give 17 distinct power levels. The Δt = 1 h ablation (§8.4) tests how much that temporal resolution matters.

**Transition.** T_in evolves by the 1R1C model of §5, discretised exactly under a zero-order hold:

T_{t+1} = T_eq + (T_t − T_eq)·e^{−Δt/RC},  T_eq = T_out + R·(Q_int(o_t) + Q_sol − COP(T_out)·P_t).

Given the exogenous series, the dynamics are deterministic. The process is stochastic only through the day drawn, the initial temperature T_0 ~ U(25.5, 28) °C, occupancy-sensor errors and forecast errors.

**Reward.**

r_t = −[ λ_E·E_t + λ_C·CI_t·E_t + λ_$·p_t·E_t + λ_K·D_t ],  E_t = P_t·Δt,

D_t = 1[o_t]·( max(0, T_{t+1} − 26) + max(0, 23 − T_{t+1}) )·Δt.

- D_t is a hinge penalty on the comfort band [23, 26] °C (SS 553 guidance for Singapore offices), in °C·h.
- The penalty is charged **only when the zone is truly occupied** (o_t, not the sensor ô_t). This choice matters for specification gaming (§9).
- Default weights are λ_E = 0, λ_C = 1 per kg, λ_$ = 0.5 per SGD and λ_K = 5 per °C·h. λ_K is swept in §8.3.
- A squared-hinge variant is implemented for the sensitivity study.

**Discount.** γ = 0.99. At 15-min steps this gives an effective horizon 1/(1−γ) = 100 steps ≈ 25 h, which spans a full pre-cool → occupancy cycle. Episodes end at a fixed horizon, with no terminal value: the zone is unoccupied from about 18:00, so letting the temperature drift at the end of the day costs nothing in reality either.

## 4.2 Carbon-awareness is temporal load shifting

Carbon (CI_t·E_t) and cost (p_t·E_t) are both linear in E_t. Any controller that saves energy therefore saves both. What separates a *carbon-aware* controller is **when** it consumes: for a fixed total E, carbon is minimised by moving consumption into low-CI steps. The thermal mass C = 15 kWh/K is the only storage. Pre-cooling by 3 K banks about 45 kWh of cooling, roughly 13 kWh of electricity at COP 3.5, which bounds how much load can move.

We report two quantities that separate energy reduction from shifting:

- the **effective CI paid**, Σ CI_t E_t / Σ E_t;
- the **share of HVAC energy consumed in below-median-CI steps**.

In our data the correlation between price and CI is ⟨r = −0.07⟩. Price peaks with afternoon demand, while CI dips with midday solar and windy periods, so the two objectives genuinely diverge. A scenario with CI constructed to be anti-correlated with price isolates carbon-awareness from cost-awareness (§8.5).

## 4.3 Partial observability

The true state includes the occupancy o_t and future exogenous values. The agent sees only a sensor ô_t that flips with probability ε ∈ {0, 0.1, 0.2}, and forecasts with multiplicative error σ·√τ. The problem is therefore a POMDP. We approximate it in two ways:

- **Feature augmentation.** Forecasts plus time features stand in for the hidden exogenous state. The occupancy *schedule* is largely predictable from (h, w_t), which compensates for a noisy sensor.
- **Recurrent policy** for the noisy-sensor variant (stretch, §8.7).

The oracle (§6) is given the true o_t and perfect forecasts, which is what makes it an upper bound.

## 4.4 Design choices

| Choice | Value | Alternative considered | Reason |
|---|---|---|---|
| Δt | 15 min | 1 h | Pre-cooling and CI dips last 1–3 h; 1 h tested as an ablation |
| Horizon | 1 day (96 steps) | 1 week | Occupancy cycle is daily; weekly episodes slow PPO with no new dynamics |
| γ | 0.99 | 0.95 | 0.95 (≈5 h horizon) cannot credit pre-cooling at 05:00 for comfort at 14:00 |
| Comfort penalty | dense hinge, occupied only | sparse violation count | Dense gradient; hinge is in physical units (°C·h) |
| Reward scaling | running return normalisation (VecNormalize) | fixed scale | Daily returns range 10–300 early in training |
| Metrics occupancy | true o_t | sensor ô_t | Prevents scoring well by exploiting sensor errors |
