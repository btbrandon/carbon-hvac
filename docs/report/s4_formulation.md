# 4. Problem Formulation

*Author: Brandon. Target length ≈ 1.25 pages. Citations in Vancouver style, numbered at the end; renumber when merging.*

We control the cooling of a single office zone over one day. Each episode starts at 00:00 and runs for H = 96 decision steps of Δt = 15 min. At every step the controller observes the zone and the grid, chooses an air-conditioning power level, and the zone temperature evolves according to the thermal model in §5.

## 4.1 Markov decision process

We formulate the problem as an MDP (S, A, P, R, γ).

**State.** The observation at step t is

  s_t = ( T_in, T_out, ô_t, p_t, CI_t, sin(2πh/24), cos(2πh/24), w_t, { T̂_out, p̂, ĈI }_{t+τ}, τ ∈ {1, 2, 4, 8} h ),

where:
- T_in and T_out are the indoor and outdoor temperatures (°C);
- ô_t is the occupancy *sensor* reading;
- p_t is the wholesale electricity price (SGD/kWh) and CI_t the grid carbon intensity (kg CO₂/kWh);
- h is the hour of day, encoded on the unit circle so that 23:45 and 00:00 are adjacent, and w_t flags a working day;
- the last block holds forecasts of outdoor temperature, price and CI at four lead times.

That gives 8 current and 12 forecast features, 20 in total. Each feature is affinely scaled to roughly [−1, 1] with fixed constants rather than a running normaliser, so every run and every controller sees an identical observation.

**Action.** a_t ∈ {0, 1, 2, 3, 4} selects electrical HVAC power P_t = (a_t / 4)·P_max, with P_max = 14 kW. We use discrete levels for three reasons:

1. Commercial units are typically staged.
2. One discrete action space admits PPO, DQN *and* an exact dynamic-programming solution (§6), so every controller solves exactly the same problem.
3. The fine time step offsets the coarse power resolution: four 15-min steps can realise 17 distinct average power levels within an hour.

**Transition.** T_in follows the 1R1C model of §5, discretised exactly under a zero-order hold:

  T_{t+1} = T_eq + (T_t − T_eq)·exp(−Δt / RC),  T_eq = T_out + R·( Q_int(o_t) + Q_sol − COP(T_out)·P_t ).

Given the exogenous series (weather, occupancy, price, CI), the transition is deterministic. Stochasticity enters only through the day drawn, the initial temperature T_0 ~ U(25.5, 28) °C, and, in the robustness experiments, sensor and forecast errors.

**Reward.** The reward is the negative of a weighted cost:

  r_t = −[ λ_E·E_t + λ_C·CI_t·E_t + λ_$·p_t·E_t + λ_K·D_t ],  E_t = P_t·Δt,

  D_t = 1[o_t]·( max(0, T_{t+1} − 26) + max(0, 23 − T_{t+1}) )·Δt.

- D_t is a hinge penalty, in °C·h, on the comfort band [23, 26] °C. This spans the 23–25 °C design range of SS 553 [1] and the 26 °C upper limit commonly applied in Singapore offices.
- The penalty is charged only when the zone is **truly** occupied (o_t, not the sensor ô_t). §9 explains why this matters for specification gaming.
- Default weights are λ_E = 0, λ_C = 1 per kg, λ_$ = 0.5 per SGD and λ_K = 5 per °C·h. Carbon is the primary objective and cost a secondary one. λ_K is swept in §8.3 to expose the carbon–comfort trade-off.

**Discount.** γ = 0.99. At 15-min steps this gives an effective horizon 1/(1−γ) = 100 steps ≈ 25 h, long enough to credit cooling at 07:00 for comfort in the afternoon. Episodes end at a fixed horizon with no terminal value. The zone is unoccupied after about 18:00, so the end-of-day temperature carries no cost in reality either.

## 4.2 Carbon-awareness is temporal load shifting

Carbon (CI_t·E_t) and cost (p_t·E_t) are both linear in energy, so any controller that saves energy saves both. What distinguishes a *carbon-aware* controller is **when** it consumes: for a fixed total energy, carbon is minimised by moving consumption into low-CI steps.

The zone's only storage is its thermal mass, C = 15 kWh/K. Pre-cooling by 3 K banks about 45 kWh of cooling, roughly 13 kWh of electricity at COP 3.5, and with τ = RC = 6 h much of it leaks back within a few hours. This bounds how much load can move, and §8 shows the bound is tight.

We therefore report two quantities that separate energy reduction from shifting:
- the **effective CI paid**, Σ CI_t·E_t / Σ E_t;
- the **share of HVAC energy consumed in below-median-CI steps**.

In our data the correlation between price and CI is r = −0.07. Price peaks with afternoon demand, while CI falls with midday solar and windy periods, so the cost-optimal and carbon-optimal schedules genuinely differ.

## 4.3 Partial observability

The true state includes the occupancy o_t and the future exogenous values, neither of which the agent observes directly. It sees a sensor ô_t, which flips with probability ε, and forecasts with multiplicative error growing with lead time. The problem is therefore a POMDP.

We approximate it by **feature augmentation**:
- the forecasts stand in for the hidden future exogenous state;
- the time features let the policy infer the occupancy *schedule*, which carries most of the information a noisy sensor would.

The main experiments use a perfect sensor and perfect forecasts (ε = 0, zero forecast error); robustness to both is tested in §8.5. A recurrent policy is the natural next step for heavier noise (§10).

The oracle (§6) is given the true o_t and the true future series, which is what makes it an upper bound. Removing each piece of its foresight in turn shows that **occupancy foresight, not carbon foresight, accounts for almost all of the oracle's advantage** over the best rule (§8). This identifies the hidden variable that matters most.

## 4.4 Design choices

| Choice | Value | Alternative | Reason / evidence |
|---|---|---|---|
| Δt | 15 min | 1 h | CI dips last 1–3 h. Ablation at Δt = 1 h: carbon 8.40 vs 8.26 kg/day and 1.7× the discomfort (§8.4) |
| Horizon | 1 day (96 steps) | 1 week | The occupancy cycle is daily; longer episodes slow learning with no new dynamics |
| γ | 0.99 | 0.995 | 0.99 spans a full day. 0.995 scored worse in tuning (validation reward −11.96 vs −11.82, §6.4) |
| Comfort penalty | dense hinge, occupied only | sparse violation count | Dense learning signal, in physical units (°C·h) |
| Reward scaling | running return normalisation | fixed scale | Early returns span −10 to −300 per day |
| Comfort scoring | true occupancy o_t | sensor ô_t | A controller cannot score well by exploiting sensor errors |

## References

1. SPRING Singapore. SS 553:2009 Code of practice for air-conditioning and mechanical ventilation in buildings. Singapore: SPRING Singapore; 2009.
