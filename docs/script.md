# Presentation Script (15 min + 5 min Q&A)

Thu 12 Nov, 2 pm. Speaker assignments follow the report split; swap freely. Timings are cumulative targets. Numbers come from `results/tables/` (5 seeds, 92 held-out test days).

---

## 1. Title · Patrick · 0:00–0:30

"We're Group 23. Our project asks whether a reinforcement-learning controller can make an office building's air-conditioning greener, not just by using less electricity, but by using it at the right time."

## 2. Why · Patrick · 0:30–1:45

"Buildings are about 30% of global final energy use, and in Singapore air-conditioning dominates commercial electricity. The carbon intensity of the grid changes hour to hour: when it's windy or sunny there's cleaner power, and in the evening peak it's dirtier. A building can exploit that, because its thermal mass stores coolness. You can cool a little earlier and coast later.

But that makes it a sequential decision problem: what you do at 6 am affects comfort at 2 pm. That's why we framed it as RL. Our question: can an RL controller beat a *well-tuned carbon-aware rule*, not just a dumb thermostat, without hurting comfort?"

## 3. What we built · Patrick · 1:45–2:30

"Five pieces: a simulator, a formal MDP, PPO and DQN against three rule-based baselines *and* an exact optimal planner, a careful statistical evaluation, and a responsible-AI analysis of what the agent actually learned. Ethan will start with the simulator."

## 4. Simulator · Ethan · 2:30–4:00

"We model one office zone as a 1R1C circuit: one thermal resistance for the walls, one capacitance for the building mass. The time constant is six hours. Left alone, the room takes six hours to close 63% of the gap to outside.

We discretise exactly rather than with Euler steps, so results don't depend on step size, and we validated the simulator with unit tests: the step response, energy use in the right range for a Singapore office, and the optimal planner never losing to any baseline.

On data: weather and occupancy are Singapore-like. For carbon intensity we deliberately substituted a UK-like grid profile. Singapore's grid is about 95% natural gas, so its carbon intensity is almost flat, and there would be nothing to shift. That's a finding in itself. We train on January to September and test on 92 unseen days in October to December."

## 5. The MDP · Brandon · 4:00–5:30

"The state has 20 features: indoor and outdoor temperature, an occupancy sensor, price, carbon intensity, time of day, and forecasts up to 8 hours ahead. The action is one of five HVAC power levels. We chose discrete actions on purpose: it means PPO, DQN *and* an exact dynamic-programming solver all work on exactly the same problem.

The reward is a weighted cost: carbon, plus half the electricity cost, plus a comfort penalty in degree-hours outside 23 to 26 °C, charged only when people are actually there. γ is 0.99, about a 25-hour effective horizon at 15-minute steps, which is enough for the agent to connect pre-cooling at 5 am with comfort at 2 pm.

The key observation: carbon and cost are both proportional to energy. So the only way to be *carbon*-aware rather than just *efficient* is to shift *when* you consume. And in our data price and carbon are nearly uncorrelated, r = −0.07, so the cheapest hour is not the cleanest."

## 6. Controllers · Jaylon (+ Brandon) · 5:30–6:45

**Jaylon:** "Our rule-based controllers: a fixed thermostat; a setback that only cools during working hours; a carbon heuristic that cools deeper when the grid is clean and lets the room drift warmer when it's dirty; and a carbon-proportional rule whose setpoint rises smoothly with carbon intensity.

The important part is that every rule was **tuned**: 972 settings in total, scored on training days with exactly the same reward PPO maximises. Our first results used untuned rules, and PPO looked 11% better. Once the rules were tuned fairly, most of that gap disappeared. We'll come back to that."

**Brandon:** "PPO got the same treatment: we searched 12 hyperparameter settings with 2 seeds each on training days, then trained the winner, a larger rollout batch, with 5 fresh seeds. And the oracle: since the dynamics are deterministic given the weather, we solve each day exactly by backward induction with perfect foresight. That's the best any controller could possibly do."

**Jaylon:** "Everything is tested once on 92 unseen days, with paired bootstrap confidence intervals."

## 7. Main result · Sean · 6:45–8:15

"Here's the main table. The fixed thermostat emits about 19 kg of CO₂ a day. Simple setback at 24 °C brings it to 11. Then just **tuning that setpoint**, to 25.9 °C, cuts another 23%, to 8.5 kg. That's the biggest single gain in the whole project, and it needs no RL.

Tuned PPO gets to 8.21: **3.7% less carbon than the best tuned rule**, with a tight confidence interval, and 1.1% better on the reward it optimises, significant at p = 0.03. Cost falls about 4%. The oracle is at 7.83, so PPO closes about a fifth of the remaining gap. The price is comfort: violations are more frequent, but tiny, under a tenth of a degree-hour per day."

## 8. Learning curves · Sean · 8:15–9:00

"PPO learns smoothly and its five seeds agree within 1.3%. DQN learns about as fast but collapses intermittently. Notice that on these 16 training days PPO ends level with the tuned rule. It's ahead on the larger validation and test sets, but the edge is small, and we want to be upfront about that."

## 9. Policy trace · Sean · 9:00–10:15

"What did PPO learn? Both controllers hold the room high in the band, around 25.7 to 25.9 °C, the same insight the tuning found. The difference is in the details: PPO cools a little harder during the clean-grid dips around 10 am and midday, then coasts up to 26 °C as carbon rises towards the evening. So there is some genuine load-shifting: the carbon PPO pays per kWh is 0.115 versus 0.118 for the rule."

## 10. Where is the headroom? · Sean (+ Brandon) · 10:15–11:30

"We asked where the oracle's advantage actually comes from by taking away one piece of its foresight at a time.

- Take away knowledge of future carbon: it barely changes, worth 0.08 per day.
- Take away knowledge of when people arrive and leave: it drops by about 1.0, below even the tuned thermostat.

So in this building **the carbon signal isn't the bottleneck; predicting occupancy is**. That's consistent with everything else: tuning switched off every carbon-reactive feature of the rules, and removing carbon from PPO's inputs costs only about 1%. A single room with a six-hour time constant is a leaky battery."

## 11. Pareto · Sean → Patrick · 11:30–12:30

"Sweeping the comfort weight traces PPO's trade-off curve; the green line sweeps the thermostat setpoint. Inside the comfort band the two curves overlap, so at matched comfort it's essentially a tie. PPO's extra savings come from accepting a bit more discomfort. How much discomfort is acceptable is a value judgement, which brings us to responsible AI."

## 12. Responsible AI · Patrick · 12:30–13:45

"Five points.

1. **The weights are a value choice.** Who sets them: facility managers or occupants? The Pareto front makes the choice visible.
2. **Specification gaming.** PPO sits right at the edge of what the reward allows. Comfort only counts when the room is occupied, so a faulty sensor could be exploited. That's why we always *score* comfort on true occupancy.
3. **Distribution.** More than half of PPO's violations happen between 5 and 6:30 pm, so the people who stay late bear the discomfort. An average hides that.
4. **Honest baselines.** Our own first result overstated the gain about threefold because the baseline wasn't tuned. Weak baselines are a quiet way to mislead, so we report the fair comparison.
5. **Transparency.** A tuned thermostat gets most of the benefit and anyone can explain it. Choosing it over RL can be the responsible deployment choice."

## 13. Limitations / next · Patrick · 13:45–14:30

"Limits: a single zone, a simplified model, synthetic data profiles, and a small RL margin. The DQN, sweep and ablation runs also used default PPO settings. For the report: occupancy prediction, since that's where the headroom is; robustness tests, including shifting the carbon signal to check the agent reads it rather than memorising the clock; constrained PPO, which treats comfort as a constraint instead of a weight; and a battery, to give the agent a real lever for shifting."

## 14. Takeaways · Patrick · 14:30–15:00

"Four takeaways. Tune your baseline first: one setpoint gives 23%, and skipping it inflated our RL result about three times. Tuned PPO still wins, by 3.7% carbon, with a little real load-shifting. In one room the carbon signal is worth little; occupancy foresight is the real headroom. And it took an oracle plus fair tuning to see all of this. Thank you, happy to take questions."

---

## Likely Q&A (owner in brackets)

| Question | Answer |
|---|---|
| Is a 3.7% gain worth RL? (Sean / Patrick) | Honestly, maybe not in one room: a tuned thermostat gets most of it. The value of the study is showing where the headroom is (occupancy) and when RL would pay (storage, multi-zone). |
| How did you make the comparison fair? (Jaylon / Brandon) | Same training days, same reward, tuned budgets for both (972 rule settings; 12 PPO settings × 2 seeds), a single final test on held-out days, paired CIs over 5 seeds. |
| Why not continuous actions / SAC? (Brandon) | Discrete actions let one problem admit PPO, DQN and an exact DP optimum. Five levels at 15-min steps already give 17 effective power levels per hour. |
| Why PPO over DQN? (Brandon) | PPO is stable across seeds (±1.3%); DQN collapses intermittently and ends 7% worse on reward. DQN wasn't hyperparameter-tuned, which we state as a limitation. |
| How is the oracle optimal? (Brandon) | Given the realised weather and occupancy, the dynamics are deterministic. Backward induction on a 0.05 °C grid with linear interpolation of V; the only approximation is the grid. Unit-tested to dominate every baseline. |
| What did tuning change in PPO? (Brandon) | The rollout length doubled (n_steps 384 → 768 per env), giving larger, less noisy batches. γ = 0.995, smaller learning rates and bigger networks did not help. |
| Why γ = 0.99? (Brandon) | 1/(1−γ) = 100 steps ≈ 25 h, enough to credit pre-cooling hours ahead. 0.995 was tested in tuning and was worse. |
| Isn't 5% violation bad? (Sean) | It's frequent but tiny (0.08 °C·h/day) and concentrated 17:00–18:30. At matched comfort PPO roughly ties the thermostat; constrained PPO is the principled fix. |
| Why not real Singapore data? (Ethan) | SG CI is nearly flat (gas grid), so there's nothing to shift. Loaders for real CSVs exist; synthetic profiles are calibrated to published ranges. |
| Is the agent just following the clock? (Sean) | Mostly it follows the schedule, as the no-CI ablation shows, but it pays 0.115 vs 0.118 kg/kWh, so it reacts to CI a little. The shifted-CI test in the report checks this directly. |
| Average vs marginal emissions? (Patrick) | We use average intensity. Marginal factors could change which hours are "clean"; flagged as a limitation. |
