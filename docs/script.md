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

## 6. Controllers · Jaylon (+ Brandon for the last three rows) · 5:30–6:45

**Jaylon:** "Our baselines go from weak to strong. A fixed 24 °C thermostat. An occupancy setback that only cools during working hours. And a carbon heuristic: it cools deeper when carbon is low and lets the room drift warmer when carbon is high, with the threshold tuned on the training months. Beating that last one is what counts."

**Brandon:** "For learners we use PPO, with the clipped objective, GAE, an entropy bonus and return normalisation, and DQN for comparison. Then the oracle: since the dynamics are deterministic given the weather, we solve the day exactly by backward induction over a fine temperature grid with perfect foresight. That's the best any controller could do, so we can say how much of the achievable saving we capture."

**Jaylon:** "Everything is evaluated with 5 seeds on the same 92 test days, with paired bootstrap confidence intervals."

## 7. Main result · Sean · 6:45–8:15

"Here's the main table. The fixed thermostat emits about 19 kg of CO₂ a day. Setback roughly halves that, and the carbon heuristic brings it to 9.3. PPO gets to 8.26: an **11.3% reduction versus the heuristic**, with a 95% interval of about 10.7 to 12%, significant across all five seeds. The oracle is at 7.83, so PPO captures 71% of the remaining gap. Cost falls by about 12% too.

The price is comfort. PPO is slightly outside the band on 5.5% of occupied 15-minute steps, but the magnitude is tiny: about a tenth of a degree-hour per day, mostly in the late afternoon."

## 8. Learning curves · Sean · 8:15–9:00

"PPO passes the heuristic after about 300,000 steps, and the five seeds end within 1.2% of each other. DQN reaches a similar level but collapses intermittently, which is why we prefer PPO here."

## 9. Policy trace · Sean · 9:00–10:15

"What did PPO actually learn? On this test day, the blue line holds the room at about 25.7 °C, just under the top of the band, all day. It doesn't pre-cool. The green heuristic dives to 23.5 °C whenever carbon dips. But with a six-hour time constant, a colder room just leaks heat in faster, so that stored cooling is mostly wasted. PPO discovered that the cheapest strategy in this building is not to store cold."

## 10. Is it carbon-aware? · Sean (+ Brandon) · 10:15–11:30

"So is PPO really carbon-aware? We tested it directly.

- If we remove carbon intensity from the agent's inputs, carbon goes up by only 1.3%.
- The average carbon intensity it pays per kWh is the same as the heuristic's, while the oracle pays noticeably less.
- Removing the clock hurts much more.

So PPO's gain is mostly from using less energy, not from timing. Real shifting is possible, as the oracle shows, but the headroom is small because this room is a leaky battery. That's exactly why the battery extension is next."

## 11. Pareto · Sean → Patrick · 11:30–12:30

"Changing the comfort weight traces this trade-off curve. Low weight: 4.6 kg a day but 8 degree-hours of discomfort. Very high weight: almost perfect comfort, but then the heuristic is actually *not* beaten. So whether PPO 'wins' depends on how much discomfort you accept, and that's a value judgement, which leads into responsible AI."

## 12. Responsible AI · Patrick · 12:30–13:45

"Four points.

1. **The weights are a value choice.** Who sets them: facility managers or occupants? The Pareto front makes the choice visible.
2. **Specification gaming.** The agent sits right at the edge of what the reward allows. Comfort is only charged when the room is occupied, so a faulty sensor could be exploited. That's why we always *score* comfort on true occupancy.
3. **Distribution.** Over 40% of violating steps fall between 5 and 6:30 pm: PPO eases off as the day ends, so the people who stay late bear the discomfort. An average hides that.
4. **Transparency.** The learned policy is simple enough to explain to a facility manager, which supports human override and a rule-based fallback."

## 13. Limitations / next · Patrick · 13:45–14:30

"Limits: a single zone, a simplified model and synthetic data profiles. For the report we're adding robustness tests: shifting the carbon signal in time, to check whether the agent reads it or memorises the clock, plus noisy occupancy and different tariff regimes. We're also adding constrained PPO, which treats comfort as a hard constraint instead of a weight, and possibly a battery to give the agent a real lever for shifting."

## 14. Takeaways · Patrick · 14:30–15:00

"PPO beats a tuned carbon-aware rule by 11% and gets 71% of the way to optimal. But it wins by being frugal at the edge of comfort, not by chasing clean hours. We only know that because we compared against an oracle and ran ablations. Thank you. Happy to take questions."

---

## Likely Q&A (owner in brackets)

| Question | Answer |
|---|---|
| Why not continuous actions / SAC? (Brandon) | Discrete actions let one problem admit PPO, DQN and an exact DP optimum. Five levels at 15-min steps already give 17 effective power levels per hour. |
| Why PPO over DQN? (Brandon) | Same final level, but PPO is far more stable across seeds (±1.2%) while DQN has collapses. PPO is also cheap here because the simulator is fast (36k steps/s). |
| How is the oracle optimal? (Brandon) | Given the realised weather and occupancy, dynamics are deterministic. Backward induction on a 0.05 °C grid with linear interpolation of V; the only approximation is grid interpolation. Unit-tested to dominate every baseline. |
| Why γ = 0.99? (Brandon) | 1/(1−γ) = 100 steps ≈ 25 h, enough to credit pre-cooling hours ahead. 0.95 would be about 5 h. |
| Isn't 5.5% violation bad? (Sean) | Small magnitude (0.1 °C·h/day), concentrated 17:00–18:30 and in the afternoon heat peak. The Pareto sweep shows tighter settings, and constrained PPO is the principled fix. |
| Why not real Singapore data? (Ethan) | SG CI is nearly flat (gas grid), so there's nothing to shift. Loaders for real CSVs exist; synthetic profiles are calibrated to published ranges. |
| Is the agent just following a schedule? (Sean) | Partly, yes: the no-CI ablation shows it. The shifted-CI test in the report checks this directly. |
| Average vs marginal emissions? (Patrick) | We use average intensity. Marginal factors could change which hours are "clean"; flagged as a limitation. |
