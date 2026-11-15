# 2. Introduction

*Author: Patrick. Target about 0.75 page in the body. Literature search and first draft done with AI assistance from Claude; declare this in the AI-use statement. References are numbered in order of first appearance, and §3 continues the numbering, so these become references 1 to 7 of the merged report. Contributions checked against the merged §8 on 15 Nov 2026: robustness (§8.5) was run for the rules and oracle only, and constrained PPO (§8.7) was not completed, so neither is listed as a contribution. The extension we did complete, the oracle foresight decomposition (§8.7), is.*

Buildings and their construction accounted for 28% of global energy consumption in 2024, and running buildings emitted about 26% of energy-related CO₂ [1]. Air conditioners and fans use nearly a fifth of the electricity in buildings, and without efficiency gains the energy used for cooling will more than triple by 2050 [2]. In a typical Singapore office, cooling takes about 60% of the electricity [3].

Carbon intensity, the CO₂ emitted per kilowatt-hour generated, varies through the day with wind, sunshine and demand. Britain's averaged 125 g/kWh in 2024 and fell as low as 19 g/kWh [4]. Because a building's structure stores heat, an office can pre-cool while the grid is clean and then drift warmer through dirtier hours without leaving the comfort band. Its emissions then depend on when it cools as well as on how much.

This makes cooling a sequential decision problem. Cooling at 07:00 affects comfort hours later. The controller has to trade carbon against cost and comfort, and in our data price and carbon intensity are almost uncorrelated, so the cheapest hour is often not the cleanest. It sees occupancy only through a sensor. A rule-based controller holds a setpoint, the indoor temperature it tries to maintain, and reacts to current conditions. Model predictive control plans ahead but needs an accurate building model. A reinforcement learning agent learns a policy from simulated experience and needs no such model.

Singapore's grid offers little to shift. Natural gas generated 94% of its electricity in 2024 [5], and its 2024 grid emission factor of 0.402 kg CO₂/kWh is an annual average over all plants [6, 7]. We found no published hourly series, and we assume intensity varies little on a grid this dependent on one fuel. We therefore pair Singapore-like weather and prices with a synthetic carbon profile shaped like Britain's, described in §5. Our carbon results describe a variable grid, not what a Singapore office would save today.

We model one office zone with a first-order thermal network and pose its cooling as a Markov decision process, or MDP, with five power levels and a reward that charges carbon, cost and discomfort. We compare PPO, our main method, and DQN with a fixed thermostat, three rule families tuned under the same protocol as PPO, and a dynamic-programming oracle with perfect foresight, over five seeds and 92 held-out test days. Our contributions are:

- §4, §5 and §7 give a simulator and MDP on which PPO, DQN and exact dynamic programming solve the same problem, and two metrics, effective carbon intensity and load shift, that separate timing from energy reduction.
- §6.4 tunes the rules and PPO under one protocol. Before we did this, PPO's carbon saving appeared to be 11.3%, about three times the 3.7% it achieves against a tuned rule.
- §8.1 shows, with paired intervals over seeds, that three public holidays account for 72% of PPO's saving over the best rule, and §8.3 that at matched comfort the tuned rule beats PPO.
- §8.7, our extension, splits the oracle's foresight and finds that on ordinary days knowing future occupancy is worth about eight times as much reward as knowing future carbon intensity.
- §9 asks who sets the comfort weight, who bears the discomfort, and how a controller could game the occupancy sensor.

Against the best tuned rule, PPO cuts carbon by 3.7% while the share of occupied time outside the comfort band rises from 0.7% to 5.2%. On ordinary days its whole advantage is timing, worth about 1% of carbon. Even a planner with perfect foresight gains only about 2% of carbon from timing, because the zone's thermal mass stores little cooling.

## References

1. United Nations Environment Programme, Global Alliance for Buildings and Construction. Global status report for buildings and construction 2025–2026 [Internet]. Paris: UNEP; 2026 [cited 2026 Nov 15]. Available from: https://globalabc.org/sites/default/files/resources/2026-05/global-status-report-for-buildings-and-construction-2025_2026.pdf
2. International Energy Agency. The future of cooling: opportunities for energy-efficient air conditioning [Internet]. Paris: IEA; 2018 [cited 2026 Nov 15]. Available from: https://www.iea.org/reports/the-future-of-cooling
3. Building and Construction Authority. Super low energy building technology roadmap [Internet]. Singapore: BCA; [date unknown] [cited 2026 Nov 15]. Available from: https://www1.bca.gov.sg/docs/default-source/docs-corp-buildsg/digitalisation/sle_tech_roadmap_report.pdf
4. National Energy System Operator. Britain's electricity explained: 2024 review [Internet]. NESO; 2025 [cited 2026 Nov 15]. Available from: https://www.neso.energy/news/britains-electricity-explained-2024-review
5. Energy Market Authority. Fuel mix for electricity generation (yearly) [Internet]. Singapore: EMA; [date unknown] [cited 2026 Nov 15]. Available from: https://www.ema.gov.sg/content/dam/corporate/resources/statistics/files/pdf/Fuel-Mix-for-Electricity-Generation-Yearly.pdf.coredownload.pdf
6. National Climate Change Secretariat. Power [Internet]. Singapore: NCCS; [date unknown] [cited 2026 Nov 15]. Available from: https://www.nccs.gov.sg/singapores-climate-action/mitigation-efforts/power/
7. Energy Market Authority. Singapore energy statistics: technical notes and glossary [Internet]. Singapore: EMA; [date unknown] [cited 2026 Nov 15]. Available from: https://ema.gov.sg/resources/singapore-energy-statistics/technical-notes-glossary
