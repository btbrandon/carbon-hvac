# 7. Experimental Setup

*Author: Jaylon (§7.1). Sean's evaluation protocol is §7.2. Target ≈ 0.4 pages in the body. Renumber when merging.*

## 7.1 Metrics

Metrics are computed per test day (96 steps of 15 minutes) and aggregated as described in §7.2. Notation follows §4: E_t is HVAC electrical energy, CI_t carbon intensity, p_t the electricity price and D_t the comfort penalty on the 23–26 °C band. Comfort is always scored against true occupancy, not the sensor reading.

**Table 7.1** Evaluation metrics.

| Metric | Definition | Unit |
|---|---|---|
| Reward | Σ r_t = −Σ (λ_C·CI_t·E_t + λ_$·p_t·E_t + λ_K·D_t), with λ_C = 1, λ_$ = 0.5, λ_K = 5; the composite score all controllers are tuned on | per day |
| Energy | Σ E_t | kWh/day |
| Carbon | Σ CI_t·E_t | kg CO₂/day |
| Cost | Σ p_t·E_t | SGD/day |
| Discomfort | Σ D_t: distance outside the comfort band, integrated over occupied time | °C·h/day |
| Violation rate | Share of occupied steps with any discomfort | % |
| Max deviation | Largest occupied distance outside the comfort band | °C |
| Effective CI | Carbon ÷ energy: average CI paid per kWh, isolating timing from quantity | kg/kWh |
| Load shift | Share of HVAC energy used in steps with CI below that day's median (50% = no shifting) | % |
| Gap closed | (best tuned rule − controller) ÷ (best tuned rule − oracle), on reward or carbon | % |

Effective CI and load shift are the two metrics that separate *carbon-aware* control from merely *efficient* control: a controller that uses less energy at the same times lowers carbon but leaves both unchanged. Gap closed is measured against the best tuned rule rather than the fixed thermostat, because against the thermostat every tuned controller scores above 90% (PPO 97%, the best rule 94%) and the measure stops discriminating.
