# Project Plan: Carbon-Aware Smart Building Energy Controller (Group 23)

CS4246/5446, Semester 1 AY2026/27. Brandon owns all the code, plus §4 and the PPO/algorithms half of §6.

**Deadlines:** presentation in the week of 10 Nov (exact slot: **TBC, find out today**), report due **Sun 15 Nov, 23:59**.

---

## 1. Technical decisions (locked unless a test proves them wrong)

| Decision | Choice | Why |
|---|---|---|
| Simulator | Our own 1R1C thermal model as a Gymnasium env | Fast (thousands of steps/s), transparent, easy to validate |
| Step Δt | 15 min, so 96 steps per 1-day episode | Resolves pre-cooling; ablation compares against 1 h |
| Action | Discrete, 5 HVAC power levels {0, 25, 50, 75, 100}% of P_max | Lets PPO, DQN *and* exact DP all run on the same problem |
| Oracle / upper bound | Finite-horizon backward induction (value iteration) on a discretised T_in grid, with perfect foresight | Exact optimum for the deterministic problem; ties directly to course material |
| RL library | stable-baselines3 (PPO, DQN), sb3-contrib (RecurrentPPO if needed) | Reliable, quick, well documented |
| Comparison algorithm | DQN | The action space is discrete |
| Tier 2 extension | Lagrangian constrained PPO (comfort treated as a constraint) | Most direct answer to "the λ weights are a value judgement"; feeds §9 |
| Data | Loaders for real CSVs (weather, USEP price, CI), with calibrated synthetic profiles as fallback | Data fetching must not block the code; the substitution is stated in the report |
| Carbon intensity | UK ESO carbon-intensity API (Singapore's CI is almost flat because the grid is ~95% gas) | Without CI variation there is nothing to shift; this is itself a finding for the report |
| Seeds | 5 per learning arm (0–4); rule-based controllers evaluated on the same fixed episodes | Matches the §7 protocol |

## 2. Repo layout

```
carbon-hvac/
├── PLAN.md                  this file
├── README.md                setup + how to reproduce every figure
├── requirements.txt
├── src/carbonhvac/
│   ├── thermal.py           1R1C model, HVAC/COP
│   ├── data.py              weather / price / CI / occupancy loaders + synthetic
│   ├── env.py               Gymnasium env, reward, observation config (for ablations)
│   ├── baselines.py         fixed setpoint, setback, carbon heuristic
│   ├── oracle.py            backward-induction DP
│   ├── lagrangian.py        constrained-PPO wrapper (extension)
│   ├── metrics.py           kWh, kgCO2, $, degree-hours, violation %, CI-per-kWh, load-shift %
│   └── stats.py             bootstrap CI, Welch, Holm-Bonferroni
├── configs/                 YAML per experiment (main, sweep, ablations, scenarios)
├── scripts/
│   ├── train.py             --algo --seed --config
│   ├── evaluate.py          runs all controllers on fixed test episodes → results/*.csv
│   ├── run_all.sh           launches the full experiment grid in parallel
│   └── make_figures.py      every report/slide figure from results/
├── tests/                   simulator validation (step response, bounds, energy magnitude)
├── results/                 csv + figures (models/ gitignored)
└── docs/
    ├── technical.md         technical documentation of the system
    ├── report/              section drafts (§4, §6 by Brandon) → handed to Patrick
    ├── slides/              Marp markdown deck → exported to PDF/PPTX
    └── script.md            presentation speaker script with timings (~15 min)
```

## 3. Experiment grid (what §8 needs)

| Exp | Runs | Report section |
|---|---|---|
| Main: PPO, DQN × 5 seeds; 3 baselines; oracle | 10 training runs | §8.1, §8.2 |
| λ_C / λ_K sweep: ~8 settings × 3 seeds | 24 | §8.3 Pareto |
| Ablations: no-CI, no-forecast, no-time, Δt=1h × 5 seeds | 20 | §8.4 |
| Scenarios (eval only): price regimes, CI regimes, sensor noise, shifted CI | 0 (eval) | §8.5 |
| Extension: Lagrangian PPO × 5 seeds | 5 | §8.7 |

**About 60 training runs.** Each is about 300k steps, which should take roughly 3–6 min on CPU, so about 1–2 h wall-clock in parallel on a laptop. This is estimated; measure it on Wednesday and fill in the §7 compute budget.

## 4. Schedule

Presentation: **Thu 12 Nov, 2 pm.** Everything the slides need must exist by Wed night.

| Stage | When | Code | Docs / presentation |
|---|---|---|---|
| 1 | **Tue 10** | Thermal model, data (synthetic), env, baselines, oracle DP, simulator tests, baseline table | Send Ethan the state/action/reward spec |
| 2 | **Tue 10 night → Wed 11 am** | PPO + DQN training, eval pipeline; launch main 5-seed runs + λ sweep + CI ablation overnight | §4 draft |
| 3 | **Wed 11** | `make_figures.py`: main table, learning curves, Pareto, CI ablation, one-day trace | Slides + speaker script; `technical.md` |
| 4 | **Thu 12 am** | Bug fixes only | Rehearse; **present 2 pm** |
| 5 | **Thu 12 pm → Fri 13** | Remaining ablations, scenario evals, Lagrangian PPO extension, real data if obtained | §6 PPO draft |
| 6 | **Sat 14** | Freeze code, final reruns, all figures to Sean/Patrick | Hand §4/§6 to Patrick; appendix tables |
| 7 | **Sun 15** | — | Final read; Patrick submits PDF |

The slides show the main comparison, the Pareto front and the CI ablation. Robustness scenarios and the extension are presented as "in progress / in the report".

## 5. Handoffs

| To | What | By |
|---|---|---|
| Ethan (§5) | Env interface, thermal parameters, validation test outputs | Wed |
| Jaylon (§6 baselines, §7 metrics) | `baselines.py` + `metrics.py` definitions, tuned heuristic threshold | Wed |
| Sean (§7 protocol, §8) | `results/*.csv`, figures, learning-curve logs | Thu (main), Sat (all) |
| Patrick (§9, integration) | §4 + §6 text, figure files, compute budget numbers | Sat |

## 6. Risks

| Risk | Mitigation |
|---|---|
| PPO doesn't beat the carbon heuristic | That's a reportable result; check reward scaling and observation normalisation first, and present the gap to the oracle honestly |
| Real data not obtained in time | Synthetic profiles calibrated to published SG/UK ranges; stated as a limitation |
| Agent memorises the clock instead of reading CI | The shifted-CI test and no-CI ablation exist exactly to show this |
| Time | Extension (Lagrangian) is the first thing to cut; Tier 1 analysis is not cut |

## 7. Progress log

**Tue 10 Nov (stages 1–3 done)**
- Simulator, env, baselines, DP oracle, 10 validation tests passing.
- 48 runs complete (main 5×PPO + 5×DQN, λK sweep 6×3, 4 ablations ×5): 26 min wall, 2.8 CPU-h.
- Headline: PPO −11.3% carbon vs carbon heuristic (95% CI −12.0 to −10.7%), 71% of the heuristic→oracle gap; 5.5% occupied-step violations (0.10 °C·h/day), concentrated 17:00–18:30.
- Key finding: PPO's gain is energy reduction, not time-shifting (effective CI 0.117 vs heuristic 0.116, oracle 0.109; no-CI ablation only +1.3%).
- Slides (`docs/slides/slides.md`), script (`docs/script.md`), §4 draft, `docs/technical.md` written.

**Next (stage 5, after Thursday):** scenario evals (shifted CI, sensor noise, price/CI regimes), Lagrangian PPO, §6 PPO draft, real data if obtained.
