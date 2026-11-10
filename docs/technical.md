# Technical Documentation

How the system is built, how to run it, and where each report number comes from.

## 1. Architecture

```
data.py ──► Profiles (15-min, 1 year: T_out, solar, occupancy, price, CI)
              │  apply_regimes(): flat / TOU / anticorrelated / shifted CI scenarios
              ▼
thermal.py ─► env.py: CarbonHVACEnv (Gymnasium) ◄── simulate_interval(): shared dynamics + cost
              │                                        ▲
   ┌──────────┼──────────────┬────────────────┐      │ same function, vectorised over a T grid
   ▼          ▼              ▼                ▼      │
baselines.py  PPO / DQN    oracle.py (DP) ────────────┘
   (rules)   (scripts/train.py)
              │
metrics.py ─► per-episode metrics on fixed test episodes ─► stats.py ─► scripts/make_figures.py
```

The env, the rule-based controllers and the DP oracle all call one function, `simulate_interval`, for dynamics and cost. The oracle therefore optimises exactly the objective the agents are scored on. The test `test_oracle_dominates_baselines` checks this.

## 2. Simulator (`thermal.py`, `data.py`)

| Parameter | Value | Meaning |
|---|---|---|
| R | 0.4 K/kW | envelope resistance |
| C | 15 kWh/K | thermal capacitance (τ = RC = 6 h) |
| P_max | 14 kW (electric) | HVAC capacity, about 49 kW thermal |
| COP | 3.5 − 0.08·(T_out − 30), clipped to [2.5, 4.5] | efficiency falls on hot afternoons |
| Internal gains | 1.5 + 12·occupancy kW | base load + people/equipment/lighting |
| Solar gains | up to 6 kW, reduced by cloud | 07:00–19:00 |

Discretisation is exact under a zero-order hold, `T' = T_eq + (T − T_eq)·exp(−Δt/τ)`, so results do not depend on step size (tested).

**Data.** `synthetic_profiles(seed)` produces one year at 15 min. Each series has its own source:

- **Outdoor temperature:** Singapore-like, 25–33 °C, with 30% rainy days.
- **Occupancy:** ASHRAE-style weekday 08:00–18:00, with arrival/departure jitter, a lunch dip and 3% holidays.
- **Price:** USEP-like, 0.14–0.23 SGD/kWh with spikes.
- **CI:** UK-like, 0.03–0.30 kg/kWh. A wind component drifts on a ~8 h timescale, plus a midday solar dip and an evening peak.

Real data can replace any series by placing `data/{t_out,price,ci}.csv` (columns `timestamp,value`) and setting `data_source="csv"`.

**Split.** Train on days 0–272 (Jan–Sep), test on days 273–364 (Oct–Dec, 92 days).

## 3. Environment (`env.py`)

- **Observation:** 20 scaled features by default (8 current + 12 forecast). `env.feature_names` lists them, and the ablation flags `obs_ci`, `obs_price`, `obs_time` and `forecast_hours` remove groups.
- **Action:** `Discrete(5)`, the HVAC power level.
- **Reward:** see §4 of the report. `info` carries every component (`energy_kwh`, `carbon_kg`, `cost_sgd`, `comfort_dh`, `pen_reward`), so a wrapper can re-weight them; the Lagrangian extension does this.
- **Reset options:** `{"day": d, "noise_seed": s, "t_init": T}` reproduce an exact episode.
- **Scenario knobs (`EnvConfig`):** `price_regime`, `ci_regime`, `ci_shift_hours`, `occ_flip_prob`, `forecast_noise`, `reward_occupancy`, `control_minutes`.

## 4. Controllers

| Controller | File | Idea |
|---|---|---|
| Fixed setpoint | `baselines.py` | Ideal staged thermostat at 24 °C, 24/7 |
| Occupancy setback | `baselines.py` | Setpoint from `lead_hours` before 08:00 on workdays or when the sensor reads occupied, otherwise off |
| Carbon heuristic | `baselines.py` | Setback that cools to `low` when CI < θ and drifts to `high` when CI ≥ θ, optional pre-cool |
| Carbon-proportional | `baselines.py` | Setback with setpoint = base + gain·(CI − median CI) |
| PPO / DQN | `scripts/train.py` | stable-baselines3, MLP 64×64; `--hp key=value` overrides; settings saved in each run's `config.json` |
| Oracle | `oracle.py` | Backward induction over a 0.05 °C grid (401 points × 96 steps × 5 actions) with perfect foresight; one-step lookahead at execution |

**Tuning (same budget rules for every controller).** `eval_baselines.py` grid-searches all three rules (972 settings) on 60 training days, maximising mean reward. It also picks a matched-comfort variant (lowest carbon with ≤1% occupied-step violations). Chosen settings go to `results/baselines_tuned.json`; all three rules converge to setback at ≈25.9 °C. `tune_ppo.sh` tries 12 PPO settings × 2 seeds × 1.5M steps on the same 60 days. `select_ppo.py` picks the best by mean validation reward (`n_steps=768`), which is then trained with 5 seeds as tag `main_tuned`. No tuning run sees the test period.

## 5. Reproducing results

```bash
pytest                                   # simulator validation (§5 of report)
python scripts/eval_baselines.py         # tune all rules on train, evaluate rules + oracle on test (~1 min)
scripts/run_all.sh                       # 48 runs: main, λK sweep, ablations (~30 min on 8 cores)
scripts/tune_ppo.sh                      # PPO hyperparameter search, validation only (~25 min)
python scripts/select_ppo.py             # rank configs -> results/ppo_selected.json
# then: train.py --tag main_tuned --hp n_steps=768 --steps 1500000 for seeds 0-4
scripts/run_all.sh main -j 4             # just one group
python scripts/make_figures.py           # results/figures/*.png, results/tables/*
```

Each run writes `results/runs/<tag>/<algo>_s<seed>/{config.json,curve.csv,test.csv}`. The model goes to `results/models/` (gitignored).

**Statistics.** Test metrics are computed on all 92 test days, each with a fixed noise seed. Learning controllers: mean over days per seed, then mean and 95% bootstrap CI over 5 seeds. Headline differences are paired per day against the heuristic, then t-tested across seeds.

## 6. Compute

The env runs at about 36k steps/s on one core (Apple Silicon). An 800k-step run takes 2.7 min for PPO (8 vectorised envs) and 6.8 min for DQN when 7 runs share the machine. The full grid of 48 runs took 26 min wall-clock with 7 parallel jobs, 2.8 CPU-hours in total (Apple Silicon, 8 cores, CPU only).
