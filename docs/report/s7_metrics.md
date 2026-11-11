# 7. Experimental Setup

*Authors: Jaylon (§7.1), Sean (§7.2). Target ≈ 0.4 pages (§7.1) + 0.6 pages (§7.2) in the body. Renumber when merging.*

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

## 7.2 Evaluation protocol

**Data split.** The simulator covers one year at 15-minute resolution (§5). Days 0–272 (January–September) are used for training and for every tuning decision; days 273–364 (October–December) form the test period. The 92 test days comprise 63 workdays, 26 weekend days and 3 weekday public holidays (days 273, 277 and 337), on which nobody arrives. Each controller is evaluated on the test period once, after all choices are fixed.

**Fixed test episodes.** One episode is one day (96 steps). Each test day is paired with a fixed noise seed, so every controller faces identical weather, occupancy, price and CI on a given day, and controllers can be compared day by day. Learned policies act greedily at test time.

**Tuning data.** All rule parameters and PPO hyperparameters were selected on the same 60 training days by mean daily reward (§6.4). DQN, the λ_K sweep and the ablations use the default PPO hyperparameters at 800k steps; only the main PPO arm uses the tuned configuration at 1.5M steps. Comparisons inside §8.3 and §8.4 are therefore always against the default-configuration PPO, never the tuned one.

**Seeds and confidence intervals.** Each learned controller is trained with 5 independent seeds (0–4), and the seed is the unit of replication: metrics are averaged over test days within a seed, and we report the mean over seeds with a 95% percentile bootstrap interval (10,000 resamples) [1]. Rules and the oracle are deterministic given a day, so their intervals are bootstrapped over the 92 days and reflect day-to-day variation only. The two kinds of interval are not comparable in width, which is why every headline comparison is paired.

**Paired comparisons.** For a learned controller A and reference B, we compute for each seed the mean per-day difference A − B over the same test days, giving 5 paired differences. We report their mean, a bootstrap 95% CI, and a one-sample t-test against zero. Pairing removes the large day-to-day variation shared by both controllers (weather, occupancy), which dominates unpaired intervals. Families of comparisons (the four ablations) are Holm–Bonferroni corrected [2]. With 5 seeds we treat p-values as supporting evidence and lean on effect sizes and intervals, following the reporting advice of [3, 4].

**Aggregating the timing metrics.** Effective CI and load shift are ratios, so they are aggregated energy-weighted: effective CI = Σ carbon / Σ energy over all test days, not the mean of daily ratios. Weekend and holiday days consume (almost) no energy, and averaging per-day ratios would score those days as 0 and make the result depend on how many idle days a controller happens to have. The load-shift baseline is also not 50%: any controller that cools only during occupied hours already places about 60% of its energy below the daily median CI, because the midday solar dip falls inside the working day. Load shift is therefore read relative to the tuned rule (60.2%) and the oracle (67.3%).

**Reference controller and day-type breakdown.** The protocol comparison is against the best tuned rule, setback at 25.9 °C (§6.1). Inspecting per-day test results showed that this rule follows its workday schedule on the 3 public holidays and cools an empty zone, while the learned controllers react to the occupancy sensor and do not. Because this is a post hoc observation, we keep the protocol result as the primary comparison and add two secondary views: (i) the same comparison restricted to the 89 non-holiday days, and (ii) a *calendar setback*, identical to the tuned setback but switched off on public holidays, as any building-management schedule would be. The calendar setback needs no new tuning or test-set choice: it equals the tuned rule on every non-holiday day.

**Gap closed and sample efficiency.** Gap closed (Table 7.1) is computed on the same days as the comparison it accompanies. Sample efficiency is the first training step at which the evaluation return on 16 fixed training days, smoothed over 5 evaluations, reaches 90% of its total improvement from the first to the final evaluation.

**Scenarios.** Table 7.2 lists the evaluation scenarios. All scenario knobs are evaluation-time changes to the environment (`EnvConfig`) and require no retraining.

**Table 7.2** Evaluation scenarios.

| Scenario | What changes | Purpose | Status |
|---|---|---|---|
| Main | Held-out test period, perfect sensor and forecasts | Main comparison (§8.1–8.4, 8.6) | Done |
| Shifted CI | CI profile shifted by k hours (`ci_shift_hours`) | Does the policy read the CI signal or rely on the clock? | Pending (§8.5) |
| Sensor noise | Occupancy-sensor flip probability ε ∈ {0.1, 0.2} (`occ_flip_prob`) | Robustness of the POMDP approximation (§4.3) | Pending (§8.5) |
| Price / CI regimes | Flat or time-of-use price; flat or price-anticorrelated CI | Isolates carbon-awareness from cost-awareness | Pending (§8.5) |

**Compute.** The environment runs at about 36,000 steps/s on one CPU core. A PPO run takes about 2.6 min at 800k steps and 4.7 min at 1.5M steps; a DQN run about 6.8 min at 800k steps. The experiment grid (main PPO and DQN, the λ_K sweep and the four ablations; 48 runs) took 26 min wall-clock with 7 parallel jobs (2.8 CPU-hours), the PPO hyperparameter search (24 runs) about 25 min, and the 5 tuned PPO seeds about 5 min. Rule tuning (972 configurations) and the oracle evaluations take about 1 min each. In total, about 5 CPU-hours on an 8-core Apple Silicon laptop, CPU only.

**Reproducibility.** Every run stores its seed, environment configuration and hyperparameters in `config.json`. All tables and figures in §8 are regenerated from the stored per-day test results by `scripts/make_figures.py` and `scripts/holiday_breakdown.py` ([REPO LINK]).

### References (§7.2)

1. Efron B, Tibshirani RJ. An introduction to the bootstrap. New York: Chapman & Hall; 1993.
2. Holm S. A simple sequentially rejective multiple test procedure. Scand J Stat. 1979;6(2):65–70.
3. Henderson P, Islam R, Bachman P, Pineau J, Precup D, Meger D. Deep reinforcement learning that matters. In: Proceedings of the AAAI Conference on Artificial Intelligence. 2018;32(1).
4. Agarwal R, Schwarzer M, Castro PS, Courville AC, Bellemare MG. Deep reinforcement learning at the edge of the statistical precipice. In: Advances in Neural Information Processing Systems 34 (NeurIPS); 2021.
