# 10. Limitations and Future Work

*Author: Patrick. Target ≈ 0.3 page in the body. No new results; every number is taken from §8. Each item states the limitation, what it means for our conclusions, and the fix, which is our future work. The cross-references to §3 assume §2–3 are merged.*

1. **A simple, uncalibrated building.** Our ranking is for one 1R1C zone with assumed parameters, daily resets and sensible heat only, ignoring the humidity that matters in the tropics (§5). A calibrated multi-zone model with latent loads, such as BOPTEST (§3.3), then shadow-mode trials in a real building, would test whether it transfers.
2. **No storage beyond thermal mass.** With τ = 6 h and a 3 K band, carbon timing is worth about 2% of carbon even to the oracle (§8.1), so learning's small value may be specific to this building. A battery or chilled-water tank, or several zones, would add a lever worth learning, the only case in which §9 accepts RL's lower explainability.
3. **Idealised occupancy information.** The main runs use a perfect sensor and forecasts (§4.3), yet occupancy holds most of the oracle's headroom and PPO's holiday response gives 72% of its saving (§8.1). Random sensor errors raise a sensor-reactive rule's carbon by 24–36% (§8.5), PPO was not re-tested under noise, and a stuck sensor (§9) is untested; occupancy prediction and a recurrent policy for noisy inputs come next.
4. **A synthetic, average carbon signal.** The CI profile is synthetic, UK-shaped and an average rather than a marginal intensity (§3.1, §5.2), so our figures are not Singapore savings, and marginal rates could rank hours differently. Measured hourly average and marginal series could replace it through the existing CSV loader (§5.2).
5. **A fixed comfort weight.** At matched comfort the tuned rule beats penalty-based PPO (§8.3, §8.8). Lagrangian PPO with a comfort limit set by stakeholders (§9) may close the gap; it was planned as our extension but not completed (§8.7).
6. **Limited breadth.** Five seeds and one synthetic year leave small effects, such as the no-CI ablation, short of significance (§8.4), and DQN was not tuned (§6.3); more seeds and years, and tuning DQN under §6.4, would fix both.

Items 2 and 3 come first: §8 finds the headroom in occupancy, and only storage would make carbon timing worth learning.
