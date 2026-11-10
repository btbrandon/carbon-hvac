# 6. Methods: baselines, learning algorithms, upper bound and tuning

*Authors: Jaylon (§6.1), Brandon (§6.2–6.4). Target ≈ 1.75 pages in the body. Tables 6.A and 6.B go in the appendix. Citations in Vancouver style; renumber when merging.*

## 6.1 Rule-based baselines

All rules share one primitive: an ideal staged thermostat that, at each step, selects the power level whose predicted next indoor temperature is closest to a target. The rules differ only in the target and when it applies.

1. **Fixed setpoint.** Holds 24 °C continuously; the reference against which all savings are measured.
2. **Occupancy setback.** Cools to a setpoint from a lead time before the scheduled 08:00 start until 18:00 on workdays, or whenever the occupancy sensor reads occupied; otherwise off. We report it both untuned (24 °C) and tuned, which isolates the value of tuning.
3. **Carbon heuristic.** Setback with a carbon-intensity threshold θ: below θ it cools to a lower setpoint, storing cooling in the thermal mass; above θ it lets the zone drift to a higher setpoint. Optional early-morning pre-cooling applies when CI is low.
4. **Carbon-proportional.** Setback whose setpoint rises linearly with CI: T_set = clip(base + gain·(CI − CI_median), 23, 26) °C, with CI_median the training-period median.

**What tuning found.** Rules 2–4 were tuned under the common protocol of §6.4 (972 configurations; Table 6.B). For each rule we also record a matched-comfort variant (lowest carbon with ≤1% of occupied steps in violation), used for the comfort-matched comparison in §8.3. **All three converged to essentially the same controller: setback at 25.9 °C with a 1-hour lead.** Tuning disabled the carbon-reactive features: the carbon-proportional gain fell to 0, making it identical to setback, and the heuristic's setpoints narrowed to 25.5/25.9 °C, leaving it marginally worse than setback. With a 6-hour thermal time constant (§5), cooling stored in a low-carbon hour largely leaks away before it is needed, so the best rule is to cool only when occupied, at the top of the comfort band. That setpoint sits just below a cliff: on training days, 25.9 °C leaves 0.7% of occupied steps in violation but 26.0 °C leaves 32%, because the staged thermostat oscillates around its target and crosses the band edge.

**Table 6.B (appendix)** Rule tuning on the 60 selection days (mean daily reward; higher is better).

| Rule | Configurations | Selected (best reward) | Matched-comfort variant | Reward |
|---|---|---|---|---|
| Setback | 84 | 25.9 °C, 1 h lead | 25.9 °C, 0.5 h lead | −11.84 |
| Carbon heuristic | 756 | θ = 0.122 kg/kWh, 25.5/25.9 °C, pre-cool from 06:00 | θ = 0.137 kg/kWh, 25.5/25.9 °C, no pre-cool | −11.92 |
| Carbon-proportional | 132 | base 25.9 °C, gain 0, 1 h lead | base 25.9 °C, gain 0, 0.5 h lead | −11.84 |

## 6.2 Proximal Policy Optimization (main method)

PPO [1] learns a stochastic policy π_θ(a|s) and a value function V_φ(s) by alternating between two phases: collecting a batch of experience with the current policy, then taking several gradient steps on a *clipped* surrogate objective:

  L^CLIP(θ) = E_t[ min( ρ_t(θ)·Â_t, clip(ρ_t(θ), 1−ε, 1+ε)·Â_t ) ],  ρ_t(θ) = π_θ(a_t|s_t) / π_θold(a_t|s_t).

The clip removes any incentive to move the probability ratio outside [1−ε, 1+ε]. This makes it safe to reuse each batch for several epochs without the destructive policy updates that unconstrained policy gradients suffer. Advantages are estimated with generalised advantage estimation (GAE) [2]:

  Â_t = Σ_l (γλ)^l δ_{t+l},  δ_t = r_t + γV_φ(s_{t+1}) − V_φ(s_t).

Here λ trades bias (λ → 0) against variance (λ → 1). The full loss adds a value-function regression term and an entropy bonus, −L^CLIP + c_v·(V_φ − V^target)² − c_e·H[π_θ], which keeps exploration alive early in training.

**Why PPO.** PPO is on-policy and therefore sample-hungry. That is a poor trade on real hardware, but our simulator runs at about 36,000 steps/s, so a 1.5M-step run takes under 5 minutes. In return we get stable, monotone learning and native support for discrete actions.

**Implementation.** We use stable-baselines3 2.9 [4]:
- **Networks:** separate actor and critic MLPs, each with two hidden layers of 64 tanh units. The actor ends in a 5-way softmax over power levels.
- **Rollouts:** 8 parallel environments × 768 steps = 6,144 transitions per update, i.e. 64 simulated days.
- **Optimisation:** 10 epochs of 256-sample minibatches.
- **Training budget:** 1.5M steps (≈244 updates, ≈4.7 min on one CPU core).

Full hyperparameters are in Table 6.A.

**Stability measures.** Training instability was the main risk named in our proposal. We address it with six measures:
1. Reward normalisation: rewards are divided by a running estimate of the discounted return's standard deviation, with the normaliser's γ matched to the agent's.
2. Per-minibatch advantage normalisation.
3. The ratio clip, ε = 0.2.
4. Gradient-norm clipping at 0.5.
5. A small entropy bonus, c_e = 0.01.
6. Longer rollouts, chosen by tuning (§6.4), which give lower-variance advantage estimates. Large-scale studies show such implementation choices matter for on-policy RL [6].

The result: across the 34 runs with the full state and default weights (5 main, 24 tuning, 5 final), the worst seed finished 0.27 reward/day below its group's median. The final 5 seeds agree within 1.3% on test-period carbon. Over all 72 PPO runs, two finished clearly below their group: one λ_K = 50 sweep seed and one seed of the no-time-features ablation. Both are harder variants. We report every seed, not the best one [7].

## 6.3 Comparison algorithm and upper bound

**DQN.** DQN [3] is our second learner. It is off-policy and value-based: it learns Q(s, a) by regressing toward r + γ·max_a′ Q_target(s′, a′), using a 100k-transition replay buffer and a target network refreshed every 1,000 steps. Exploration is ε-greedy, annealed from 1.0 to 0.02 over the first 30% of training. The network is a 64×64 ReLU MLP trained for 800k steps (Table 6.A). DQN is the natural comparison because it uses experience more efficiently and handles our discrete action space directly. It was **not** hyperparameter-tuned, which we note as a limitation (§10).

**Oracle (perfect-foresight dynamic programming).** Given one day's realised weather, occupancy, price and CI, the dynamics are deterministic and the state is the scalar T_in. We can therefore solve each day exactly by finite-horizon backward induction (value iteration) [5]:

  V_H(T) = 0,  V_t(T) = min_a [ c_t(T, a) + V_{t+1}( f_t(T, a) ) ],

where c_t is the per-step cost (the negative reward) and f_t the 1R1C transition.
- **Grid:** V is computed on a 401-point temperature grid over [15, 35] °C at 0.05 °C, with linear interpolation between grid points.
- **Execution:** the oracle acts by one-step lookahead from the exact current temperature, so interpolation is the only approximation.
- **Shared code:** dynamics and cost come from the same function the environment uses, so the oracle optimises exactly the objective every controller is scored on.
- **Checks:** unit tests confirm, on sampled test days, that it never loses to a rule-based controller and that its predicted value matches the realised return within 2%. On all 92 test days it scores above every tuned rule on average.
- **Decomposition:** two variants each remove one piece of foresight. One plans with a flat CI; the other plans with the nominal 08:00–18:00 schedule instead of true occupancy. Both are scored on the true outcome, which lets us attribute the oracle's advantage (§8).

## 6.4 Hyperparameter tuning protocol

A learned controller is only meaningfully better than a rule if both have been tuned with comparable care. Our first results compared default-setting PPO against a carbon heuristic with only two of its settings tuned, and overstated PPO's carbon advantage roughly threefold (11.3% vs 3.7%). We therefore tuned every controller under the same protocol:

- **Selection data:** the same 60 fixed days from the training period (January–September) for every controller. The test period (October–December) is never used for any choice.
- **Objective:** mean daily reward, exactly the quantity PPO maximises.
- **Rules:** grid search over 972 configurations of the three rule families (§6.1).
- **PPO:** 12 configurations (Table 6.1), each changing one or two hyperparameters from the defaults, with 2 seeds × 1.5M steps each (24 runs, ≈25 min on 7 parallel jobs).
- **Final run:** the selected configuration is retrained with 5 fresh seeds and evaluated **once** on the 92 test days.

**Table 6.1** PPO tuning on validation days (mean daily reward over 2 seeds; higher is better). For reference, the best tuned rule scores −11.84 and the oracle −11.21 on the same days.

| Configuration | Change from defaults | Validation reward |
|---|---|---|
| **Selected** | rollout length 384 → 768 | **−11.71** |
| ent0 | entropy coefficient 0.01 → 0 | −11.78 |
| clip01 | clip ε 0.2 → 0.1 | −11.79 |
| default | — | −11.82 |
| lr1e4 | learning rate 3×10⁻⁴ → 1×10⁻⁴ | −11.85 |
| net128 | 128×128 networks | −11.93 |
| g995 | γ 0.99 → 0.995 | −11.96 |
| gae098 | GAE λ 0.95 → 0.98 | −12.02 |
| *4 further variants* | entropy 0.001, and combinations of the above | −11.82 to −11.96 |

Two observations matter for interpreting the main results.

1. **The spread across configurations (≈0.3) is only about three times the seed-to-seed spread (≈0.1).** Picking the best of 12 therefore risks a winner's-curse bias. Retraining with 5 fresh seeds and testing once on held-out days is our guard against it.
2. **The two worst single changes, γ = 0.995 and GAE λ = 0.98, both lengthen the credit-assignment horizon. Larger networks did not help either; a longer rollout did.** This is consistent with a problem whose relevant planning horizon is under a day and whose reward is dense. The remaining differences are within seed noise.

**Table 6.A (appendix)** Final hyperparameters.

| | PPO (selected) | DQN |
|---|---|---|
| Library | stable-baselines3 2.9.2 | stable-baselines3 2.9.2 |
| Network | actor and critic 64×64 tanh (separate) | Q-network 64×64 ReLU |
| Parallel environments | 8 | 1 |
| Rollout length / replay buffer | 768 steps per env | 100,000 transitions |
| Minibatch / epochs | 256 / 10 | 128, 1 gradient step per 4 env steps |
| Learning rate | 3×10⁻⁴ | 5×10⁻⁴ |
| γ / GAE λ | 0.99 / 0.95 | 0.99 / — |
| Clip ε / entropy c_e / value c_v | 0.2 / 0.01 / 0.5 | — |
| Exploration | stochastic policy + entropy bonus | ε-greedy 1.0 → 0.02 over 30% of training |
| Target network update | — | every 1,000 steps |
| Gradient-norm clip | 0.5 | 10 (library default) |
| Reward normalisation | running return std (γ-matched) | running return std |
| Training steps | 1.5M | 800k |
| Wall-clock per run | ≈4.7 min | ≈6.8 min |
| Seeds | 5 (0–4) | 5 (0–4) |

## References

1. Schulman J, Wolski F, Dhariwal P, Radford A, Klimov O. Proximal policy optimization algorithms [Preprint]. arXiv:1707.06347. 2017.
2. Schulman J, Moritz P, Levine S, Jordan M, Abbeel P. High-dimensional continuous control using generalized advantage estimation. In: Proceedings of the 4th International Conference on Learning Representations (ICLR); 2016.
3. Mnih V, Kavukcuoglu K, Silver D, Rusu AA, Veness J, Bellemare MG, et al. Human-level control through deep reinforcement learning. Nature. 2015;518(7540):529–33.
4. Raffin A, Hill A, Gleave A, Kanervisto A, Ernestus M, Dormann N. Stable-Baselines3: reliable reinforcement learning implementations. J Mach Learn Res. 2021;22(268):1–8.
5. Bellman R. Dynamic programming. Princeton (NJ): Princeton University Press; 1957.
6. Andrychowicz M, Raichuk A, Stańczyk P, Orsini M, Girgin S, Marinier R, et al. What matters in on-policy reinforcement learning? A large-scale empirical study. In: Proceedings of the 9th International Conference on Learning Representations (ICLR); 2021.
7. Henderson P, Islam R, Bachman P, Pineau J, Precup D, Meger D. Deep reinforcement learning that matters. In: Proceedings of the AAAI Conference on Artificial Intelligence. 2018;32(1).
