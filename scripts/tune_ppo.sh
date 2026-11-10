#!/usr/bin/env bash
# PPO hyperparameter search. Each config: 2 seeds x 1.5M steps, scored on the 60 training
# days used to tune the rule-based baselines. Never evaluates on the test period.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
STEPS=1500000
declare -a CFGS=(
  "default|"
  "g995|gamma=0.995"
  "lr1e4|learning_rate=0.0001"
  "ent0|ent_coef=0.0"
  "ent001|ent_coef=0.001"
  "net128|net=128"
  "nsteps768|n_steps=768"
  "clip01|clip_range=0.1"
  "gae098|gae_lambda=0.98"
  "g995_lr1e4|gamma=0.995 learning_rate=0.0001"
  "g995_ent001|gamma=0.995 ent_coef=0.001"
  "lr1e4_ent001_net128|learning_rate=0.0001 ent_coef=0.001 net=128"
)
cmds=()
for c in "${CFGS[@]}"; do
  name="${c%%|*}"; hp="${c#*|}"
  for s in 0 1; do
    cmds+=("$PY scripts/train.py --algo ppo --seed $s --steps $STEPS --tag tune_$name --val --no-test --eval-every 100000 ${hp:+--hp $hp}")
  done
done
mkdir -p results/logs
echo "launching ${#cmds[@]} tuning runs"
printf '%s\n' "${cmds[@]}" | xargs -P 7 -I{} sh -c '{} 2>>results/logs/stderr_tune.log' | tee -a results/logs/tune.log
