#!/usr/bin/env bash
# Launch the full experiment grid in parallel. Usage: scripts/run_all.sh [group ...] [-j N]
#   groups: main sweep ablations   (default: all)
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=7
GROUPS_=()
while [[ $# -gt 0 ]]; do
  case "$1" in -j) JOBS="$2"; shift 2;; *) GROUPS_+=("$1"); shift;; esac
done
[[ ${#GROUPS_[@]} -eq 0 ]] && GROUPS_=(main sweep ablations)

PY=.venv/bin/python
STEPS=800000
SEEDS="0 1 2 3 4"
cmds=()
for g in "${GROUPS_[@]}"; do
  case "$g" in
    main)
      for s in $SEEDS; do
        cmds+=("$PY scripts/train.py --algo ppo --seed $s --steps $STEPS --tag main")
        cmds+=("$PY scripts/train.py --algo dqn --seed $s --steps $STEPS --tag main")
      done ;;
    sweep)   # comfort weight sweep -> carbon/comfort Pareto front (3 seeds each)
      for lk in 0.5 1 2 10 20 50; do for s in 0 1 2; do
        cmds+=("$PY scripts/train.py --algo ppo --seed $s --steps $STEPS --tag sweep_lamK$lk --set lam_comfort=$lk")
      done; done ;;
    ablations)
      for s in $SEEDS; do
        cmds+=("$PY scripts/train.py --algo ppo --seed $s --steps $STEPS --tag abl_no_ci --set obs_ci=false")
        cmds+=("$PY scripts/train.py --algo ppo --seed $s --steps $STEPS --tag abl_no_forecast --set forecast_hours=[]")
        cmds+=("$PY scripts/train.py --algo ppo --seed $s --steps $STEPS --tag abl_no_time --set obs_time=false")
        cmds+=("$PY scripts/train.py --algo ppo --seed $s --steps $STEPS --tag abl_dt60 --set control_minutes=60")
      done ;;
    *) echo "unknown group $g"; exit 1 ;;
  esac
done

mkdir -p results/logs
echo "launching ${#cmds[@]} runs with $JOBS parallel jobs"
printf '%s\n' "${cmds[@]}" | xargs -P "$JOBS" -I{} sh -c '{} 2>>results/logs/stderr.log' | tee -a results/logs/summary.log
