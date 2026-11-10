# Carbon-Aware Smart Building Energy Controller (CS4246 Group 23)

RL control of a single-zone office HVAC system that shifts cooling toward low-carbon hours
while keeping occupants comfortable. PPO and DQN are compared against rule-based baselines
and a perfect-foresight dynamic-programming upper bound.

## Setup
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
pytest                                  # simulator validation
python scripts/eval_baselines.py        # tune heuristic, baseline + oracle table
```

## Layout
See `PLAN.md` for the full plan, experiment grid and schedule.
