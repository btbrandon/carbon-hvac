"""Tune every rule-based controller on the training period, then evaluate on test.

Baselines get the same data and objective as PPO: each rule's parameters are chosen by
grid search to maximise mean reward on training days (never test days). We also select a
"matched-comfort" variant of each rule: the lowest-carbon setting whose occupied-step
violation rate on training days is <= MAX_VIOL_PCT.

    python scripts/eval_baselines.py [--n-train 60] [--jobs 8]

Writes results/baselines.csv (test, per episode), results/baselines_tuned.json (chosen
parameters + best rule), results/tuning_grid.csv (every candidate's train score),
results/rule_front.csv (setback setpoint sweep on test, for the Pareto plot) and
results/baselines_train_curve_ref.csv (reference lines for learning curves).
"""
import argparse
import itertools
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from carbonhvac.baselines import CarbonHeuristic, CarbonProportional, FixedSetpoint, OccupancySetback
from carbonhvac.data import split_days
from carbonhvac.env import CarbonHVACEnv, EnvConfig
from carbonhvac.metrics import eval_episodes, evaluate
from carbonhvac.oracle import Oracle

COLS = ["reward", "kwh", "kg_co2", "cost_sgd", "comfort_dh", "viol_pct", "ci_per_kwh", "low_ci_share_pct"]
MAX_VIOL_PCT = 1.0
RULES = {"setback": OccupancySetback, "carbon_heuristic": CarbonHeuristic, "carbon_prop": CarbonProportional}

_env = _episodes = None


def _init(split, episodes):
    global _env, _episodes
    _env, _episodes = CarbonHVACEnv(EnvConfig(split=split)), episodes


def _score(job):
    rule, params = job
    m = evaluate(_env, RULES[rule](**params), _episodes)[COLS].mean().to_dict()
    return {"rule": rule, "params": json.dumps(params), **m}


def grids(ci_train):
    q = {round(x, 1): float(np.quantile(ci_train, x)) for x in np.arange(0.2, 0.81, 0.1)}
    ci_ref = float(np.median(ci_train))
    jobs = []
    for sp, lead in itertools.product(np.round(np.arange(24.0, 26.01, 0.1), 1), (0.0, 0.5, 1.0, 2.0)):
        jobs.append(("setback", {"setpoint": float(sp), "lead_hours": lead}))
    for th, low, high, pre in itertools.product(q.values(), (23.5, 24.0, 24.5, 25.0, 25.5),
                                                (25.0, 25.5, 25.7, 25.8, 25.9, 26.0), (None, 4.0, 5.0, 6.0)):
        if low < high:
            jobs.append(("carbon_heuristic", {"ci_threshold": th, "low": low, "high": high, "precool_start": pre}))
    for base, gain, lead in itertools.product(np.round(np.arange(25.0, 26.01, 0.1), 1),
                                              (0.0, 5.0, 10.0, 15.0, 20.0, 30.0), (0.5, 1.0)):
        jobs.append(("carbon_prop", {"base": float(base), "gain": gain, "ci_ref": ci_ref, "lead_hours": lead}))
    return jobs


def run(pool_args, jobs, n_jobs):
    with Pool(n_jobs, initializer=_init, initargs=pool_args) as pool:
        return pd.DataFrame(pool.map(_score, jobs, chunksize=4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=60)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    Path("results").mkdir(exist_ok=True)

    train_env = CarbonHVACEnv(EnvConfig(split="train"))
    ci_train = train_env.prof.ci[np.concatenate([np.arange(d * 96, (d + 1) * 96) for d in split_days("train")])]
    jobs = grids(ci_train)
    print(f"tuning {len(jobs)} rule configurations on {args.n_train} training days")
    grid = run(("train", eval_episodes("train", n=args.n_train, seed=0)), jobs, args.jobs)
    grid.to_csv("results/tuning_grid.csv", index=False)

    chosen = {}
    for rule in RULES:
        g = grid[grid.rule == rule]
        best = g.loc[g.reward.idxmax()]
        ok = g[g.viol_pct <= MAX_VIOL_PCT]
        mc = ok.loc[ok.kg_co2.idxmin()]
        chosen[rule] = {"best_reward": json.loads(best.params), "matched_comfort": json.loads(mc.params),
                        "train_reward": best.reward, "train_reward_mc": mc.reward}
    best_rule = max(RULES, key=lambda r: chosen[r]["train_reward"])
    meta = {"n_configs": len(jobs), "n_train_days": args.n_train, "max_viol_pct": MAX_VIOL_PCT,
            "best_rule": best_rule, "rules": chosen}
    Path("results/baselines_tuned.json").write_text(json.dumps(meta, indent=2))
    for r, c in chosen.items():
        print(f"{r:17s} best-reward {c['best_reward']}  |  matched-comfort {c['matched_comfort']}")
    print(f"best rule on train: {best_rule}")

    # ---- test evaluation: naive references, tuned rules, matched-comfort rules, oracle ----
    test_env = CarbonHVACEnv(EnvConfig(split="test"))
    episodes = eval_episodes("test")
    ctrls = [FixedSetpoint(), OccupancySetback(name="setback_24")]
    for r, c in chosen.items():
        ctrls.append(RULES[r](**{**c["best_reward"], "name": r}))
        ctrls.append(RULES[r](**{**c["matched_comfort"], "name": f"{r}_mc"}))
    ctrls.append(Oracle())
    df = pd.concat([evaluate(test_env, c, episodes) for c in ctrls])
    df.to_csv("results/baselines.csv", index=False)
    print(df.groupby("controller", sort=False)[COLS].mean().round(3).to_string())

    # setback setpoint sweep on test: the rule-based trade-off curve for the Pareto plot
    front = []
    for sp in np.round(np.arange(24.0, 26.01, 0.1), 1):
        m = evaluate(test_env, OccupancySetback(setpoint=float(sp), lead_hours=chosen["setback"]["best_reward"]["lead_hours"]), episodes)
        front.append({"setpoint": sp, **m[COLS].mean().to_dict()})
    pd.DataFrame(front).to_csv("results/rule_front.csv", index=False)

    # reference lines for the learning-curve plot: same 16 training days as scripts/train.py
    curve_eps = eval_episodes("train", n=16, seed=99)
    refs = [RULES[best_rule](**{**chosen[best_rule]["best_reward"], "name": best_rule}), Oracle()]
    ref = pd.concat([evaluate(train_env, c, curve_eps) for c in refs])
    ref.groupby("controller", sort=False)[["reward"]].mean().reset_index().to_csv(
        "results/baselines_train_curve_ref.csv", index=False)


if __name__ == "__main__":
    main()
