"""Tune the carbon heuristic on the training period, then evaluate rule-based controllers + oracle on test.

    python scripts/eval_baselines.py [--n-train 60] [--out results/baselines.csv]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from carbonhvac.baselines import CarbonHeuristic, FixedSetpoint, OccupancySetback
from carbonhvac.env import CarbonHVACEnv, EnvConfig
from carbonhvac.metrics import eval_episodes, evaluate
from carbonhvac.oracle import Oracle

COLS = ["reward", "kwh", "kg_co2", "cost_sgd", "comfort_dh", "viol_pct", "ci_per_kwh", "low_ci_share_pct"]


def tune_heuristic(n_train, seed=0):
    env = CarbonHVACEnv(EnvConfig(split="train"))
    episodes = eval_episodes("train", n=n_train, seed=seed)
    ci = env.prof.ci
    rows = []
    for q in np.arange(0.2, 0.81, 0.1):
        theta = float(np.quantile(ci, q))
        for low in (23.0, 23.5):
            ctrl = CarbonHeuristic(ci_threshold=theta, low=low)
            r = evaluate(env, ctrl, episodes)["reward"].mean()
            rows.append({"quantile": round(q, 1), "ci_threshold": theta, "low": low, "reward": r})
    df = pd.DataFrame(rows)
    best = df.loc[df["reward"].idxmax()]
    return CarbonHeuristic(ci_threshold=float(best.ci_threshold), low=float(best.low)), df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=60)
    ap.add_argument("--out", default="results/baselines.csv")
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    heuristic, grid = tune_heuristic(args.n_train)
    grid.to_csv("results/heuristic_tuning.csv", index=False)
    Path("results/heuristic.json").write_text(json.dumps(
        {"ci_threshold": heuristic.ci_threshold, "low": heuristic.low}, indent=2))
    print(f"tuned heuristic: threshold={heuristic.ci_threshold:.3f} kg/kWh, low={heuristic.low}")

    env = CarbonHVACEnv(EnvConfig(split="test"))
    episodes = eval_episodes("test")
    frames = [evaluate(env, c, episodes) for c in
              (FixedSetpoint(), OccupancySetback(), heuristic, Oracle())]
    df = pd.concat(frames)
    df.to_csv(args.out, index=False)
    summary = df.groupby("controller", sort=False)[COLS].mean()
    print(summary.round(3).to_string())

    # reference lines for the learning-curve plot: same 16 training days as scripts/train.py
    train_env = CarbonHVACEnv(EnvConfig(split="train"))
    curve_eps = eval_episodes("train", n=16, seed=99)
    ref = pd.concat([evaluate(train_env, c, curve_eps) for c in (OccupancySetback(), heuristic, Oracle())])
    ref.groupby("controller", sort=False)[["reward"]].mean().reset_index().to_csv(
        "results/baselines_train_curve_ref.csv", index=False)


if __name__ == "__main__":
    main()
