"""Train PPO or DQN on the carbon-aware HVAC env, log learning curves, evaluate on the test period.

    python scripts/train.py --algo ppo --seed 0 --tag main
    python scripts/train.py --algo ppo --seed 0 --tag no_ci --set obs_ci=false
    python scripts/train.py --algo ppo --seed 0 --tag lamC2 --set lam_carbon=2.0

Outputs in results/runs/<tag>/<algo>_s<seed>/:
  config.json, curve.csv (eval reward and components vs steps), test.csv (per-episode metrics),
  model.zip (gitignored via results/models symlink-free path: see .gitignore).
"""
import argparse
import json
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from carbonhvac.env import CarbonHVACEnv, EnvConfig
from carbonhvac.metrics import eval_episodes, evaluate

PPO_KW = dict(n_steps=384, batch_size=256, n_epochs=10, learning_rate=3e-4, gamma=0.99,
              gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
              policy_kwargs=dict(net_arch=[64, 64]))
DQN_KW = dict(learning_rate=5e-4, buffer_size=100_000, learning_starts=5_000, batch_size=128,
              gamma=0.99, train_freq=4, target_update_interval=1_000,
              exploration_fraction=0.3, exploration_final_eps=0.02,
              policy_kwargs=dict(net_arch=[64, 64]))
N_ENVS = {"ppo": 8, "dqn": 1}
CURVE_COLS = ["reward", "kwh", "kg_co2", "cost_sgd", "comfort_dh", "viol_pct"]


def parse_overrides(pairs):
    names = {f.name for f in fields(EnvConfig)}
    out = {}
    for pair in pairs:
        k, v = pair.split("=", 1)
        if k not in names:
            raise SystemExit(f"unknown EnvConfig field {k!r}")
        v = yaml.safe_load(v)
        out[k] = tuple(v) if isinstance(v, list) else v
    return out


def greedy(model):
    def act(obs, env):
        return int(model.predict(obs, deterministic=True)[0])
    act.name = type(model).__name__.lower()
    return act


class CurveCallback(BaseCallback):
    """Every `every` env steps, evaluate the deterministic policy on fixed training-period days."""

    def __init__(self, cfg, every, n_eval=16):
        super().__init__()
        self.env = CarbonHVACEnv(EnvConfig(**{**cfg.__dict__, "split": "train"}))
        self.episodes = eval_episodes("train", n=n_eval, seed=99)
        self.every, self.next_at, self.rows = every, 0, []

    def _on_step(self):
        if self.num_timesteps >= self.next_at:
            self.next_at += self.every
            m = evaluate(self.env, greedy(self.model), self.episodes)[CURVE_COLS].mean()
            self.rows.append({"steps": self.num_timesteps, **m.to_dict()})
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["ppo", "dqn"], default="ppo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--set", nargs="*", default=[], help="EnvConfig overrides, key=value")
    ap.add_argument("--eval-every", type=int, default=15_000)
    ap.add_argument("--hp", nargs="*", default=[], help="algorithm hyperparameter overrides, key=value (net=128 sets net_arch)")
    ap.add_argument("--val", action="store_true", help="score on the 60 training days used to tune the rules (val.csv)")
    ap.add_argument("--no-test", action="store_true", help="skip the test-period evaluation (tuning runs)")
    args = ap.parse_args()

    torch.set_num_threads(1)
    cfg = EnvConfig(split="train", **parse_overrides(args.set))
    out = Path("results/runs") / args.tag / f"{args.algo}_s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    Algo, kw = (PPO, dict(PPO_KW)) if args.algo == "ppo" else (DQN, dict(DQN_KW))
    for pair in args.hp:
        k, v = pair.split("=", 1)
        v = yaml.safe_load(v)
        if k == "net":
            kw["policy_kwargs"] = dict(net_arch=[v, v])
        elif k in kw:
            kw[k] = v
        else:
            raise SystemExit(f"unknown hyperparameter {k!r}")
    venv = make_vec_env(lambda: CarbonHVACEnv(cfg), n_envs=N_ENVS[args.algo], seed=args.seed)
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=kw["gamma"])
    model = Algo("MlpPolicy", venv, seed=args.seed, verbose=0, device="cpu", **kw)

    cb = CurveCallback(cfg, args.eval_every)
    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=cb)
    train_min = (time.time() - t0) / 60
    cb.rows.append({"steps": model.num_timesteps,
                    **evaluate(cb.env, greedy(model), cb.episodes)[CURVE_COLS].mean().to_dict()})
    pd.DataFrame(cb.rows).to_csv(out / "curve.csv", index=False)

    models_dir = Path("results/models") / args.tag
    models_dir.mkdir(parents=True, exist_ok=True)
    model.save(models_dir / f"{args.algo}_s{args.seed}.zip")

    if args.val:
        val = evaluate(cb.env, greedy(model), eval_episodes("train", n=60, seed=0), name=args.algo)
        val["seed"], val["tag"] = args.seed, args.tag
        val.to_csv(out / "val.csv", index=False)
    if args.no_test:
        test = val if args.val else pd.DataFrame(cb.rows[-1:])
    else:
        test_env = CarbonHVACEnv(EnvConfig(**{**cfg.__dict__, "split": "test"}))
        test = evaluate(test_env, greedy(model), eval_episodes("test"), name=args.algo)
        test["seed"], test["tag"] = args.seed, args.tag
        test.to_csv(out / "test.csv", index=False)

    meta = {"algo": args.algo, "seed": args.seed, "steps": args.steps, "tag": args.tag,
            "train_minutes": round(train_min, 2), "env": cfg.to_dict(),
            "hp_overrides": args.hp, "hyperparams": {k: str(v) for k, v in kw.items()}}
    (out / "config.json").write_text(json.dumps(meta, indent=2, default=str))
    s = test[["reward", "kg_co2", "kwh", "comfort_dh", "viol_pct"]].mean()
    split = "val" if args.no_test else "test"
    print(f"[{args.tag} {args.algo} s{args.seed}] {train_min:.1f} min | {split} " +
          " ".join(f"{k}={v:.3f}" for k, v in s.items()))


if __name__ == "__main__":
    main()
