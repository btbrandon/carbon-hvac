"""Episode rollouts and the §7 metrics."""
import numpy as np
import pandas as pd

from .data import split_days


def eval_episodes(split="test", n=None, seed=1234):
    """Fixed (day, noise_seed) pairs so every controller faces identical conditions."""
    rng = np.random.default_rng(seed)
    days = split_days(split)
    if n is not None and n < len(days):
        days = np.sort(rng.choice(days, n, replace=False))
    return [(int(d), int(rng.integers(2**31))) for d in days]


def run_episode(env, policy, day, noise_seed):
    obs, _ = env.reset(options={"day": day, "noise_seed": noise_seed})
    infos, total = [], 0.0
    done = False
    while not done:
        obs, r, done, _, info = env.step(policy(obs, env))
        info["reward"] = r
        infos.append(info)
        total += r
    return pd.DataFrame(infos)


def episode_metrics(trace: pd.DataFrame) -> dict:
    e = trace["energy_kwh"]
    occ = trace["occupied"]
    below_median = trace["ci"] < trace["ci"].median()
    return {
        "reward": trace["reward"].sum(),
        "kwh": e.sum(),
        "kg_co2": trace["carbon_kg"].sum(),
        "cost_sgd": trace["cost_sgd"].sum(),
        "comfort_dh": trace["comfort_dh"].sum(),
        "viol_pct": 100 * ((trace["comfort_dh"] > 1e-6) & occ).sum() / max(occ.sum(), 1),
        "max_dev": trace["max_dev"].max(),
        "ci_per_kwh": trace["carbon_kg"].sum() / max(e.sum(), 1e-9),
        "low_ci_share_pct": 100 * e[below_median].sum() / max(e.sum(), 1e-9),
    }


def evaluate(env, policy, episodes, name=None) -> pd.DataFrame:
    rows = []
    for day, ns in episodes:
        m = episode_metrics(run_episode(env, policy, day, ns))
        m.update(day=day, controller=name or getattr(policy, "name", "policy"))
        rows.append(m)
    return pd.DataFrame(rows)
