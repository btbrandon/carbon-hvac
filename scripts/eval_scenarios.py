"""§8.5 robustness: evaluate every controller under the Table 7.2 scenarios (no retraining).

    python scripts/eval_scenarios.py               # rules + oracle (+ PPO/DQN if models exist)
    python scripts/eval_scenarios.py --only ppo    # learned controllers only (appends)

Scenarios are evaluation-time changes to EnvConfig on the 92 test days, with the same
(day, noise_seed) pairs as the main evaluation. Learned controllers are loaded from
results/models/<tag>/<algo>_s<seed>.zip when present (these files are gitignored, so they only
exist on the machine that trained them; retrain with scripts/train.py if needed).

Writes results/tables/s8_scenarios_days.csv (one row per scenario x controller x seed x day)
and results/tables/s8_scenarios.csv (summary), and results/figures/scenarios.png.
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from carbonhvac.baselines import CarbonHeuristic, FixedSetpoint, OccupancySetback, _occupied_window, track
from carbonhvac.data import STEPS_PER_DAY, weekday
from carbonhvac.env import CarbonHVACEnv, EnvConfig
from carbonhvac.metrics import eval_episodes, evaluate
from carbonhvac.oracle import Oracle
from carbonhvac.stats import bootstrap_ci

RES = Path("results")
SCENARIOS = {
    "main": {},
    "ci_shift_+3h": {"ci_shift_hours": 3.0},
    "ci_shift_-3h": {"ci_shift_hours": -3.0},
    "sensor_flip_0.1": {"occ_flip_prob": 0.1},
    "sensor_flip_0.2": {"occ_flip_prob": 0.2},
    "price_flat": {"price_regime": "flat"},
    "price_tou": {"price_regime": "tou"},
    "ci_flat": {"ci_regime": "flat"},
    "ci_anticorrelated": {"ci_regime": "anticorrelated"},
}


@dataclass
class CalendarSetback(OccupancySetback):
    """Tuned setback that knows the public-holiday calendar: holidays are treated like weekends,
    so the zone is cooled only if the occupancy sensor reads occupied."""
    name: str = "setback_calendar"

    def __call__(self, obs, env):
        r = dict(env.raw_obs())
        d = env.day
        if weekday(d) < 5 and env.prof.occ[d * STEPS_PER_DAY:(d + 1) * STEPS_PER_DAY].sum() == 0:
            r["is_workday"] = 0.0
        on = _occupied_window(r, start=8.0 - self.lead_hours)
        return track(env, self.setpoint if on else None)


def rules():
    t = json.loads((RES / "baselines_tuned.json").read_text())["rules"]
    sb = t["setback"]["best_reward"]
    return [FixedSetpoint(),
            OccupancySetback(**{**sb, "name": "setback"}),
            CarbonHeuristic(**{**t["carbon_heuristic"]["best_reward"], "name": "carbon_heuristic"}),
            CalendarSetback(**{k: v for k, v in sb.items() if k != "name"}),
            Oracle()]


def learned(tag, algo):
    paths = sorted((RES / "models" / tag).glob(f"{algo}_s*.zip"))
    if not paths:
        return []
    from stable_baselines3 import DQN, PPO
    Algo = PPO if algo == "ppo" else DQN
    out = []
    for p in paths:
        m = Algo.load(p, device="cpu")
        seed = int(p.stem.split("_s")[-1])

        def act(obs, env, m=m):
            return int(m.predict(obs, deterministic=True)[0])
        out.append((f"{algo}" if tag == "main_tuned" else f"{algo}_{tag}", seed, act))
    return out


def summarise(days):
    rows = []
    for (scen, ctrl), g in days.groupby(["scenario", "controller"], sort=False):
        per = g.groupby("seed").apply(lambda x: pd.Series({
            "reward": x.reward.mean(), "kg_co2": x.kg_co2.mean(), "kwh": x.kwh.mean(),
            "comfort_dh": x.comfort_dh.mean(), "viol_pct": x.viol_pct.mean(),
            "eff_ci": x.kg_co2.sum() / max(x.kwh.sum(), 1e-9)}), include_groups=False)
        r = {"scenario": scen, "controller": ctrl, "n_seeds": len(per)}
        for c in per:
            if len(per) > 1:
                r[c], r[c + "_lo"], r[c + "_hi"] = bootstrap_ci(per[c].values)
            else:  # deterministic controller: CI over days
                x = g[c] if c != "eff_ci" else None
                if x is None:
                    r[c] = per[c].iloc[0]
                else:
                    r[c], r[c + "_lo"], r[c + "_hi"] = bootstrap_ci(x.values)
        rows.append(r)
    return pd.DataFrame(rows)


def figure(summ):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = {"setback": "#1baf7a", "setback_calendar": "#7fd6b6", "ppo": "#2a78d6", "oracle": "#0b0b0b"}
    show = [c for c in ["setback", "setback_calendar", "ppo", "oracle"] if c in set(summ.controller)]
    scen = list(SCENARIOS)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    w = 0.8 / len(show)
    for ax, m, lab in ((axes[0], "kg_co2", "Carbon (kg CO2 per day)"), (axes[1], "viol_pct", "Violations (% of occupied steps)")):
        for i, c in enumerate(show):
            s = summ[summ.controller == c].set_index("scenario").reindex(scen)
            x = np.arange(len(scen)) + (i - (len(show) - 1) / 2) * w
            err = None
            if m + "_lo" in s:
                err = np.vstack([s[m] - s[m + "_lo"], s[m + "_hi"] - s[m]]).clip(min=0)
            ax.bar(x, s[m], w * 0.92, color=C[c], label=c.replace("_", " "), yerr=err, capsize=2,
                   error_kw={"elinewidth": 0.8})
        ax.set_xticks(np.arange(len(scen)), [s.replace("_", "\n", 1) for s in scen], fontsize=8)
        ax.set_ylabel(lab)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(RES / "figures" / "scenarios.png", dpi=150, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["rules", "ppo", "all"], default="all")
    args = ap.parse_args()
    eps = eval_episodes("test")
    out_days = RES / "tables" / "s8_scenarios_days.csv"
    rows = [pd.read_csv(out_days)] if (args.only == "ppo" and out_days.exists()) else []
    for scen, knobs in SCENARIOS.items():
        env = CarbonHVACEnv(EnvConfig(split="test", **knobs))
        if args.only in ("rules", "all"):
            for c in rules():
                df = evaluate(env, c, eps)
                df["scenario"], df["seed"] = scen, 0
                rows.append(df)
        if args.only in ("ppo", "all"):
            for name, seed, act in learned("main_tuned", "ppo"):
                df = evaluate(env, act, eps, name=name)
                df["scenario"], df["seed"] = scen, seed
                rows.append(df)
        print("done", scen, flush=True)
    days = pd.concat(rows, ignore_index=True)
    if args.only == "ppo":
        days = days.drop_duplicates(["scenario", "controller", "seed", "day"], keep="last")
    days.to_csv(out_days, index=False)
    summ = summarise(days)
    summ.to_csv(RES / "tables" / "s8_scenarios.csv", index=False)
    figure(summ)
    print(summ[["scenario", "controller", "n_seeds", "reward", "kg_co2", "kwh", "comfort_dh", "viol_pct", "eff_ci"]]
          .round(3).to_string())


if __name__ == "__main__":
    main()
