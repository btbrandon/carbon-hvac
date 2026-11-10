"""Build every table and figure for the slides/report from results/.

    python scripts/make_figures.py
Writes results/figures/*.png and results/tables/*.{csv,md}.

Statistics follow §7: learning controllers are summarised per seed (mean over test
episodes), then mean +- 95% bootstrap CI across seeds; rule-based controllers and the
oracle are deterministic per episode, so their CI is over episodes.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from scipy import stats as sps

from carbonhvac.baselines import CarbonHeuristic, OccupancySetback
from carbonhvac.env import CarbonHVACEnv, EnvConfig
from carbonhvac.metrics import run_episode
from carbonhvac.stats import bootstrap_ci

RES = Path("results")
FIG, TAB = RES / "figures", RES / "tables"
METRICS = {"reward": "Reward (per day)", "kwh": "Energy (kWh/day)", "kg_co2": "Carbon (kg CO2/day)", "cost_sgd": "Cost (SGD/day)",
           "comfort_dh": "Discomfort (degC·h/day)", "viol_pct": "Occupied steps in violation (%)",
           "max_dev": "Max deviation (degC)", "ci_per_kwh": "Effective CI (kg/kWh)"}
ORDER = ["fixed_setpoint", "setback_24", "setback", "carbon_heuristic", "carbon_prop", "dqn", "ppo", "oracle"]
LABEL = {"fixed_setpoint": "Fixed setpoint (24 °C)", "setback_24": "Setback, 24 °C (untuned)",
         "setback": "Setback, tuned", "carbon_heuristic": "Carbon heuristic, tuned",
         "carbon_prop": "Carbon-proportional, tuned", "dqn": "DQN", "ppo": "PPO", "oracle": "Oracle (DP)",
         "setback_mc": "Setback, tuned (≤1% viol.)"}
TUNED = json.loads((RES / "baselines_tuned.json").read_text())
BEST_RULE = TUNED["best_rule"]
# reference palette (light): blue, orange, aqua; baselines in neutral ink
C = {"ppo": "#2a78d6", "dqn": "#eb6834", "heur": "#1baf7a", "ink": "#0b0b0b", "muted": "#8a8984", "grid": "#e6e5e0"}

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": C["muted"], "axes.grid": True, "grid.color": C["grid"],
                     "grid.linewidth": 0.8, "axes.axisbelow": True, "lines.linewidth": 2,
                     "figure.dpi": 150, "savefig.bbox": "tight", "legend.frameon": False})


def load_runs(tag):
    files = sorted((RES / "runs" / tag).glob("*/test.csv"))
    return pd.concat([pd.read_csv(f) for f in files]) if files else pd.DataFrame()


def per_seed(df):
    return df.groupby(["controller", "seed"]).mean(numeric_only=True).reset_index()


def summarise(baselines, learned):
    """controller x metric -> (mean, lo, hi)."""
    rows = []
    for ctrl in ORDER:
        if ctrl in ("dqn", "ppo"):
            sub = per_seed(learned[learned.controller == ctrl]) if len(learned) else None
        else:
            sub = baselines[baselines.controller == ctrl]
        if sub is None or not len(sub):
            continue
        row = {"controller": ctrl, "n": len(sub)}
        for m in METRICS:
            row[m], row[m + "_lo"], row[m + "_hi"] = bootstrap_ci(sub[m])
        rows.append(row)
    return pd.DataFrame(rows)


def main_table(summary):
    lines = ["| Controller | " + " | ".join(METRICS.values()) + " |",
             "|---|" + "---|" * len(METRICS)]
    for _, r in summary.iterrows():
        cells = [f"{r[m]:.2f} [{r[m+'_lo']:.2f}, {r[m+'_hi']:.2f}]" for m in METRICS]
        lines.append(f"| {LABEL[r.controller]} | " + " | ".join(cells) + " |")
    (TAB / "main.md").write_text("\n".join(lines) + "\n")
    summary.to_csv(TAB / "main.csv", index=False)


def headline(baselines, learned):
    """Paired comparison on identical test episodes, replicated over seeds."""
    heur = baselines[baselines.controller == BEST_RULE].set_index("day")
    fixed = baselines[baselines.controller == "fixed_setpoint"].set_index("day")
    oracle = baselines[baselines.controller == "oracle"].set_index("day")
    out = {}
    for algo in ("ppo", "dqn"):
        sub = learned[learned.controller == algo]
        if not len(sub):
            continue
        res = {}
        for m in ("reward", "kg_co2", "cost_sgd", "kwh", "comfort_dh", "viol_pct"):
            diffs = [(g.set_index("day")[m] - heur[m]).mean() for _, g in sub.groupby("seed")]
            mean, lo, hi = bootstrap_ci(diffs)
            p = sps.ttest_1samp(diffs, 0).pvalue if len(diffs) > 1 else float("nan")
            res[m] = {"diff_vs_best_rule": mean, "ci": [lo, hi], "p": p,
                      "pct_vs_best_rule": 100 * mean / abs(heur[m].mean()) if abs(heur[m].mean()) > 0.05 else None}
        ctrl_c = per_seed(sub)["kg_co2"].mean()
        res["oracle_gap_capture_vs_fixed_pct"] = 100 * (fixed.kg_co2.mean() - ctrl_c) / (fixed.kg_co2.mean() - oracle.kg_co2.mean())
        res["oracle_gap_capture_vs_best_rule_pct"] = 100 * (heur.kg_co2.mean() - ctrl_c) / (heur.kg_co2.mean() - oracle.kg_co2.mean())
        ctrl_r = per_seed(sub)["reward"].mean()
        res["reward_gap_capture_vs_best_rule_pct"] = 100 * (ctrl_r - heur.reward.mean()) / (oracle.reward.mean() - heur.reward.mean())
        out[algo] = res
    out["best_rule"] = BEST_RULE
    (TAB / "headline.json").write_text(json.dumps(out, indent=2, default=float))
    return out


def fig_learning_curves():
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for algo in ("ppo", "dqn"):
        curves = [pd.read_csv(f) for f in sorted((RES / "runs/main").glob(f"{algo}_s*/curve.csv"))]
        if not curves:
            continue
        steps = curves[0].steps.values
        n = min(len(c) for c in curves)
        r = np.stack([c.reward.values[:n] for c in curves])
        ax.plot(steps[:n], r.mean(0), color=C[algo], label=f"{algo.upper()} (mean of {len(curves)} seeds)")
        ax.fill_between(steps[:n], r.min(0), r.max(0), color=C[algo], alpha=0.15, linewidth=0)
    b = pd.read_csv(RES / "baselines_train_curve_ref.csv") if (RES / "baselines_train_curve_ref.csv").exists() else None
    if b is not None:
        for _, row in b.iterrows():
            ax.axhline(row.reward, color=C["muted"], linestyle="--", linewidth=1)
            ax.text(ax.get_xlim()[1], row.reward, " " + LABEL[row.controller], va="center", fontsize=9, color=C["muted"])
    ax.set_ylim(-40, None)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Eval return per day (train days)")
    ax.set_title("Learning curves (band = min/max over seeds)", loc="left")
    ax.legend(loc="lower right")
    fig.savefig(FIG / "learning_curves.png")
    plt.close(fig)


def fig_pareto(baselines):
    pts = []
    for d in sorted((RES / "runs").glob("sweep_lamK*")) + [RES / "runs/main"]:
        df = load_runs(d.name)
        if not len(df) or not len(df := df[df.controller == "ppo"]):
            continue
        lam = 5.0 if d.name == "main" else float(d.name.replace("sweep_lamK", ""))
        for s, g in df.groupby("seed"):
            pts.append({"lam": lam, "seed": s, "kg_co2": g.kg_co2.mean(), "comfort_dh": g.comfort_dh.mean()})
    if not pts:
        return
    pts = pd.DataFrame(pts)
    means = pts.groupby("lam")[["kg_co2", "comfort_dh"]].mean().reset_index().sort_values("comfort_dh")
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.scatter(pts.comfort_dh, pts.kg_co2, s=18, color=C["muted"], alpha=0.5, label="PPO, single seed")
    front, best = [], np.inf
    for _, r in means.iterrows():
        if r.kg_co2 < best:
            front.append(r)
            best = r.kg_co2
    front = pd.DataFrame(front)
    ax.plot(front.comfort_dh, front.kg_co2, color=C["ppo"], marker="o", markersize=7,
            markeredgecolor="white", markeredgewidth=1.5, label="PPO non-dominated (mean over seeds)")
    for _, r in means.iterrows():
        ax.annotate(f"λK={r.lam:g}", (r.comfort_dh, r.kg_co2), textcoords="offset points",
                    xytext=(6, 4), fontsize=8, color=C["muted"])
    rf = pd.read_csv(RES / "rule_front.csv").sort_values("comfort_dh")
    rf = rf[rf.kg_co2 <= 11.5]
    ax.plot(rf.comfort_dh, rf.kg_co2, color=C["heur"], marker="s", markersize=5, linewidth=1.5,
            label="Setback, setpoint 24.0–26.0 °C")
    for ctrl, mk in ((BEST_RULE, "s"), ("oracle", "*")):
        b = baselines[baselines.controller == ctrl]
        ax.scatter(b.comfort_dh.mean(), b.kg_co2.mean(), marker=mk, s=130 if mk == "*" else 70,
                   color=C["heur"] if mk == "s" else C["ink"], zorder=5, edgecolor="white", linewidth=1.5)
        ax.annotate(LABEL[ctrl], (b.comfort_dh.mean(), b.kg_co2.mean()), textcoords="offset points",
                    xytext=(8, -14), fontsize=9)
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_xlabel("Discomfort (degC·h per day, occupied, symlog)")
    ax.set_ylabel("Carbon (kg CO2 per day)")
    ax.set_title("Carbon vs comfort trade-off (test period)", loc="left")
    ax.legend(loc="upper right", fontsize=9)
    fig.savefig(FIG / "pareto.png")
    plt.close(fig)
    pts.to_csv(TAB / "pareto_points.csv", index=False)


def fig_ablations():
    tags = [("main", "Full PPO"), ("abl_no_ci", "No CI"), ("abl_no_forecast", "No forecasts"),
            ("abl_no_time", "No time features"), ("abl_dt60", "Δt = 1 h")]
    rows = []
    for tag, label in tags:
        df = load_runs(tag)
        df = df[df.controller == "ppo"] if len(df) else df
        if not len(df):
            continue
        s = per_seed(df)
        r = {"variant": label}
        for m in ("kg_co2", "ci_per_kwh", "comfort_dh", "kwh", "viol_pct"):
            r[m], r[m + "_lo"], r[m + "_hi"] = bootstrap_ci(s[m])
        rows.append(r)
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "ablations.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    y = np.arange(len(df))
    for ax, m, title in ((axes[0], "kg_co2", "Carbon (kg CO2/day)"), (axes[1], "ci_per_kwh", "Effective CI paid (kg/kWh)")):
        # dot + 95% CI: honest on a zoomed axis, unlike truncated bars
        ax.errorbar(df[m], y, xerr=[df[m] - df[m + "_lo"], df[m + "_hi"] - df[m]], fmt="none",
                    ecolor=C["muted"], capsize=0, linewidth=2)
        ax.scatter(df[m], y, s=60, zorder=3, edgecolor="white", linewidth=1.5,
                   color=[C["ppo"]] + [C["ink"]] * (len(df) - 1))
        ax.axvline(df[m].iloc[0], color=C["ppo"], linewidth=1, linestyle=":")
        ax.set_title(title, loc="left", fontsize=11)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, df.variant)
    axes[0].invert_yaxis()
    fig.suptitle("State ablations (PPO, 5 seeds, mean and 95% CI)", x=0.01, y=1.04, ha="left")
    fig.savefig(FIG / "ablations.png")
    plt.close(fig)


def fig_day_trace(day=None):
    from stable_baselines3 import PPO
    path = RES / "models/main/ppo_s0.zip"
    if not path.exists():
        return
    model = PPO.load(path, device="cpu")
    env = CarbonHVACEnv(EnvConfig(split="test"))
    from carbonhvac.baselines import CarbonProportional
    rules = {"setback": OccupancySetback, "carbon_heuristic": CarbonHeuristic, "carbon_prop": CarbonProportional}
    heur = rules[BEST_RULE](**TUNED["rules"][BEST_RULE]["best_reward"])
    if day is None:  # a representative workday: median CI swing among test workdays
        cands = [d for d in env.days if (d + 3) % 7 < 5]
        swings = [np.ptp(env.prof.ci[d * 96 + 32:d * 96 + 72]) for d in cands]
        day = cands[int(np.argsort(swings)[len(swings) // 2])]
    ppo = run_episode(env, lambda o, e: int(model.predict(o, deterministic=True)[0]), day, 7)
    hr = run_episode(env, heur, day, 7)
    h = ppo.hour.values
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 6.8), sharex=True, gridspec_kw={"height_ratios": [1.1, 1, 0.8]})
    for ax in axes:
        occ = ppo.occupied.values
        starts = h[np.flatnonzero(np.diff(np.r_[0, occ.astype(int)]) == 1)]
        ends = h[np.flatnonzero(np.diff(np.r_[occ.astype(int), 0]) == -1)] + 0.25
        for s, e in zip(starts, ends):
            ax.axvspan(s, e, color="#f1f0ec", zorder=0, linewidth=0)
    ax = axes[0]
    ax.axhspan(env.cfg.comfort_low, env.cfg.comfort_high, color=C["heur"], alpha=0.08, linewidth=0)
    ax.step(h, ppo.t_in, where="post", color=C["ppo"], label="PPO")
    ax.step(h, hr.t_in, where="post", color=C["heur"], label=LABEL[BEST_RULE])
    ax.set_ylabel("Indoor temp (degC)")
    ax.legend(loc="upper right", ncol=2)
    ax.set_title(f"Test day {day}: shaded = occupied, green band = comfort", loc="left")
    ax = axes[1]
    ax.step(h, ppo.energy_kwh * 4, where="post", color=C["ppo"])
    ax.step(h, hr.energy_kwh * 4, where="post", color=C["heur"])
    ax.set_ylabel("HVAC power (kW)")
    ax = axes[2]
    ax.plot(h, ppo.ci, color=C["ink"], linewidth=1.5)
    ax.set_ylabel("CI (kg/kWh)")
    ax.set_xlabel("Hour of day")
    ax.set_xticks(range(0, 25, 3))
    fig.savefig(FIG / "day_trace.png")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    baselines = pd.read_csv(RES / "baselines.csv")
    learned = load_runs("main")
    summary = summarise(baselines, learned)
    main_table(summary)
    hl = headline(baselines, learned)
    fig_learning_curves()
    fig_pareto(baselines)
    fig_ablations()
    fig_day_trace()
    print(summary[["controller", "n", "reward", "kg_co2", "kg_co2_lo", "kg_co2_hi", "comfort_dh", "viol_pct", "ci_per_kwh"]].round(3).to_string(index=False))
    print(json.dumps(hl, indent=1, default=lambda x: round(float(x), 3)))


if __name__ == "__main__":
    main()
