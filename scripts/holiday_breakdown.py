"""Sean's §8 analysis: day-type breakdown, holiday-calendar baseline, oracle foresight
decomposition and energy-weighted timing metrics.

    python scripts/holiday_breakdown.py

Why this exists
- The test period (days 273-364) contains 3 weekday public holidays (273, 277, 337).
  The tuned setback follows its workday schedule on them and cools an empty zone; PPO and
  DQN react to the occupancy sensor and do not. We therefore report results on the 89
  normal days as well, and against a setback that is given the holiday calendar (as any
  building management system would be).
- `ci_per_kwh` and `low_ci_share_pct` in test.csv are per-day ratios. Zero-energy days
  (weekends, holidays) score 0, so averaging them per day is biased and depends on how many
  idle days a controller has. Here they are aggregated energy-weighted:
  effective CI = sum(carbon) / sum(energy).

Reads results/baselines.csv, results/baselines_tuned.json and results/runs/*/*/test.csv.
Writes results/tables/s8_*.csv|json and results/figures/pareto_normal_days.png.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from carbonhvac.baselines import OccupancySetback
from carbonhvac.data import weekday
from carbonhvac.env import CarbonHVACEnv, EnvConfig
from carbonhvac.metrics import eval_episodes, evaluate
from carbonhvac.oracle import Oracle
from carbonhvac.stats import bootstrap_ci

RES = Path("results")
TAB, FIG = RES / "tables", RES / "figures"
C = {"ppo": "#2a78d6", "dqn": "#eb6834", "heur": "#1baf7a", "ink": "#0b0b0b", "muted": "#8a8984", "grid": "#e6e5e0"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": C["muted"], "axes.grid": True, "grid.color": C["grid"],
                     "grid.linewidth": 0.8, "axes.axisbelow": True, "lines.linewidth": 2,
                     "figure.dpi": 150, "savefig.bbox": "tight", "legend.frameon": False})
M = ["reward", "kg_co2", "cost_sgd", "kwh", "comfort_dh", "viol_pct"]


def load(tag, algo):
    fs = sorted((RES / "runs" / tag).glob(f"{algo}_s*/test.csv"))
    return pd.concat([pd.read_csv(f).assign(seed=i) for i, f in enumerate(fs)]) if fs else None


def agg(df):
    """Per-controller summary with energy-weighted timing metrics."""
    e = df.kwh.sum()
    return {**df[M].mean().to_dict(), "eff_ci": df.kg_co2.sum() / e,
            "low_ci_share": (df.low_ci_share_pct * df.kwh).sum() / e}


def summarise_learner(L, keep):
    per = pd.DataFrame([agg(g[g.day.isin(keep)]) for _, g in L.groupby("seed")])
    out = {}
    for c in per:
        m, lo, hi = bootstrap_ci(per[c].values)
        out[c], out[c + "_lo"], out[c + "_hi"] = m, lo, hi
    return out


def paired(L, ref, keep):
    """Mean per-day difference vs reference, per seed; bootstrap CI and t-test over seeds."""
    rows = {}
    for m in M:
        d = [(g.set_index("day")[m] - ref[m]).loc[keep].mean() for _, g in L.groupby("seed")]
        mean, lo, hi = bootstrap_ci(d)
        base = abs(ref[m].loc[keep].mean())
        pct = (lambda x: 100 * x / base) if base > 0.05 else (lambda x: np.nan)
        rows[m] = {"diff": mean, "lo": lo, "hi": hi, "pct": pct(mean), "pct_lo": pct(lo),
                   "pct_hi": pct(hi), "p": stats.ttest_1samp(d, 0).pvalue}
    return rows


def main():
    env = CarbonHVACEnv(EnvConfig(split="test"))
    eps = eval_episodes("test")
    occ = env.prof.occ.reshape(-1, 96)
    days = [d for d, _ in eps]
    cat = {d: "weekend" if weekday(d) >= 5 else ("holiday" if (occ[d] > 0).sum() == 0 else "workday") for d in days}
    hol = [d for d in days if cat[d] == "holiday"]
    normal = [d for d in days if cat[d] != "holiday"]

    tuned = json.loads((RES / "baselines_tuned.json").read_text())
    sb = tuned["rules"]["setback"]["best_reward"]

    # oracle foresight decomposition + rule front, per day
    ctrls = [Oracle(ci_mode="flat"), Oracle(occ_mode="schedule"), Oracle(occ_mode="schedule", ci_mode="flat")]
    extra = pd.concat([evaluate(env, c, eps) for c in ctrls])
    front = pd.concat([evaluate(env, OccupancySetback(setpoint=float(sp), lead_hours=sb["lead_hours"]), eps).assign(setpoint=sp)
                       for sp in np.round(np.arange(24.0, 26.01, 0.1), 1)])

    base = pd.read_csv(RES / "baselines.csv")
    ctl = pd.concat([base, extra])
    by = {c: g.set_index("day") for c, g in ctl.groupby("controller")}
    rule = by["setback"]
    cal = rule.copy()
    cal.loc[hol, M] = 0.0  # setback with a holiday calendar: off on public holidays
    cal["low_ci_share_pct"] = cal["low_ci_share_pct"].where(cal.index.isin(normal), 0.0)
    by["setback_calendar"] = cal

    ppo, dqn = load("main_tuned", "ppo"), load("main", "dqn")

    # 1. day-type breakdown
    rows = []
    for c in ("workday", "holiday", "weekend"):
        k = [d for d in days if cat[d] == c]
        r = {"day_type": c, "n_days": len(k)}
        for name, df in (("rule", rule), ("oracle", by["oracle"])):
            r[f"{name}_kwh"], r[f"{name}_co2"] = df.kwh.loc[k].mean(), df.kg_co2.loc[k].mean()
        for name, L in (("ppo", ppo), ("dqn", dqn)):
            r[f"{name}_kwh"], r[f"{name}_co2"] = L[L.day.isin(k)].kwh.mean(), L[L.day.isin(k)].kg_co2.mean()
        rows.append(r)
    pd.DataFrame(rows).to_csv(TAB / "s8_daytype.csv", index=False)

    # 2. headline comparisons: protocol (all days, tuned setback), normal days, calendar rule
    head = {}
    for lab, ref, keep in (("all_days_vs_setback", rule, days), ("normal_days_vs_setback", rule, normal),
                           ("all_days_vs_setback_calendar", cal, days)):
        head[lab] = {"ppo": paired(ppo, ref, keep), "dqn": paired(dqn, ref, keep)}
        p_c = np.mean([g.set_index("day").kg_co2.loc[keep].mean() for _, g in ppo.groupby("seed")])
        p_r = np.mean([g.set_index("day").reward.loc[keep].mean() for _, g in ppo.groupby("seed")])
        o = by["oracle"]
        head[lab]["ppo_gap_closed_co2_pct"] = 100 * (ref.kg_co2.loc[keep].mean() - p_c) / (ref.kg_co2.loc[keep].mean() - o.kg_co2.loc[keep].mean())
        head[lab]["ppo_gap_closed_reward_pct"] = 100 * (p_r - ref.reward.loc[keep].mean()) / (o.reward.loc[keep].mean() - ref.reward.loc[keep].mean())
    hol_share = np.mean([(g.set_index("day").kg_co2 - rule.kg_co2).loc[hol].sum() for _, g in ppo.groupby("seed")]) / \
        np.mean([(g.set_index("day").kg_co2 - rule.kg_co2).sum() for _, g in ppo.groupby("seed")])
    head["ppo_co2_saving_share_from_holidays_pct"] = 100 * hol_share
    head["holidays"] = hol
    (TAB / "s8_headline.json").write_text(json.dumps(head, indent=1, default=float))

    # 3. controller table on normal days, energy-weighted timing metrics
    tbl = []
    for c in ("fixed_setpoint", "setback_24", "setback", "setback_mc", "carbon_heuristic", "oracle",
              "oracle_flat_ci", "oracle_sched_occ", "oracle_sched_occ_flat_ci"):
        for lab, keep in (("all", days), ("normal", normal)):
            tbl.append({"controller": c, "days": lab, **agg(by[c].loc[keep].reset_index())})
    for c, L in (("ppo", ppo), ("dqn", dqn)):
        for lab, keep in (("all", days), ("normal", normal)):
            tbl.append({"controller": c, "days": lab, **summarise_learner(L, keep)})
    pd.DataFrame(tbl).to_csv(TAB / "s8_controllers.csv", index=False)

    # 4. oracle decomposition (paired over days, normal days)
    dec = []
    for v in ("oracle_flat_ci", "oracle_sched_occ", "oracle_sched_occ_flat_ci", "setback"):
        for m in ("reward", "kg_co2"):
            for lab, keep in (("all", days), ("normal", normal)):
                x = (by["oracle"][m] - by[v][m]).loc[keep]
                mean, lo, hi = bootstrap_ci(x.values)
                dec.append({"variant": v, "metric": m, "days": lab, "oracle_minus_variant": mean, "lo": lo, "hi": hi})
    pd.DataFrame(dec).to_csv(TAB / "s8_oracle_decomposition.csv", index=False)

    # 5. all learned runs (ablations, sweep), normal days, energy-weighted
    runs = []
    for d in sorted((RES / "runs").glob("*")):
        if d.name.startswith("tune"):
            continue
        for a in ("ppo", "dqn"):
            L = load(d.name, a)
            if L is None:
                continue
            for lab, keep in (("all", days), ("normal", normal)):
                runs.append({"tag": d.name, "algo": a, "days": lab, "seeds": L.seed.nunique(), **summarise_learner(L, keep)})
    runs = pd.DataFrame(runs)
    runs.to_csv(TAB / "s8_runs.csv", index=False)

    # 6. Pareto on normal days
    rf = front[front.day.isin(normal)].groupby("setpoint")[["kg_co2", "comfort_dh"]].mean().reset_index()
    rf = rf[rf.kg_co2 <= 11.5].sort_values("comfort_dh")
    pts = []
    for d in sorted((RES / "runs").glob("sweep_lamK*")) + [RES / "runs/main"]:
        L = load(d.name, "ppo")
        lam = 5.0 if d.name == "main" else float(d.name.replace("sweep_lamK", ""))
        for s, g in L[L.day.isin(normal)].groupby("seed"):
            pts.append({"lam": lam, "seed": s, "kg_co2": g.kg_co2.mean(), "comfort_dh": g.comfort_dh.mean()})
    pts = pd.DataFrame(pts)
    pts.to_csv(TAB / "s8_pareto_points_normal_days.csv", index=False)
    means = pts.groupby("lam")[["kg_co2", "comfort_dh"]].mean().reset_index().sort_values("comfort_dh")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.scatter(pts.comfort_dh, pts.kg_co2, s=18, color=C["muted"], alpha=0.5, label="PPO, single seed")
    ax.plot(means.comfort_dh, means.kg_co2, color=C["ppo"], marker="o", markersize=7, markeredgecolor="white",
            markeredgewidth=1.5, label="PPO, λK sweep (mean over seeds)")
    for _, r in means.iterrows():
        ax.annotate(f"λK={r.lam:g}", (r.comfort_dh, r.kg_co2), textcoords="offset points", xytext=(6, 4),
                    fontsize=8, color=C["muted"])
    ax.plot(rf.comfort_dh, rf.kg_co2, color=C["heur"], marker="s", markersize=5, linewidth=1.5,
            label="Setback, setpoint 24.0–26.0 °C")
    for _, r in rf[rf.setpoint.isin([25.9, 26.0])].iterrows():
        ax.annotate(f"{r.setpoint:.1f} °C", (r.comfort_dh, r.kg_co2), textcoords="offset points",
                    xytext=(4, -16) if r.setpoint == 25.9 else (6, -16), fontsize=8, color=C["heur"])
    tp = ppo[ppo.day.isin(normal)].groupby("seed")[["kg_co2", "comfort_dh"]].mean().mean()
    ax.scatter(tp.comfort_dh, tp.kg_co2, marker="D", s=60, color=C["ppo"], edgecolor="white", linewidth=1.5, zorder=6)
    ax.annotate("PPO, tuned", (tp.comfort_dh, tp.kg_co2), textcoords="offset points", xytext=(14, -28), fontsize=9,
                arrowprops=dict(arrowstyle="-", color=C["muted"], lw=0.8))
    o = by["oracle"].loc[normal]
    ax.scatter(o.comfort_dh.mean(), o.kg_co2.mean(), marker="*", s=130, color=C["ink"], edgecolor="white", linewidth=1.5, zorder=6)
    ax.annotate("Oracle (DP)", (o.comfort_dh.mean(), o.kg_co2.mean()), textcoords="offset points", xytext=(-34, -18), fontsize=9)
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_xlabel("Discomfort (degC·h per day, occupied, symlog)")
    ax.set_ylabel("Carbon (kg CO2 per day)")
    ax.set_title("Carbon vs comfort, 89 test days excluding holidays", loc="left")
    ax.legend(loc="upper right", fontsize=9)
    fig.savefig(FIG / "pareto_normal_days.png")
    plt.close(fig)
    print(json.dumps({k: v for k, v in head.items() if not isinstance(v, dict)}, default=float))


if __name__ == "__main__":
    main()
