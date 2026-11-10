"""Pick the PPO hyperparameters with the best mean validation reward (training days only).

    python scripts/select_ppo.py
Prints the ranking, writes results/tables/ppo_tuning.csv and results/ppo_selected.json.
"""
import json
from pathlib import Path

import pandas as pd

RES = Path("results")
rows = []
for d in sorted((RES / "runs").glob("tune_*")):
    for run in sorted(d.glob("ppo_s*")):
        if not (run / "val.csv").exists():
            continue
        v = pd.read_csv(run / "val.csv")
        cfg = json.loads((run / "config.json").read_text())
        rows.append({"config": d.name.removeprefix("tune_"), "seed": cfg["seed"],
                     "hp": " ".join(cfg["hp_overrides"]) or "(defaults)",
                     **v[["reward", "kg_co2", "kwh", "comfort_dh", "viol_pct"]].mean().to_dict()})
df = pd.DataFrame(rows)
agg = (df.groupby(["config", "hp"]).agg(seeds=("seed", "count"), reward=("reward", "mean"),
                                        reward_min=("reward", "min"), kg_co2=("kg_co2", "mean"),
                                        comfort_dh=("comfort_dh", "mean"), viol_pct=("viol_pct", "mean"))
       .reset_index().sort_values("reward", ascending=False))
Path(RES / "tables").mkdir(exist_ok=True)
agg.to_csv(RES / "tables/ppo_tuning.csv", index=False)
print(agg.round(3).to_string(index=False))
best = agg.iloc[0]
(RES / "ppo_selected.json").write_text(json.dumps({"config": best.config, "hp": best.hp,
                                                    "val_reward": best.reward}, indent=2))
print(f"\nselected: {best.config} ({best.hp})")
