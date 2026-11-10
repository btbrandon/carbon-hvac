"""Exogenous profiles at 15-minute resolution for one year: weather, gains, occupancy, price, CI.

`synthetic_profiles` produces calibrated stand-ins so the code never blocks on data:
  - outdoor temperature / solar: Singapore-like (25-33 degC, diurnal, rainy days)
  - price: USEP-like wholesale price in SGD/kWh with a daytime demand bump and spikes
  - carbon intensity: UK-grid-like kg/kWh (wind-driven daily level, midday solar dip,
    evening peak). Singapore's grid is ~95% gas and its CI is nearly flat, which leaves
    nothing to shift, so we substitute a variable grid (stated in the report).
`load_profiles(source="csv")` replaces any series for which data/<name>.csv exists.
"""
from dataclasses import dataclass, replace
from pathlib import Path
import warnings

import numpy as np

STEPS_PER_DAY = 96
DT = 0.25  # hours
N_DAYS = 365
FIRST_WEEKDAY = 3  # 1 Jan 2026 is a Thursday (Mon=0)
TEST_FIRST_DAY = 273  # 1 Oct: train on Jan-Sep, test on Oct-Dec


@dataclass(frozen=True)
class Profiles:
    t_out: np.ndarray   # degC
    solar: np.ndarray   # kW solar heat gain into the zone
    occ: np.ndarray     # true occupancy fraction in [0, 1]
    price: np.ndarray   # SGD/kWh
    ci: np.ndarray      # kg CO2 / kWh

    @property
    def n_steps(self) -> int:
        return len(self.t_out)

    @property
    def n_days(self) -> int:
        return self.n_steps // STEPS_PER_DAY

    def internal_gain(self, idx):
        """Internal gains in kW: base load plus people/equipment/lighting scaled by occupancy."""
        return 1.5 + 12.0 * self.occ[idx]


def weekday(day: int) -> int:
    return (day + FIRST_WEEKDAY) % 7


def split_days(split: str, n_days: int = N_DAYS) -> np.ndarray:
    days = np.arange(n_days)
    if split == "train":
        return days[days < TEST_FIRST_DAY]
    if split == "test":
        return days[days >= TEST_FIRST_DAY]
    if split == "all":
        return days
    raise ValueError(f"unknown split {split!r}")


def _ar1(rng, n, phi, sigma):
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal(0, sigma)
    return x


def synthetic_profiles(seed: int = 0, n_days: int = N_DAYS) -> Profiles:
    rng = np.random.default_rng(seed)
    n = n_days * STEPS_PER_DAY
    step = np.arange(n)
    hour = (step % STEPS_PER_DAY) * DT
    day = step // STEPS_PER_DAY
    doy = day.astype(float)

    # --- weather (Singapore-like) ---
    rainy_d = rng.random(n_days) < 0.3
    mean_d = 28.0 + 0.6 * np.sin(2 * np.pi * (np.arange(n_days) - 120) / 365) \
        + _ar1(rng, n_days, 0.7, 0.5) - 1.0 * rainy_d
    amp_d = np.where(rainy_d, 1.8, 3.2)
    t_out = mean_d[day] + amp_d[day] * np.cos(2 * np.pi * (hour - 15) / 24) \
        + _ar1(rng, n, 0.95, 0.08)

    cloud_d = np.where(rainy_d, 0.4, rng.uniform(0.7, 1.0, n_days))
    solar = 6.0 * np.clip(np.sin(np.pi * (hour - 7) / 12), 0, None) * cloud_d[day]

    # --- occupancy: weekdays ~08:00-18:00, jittered arrival/departure, lunch dip, holidays ---
    occ = np.zeros(n)
    holiday = rng.random(n_days) < 0.03
    for d in range(n_days):
        if weekday(d) >= 5 or holiday[d]:
            continue
        start = 8.0 + rng.integers(-2, 3) * DT
        end = 18.0 + rng.integers(-2, 4) * DT
        h = hour[d * STEPS_PER_DAY:(d + 1) * STEPS_PER_DAY]
        frac = np.where((h >= start) & (h < end), rng.uniform(0.8, 1.0), 0.0)
        frac = np.where((h >= 12) & (h < 13) & (frac > 0), 0.6, frac)
        occ[d * STEPS_PER_DAY:(d + 1) * STEPS_PER_DAY] = frac

    # --- price (USEP-like, SGD/kWh) ---
    demand = np.exp(-0.5 * ((hour - 14) / 4.0) ** 2) + 0.6 * np.exp(-0.5 * ((hour - 19.5) / 1.5) ** 2)
    level_d = np.exp(_ar1(rng, n_days, 0.8, 0.08))
    price = (0.14 + 0.09 * demand) * level_d[day] * (1 + _ar1(rng, n, 0.9, 0.02))
    spikes = rng.random(n) < 0.004
    price = np.where(spikes, price * rng.uniform(2, 4, n), price)

    # --- carbon intensity (UK-like, kg/kWh) ---
    # wind output drifts on a ~8 h timescale, so CI swings are not tied to the clock
    wind = np.clip(0.5 + _ar1(rng, n, 0.97, 0.06), 0, 1)
    solar_season = 0.6 + 0.4 * np.sin(2 * np.pi * (doy - 80) / 365)
    solar_dip = np.clip(np.sin(np.pi * (hour - 6) / 14), 0, None) * solar_season
    evening = np.exp(-0.5 * ((hour - 18.5) / 1.5) ** 2)
    ci = 0.25 - 0.16 * wind - 0.06 * solar_dip + 0.05 * evening
    ci = np.clip(ci, 0.03, 0.40)

    return Profiles(t_out=t_out, solar=solar, occ=occ, price=price, ci=ci)


def _load_csv_series(path: Path, n: int) -> np.ndarray:
    """CSV with columns timestamp,value at any resolution -> 15-min array of length n."""
    import pandas as pd
    df = pd.read_csv(path, parse_dates=[0], index_col=0)
    s = df.iloc[:, 0].astype(float).resample("15min").mean().interpolate().ffill().bfill()
    if len(s) < n:
        raise ValueError(f"{path} covers {len(s)} 15-min steps, need {n}")
    return s.to_numpy()[:n]


def load_profiles(source: str = "synthetic", data_dir: str | Path = "data", seed: int = 0) -> Profiles:
    base = synthetic_profiles(seed)
    if source == "synthetic":
        return base
    if source != "csv":
        raise ValueError(f"unknown source {source!r}")
    data_dir = Path(data_dir)
    updates = {}
    for name in ("t_out", "price", "ci"):
        path = data_dir / f"{name}.csv"
        if path.exists():
            updates[name] = _load_csv_series(path, base.n_steps)
        else:
            warnings.warn(f"{path} not found, using synthetic {name}")
    return replace(base, **updates)


def apply_regimes(p: Profiles, price_regime: str = "real", ci_regime: str = "real",
                  ci_shift_hours: float = 0.0) -> Profiles:
    """Scenario transforms used in the robustness experiments (§7 scenarios)."""
    hour = (np.arange(p.n_steps) % STEPS_PER_DAY) * DT
    price, ci = p.price, p.ci
    if price_regime == "flat":
        price = np.full_like(price, price.mean())
    elif price_regime == "tou":
        price = np.where((hour >= 9) & (hour < 21), 0.26, 0.14)
    elif price_regime != "real":
        raise ValueError(price_regime)

    if ci_regime == "flat":
        ci = np.full_like(ci, ci.mean())
    elif ci_regime == "anticorrelated":
        # CI moves opposite to price, so cutting cost and cutting carbon conflict
        z = (p.price - p.price.mean()) / p.price.std()
        ci = np.clip(ci.mean() - 0.8 * ci.std() * z, 0.03, 0.40)
    elif ci_regime != "real":
        raise ValueError(ci_regime)

    if ci_shift_hours:
        ci = np.roll(ci, int(round(ci_shift_hours / DT)))
    return replace(p, price=price, ci=ci)
