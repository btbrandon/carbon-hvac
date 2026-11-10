"""Gymnasium environment: carbon-aware HVAC control of one office zone, one day per episode.

MDP (see docs/report/s4_formulation.md):
  state   T_in, T_out, occupancy sensor, price, CI, sin/cos hour, weekday, k-step forecasts
  action  HVAC electrical power level in {0, 1/(n-1), ..., 1} * P_max
  reward  -(lam_E*E + lam_C*CI*E + lam_$*price*E + lam_K*discomfort), discomfort only when occupied
Dynamics are deterministic given the exogenous series, so stochasticity comes from the
day drawn, the initial temperature, the occupancy sensor and the forecast errors.
"""
from dataclasses import dataclass, field, asdict

import gymnasium as gym
import numpy as np

from .data import (DT, STEPS_PER_DAY, Profiles, apply_regimes, load_profiles,
                   split_days, weekday)
from .thermal import ThermalParams, step_temperature


@dataclass
class EnvConfig:
    split: str = "train"
    control_minutes: int = 15          # 15 or 60 (ablation); simulation always runs at 15 min
    n_actions: int = 5
    comfort_low: float = 23.0
    comfort_high: float = 26.0
    lam_energy: float = 0.0            # per kWh
    lam_carbon: float = 1.0            # per kg CO2
    lam_cost: float = 0.5              # per SGD
    lam_comfort: float = 5.0           # per degC-hour (or degC^2-hour if squared)
    comfort_penalty: str = "linear"    # "linear" | "squared"
    obs_ci: bool = True
    obs_price: bool = True
    obs_time: bool = True
    forecast_hours: tuple = (1, 2, 4, 8)
    forecast_noise: float = 0.0        # relative std per sqrt(hour) of lead time
    occ_flip_prob: float = 0.0         # occupancy-sensor error rate
    reward_occupancy: str = "true"     # "true" | "sensor": which occupancy the comfort reward uses
    price_regime: str = "real"
    ci_regime: str = "real"
    ci_shift_hours: float = 0.0
    data_source: str = "synthetic"
    data_seed: int = 0
    thermal: ThermalParams = field(default_factory=ThermalParams)

    def to_dict(self):
        return asdict(self)


def comfort_deviation(t, low, high):
    return np.maximum(t - high, 0.0) + np.maximum(low - t, 0.0)


def simulate_interval(cfg: EnvConfig, prof: Profiles, idx0: int, n_sub: int, t_in, level):
    """Run one control interval (n_sub x 15 min) with a fixed action.

    Vectorised over t_in so the oracle DP can evaluate a whole temperature grid at once.
    Returns the end temperature, per-interval energy/carbon/cost, and per-substep comfort
    deviations (shape (n_sub, *t_in.shape)) so callers can weight them by any occupancy.
    """
    params = cfg.thermal
    p_elec = level / (cfg.n_actions - 1) * params.p_max
    t = np.asarray(t_in, dtype=float)
    energy = carbon = cost = 0.0
    devs = []
    for j in range(n_sub):
        i = (idx0 + j) % prof.n_steps
        q_gain = prof.internal_gain(i) + prof.solar[i]
        t = step_temperature(params, t, prof.t_out[i], q_gain, p_elec, DT)
        e = p_elec * DT
        energy += e
        carbon += e * prof.ci[i]
        cost += e * prof.price[i]
        devs.append(comfort_deviation(t, cfg.comfort_low, cfg.comfort_high))
    return t, energy, carbon, cost, np.stack(devs)


def discomfort(cfg: EnvConfig, devs, occ_weights):
    """Comfort penalty over an interval: sum_j occ_j * f(dev_j) * dt."""
    occ = (np.asarray(occ_weights) > 0).astype(float).reshape((-1,) + (1,) * (devs.ndim - 1))
    f = devs ** 2 if cfg.comfort_penalty == "squared" else devs
    return (occ * f).sum(axis=0) * DT


def step_cost(cfg: EnvConfig, energy, carbon, cost, pen):
    return (cfg.lam_energy * energy + cfg.lam_carbon * carbon
            + cfg.lam_cost * cost + cfg.lam_comfort * pen)


class CarbonHVACEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None, profiles: Profiles | None = None):
        super().__init__()
        self.cfg = config or EnvConfig()
        if 60 % self.cfg.control_minutes or self.cfg.control_minutes % 15:
            raise ValueError("control_minutes must be 15, 30 or 60")
        self.n_sub = self.cfg.control_minutes // 15
        self.horizon = STEPS_PER_DAY // self.n_sub
        base = profiles or load_profiles(self.cfg.data_source, seed=self.cfg.data_seed)
        self.prof = apply_regimes(base, self.cfg.price_regime, self.cfg.ci_regime,
                                  self.cfg.ci_shift_hours)
        self.days = split_days(self.cfg.split, self.prof.n_days)
        self.feature_names = self._feature_names()
        self.action_space = gym.spaces.Discrete(self.cfg.n_actions)
        self.observation_space = gym.spaces.Box(-10, 10, (len(self.feature_names),), np.float32)

    # ---------- observation ----------
    def _feature_names(self):
        c = self.cfg
        names = ["t_in", "t_out", "occ_sensor"]
        if c.obs_price:
            names.append("price")
        if c.obs_ci:
            names.append("ci")
        if c.obs_time:
            names += ["hour_sin", "hour_cos", "weekday"]
        for h in c.forecast_hours:
            names.append(f"t_out_f{h}h")
            if c.obs_price:
                names.append(f"price_f{h}h")
            if c.obs_ci:
                names.append(f"ci_f{h}h")
        return names

    @staticmethod
    def _scale(name, v):
        if name.startswith("t_in"):
            return (v - 24.5) / 3.0
        if name.startswith("t_out"):
            return (v - 28.5) / 3.0
        if name.startswith("price"):
            return (v - 0.2) / 0.1
        if name.startswith("ci"):
            return (v - 0.17) / 0.08
        return v

    def _forecast(self, series, idx, h):
        v = series[(idx + int(h / DT)) % self.prof.n_steps]
        if self.cfg.forecast_noise:
            v = v * (1 + self.cfg.forecast_noise * np.sqrt(h) * self._rng.normal())
        return v

    def raw_obs(self) -> dict:
        """Unscaled current observation, also used by the rule-based controllers."""
        i = self.idx
        hour = (self.k * self.n_sub * DT) % 24
        return {
            "t_in": self.t_in, "t_out": self.prof.t_out[i], "occ_sensor": self.occ_sensor[self.k * self.n_sub],
            "price": self.prof.price[i], "ci": self.prof.ci[i], "hour": hour,
            "weekday": weekday(self.day), "is_workday": float(weekday(self.day) < 5),
        }

    def _obs(self):
        r = self.raw_obs()
        vals = {"t_in": r["t_in"], "t_out": r["t_out"], "occ_sensor": r["occ_sensor"],
                "price": r["price"], "ci": r["ci"],
                "hour_sin": np.sin(2 * np.pi * r["hour"] / 24),
                "hour_cos": np.cos(2 * np.pi * r["hour"] / 24),
                "weekday": r["is_workday"]}
        for h in self.cfg.forecast_hours:
            vals[f"t_out_f{h}h"] = self._forecast(self.prof.t_out, self.idx, h)
            vals[f"price_f{h}h"] = self._forecast(self.prof.price, self.idx, h)
            vals[f"ci_f{h}h"] = self._forecast(self.prof.ci, self.idx, h)
        return np.array([self._scale(n, vals[n]) for n in self.feature_names], dtype=np.float32)

    # ---------- episode ----------
    @property
    def idx(self) -> int:
        return self.day * STEPS_PER_DAY + self.k * self.n_sub

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.day = int(options["day"]) if "day" in options else int(self.np_random.choice(self.days))
        noise_seed = options.get("noise_seed", int(self.np_random.integers(2**31)))
        self._rng = np.random.default_rng(noise_seed)
        self.k = 0
        self.t_in = float(options.get("t_init", self._rng.uniform(25.5, 28.0)))
        sl = slice(self.day * STEPS_PER_DAY, (self.day + 1) * STEPS_PER_DAY)
        self.occ_true = self.prof.occ[sl] > 0
        flips = self._rng.random(STEPS_PER_DAY) < self.cfg.occ_flip_prob
        self.occ_sensor = (self.occ_true ^ flips).astype(float)
        return self._obs(), {"day": self.day}

    def predict_next_t(self, level: int) -> float:
        t, *_ = simulate_interval(self.cfg, self.prof, self.idx, self.n_sub, self.t_in, level)
        return float(t)

    def step(self, action):
        level = int(action)
        s = slice(self.k * self.n_sub, (self.k + 1) * self.n_sub)
        t, energy, carbon, cost, devs = simulate_interval(
            self.cfg, self.prof, self.idx, self.n_sub, self.t_in, level)
        occ_r = self.occ_true[s] if self.cfg.reward_occupancy == "true" else self.occ_sensor[s]
        pen_reward = float(discomfort(self.cfg, devs, occ_r))
        info = {
            "hour": self.k * self.n_sub * DT, "action": level, "t_in": float(t),
            "t_out": self.prof.t_out[self.idx], "ci": self.prof.ci[self.idx],
            "price": self.prof.price[self.idx], "occupied": bool(self.occ_true[s].any()),
            "energy_kwh": energy, "carbon_kg": carbon, "cost_sgd": cost,
            # metrics always use true occupancy, whatever the reward uses (§7 comfort scoring)
            "comfort_dh": float((devs * self.occ_true[s]).sum() * DT),
            "max_dev": float((devs.reshape(len(devs), -1).max(axis=1) * self.occ_true[s]).max()),
            "pen_reward": pen_reward,
        }
        reward = -float(step_cost(self.cfg, energy, carbon, cost, pen_reward))
        self.t_in = float(t)
        self.k += 1
        terminated = self.k >= self.horizon
        obs = np.zeros(self.observation_space.shape, np.float32) if terminated else self._obs()
        return obs, reward, terminated, False, info
