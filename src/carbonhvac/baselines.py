"""Rule-based controllers, weakest to strongest.

All three use the same thermostat primitive: pick the power level whose predicted next
temperature is closest to the target (an ideal staged thermostat). They differ only in
which target they choose and when.
"""
from dataclasses import dataclass


def track(env, target: float | None) -> int:
    if target is None:
        return 0
    preds = [env.predict_next_t(a) for a in range(env.action_space.n)]
    return min(range(len(preds)), key=lambda a: (abs(preds[a] - target), a))


def _occupied_window(r, start=8.0, end=18.0):
    return (r["is_workday"] and start <= r["hour"] < end) or r["occ_sensor"] > 0


@dataclass
class FixedSetpoint:
    setpoint: float = 24.0
    name: str = "fixed_setpoint"

    def __call__(self, obs, env):
        return track(env, self.setpoint)


@dataclass
class OccupancySetback:
    """Cool to the setpoint from 1 h before the scheduled start, or whenever the sensor says occupied."""
    setpoint: float = 24.0
    lead_hours: float = 1.0
    name: str = "setback"

    def __call__(self, obs, env):
        r = env.raw_obs()
        on = _occupied_window(r, start=8.0 - self.lead_hours)
        return track(env, self.setpoint if on else None)


@dataclass
class CarbonHeuristic:
    """Setback plus carbon-reactive load shifting.

    When CI is below `ci_threshold` it cools to `low` (stores cooling in the thermal mass,
    including pre-cooling from `precool_start`); when CI is above, it lets the zone drift
    up to `high`. The threshold is tuned on the training period (scripts/tune_heuristic.py).
    """
    ci_threshold: float = 0.15
    low: float = 23.3
    high: float = 25.5
    precool_start: float = 5.0
    name: str = "carbon_heuristic"

    def __call__(self, obs, env):
        r = env.raw_obs()
        cheap = r["ci"] < self.ci_threshold
        if r["is_workday"] and self.precool_start <= r["hour"] < 8.0:
            if cheap:
                return track(env, self.low)
            return track(env, 24.0 if r["hour"] >= 7.0 else None)
        if _occupied_window(r):
            return track(env, self.low if cheap else self.high)
        return 0
